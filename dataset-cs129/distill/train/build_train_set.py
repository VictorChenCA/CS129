import json
import argparse
from pathlib import Path
from typing import Any
import tempfile
import os
import re
import sys

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pseudo_rationales.tokenizer import load_math_dataset
from gcs_utils import parse_gcs_uri, upload_dir_to_gcs

def load_pseudo_rationales_jsonl(jsonl_path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Load the JSONL created by `extract_pseudo_rationales.py`.

    Returns a mapping keyed by (subset, problem_text) -> row.
    (We include `subset` so rationales from different subsets can’t collide.)
    """
    path = Path(jsonl_path)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            subset = row.get("subset")
            prob = row.get("problem")
            if not subset or not prob:
                continue
            by_key[(subset, prob)] = row
    return by_key



def _download_from_gcs(gcs_uri: str, *, project: str) -> Path:
    """Download a single file from GCS (unique to this script)."""
    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "To read from GCS, install google-cloud-storage: `pip install google-cloud-storage`"
        ) from e

    bucket_name, blob_name = parse_gcs_uri(gcs_uri)
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    tmp_dir = Path(tempfile.mkdtemp(prefix="pseudo_rationales_dl_"))
    local_path = tmp_dir / Path(blob_name).name
    blob.download_to_filename(str(local_path))
    return local_path

# dataset mainly persisted via google cloud storage. Check there for past built training sets
def persist_dataset(ds: Dataset, out_path: str, *, gcp_project: str, overwrite: bool = False) -> str:
    """
    Persist a processed HF Dataset either locally or to GCS.

    - Local: saves to a directory at `out_path`
    - GCS: saves to a temp local directory, uploads it to the `gs://...` prefix

    Returns the location string you should use later (local path or gs:// prefix).
    """
    if  str(out_path).startswith("gs://"):
        tmp_dir = Path(tempfile.mkdtemp(prefix="processed_ds_"))
        local_save_dir = tmp_dir / "dataset"
        ds.save_to_disk(str(local_save_dir))
        upload_dir_to_gcs(local_save_dir, out_path, project=gcp_project)
        return out_path

    out_dir = Path(out_path)
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing path: {out_dir} (pass --overwrite)")
        # datasets.save_to_disk requires an empty/non-existent dir
        for root, dirs, files in os.walk(out_dir, topdown=False):
            for f in files:
                Path(root, f).unlink(missing_ok=True)
            for d in dirs:
                Path(root, d).rmdir()
        out_dir.rmdir()

    ds.save_to_disk(str(out_dir))
    return str(out_dir)


def build_student_messages(problem_text: str, rationale: str | None, answer_text: str) -> list[dict[str, str]]:
    """
    Builds chat-style messages for Qwen2.5-Instruct student SFT.

    User: sees only the problem.
    Assistant: emits rationale + final answer (from dataset `answer` column).
    """
    user_content = f"Solve the following math problem.\n\nProblem:\n{problem_text}\n"

    # user content tokens ignored for loss and masked out with [-100]

    # assistant content used as target label(s) for loss
    rat = "" if rationale is None else rationale
    assistant_content = f"<rationale>{rat}</rationale>\n<answer>{answer_text}</answer>"


    return [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]


def ensure_answer_column(ds: Dataset) -> Dataset:
    """
    Ensure examples expose an explicit `answer` field from the merged base dataset.
    Preference order:
      1) answer
      2) final_answer
      3) parsed from solution (inner payload of last \boxed{...})
    """
    def _extract_boxed_payload(text: Any) -> str:
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        saw_boxed = "\\boxed" in text
        starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
        for s in reversed(starts):
            brace_idx = text.find("{", s)
            if brace_idx < 0:
                continue
            depth = 0
            end_idx = -1
            for i in range(brace_idx, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end_idx = i
                        break
            if end_idx > brace_idx:
                payload = text[brace_idx + 1 : end_idx].strip()
                return payload

        # Handle compact form like "\boxed 2" (no braces).
        no_brace = re.findall(r"\\boxed\s+([^\s$.,;]+)", text)
        if no_brace:
            return no_brace[-1].strip()

        # If the source explicitly used boxed formatting but payload is empty/unparseable.
        if saw_boxed:
            return ""
        return text.strip()

    cols = set(ds.column_names)

    # ind answer from golden MATH set labels

    # hopefully answer column is contained within set
    if "answer" in cols:
        return ds

    # if "final_answer" in cols:
    #     return ds.map(lambda ex: {"answer": ex["final_answer"]})

    # most probable and just extracts boxed answer using heinous method above
    if "solution" in cols:
        return ds.map(lambda ex: {"answer": _extract_boxed_payload(ex["solution"])})

    raise KeyError(
        f"Could not find an answer-like column in dataset. Available columns: {sorted(cols)}"
    )


def attach_rationales(ds: Dataset, rationales_by_key: dict[tuple[str, str], dict[str, Any]]) -> Dataset:
    """Adds a `rationale` column by matching on (`subset`, `problem`)."""

    def _fallback_rationale_from_raw(raw: Any) -> str | None:
        """
        If the teacher didn't follow the <rationale>...</rationale> format, we may still
        have a useful explanation in `raw_response`. Use it as a fallback.
        """
        if raw is None:
            return None
        if not isinstance(raw, str):
            raw = str(raw)
        raw = raw.strip()
        if not raw:
            return None

        # Prefer extracting inner text if tags exist.
        m = re.search(r"<rationale>(.*?)</rationale>", raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            inner = m.group(1).strip()
            return inner if inner else None
        return raw

    def _add(ex):
        row = rationales_by_key.get((ex.get("subset"), ex["problem"]))
        if row is None:
            return {"rationale": None}

        rat = row.get("rationale")
        if isinstance(rat, str) and rat.strip():
            return {"rationale": rat}

        # rationale is None/empty -> fallback to raw_response
        fallback = _fallback_rationale_from_raw(row.get("raw_response"))
        return {"rationale": fallback}

    
    # either map raw rationale onto each example if rationale tags are not present, or extract rationale from between rationale tags.
    return ds.map(_add)


# SHOULD NOT BE NECESSARY:
def filter_missing_rationales(ds: Dataset) -> Dataset:
    """
    Keep only examples with a non-empty rationale.

    The pseudo-rationales join can produce:
      - None (no match in JSONL)
      - "" (match exists but rationale missing/empty)
    We drop both so the final training set contains only teacher-rationale examples.
    """

    def _keep(ex):
        r = ex.get("rationale")
        if r is None:
            return False
        if isinstance(r, str) and r.strip() == "":
            return False
        return True

    return ds.filter(_keep)


def tokenize_for_causal_lm_sft(
    ds: Dataset,
    student_model_name: str,
    max_length: int = 2048,
) -> Dataset:
    """
    Creates `input_ids`, `attention_mask`, `labels` for causal LM SFT with prompt-masking.

    Labels:
      - prompt tokens (everything up through the end of the user message) are set to -100
      - assistant tokens are the target ids
    """
    tok = AutoTokenizer.from_pretrained(student_model_name, use_fast=True)

    def _tok(ex):
        answer_text = ex.get("answer")

        # quick sanity check, should now be sorted out via boxed extraction
        if answer_text is None:
            raise KeyError(
                "Expected `answer` field in dataset examples. "
                "Use a dataset/subset configuration that includes `answer`."
            )


        messages = build_student_messages(ex["problem"], ex.get("rationale"), answer_text)

        # add assistant prompt token and only apply template to user message with the token
        prompt_text = tok.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)

        # apply chat template to full message so no assistant token needed
        full_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = tok(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
        full = tok(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
        input_ids = full["input_ids"]

        labels = input_ids.copy()
        n_prompt = min(len(prompt_ids), len(labels))
        labels[:n_prompt] = [-100] * n_prompt

        return {
            "messages": messages,
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    return ds.map(_tok)


# append rationales to concatenated dataset then tokenize
def build_training_dataset(
    student_model_name: str,
    pseudo_rationales_jsonl: str | Path,
    dataset_name: str = "EleutherAI/hendrycks_math",
    subset: str = "algebra",
    subsets: list[str] | None = None,
    split: str = "train",
    max_length: int = 2048,
    drop_missing_rationales: bool = False,
) -> Dataset:
    """
    End-to-end data prep:
      1) load math dataset
      2) load pseudo rationales JSONL
      3) join (adds `rationale`)
      4) tokenize into SFT fields for a causal LM student
    """
    if subsets:
        parts = []
        for s in subsets:
            ds = load_math_dataset(dataset_name=dataset_name, subset=s, split=split)
            ds = ds.map(lambda ex, _s=s: {"subset": _s})
            parts.append(ds)
        base = concatenate_datasets(parts)
    else:
        base = load_math_dataset(dataset_name=dataset_name, subset=subset, split=split)
        base = base.map(lambda ex: {"subset": subset})

    rats = load_pseudo_rationales_jsonl(pseudo_rationales_jsonl)
    joined = attach_rationales(base, rats)
    joined = ensure_answer_column(joined)
    if drop_missing_rationales:
        joined = filter_missing_rationales(joined)
    tokenized = tokenize_for_causal_lm_sft(joined, student_model_name=student_model_name, max_length=max_length)
    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Prepare a Qwen2.5 SFT dataset from problems + QwQ rationales.")
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--pseudo_rationales_jsonl", type=str, required=True)
    parser.add_argument("--gcp_project", type=str, default="cs224n-dapo-distill")
    parser.add_argument("--subset", type=str, default="algebra")
    parser.add_argument(
        "--subsets",
        type=str,
        default=None,
        help='Optional comma-separated subset list (e.g. "algebra,geometry,prealgebra"). If set, overrides --subset.',
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--print_n", type=int, default=3)
    parser.add_argument(
        "--drop_missing_rationales",
        action="store_true",
        help="If set, drop examples where rationale is missing/empty after join (default keeps all).",
    )
    parser.add_argument(
        "--out_dataset",
        type=str,
        default=None,
        help="Optional. Persist processed dataset to a local dir or GCS prefix (gs://bucket/path/).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting local --out_dataset path.")
    args = parser.parse_args()

    subset_list = None
    if args.subsets:
        subset_list = [s.strip() for s in args.subsets.split(",") if s.strip()]

    pseudo_path = args.pseudo_rationales_jsonl
    if str(pseudo_path).startswith("gs://"):
        pseudo_path = str(_download_from_gcs(pseudo_path, project=args.gcp_project))

    ds = build_training_dataset(
        student_model_name=args.student_model_name,
        pseudo_rationales_jsonl=pseudo_path,
        subset=args.subset,
        subsets=subset_list,
        split=args.split,
        max_length=args.max_length,
        drop_missing_rationales=args.drop_missing_rationales,
    )

    # quick sanity check
    for i in range(args.print_n):
        ex = ds[i]
        print(
            {
                "problem": ex["problem"][:120] + ("..." if len(ex["problem"]) > 120 else ""),
                "has_rationale": ex.get("rationale") is not None,
                "input_ids_len": len(ex["input_ids"]),
                "labels_ignore_count": sum(1 for t in ex["labels"] if t == -100),
            }
        )

    if args.out_dataset:
        loc = persist_dataset(ds, args.out_dataset, gcp_project=args.gcp_project, overwrite=args.overwrite)
        print(f"Saved processed dataset to: {loc}")


if __name__ == "__main__":
    main()
