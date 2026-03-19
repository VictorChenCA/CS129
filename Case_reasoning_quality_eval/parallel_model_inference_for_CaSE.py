"""Generate CaSE-ready JSONL outputs for the four final algebra models on Modal.

Each run handles one model so multiple `modal run --detach ... --model-key ...`
invocations can execute in parallel on separate GPUs.

Output rows are compatible with `Case_reasoning_quality_eval/case_eval.py`:
  - `id`
  - `question`
  - `rationale`
  - `answer`

Extra fields preserve the full raw generation, prompt, and prompt provenance.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


DATASET_HF = "EleutherAI/hendrycks_math"
DATASET_SUBSET = "algebra"
DATASET_SPLIT = "test"
DEFAULT_LIMIT = 100
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_PROMPT_MAX_LENGTH = 8192
DEFAULT_BATCH_SIZE = 1

CHECKPOINT_PROJECT = "cs224n-project"
CHECKPOINT_BUCKET = "checkpoints-cs224n"
BASE_GCS_PREFIX = f"gs://{CHECKPOINT_BUCKET}/final_best"
REAL_FINAL_BEST_GCS_PREFIX = f"gs://{CHECKPOINT_BUCKET}/real_final_best/v3_optimized"
DEFAULT_RESULTS_GCS_PREFIX = (
    f"{BASE_GCS_PREFIX}/case_reasoning_quality_eval/jsonl_model_output_for_CaSE"
)

OUTPUTS_VOLUME_NAME = "case-reasoning-quality-eval-outputs"
OUTPUTS_DIR = "/outputs/case_reasoning_quality_eval"
DEFAULT_MODAL_SECRET_NAME = "joseph-cs224n-project"

RL_SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve problems step by step, "
    "showing your reasoning clearly. Put your final answer in \\boxed{}."
)
RL_USER_TEMPLATE = "Solve this math problem:\n\n{problem}"
DISTILLED_USER_TEMPLATE = "Solve the following math problem.\n\nProblem:\n{problem}\n"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_path: str
    prompt_style: str
    prompt_source_files: tuple[str, ...]
    prompt_source_note: str
    expected_output_format: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "base_qwen_oob": ModelSpec(
        key="base_qwen_oob",
        display_name="Qwen2.5-0.5B Out-of-Box",
        model_path=DEFAULT_BASE_MODEL,
        prompt_style="distilled_multitask",
        prompt_source_files=(
            "distilling_step_by_step-gemini/train/train_model_multitask_stratified.py",
            "distilling_step_by_step-gemini/eval/eval_stratified_heldout_modal.py",
        ),
        prompt_source_note=(
            "Out-of-box Qwen baseline evaluated with the same user-only prompt style "
            "as the distilled model for direct CaSE comparison."
        ),
        expected_output_format="free-form reasoning; answer extracted from <answer>, \\boxed{}, or final line",
    ),
    "algebra_distilled_multitask": ModelSpec(
        key="algebra_distilled_multitask",
        display_name="Algebra Distilled Multitask",
        model_path=(
            "gs://cs224n-project-data/distillation_model_checkpoints/final_report_subject_sweep/"
            "final_report_subject_sweep_local_20260314_042600_train_algebra/final_model"
        ),
        prompt_style="distilled_multitask",
        prompt_source_files=(
            "distilling_step_by_step-gemini/train/train_model_multitask_stratified.py",
            "distilling_step_by_step-gemini/eval/eval_stratified_heldout_modal.py",
        ),
        prompt_source_note=(
            "Final report algebra-only multitask-distilled checkpoint "
            "(run final_report_subject_sweep_local_20260314_042600_train_algebra)."
        ),
        expected_output_format="<rationale>...</rationale>\\n<answer>...</answer>",
    ),
    "sft": ModelSpec(
        key="sft",
        display_name="SFT",
        model_path=f"{REAL_FINAL_BEST_GCS_PREFIX}/sft_qwen_math/merged",
        prompt_style="rl_math_tutor",
        prompt_source_files=(
            "reinforcement_learning/sft_trainer.py",
            "reinforcement_learning/train_targeted.py",
        ),
        prompt_source_note=(
            "SFT training uses the RL math-tutor system prompt plus "
            "`Solve this math problem:` as the user turn."
        ),
        expected_output_format="free-form reasoning ending with \\boxed{...}",
    ),
    "grpo": ModelSpec(
        key="grpo",
        display_name="SFT+GRPO",
        model_path=f"{REAL_FINAL_BEST_GCS_PREFIX}/grpo_qwen_math_merged",
        prompt_style="rl_math_tutor",
        prompt_source_files=(
            "reinforcement_learning/train_targeted.py",
            "reinforcement_learning/modal_evaluate.py",
        ),
        prompt_source_note=(
            "GRPO continues from the same RL algebra prompt family as SFT, "
            "and evaluation also uses the same system/user template."
        ),
        expected_output_format="free-form reasoning ending with \\boxed{...}",
    ),
    "targeted_dapo": ModelSpec(
        key="targeted_dapo",
        display_name="SFT+GRPO+TargetedDAPO",
        model_path=f"{REAL_FINAL_BEST_GCS_PREFIX}/targeted_dapo_qwen_math/merged",
        prompt_style="rl_math_tutor",
        prompt_source_files=(
            "reinforcement_learning/train_targeted.py",
            "reinforcement_learning/modal_evaluate.py",
        ),
        prompt_source_note=(
            "Targeted DAPO repair is trained and evaluated with the same RL "
            "math-tutor prompt family as SFT/GRPO."
        ),
        expected_output_format="free-form reasoning ending with \\boxed{...}",
    ),
    "distilled_multitask": ModelSpec(
        key="distilled_multitask",
        display_name="Distilled Multitask",
        model_path=(
            f"{BASE_GCS_PREFIX}/distilled_models/"
            "multi_task_algebra_distilled_build_train_set_final_20260309"
        ),
        prompt_style="distilled_multitask",
        prompt_source_files=(
            "distilling_step_by_step-gemini/train/train_model_multitask_stratified.py",
            "distilling_step_by_step-gemini/eval/eval_stratified_heldout_modal.py",
        ),
        prompt_source_note=(
            "The multitask distilled model is trained with a user-only prompt "
            "and generates `<rationale>...</rationale>` plus `<answer>...</answer>`."
        ),
        expected_output_format="<rationale>...</rationale>\\n<answer>...</answer>",
    ),
}


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.49.0",
        "datasets",
        "peft",
        "accelerate",
        "huggingface_hub",
        "google-cloud-storage",
    )
)

app = modal.App("case-reasoning-quality-inference")
outputs_volume = modal.Volume.from_name(OUTPUTS_VOLUME_NAME, create_if_missing=True)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[len("gs://") :]
    bucket, _, blob = no_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def now_utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_service_account_info() -> dict[str, Any] | None:
    for key in (
        "SERVICE_ACCOUNT_JSON",
        "CHECKPOINT_SERVICE_ACCOUNT_JSON",
        "JOSEPH_SERVICE_ACCOUNT_JSON",
        "SERVICE_ACCOUNT_JSON_JOSEPH",
    ):
        raw_json = os.environ.get(key, "").strip()
        if raw_json:
            return json.loads(raw_json)
    return None


def build_storage_client(project: str):
    from google.cloud import storage

    service_account_info = load_service_account_info()
    if service_account_info is not None:
        return storage.Client.from_service_account_info(service_account_info, project=project)
    return storage.Client(project=project)


def download_gcs_prefix_to_dir(gcs_prefix: str, local_dir: Path, *, project: str) -> None:
    client = build_storage_client(project)
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    if prefix and not prefix.endswith("/"):
        prefix = f"{prefix}/"

    bucket = client.bucket(bucket_name)
    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix) :] if prefix else blob.name
        dst = local_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dst))
        downloaded += 1

    if downloaded == 0:
        raise FileNotFoundError(f"No files found under {gcs_prefix}")


def upload_dir_to_gcs(local_dir: Path, gcs_uri_prefix: str, *, project: str) -> int:
    client = build_storage_client(project)
    bucket_name, blob_prefix = parse_gcs_uri(
        gcs_uri_prefix if gcs_uri_prefix.endswith("/") else f"{gcs_uri_prefix}/"
    )
    bucket = client.bucket(bucket_name)

    uploaded = 0
    local_dir = local_dir.resolve()
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        blob_name = f"{blob_prefix.rstrip('/')}/{rel}"
        bucket.blob(blob_name).upload_from_filename(str(path))
        uploaded += 1
    return uploaded


def extract_tag_payload(text: str, tag: str) -> str:
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def extract_boxed_payload(text: str) -> str:
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        brace_idx = text.find("{", start)
        if brace_idx < 0:
            continue
        depth = 0
        for idx in range(brace_idx, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_idx + 1 : idx].strip()

    no_brace = re.findall(r"\\boxed\s+([^\s$.,;]+)", text)
    if no_brace:
        return no_brace[-1].strip()
    return ""


def extract_answer_from_generation(text: str) -> str:
    tagged = extract_tag_payload(text, "answer")
    if tagged:
        return tagged

    boxed = extract_boxed_payload(text)
    if boxed:
        return boxed

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    last_line = lines[-1]
    phrase_match = re.search(
        r"(?:answer\s+is|result\s+is|equals?|=)\s*([^\s,;]+)",
        last_line,
        flags=re.IGNORECASE,
    )
    if phrase_match:
        return phrase_match.group(1).strip().rstrip(".,;")

    numbers = re.findall(r"[-+]?\d+(?:[./]\d+)?", last_line)
    if numbers:
        return numbers[-1]
    return last_line


def get_prompt_metadata(spec: ModelSpec) -> dict[str, Any]:
    if spec.prompt_style == "rl_math_tutor":
        return {
            "style": spec.prompt_style,
            "system_prompt": RL_SYSTEM_PROMPT,
            "user_template": RL_USER_TEMPLATE,
            "source_files": list(spec.prompt_source_files),
            "source_note": spec.prompt_source_note,
            "expected_output_format": spec.expected_output_format,
        }
    if spec.prompt_style == "distilled_multitask":
        return {
            "style": spec.prompt_style,
            "system_prompt": None,
            "user_template": DISTILLED_USER_TEMPLATE,
            "source_files": list(spec.prompt_source_files),
            "source_note": spec.prompt_source_note,
            "expected_output_format": spec.expected_output_format,
        }
    raise ValueError(f"Unsupported prompt style: {spec.prompt_style}")


def build_prompt(problem: str, tokenizer: Any, spec: ModelSpec) -> str:
    if spec.prompt_style == "rl_math_tutor":
        messages = [
            {"role": "system", "content": RL_SYSTEM_PROMPT},
            {"role": "user", "content": RL_USER_TEMPLATE.format(problem=problem)},
        ]
    elif spec.prompt_style == "distilled_multitask":
        messages = [
            {"role": "user", "content": DISTILLED_USER_TEMPLATE.format(problem=problem)},
        ]
    else:
        raise ValueError(f"Unsupported prompt style: {spec.prompt_style}")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model_and_tokenizer(
    model_path: str,
    *,
    base_model_name: str,
    project: str,
):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_path: Path | None = None
    hf_model_id: str | None = None

    if model_path.startswith("gs://"):
        tmp_dir = Path(tempfile.mkdtemp(prefix="case_model_dl_"))
        download_gcs_prefix_to_dir(model_path, tmp_dir, project=project)
        resolved_path = tmp_dir
    else:
        candidate = Path(model_path)
        if candidate.exists():
            resolved_path = candidate
        else:
            hf_model_id = model_path

    tokenizer_source = (
        str(resolved_path)
        if resolved_path is not None and (resolved_path / "tokenizer_config.json").exists()
        else (hf_model_id or base_model_name)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if resolved_path is not None and (resolved_path / "adapter_config.json").exists():
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=model_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(resolved_path))
        model = model.merge_and_unload()
        resolved_model_source = str(resolved_path)
        model_kind = "lora_adapter_merged_for_inference"
    else:
        model_source = str(resolved_path) if resolved_path is not None else hf_model_id
        if not model_source:
            raise ValueError(f"Unable to resolve model source: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=model_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        resolved_model_source = model_source
        model_kind = "full_model"

    model.eval()
    return model, tokenizer, resolved_model_source, model_kind


def load_algebra_test_rows(limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_HF, DATASET_SUBSET, split=DATASET_SPLIT)
    if limit >= 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        rows.append(
            {
                "row_index": idx,
                "problem": str(row.get("problem", "")),
                "solution": str(row.get("solution", "")),
                "type": str(row.get("type", "")),
                "id": row.get("id"),
            }
        )
    return rows


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 8,
    memory=32768,
    volumes={"/outputs": outputs_volume},
    secrets=[
        modal.Secret.from_name(DEFAULT_MODAL_SECRET_NAME),
        modal.Secret.from_name("googlecloud-secret"),
    ],
)
def run_case_model_inference(request: dict[str, Any]) -> dict[str, Any]:
    import torch

    model_key = str(request["model_key"]).strip()
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Unknown model_key={model_key!r}. Expected one of {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[model_key]

    run_name = str(request.get("run_name", "")).strip() or f"case_{now_utc_timestamp()}"
    limit = int(request.get("limit", DEFAULT_LIMIT))
    max_new_tokens = int(request.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    prompt_max_length = int(request.get("prompt_max_length", DEFAULT_PROMPT_MAX_LENGTH))
    batch_size = int(request.get("batch_size", DEFAULT_BATCH_SIZE))
    results_gcs_prefix = str(request.get("results_gcs_prefix", DEFAULT_RESULTS_GCS_PREFIX)).strip()
    checkpoint_project = str(request.get("checkpoint_project", CHECKPOINT_PROJECT))
    base_model_name = str(request.get("base_model_name", DEFAULT_BASE_MODEL))

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    rows = load_algebra_test_rows(limit)
    if not rows:
        raise RuntimeError("No algebra test rows were loaded.")

    model, tokenizer, resolved_model_source, model_kind = load_model_and_tokenizer(
        spec.model_path,
        base_model_name=base_model_name,
        project=checkpoint_project,
    )
    print(
        f"[start] model_key={spec.key} run_name={run_name} num_examples={len(rows)} "
        f"batch_size={batch_size} max_new_tokens={max_new_tokens}",
        flush=True,
    )

    output_dir = Path(OUTPUTS_DIR) / run_name / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / f"{model_key}_case_eval.jsonl"
    meta_json = output_dir / f"{model_key}_run_meta.json"

    prompt_meta = get_prompt_metadata(spec)
    model_device = next(model.parameters()).device
    records_written = 0

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            prompts = [
                build_prompt(row["problem"], tokenizer, spec)
                for row in batch_rows
            ]
            tokenized = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=prompt_max_length,
            )
            tokenized = {key: value.to(model_device) for key, value in tokenized.items()}
            input_len = int(tokenized["input_ids"].shape[1])

            with torch.inference_mode():
                generated = model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            continuations = generated[:, input_len:]
            decoded_outputs = tokenizer.batch_decode(continuations, skip_special_tokens=True)

            for batch_idx, raw_output in enumerate(decoded_outputs):
                row = batch_rows[batch_idx]
                example_number = start + batch_idx + 1
                prompt = prompts[batch_idx]
                generated_ids = continuations[batch_idx]
                generated_token_count = int(generated_ids.shape[0])
                hit_max_new_tokens = generated_token_count >= max_new_tokens

                predicted_answer = extract_answer_from_generation(raw_output)
                gold_answer = extract_boxed_payload(row["solution"])
                example_id = row["id"]
                if example_id in (None, ""):
                    example_id = f"{DATASET_SUBSET}_{DATASET_SPLIT}_{row['row_index']:04d}"

                record = {
                    "id": str(example_id),
                    "question": row["problem"],
                    "problem": row["problem"],
                    "rationale": raw_output,
                    "answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "raw_output": raw_output,
                    "reference_solution": row["solution"],
                    "model_key": spec.key,
                    "model_name": spec.display_name,
                    "model_path": spec.model_path,
                    "prompt": prompt,
                    "prompt_style": prompt_meta["style"],
                    "prompt_system": prompt_meta["system_prompt"],
                    "prompt_user_template": prompt_meta["user_template"],
                    "prompt_source_files": prompt_meta["source_files"],
                    "prompt_source_note": prompt_meta["source_note"],
                    "expected_output_format": prompt_meta["expected_output_format"],
                    "generation_config": {
                        "max_new_tokens": max_new_tokens,
                        "prompt_max_length": prompt_max_length,
                        "do_sample": False,
                        "num_beams": 1,
                    },
                    "generation_stats": {
                        "generated_token_count": generated_token_count,
                        "hit_max_new_tokens": hit_max_new_tokens,
                    },
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_written += 1
                print(
                    f"[example] {example_number}/{len(rows)} id={record['id']} "
                    f"generated_tokens={generated_token_count} "
                    f"hit_max_new_tokens={hit_max_new_tokens}",
                    flush=True,
                )

    run_meta = {
        "status": "complete",
        "run_name": run_name,
        "dataset": {
            "hf_dataset": DATASET_HF,
            "subset": DATASET_SUBSET,
            "split": DATASET_SPLIT,
            "num_examples": len(rows),
        },
        "model": {
            "key": spec.key,
            "display_name": spec.display_name,
            "path": spec.model_path,
            "resolved_model_source": resolved_model_source,
            "model_kind": model_kind,
            "base_model_name": base_model_name,
        },
        "prompt": prompt_meta,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "prompt_max_length": prompt_max_length,
            "batch_size": batch_size,
            "do_sample": False,
            "num_beams": 1,
        },
        "artifacts": {
            "output_jsonl": str(output_jsonl),
            "output_meta_json": str(meta_json),
        },
        "service_account_env_present": load_service_account_info() is not None,
    }
    meta_json.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    outputs_volume.commit()

    gcs_output_prefix = ""
    uploaded_files = 0
    if results_gcs_prefix:
        gcs_output_prefix = f"{results_gcs_prefix.rstrip('/')}/{run_name}/{model_key}"
        uploaded_files = upload_dir_to_gcs(output_dir, gcs_output_prefix, project=checkpoint_project)

    del model
    torch.cuda.empty_cache()

    return {
        "status": "complete",
        "model_key": spec.key,
        "run_name": run_name,
        "examples_written": records_written,
        "output_jsonl": str(output_jsonl),
        "output_meta_json": str(meta_json),
        "volume_download_hint": (
            f"modal volume get {OUTPUTS_VOLUME_NAME} "
            f"/case_reasoning_quality_eval/{run_name}/{model_key} "
            f"./Case_reasoning_quality_eval/jsonl_model_output_for_CaSE/{run_name}/{model_key}/"
        ),
        "gcs_output_prefix": gcs_output_prefix,
        "uploaded_files": uploaded_files,
    }


@app.local_entrypoint()
def main(
    model_key: str,
    run_name: str = "",
    limit: int = DEFAULT_LIMIT,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    prompt_max_length: int = DEFAULT_PROMPT_MAX_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    results_gcs_prefix: str = DEFAULT_RESULTS_GCS_PREFIX,
    checkpoint_project: str = CHECKPOINT_PROJECT,
    base_model_name: str = DEFAULT_BASE_MODEL,
) -> None:
    if model_key not in MODEL_SPECS:
        raise SystemExit(f"Unknown --model-key={model_key!r}. Choose from: {', '.join(sorted(MODEL_SPECS))}")

    if not run_name:
        run_name = f"case_{now_utc_timestamp()}"

    request = {
        "model_key": model_key,
        "run_name": run_name,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "prompt_max_length": prompt_max_length,
        "batch_size": batch_size,
        "results_gcs_prefix": results_gcs_prefix,
        "checkpoint_project": checkpoint_project,
        "base_model_name": base_model_name,
    }

    print(
        f"Launching CaSE inference for model_key={model_key} "
        f"(run_name={run_name}, limit={limit}, max_new_tokens={max_new_tokens})..."
    )
    result = run_case_model_inference.remote(request)
    print(json.dumps(result, indent=2))
