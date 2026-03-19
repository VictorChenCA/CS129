"""Modal trainer for stratified MATH — supports all paper model variants.

Training objectives (--training_objective):
  multitask    Answer + rationale loss. Covers:
               - Multitask SFT (human):    --rationale_source solution
               - Multitask SFT (synthetic): --rationale_source gs://...
  answer_only  Answer loss only. Covers:
               - Answer-only SFT:          (default)
               - Rationale Prompting:      --rationale_in_prompt true --rationale_source gs://...

Features:
- Runs on Modal GPU.
- Grid search stage for short hyperparameter trials.
- Optional full finetune on combined easy/medium/hard training sets.
- Frequent train/validation loss + accuracy logging for smooth curves.
- Checkpoints saved both locally (Modal volume + downloaded copy) and to GCS.

Usage examples:
  modal run dataset-cs129/distill/train/train_model_multitask_stratified.py
  modal run dataset-cs129/distill/train/train_model_multitask_stratified.py --stage search_then_full --training_objective answer_only
  modal run dataset-cs129/distill/train/train_model_multitask_stratified.py --stage search_then_full --training_objective multitask --rationale_source gs://cs224n-dapo-distill-data/pseudo_label_rationales/gemini_all.jsonl
  modal run dataset-cs129/distill/train/train_model_multitask_stratified.py --stage full_train --best_hparams_json '{"learning_rate":2e-5,"lora_r":16,"lora_alpha":32,"lora_dropout":0.05}'
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import modal
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_GCP_PROJECT = "cs224n-dapo-distill"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_TRAIN_JSONL_PATHS = [
    "dataset-cs129/stratified/easy_train.jsonl",
    "dataset-cs129/stratified/medium_train.jsonl",
    "dataset-cs129/stratified/hard_train.jsonl",
]

DEFAULT_VAL_JSONL_PATHS = [
    "dataset-cs129/stratified/easy_valid.jsonl",
    "dataset-cs129/stratified/medium_valid.jsonl",
    "dataset-cs129/stratified/hard_valid.jsonl",
]

DEFAULT_GCS_CHECKPOINT_ROOT = "gs://cs224n-dapo-distill-data/distillation_model_checkpoints"
DEFAULT_RUN_PREFIX = "qwen25_05b_lora_reasoning_run"

DEFAULT_GRID = [
    {"learning_rate": 5e-5, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
    {"learning_rate": 2e-5, "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
    {"learning_rate": 5e-5, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
    {"learning_rate": 2e-5, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05},
]


# -----------------------------------------------------------------------------
# Modal app config
# -----------------------------------------------------------------------------
APP_NAME = "distill-multitask-stratified-trainer"
ARTIFACTS_VOLUME_NAME = "distill-multitask-artifacts"
ARTIFACTS_DIR = "/artifacts"

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.49.0",
        "datasets",
        "peft",
        "accelerate",
        "tqdm",
        "matplotlib",
        "numpy",
        "google-cloud-storage",
    )
)

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[len("gs://") :]
    bucket, _, blob = no_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def upload_dir_to_gcs(local_dir: Path, gcs_uri_prefix: str, *, project: str) -> int:
    try:
        from google.cloud import storage
    except Exception as exc:
        raise ImportError("Install google-cloud-storage: `pip install google-cloud-storage`") from exc

    bucket_name, blob_prefix = parse_gcs_uri(
        gcs_uri_prefix if gcs_uri_prefix.endswith("/") else f"{gcs_uri_prefix}/"
    )
    client = storage.Client(project=project)
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


def load_jsonl_rows(path_or_gs: str, gcp_project: str) -> list[dict[str, Any]]:
    if path_or_gs.startswith("gs://"):
        try:
            from google.cloud import storage
        except Exception as exc:
            raise ImportError("Install google-cloud-storage: `pip install google-cloud-storage`") from exc

        bucket_name, blob_name = parse_gcs_uri(path_or_gs)
        client = storage.Client(project=gcp_project)
        text = client.bucket(bucket_name).blob(blob_name).download_as_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    with open(path_or_gs, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_rationale_by_problem(rationale_source: str, gcp_project: str) -> dict[str, str]:
    """Load a pseudo-rationales JSONL (local or GCS) and return a problem→rationale dict."""
    rows = load_jsonl_rows(rationale_source, gcp_project)
    result: dict[str, str] = {}
    for row in rows:
        problem = str(row.get("problem", ""))
        rationale = str(row.get("rationale", row.get("solution", "")))
        if problem:
            result[problem] = rationale
    return result


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_boxed_payload(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
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
    return text.strip()


def build_student_messages(
    problem_text: str,
    rationale: str,
    answer_text: str,
    *,
    training_objective: str = "multitask",
    rationale_in_prompt: bool = False,
) -> list[dict[str, str]]:
    if rationale_in_prompt:
        # Rationale Prompting: rationale in user context, only answer supervised
        user_content = (
            f"Solve the following math problem.\n\nProblem:\n{problem_text}\n"
            f"\nReasoning:\n{rationale}\n"
        )
        assistant_content = f"<answer>{answer_text}</answer>"
    elif training_objective == "answer_only":
        # Answer-only SFT: no rationale in input or output
        user_content = f"Solve the following math problem.\n\nProblem:\n{problem_text}\n"
        assistant_content = f"<answer>{answer_text}</answer>"
    else:
        # Multitask: rationale + answer in assistant output
        user_content = f"Solve the following math problem.\n\nProblem:\n{problem_text}\n"
        assistant_content = f"<rationale>{rationale}</rationale>\n<answer>{answer_text}</answer>"
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def find_subsequence(token_ids: list[int], pattern_ids: list[int], start_idx: int = 0) -> int:
    if not pattern_ids:
        return -1
    max_idx = len(token_ids) - len(pattern_ids)
    for idx in range(start_idx, max_idx + 1):
        if token_ids[idx : idx + len(pattern_ids)] == pattern_ids:
            return idx
    return -1


def build_multitask_dataset_from_rows(
    rows: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_length: int,
    *,
    training_objective: str = "multitask",
    rationale_by_problem: dict[str, str] | None = None,
    rationale_in_prompt: bool = False,
) -> Dataset:
    rationale_start_ids = tokenizer.encode("<rationale>", add_special_tokens=False)
    rationale_end_ids = tokenizer.encode("</rationale>", add_special_tokens=False)
    answer_start_ids = tokenizer.encode("<answer>", add_special_tokens=False)
    answer_end_ids = tokenizer.encode("</answer>", add_special_tokens=False)

    examples: list[dict[str, Any]] = []
    for row in rows:
        problem = str(row.get("problem", ""))
        solution = str(row.get("solution", ""))
        # Use synthetic rationale if provided, otherwise fall back to dataset solution
        if rationale_by_problem is not None:
            rationale = rationale_by_problem.get(problem, solution)
        else:
            rationale = solution
        answer = extract_boxed_payload(solution)
        messages = build_student_messages(
            problem, rationale, answer,
            training_objective=training_objective,
            rationale_in_prompt=rationale_in_prompt,
        )

        prompt_text = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_attention_mask=True,
        )
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]
        labels = input_ids.copy()

        n_prompt = min(len(prompt_ids), len(labels))
        labels[:n_prompt] = [-100] * n_prompt

        rationale_mask = [0] * len(input_ids)
        answer_mask = [0] * len(input_ids)

        rat_tag_start = find_subsequence(input_ids, rationale_start_ids, start_idx=0)
        if rat_tag_start >= 0:
            rat_content_start = rat_tag_start + len(rationale_start_ids)
            rat_content_end = find_subsequence(input_ids, rationale_end_ids, start_idx=rat_content_start)
            if rat_content_end >= 0:
                for idx in range(rat_content_start, min(rat_content_end, len(labels))):
                    if labels[idx] != -100:
                        rationale_mask[idx] = 1

        ans_tag_start = find_subsequence(input_ids, answer_start_ids, start_idx=0)
        if ans_tag_start >= 0:
            ans_content_start = ans_tag_start + len(answer_start_ids)
            ans_content_end = find_subsequence(input_ids, answer_end_ids, start_idx=ans_content_start)
            if ans_content_end >= 0:
                for idx in range(ans_content_start, min(ans_content_end, len(labels))):
                    if labels[idx] != -100:
                        answer_mask[idx] = 1

        examples.append(
            {
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "rationale_mask": rationale_mask,
                "answer_mask": answer_mask,
            }
        )

    return Dataset.from_list(examples)

class AnswerAwareDistillCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_size = len(examples)
        max_len = max(len(ex["input_ids"]) for ex in examples)

        input_ids = torch.full((batch_size, max_len), fill_value=self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)
        rationale_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        answer_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for row_idx, ex in enumerate(examples):
            seq_len = len(ex["input_ids"])
            input_ids[row_idx, :seq_len] = torch.tensor(ex["input_ids"], dtype=torch.long)
            attention_mask[row_idx, :seq_len] = torch.tensor(ex["attention_mask"], dtype=torch.long)
            labels[row_idx, :seq_len] = torch.tensor(ex["labels"], dtype=torch.long)
            rationale_mask[row_idx, :seq_len] = torch.tensor(ex["rationale_mask"], dtype=torch.bool)
            answer_mask[row_idx, :seq_len] = torch.tensor(ex["answer_mask"], dtype=torch.bool)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rationale_mask": rationale_mask,
            "answer_mask": answer_mask,
        }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    denom = mask.float().sum()
    if denom.item() <= 0:
        return None
    return (values * mask.float()).sum() / denom


def compute_weighted_distill_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    rationale_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    lambda_rationale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_rationale = rationale_mask[:, 1:].contiguous()
    shift_answer = answer_mask[:, 1:].contiguous()

    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shift_labels)

    valid_mask = shift_labels != -100
    fallback_loss = _masked_mean(token_loss, valid_mask)
    if fallback_loss is None:
        raise RuntimeError("No valid labels in batch.")

    rationale_loss = _masked_mean(token_loss, shift_rationale & valid_mask)
    answer_loss = _masked_mean(token_loss, shift_answer & valid_mask)
    if rationale_loss is None:
        rationale_loss = fallback_loss
    if answer_loss is None:
        answer_loss = fallback_loss

    total_loss = answer_loss + lambda_rationale * rationale_loss

    preds = shift_logits.argmax(dim=-1)
    answer_valid_mask = shift_answer & valid_mask
    if answer_valid_mask.sum().item() <= 0:
        answer_valid_mask = valid_mask
    answer_correct = ((preds == shift_labels) & answer_valid_mask).sum()
    answer_total = answer_valid_mask.sum()

    return total_loss, rationale_loss.detach(), answer_loss.detach(), answer_correct.detach(), answer_total.detach()


def evaluate_model(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    device: str,
    lambda_rationale: float,
    use_bf16: bool,
    use_fp16: bool,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total_batches = 0
    sum_total_loss = 0.0
    sum_rationale_loss = 0.0
    sum_answer_loss = 0.0
    answer_correct = 0.0
    answer_total = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            total_batches += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=(use_bf16 or use_fp16),
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss, rat_loss, ans_loss, ans_correct, ans_total = compute_weighted_distill_loss(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    rationale_mask=batch["rationale_mask"],
                    answer_mask=batch["answer_mask"],
                    lambda_rationale=lambda_rationale,
                )

            sum_total_loss += float(loss.item())
            sum_rationale_loss += float(rat_loss.item())
            sum_answer_loss += float(ans_loss.item())
            answer_correct += float(ans_correct.item())
            answer_total += float(ans_total.item())

    model.train()

    if total_batches == 0:
        return {
            "loss": float("nan"),
            "rationale_loss": float("nan"),
            "answer_loss": float("nan"),
            "answer_accuracy": float("nan"),
        }

    return {
        "loss": sum_total_loss / total_batches,
        "rationale_loss": sum_rationale_loss / total_batches,
        "answer_loss": sum_answer_loss / total_batches,
        "answer_accuracy": (answer_correct / max(answer_total, 1.0)),
    }


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 1:
        return values
    running = 0.0
    out: list[float] = []
    for idx, value in enumerate(values):
        running += value
        if idx >= window:
            running -= values[idx - window]
        denom = min(idx + 1, window)
        out.append(running / denom)
    return out


def plot_training_curves(
    output_dir: Path,
    train_history: list[dict[str, float]],
    eval_history: list[dict[str, float]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "train_step_history.json", train_history)
    write_json(output_dir / "eval_step_history.json", eval_history)

    if train_history or eval_history:
        train_steps = [int(row["step"]) for row in train_history]
        train_loss = [float(row["total_loss"]) for row in train_history]
        eval_steps = [int(row["step"]) for row in eval_history]
        eval_train_loss = [float(row["train_eval_loss"]) for row in eval_history]
        eval_val_loss = [float(row["val_loss"]) for row in eval_history]

        loss_window = max(3, min(25, max(3, len(train_loss) // 20)))
        smoothed_train_loss = moving_average(train_loss, loss_window)

        plt.figure(figsize=(10, 5))
        if train_steps:
            plt.plot(train_steps, train_loss, alpha=0.2, linewidth=1.0, label="train_loss_raw")
            plt.plot(train_steps, smoothed_train_loss, linewidth=2.0, label=f"train_loss_smooth_w{loss_window}")
        if eval_steps:
            plt.plot(eval_steps, eval_train_loss, linewidth=2.0, label="train_loss_eval_subset")
            plt.plot(eval_steps, eval_val_loss, linewidth=2.0, label="val_loss")
        plt.xlabel("Optimizer Step")
        plt.ylabel("Loss")
        plt.title("Multitask Stratified Loss Curves")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "loss_curve.png", dpi=150)
        plt.close()

    if train_history or eval_history:
        train_steps = [int(row["step"]) for row in train_history]
        train_acc = [float(row["batch_answer_accuracy"]) for row in train_history]
        eval_steps = [int(row["step"]) for row in eval_history]
        eval_train_acc = [float(row["train_eval_answer_accuracy"]) for row in eval_history]
        eval_val_acc = [float(row["val_answer_accuracy"]) for row in eval_history]

        acc_window = max(3, min(25, max(3, len(train_acc) // 20)))
        smoothed_train_acc = moving_average(train_acc, acc_window)

        plt.figure(figsize=(10, 5))
        if train_steps:
            plt.plot(train_steps, train_acc, alpha=0.2, linewidth=1.0, label="train_acc_raw")
            plt.plot(train_steps, smoothed_train_acc, linewidth=2.0, label=f"train_acc_smooth_w{acc_window}")
        if eval_steps:
            plt.plot(eval_steps, eval_train_acc, linewidth=2.0, label="train_acc_eval_subset")
            plt.plot(eval_steps, eval_val_acc, linewidth=2.0, label="val_acc")
        plt.xlabel("Optimizer Step")
        plt.ylabel("Answer Token Accuracy")
        plt.title("Multitask Stratified Accuracy Curves")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
        plt.close()


def save_model_checkpoint(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    local_dir: Path,
    checkpoint_meta: dict[str, Any],
    gcs_uri_prefix: str | None,
    gcp_project: str,
    upload_to_gcs: bool,
) -> dict[str, Any]:
    local_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(local_dir))
    tokenizer.save_pretrained(str(local_dir))
    write_json(local_dir / "checkpoint_meta.json", checkpoint_meta)

    upload_count = 0
    if upload_to_gcs and gcs_uri_prefix:
        upload_count = upload_dir_to_gcs(local_dir, gcs_uri_prefix, project=gcp_project)

    return {
        "local_dir": str(local_dir),
        "gcs_uri_prefix": gcs_uri_prefix,
        "uploaded_files": upload_count,
    }


def choose_best_grid_result(grid_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not grid_results:
        raise ValueError("Grid search results are empty.")
    return min(
        grid_results,
        key=lambda row: (
            float(row["best_val_loss"]),
            -float(row["best_val_answer_accuracy"]),
        ),
    )


def parse_grid_json(grid_json: str) -> list[dict[str, Any]]:
    parsed = json.loads(grid_json)
    if not isinstance(parsed, list):
        raise ValueError("grid_json must decode to a list of objects.")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, dict):
            raise ValueError(f"grid_json item at index {idx} is not an object.")
        out.append(dict(item))
    if not out:
        raise ValueError("grid_json cannot be empty.")
    return out


def sample_rows(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    rnd = random.Random(seed)
    idxs = list(range(len(rows)))
    rnd.shuffle(idxs)
    chosen = idxs[:limit]
    return [rows[i] for i in chosen]


def train_single_run(
    *,
    run_name: str,
    run_dir: Path,
    model_name: str,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    gcp_project: str,
    gcs_run_root: str | None,
    max_length: int,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    max_steps: int,
    logging_steps: int,
    eval_steps: int,
    checkpoint_steps: int,
    lambda_rationale: float,
    use_lora: bool,
    lora_target_modules: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    seed: int,
    train_eval_samples: int,
    max_train_eval_batches: int,
    max_val_eval_batches: int,
    upload_checkpoints_to_gcs: bool,
    training_objective: str = "multitask",
    rationale_by_problem: dict[str, str] | None = None,
    rationale_in_prompt: bool = False,
) -> dict[str, Any]:
    from peft import LoraConfig, TaskType, get_peft_model

    # answer_only mode never supervises rationale tokens
    if training_objective == "answer_only":
        lambda_rationale = 0.0

    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=model_dtype,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if use_lora:
        target_modules = [name.strip() for name in lora_target_modules.split(",") if name.strip()]
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)

    model.to(device)
    model.train()

    ds_kwargs = dict(
        training_objective=training_objective,
        rationale_by_problem=rationale_by_problem,
        rationale_in_prompt=rationale_in_prompt,
    )
    train_ds = build_multitask_dataset_from_rows(train_rows, tokenizer=tokenizer, max_length=max_length, **ds_kwargs)
    val_ds = build_multitask_dataset_from_rows(val_rows, tokenizer=tokenizer, max_length=max_length, **ds_kwargs)
    train_eval_rows = sample_rows(train_rows, limit=train_eval_samples, seed=seed + 17)
    train_eval_ds = build_multitask_dataset_from_rows(train_eval_rows, tokenizer=tokenizer, max_length=max_length, **ds_kwargs)

    collator = AnswerAwareDistillCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    grad_accum = max(gradient_accumulation_steps, 1)
    if max_steps > 0:
        total_optim_steps = max_steps
    else:
        steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
        total_optim_steps = max(1, math.ceil(steps_per_epoch * num_train_epochs))

    global_micro_step = 0
    optim_step = 0
    epoch = 0
    train_history: list[dict[str, float]] = []
    eval_history: list[dict[str, float]] = []
    saved_checkpoints: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0

    progress = tqdm(total=total_optim_steps, desc=f"{run_name}", unit="step", dynamic_ncols=True)
    optimizer.zero_grad(set_to_none=True)

    # Baseline eval at step 0
    baseline_train_eval = evaluate_model(
        model=model,
        dataloader=train_eval_loader,
        device=device,
        lambda_rationale=lambda_rationale,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        max_batches=max_train_eval_batches,
    )
    baseline_val_eval = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        lambda_rationale=lambda_rationale,
        use_bf16=use_bf16,
        use_fp16=use_fp16,
        max_batches=max_val_eval_batches,
    )
    eval_history.append(
        {
            "step": 0,
            "train_eval_loss": float(baseline_train_eval["loss"]),
            "train_eval_answer_accuracy": float(baseline_train_eval["answer_accuracy"]),
            "val_loss": float(baseline_val_eval["loss"]),
            "val_answer_accuracy": float(baseline_val_eval["answer_accuracy"]),
        }
    )

    while optim_step < total_optim_steps:
        epoch += 1
        for batch in train_loader:
            global_micro_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if use_bf16 else torch.float16,
                enabled=(use_bf16 or use_fp16),
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss, rationale_loss, answer_loss, ans_correct, ans_total = compute_weighted_distill_loss(
                    logits=outputs.logits,
                    labels=batch["labels"],
                    rationale_mask=batch["rationale_mask"],
                    answer_mask=batch["answer_mask"],
                    lambda_rationale=lambda_rationale,
                )
                scaled_loss = loss / grad_accum

            if use_fp16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if global_micro_step % grad_accum != 0:
                continue

            if use_fp16:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            optim_step += 1
            progress.update(1)

            batch_answer_accuracy = float(ans_correct.item() / max(ans_total.item(), 1))
            train_row = {
                "step": float(optim_step),
                "total_loss": float(loss.item()),
                "rationale_loss": float(rationale_loss.item()),
                "answer_loss": float(answer_loss.item()),
                "batch_answer_accuracy": batch_answer_accuracy,
                "epoch": float(epoch),
            }
            train_history.append(train_row)

            if optim_step % max(logging_steps, 1) == 0:
                progress.set_postfix(
                    loss=f"{train_row['total_loss']:.4f}",
                    ans_acc=f"{train_row['batch_answer_accuracy']:.4f}",
                )

            should_eval = (
                optim_step % max(eval_steps, 1) == 0
                or optim_step == total_optim_steps
            )
            if should_eval:
                train_eval_metrics = evaluate_model(
                    model=model,
                    dataloader=train_eval_loader,
                    device=device,
                    lambda_rationale=lambda_rationale,
                    use_bf16=use_bf16,
                    use_fp16=use_fp16,
                    max_batches=max_train_eval_batches,
                )
                val_metrics = evaluate_model(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    lambda_rationale=lambda_rationale,
                    use_bf16=use_bf16,
                    use_fp16=use_fp16,
                    max_batches=max_val_eval_batches,
                )
                eval_row = {
                    "step": float(optim_step),
                    "train_eval_loss": float(train_eval_metrics["loss"]),
                    "train_eval_answer_accuracy": float(train_eval_metrics["answer_accuracy"]),
                    "val_loss": float(val_metrics["loss"]),
                    "val_answer_accuracy": float(val_metrics["answer_accuracy"]),
                }
                eval_history.append(eval_row)

                current_val_loss = float(val_metrics["loss"])
                current_val_acc = float(val_metrics["answer_accuracy"])
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                if current_val_acc > best_val_acc:
                    best_val_acc = current_val_acc

            should_checkpoint = (
                checkpoint_steps > 0
                and optim_step % checkpoint_steps == 0
            ) or optim_step == total_optim_steps
            if should_checkpoint:
                ckpt_local = run_dir / "checkpoints" / f"step_{optim_step:06d}"
                ckpt_gcs = None
                if gcs_run_root:
                    ckpt_gcs = f"{gcs_run_root.rstrip('/')}/checkpoints/step_{optim_step:06d}"
                saved = save_model_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    local_dir=ckpt_local,
                    checkpoint_meta={
                        "step": optim_step,
                        "best_val_loss_so_far": best_val_loss,
                        "best_val_answer_accuracy_so_far": best_val_acc,
                    },
                    gcs_uri_prefix=ckpt_gcs,
                    gcp_project=gcp_project,
                    upload_to_gcs=upload_checkpoints_to_gcs,
                )
                saved_checkpoints.append(saved)

            if optim_step >= total_optim_steps:
                break
        if optim_step >= total_optim_steps:
            break

    progress.close()

    metrics_dir = run_dir / "metrics"
    plot_training_curves(metrics_dir, train_history=train_history, eval_history=eval_history)

    final_local = run_dir / "final_model"
    final_gcs = f"{gcs_run_root.rstrip('/')}/final_model" if gcs_run_root else None
    final_checkpoint = save_model_checkpoint(
        model=model,
        tokenizer=tokenizer,
        local_dir=final_local,
        checkpoint_meta={
            "step": optim_step,
            "best_val_loss": best_val_loss,
            "best_val_answer_accuracy": best_val_acc,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        },
        gcs_uri_prefix=final_gcs,
        gcp_project=gcp_project,
        upload_to_gcs=upload_checkpoints_to_gcs,
    )

    summary = {
        "run_name": run_name,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "total_optimizer_steps": optim_step,
        "best_val_loss": best_val_loss,
        "best_val_answer_accuracy": best_val_acc,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lambda_rationale": lambda_rationale,
            "max_length": max_length,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "training_objective": training_objective,
            "rationale_in_prompt": rationale_in_prompt,
        },
        "last_train_point": train_history[-1] if train_history else None,
        "last_eval_point": eval_history[-1] if eval_history else None,
        "saved_checkpoints": saved_checkpoints,
        "final_checkpoint": final_checkpoint,
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "run_summary.json", summary)
    return summary


@app.function(
    image=training_image,
    gpu="A100",
    timeout=60 * 60 * 16,
    memory=32768,
    volumes={ARTIFACTS_DIR: artifacts_volume},
)
def run_training_pipeline(request: dict[str, Any]) -> dict[str, Any]:
    stage = str(request.get("stage", "grid_search")).strip()
    if stage not in {"grid_search", "search_then_full", "full_train"}:
        raise ValueError("stage must be one of: grid_search, search_then_full, full_train")

    run_name = str(request.get("run_name") or "").strip()
    if not run_name:
        run_name = f"{DEFAULT_RUN_PREFIX}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    gcp_project = str(request.get("gcp_project", DEFAULT_GCP_PROJECT))
    model_name = str(request.get("model_name", DEFAULT_MODEL_NAME))
    gcs_checkpoint_root = str(request.get("gcs_checkpoint_root", DEFAULT_GCS_CHECKPOINT_ROOT))
    upload_checkpoints_to_gcs = bool(request.get("upload_checkpoints_to_gcs", True))

    train_paths = list(request.get("train_jsonl_paths", DEFAULT_TRAIN_JSONL_PATHS))
    val_paths = list(request.get("val_jsonl_paths", DEFAULT_VAL_JSONL_PATHS))

    provided_train_rows = request.get("train_rows")
    provided_val_rows = request.get("val_rows")

    all_train_rows: list[dict[str, Any]] = []
    all_val_rows: list[dict[str, Any]] = []
    if isinstance(provided_train_rows, list) and provided_train_rows:
        all_train_rows = [dict(row) for row in provided_train_rows]
    else:
        for path in train_paths:
            all_train_rows.extend(load_jsonl_rows(str(path), gcp_project))

    if isinstance(provided_val_rows, list) and provided_val_rows:
        all_val_rows = [dict(row) for row in provided_val_rows]
    else:
        for path in val_paths:
            all_val_rows.extend(load_jsonl_rows(str(path), gcp_project))

    run_root = Path(ARTIFACTS_DIR) / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    training_objective = str(request.get("training_objective", "multitask")).strip()
    if training_objective not in {"multitask", "answer_only"}:
        raise ValueError("training_objective must be 'multitask' or 'answer_only'")
    rationale_source = str(request.get("rationale_source", "solution")).strip()
    rationale_in_prompt = bool(request.get("rationale_in_prompt", False))

    # Load synthetic rationales if a GCS path or local path is given
    rationale_by_problem: dict[str, str] | None = None
    provided_rationale = request.get("rationale_by_problem")
    if isinstance(provided_rationale, dict) and provided_rationale:
        rationale_by_problem = provided_rationale
    elif rationale_source and rationale_source != "solution":
        rationale_by_problem = load_rationale_by_problem(rationale_source, gcp_project)

    write_json(
        run_root / "dataset_manifest.json",
        {
            "stage": stage,
            "training_objective": training_objective,
            "rationale_source": rationale_source,
            "rationale_in_prompt": rationale_in_prompt,
            "train_jsonl_paths": train_paths,
            "val_jsonl_paths": val_paths,
            "train_rows_total": len(all_train_rows),
            "val_rows_total": len(all_val_rows),
            "rationale_by_problem_count": len(rationale_by_problem) if rationale_by_problem else 0,
        },
    )

    seed = int(request.get("seed", 42))
    max_length = int(request.get("max_length", 2048))
    lambda_rationale = float(request.get("lambda_rationale", 1.0))
    use_lora = bool(request.get("use_lora", True))
    lora_target_modules = str(
        request.get(
            "lora_target_modules",
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        )
    )

    weight_decay = float(request.get("weight_decay", 1e-4))
    max_grad_norm = float(request.get("max_grad_norm", 1.0))
    per_device_train_batch_size = int(request.get("per_device_train_batch_size", 1))
    per_device_eval_batch_size = int(request.get("per_device_eval_batch_size", 2))
    gradient_accumulation_steps = int(request.get("gradient_accumulation_steps", 8))
    train_eval_samples = int(request.get("train_eval_samples", 256))

    grid_results: list[dict[str, Any]] = []
    best_hparams: dict[str, Any] | None = None

    if stage in {"grid_search", "search_then_full"}:
        grid = request.get("grid", DEFAULT_GRID)
        if not isinstance(grid, list):
            raise ValueError("request['grid'] must be a list")

        grid_train_sample_rows = int(request.get("grid_train_sample_rows", 512))
        grid_val_sample_rows = int(request.get("grid_val_sample_rows", 256))
        grid_max_steps = int(request.get("grid_max_steps", 80))
        grid_logging_steps = int(request.get("grid_logging_steps", 2))
        grid_eval_steps = int(request.get("grid_eval_steps", 10))

        search_train_rows = sample_rows(all_train_rows, grid_train_sample_rows, seed=seed + 1001)
        search_val_rows = sample_rows(all_val_rows, grid_val_sample_rows, seed=seed + 2001)

        for idx, hp in enumerate(grid):
            run_hp = dict(hp)
            run_dir = run_root / "grid_search" / f"run_{idx:02d}"
            result = train_single_run(
                run_name=f"grid_{idx:02d}",
                run_dir=run_dir,
                model_name=model_name,
                train_rows=search_train_rows,
                val_rows=search_val_rows,
                gcp_project=gcp_project,
                gcs_run_root=None,
                max_length=max_length,
                num_train_epochs=1.0,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                max_steps=grid_max_steps,
                logging_steps=grid_logging_steps,
                eval_steps=grid_eval_steps,
                checkpoint_steps=0,
                lambda_rationale=lambda_rationale,
                use_lora=use_lora,
                lora_target_modules=lora_target_modules,
                lora_r=int(run_hp["lora_r"]),
                lora_alpha=int(run_hp["lora_alpha"]),
                lora_dropout=float(run_hp["lora_dropout"]),
                learning_rate=float(run_hp["learning_rate"]),
                seed=seed + idx,
                train_eval_samples=min(train_eval_samples, len(search_train_rows)),
                max_train_eval_batches=int(request.get("grid_max_train_eval_batches", 12)),
                max_val_eval_batches=int(request.get("grid_max_val_eval_batches", 12)),
                upload_checkpoints_to_gcs=False,
                training_objective=training_objective,
                rationale_by_problem=rationale_by_problem,
                rationale_in_prompt=rationale_in_prompt,
            )
            row = {
                "run_index": idx,
                "hyperparameters": run_hp,
                "best_val_loss": float(result["best_val_loss"]),
                "best_val_answer_accuracy": float(result["best_val_answer_accuracy"]),
                "run_dir": result["run_dir"],
            }
            grid_results.append(row)

        best_grid = choose_best_grid_result(grid_results)
        best_hparams = dict(best_grid["hyperparameters"])
        write_json(run_root / "grid_search_results.json", grid_results)
        write_json(run_root / "best_hparams.json", best_hparams)

    if stage == "full_train":
        provided_best = request.get("best_hparams")
        if not isinstance(provided_best, dict):
            raise ValueError("For stage=full_train provide best_hparams object.")
        best_hparams = dict(provided_best)

    full_train_result = None
    if stage in {"search_then_full", "full_train"}:
        if not best_hparams:
            raise RuntimeError("best_hparams missing before full training.")

        full_num_train_epochs = float(request.get("full_num_train_epochs", 1.0))
        full_max_steps = int(request.get("full_max_steps", -1))
        full_logging_steps = int(request.get("full_logging_steps", 5))
        full_eval_steps = int(request.get("full_eval_steps", 20))
        full_checkpoint_steps = int(request.get("full_checkpoint_steps", 100))
        gcs_run_root = f"{gcs_checkpoint_root.rstrip('/')}/{run_name}"

        full_train_result = train_single_run(
            run_name="full_train",
            run_dir=run_root / "full_train",
            model_name=model_name,
            train_rows=all_train_rows,
            val_rows=all_val_rows,
            gcp_project=gcp_project,
            gcs_run_root=gcs_run_root,
            max_length=max_length,
            num_train_epochs=full_num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            max_steps=full_max_steps,
            logging_steps=full_logging_steps,
            eval_steps=full_eval_steps,
            checkpoint_steps=full_checkpoint_steps,
            lambda_rationale=lambda_rationale,
            use_lora=use_lora,
            lora_target_modules=lora_target_modules,
            lora_r=int(best_hparams["lora_r"]),
            lora_alpha=int(best_hparams["lora_alpha"]),
            lora_dropout=float(best_hparams["lora_dropout"]),
            learning_rate=float(best_hparams["learning_rate"]),
            seed=seed,
            train_eval_samples=min(train_eval_samples, len(all_train_rows)),
            max_train_eval_batches=int(request.get("full_max_train_eval_batches", 32)),
            max_val_eval_batches=int(request.get("full_max_val_eval_batches", -1)),
            upload_checkpoints_to_gcs=upload_checkpoints_to_gcs,
            training_objective=training_objective,
            rationale_by_problem=rationale_by_problem,
            rationale_in_prompt=rationale_in_prompt,
        )

    summary = {
        "stage": stage,
        "run_name": run_name,
        "model_name": model_name,
        "gcp_project": gcp_project,
        "training_objective": training_objective,
        "rationale_source": rationale_source,
        "rationale_in_prompt": rationale_in_prompt,
        "train_rows_total": len(all_train_rows),
        "val_rows_total": len(all_val_rows),
        "train_jsonl_paths": train_paths,
        "val_jsonl_paths": val_paths,
        "grid_results": grid_results,
        "best_hparams": best_hparams,
        "full_train_result": full_train_result,
        "gcs_checkpoint_root": gcs_checkpoint_root,
        "upload_checkpoints_to_gcs": upload_checkpoints_to_gcs,
        "artifacts_volume_name": ARTIFACTS_VOLUME_NAME,
        "artifacts_volume_run_path": f"/{run_name}",
    }

    write_json(run_root / "summary.json", summary)
    artifacts_volume.commit()
    return summary


def download_from_modal_volume(volume_name: str, remote_path: str, local_destination: Path) -> None:
    local_destination.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "modal",
        "volume",
        "get",
        volume_name,
        remote_path,
        str(local_destination),
        "--force",
    ]
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
def main(
    stage: str = "grid_search",
    run_name: str = "",
    gcp_project: str = DEFAULT_GCP_PROJECT,
    model_name: str = DEFAULT_MODEL_NAME,
    gcs_checkpoint_root: str = DEFAULT_GCS_CHECKPOINT_ROOT,
    train_jsonl_paths: str = ",".join(DEFAULT_TRAIN_JSONL_PATHS),
    val_jsonl_paths: str = ",".join(DEFAULT_VAL_JSONL_PATHS),
    grid_json: str = "",
    best_hparams_json: str = "",
    output_dir: str = "",
    upload_checkpoints_to_gcs: bool = False,
    training_objective: str = "multitask",
    rationale_source: str = "solution",
    rationale_in_prompt: bool = False,
    lambda_rationale: float = 1.0,
) -> None:
    if not run_name:
        run_name = f"{DEFAULT_RUN_PREFIX}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    train_paths = [entry.strip() for entry in train_jsonl_paths.split(",") if entry.strip()]
    val_paths = [entry.strip() for entry in val_jsonl_paths.split(",") if entry.strip()]

    request: dict[str, Any] = {
        "stage": stage,
        "run_name": run_name,
        "gcp_project": gcp_project,
        "model_name": model_name,
        "gcs_checkpoint_root": gcs_checkpoint_root,
        "train_jsonl_paths": train_paths,
        "val_jsonl_paths": val_paths,
        "upload_checkpoints_to_gcs": upload_checkpoints_to_gcs,
        "training_objective": training_objective,
        "rationale_source": rationale_source,
        "rationale_in_prompt": rationale_in_prompt,
        "seed": 42,
        "max_length": 2048,
        "lambda_rationale": lambda_rationale,
        "use_lora": True,
        "lora_target_modules": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "train_eval_samples": 256,
        "grid_train_sample_rows": 512,
        "grid_val_sample_rows": 256,
        "grid_max_steps": 80,
        "grid_logging_steps": 2,
        "grid_eval_steps": 10,
        "grid_max_train_eval_batches": 12,
        "grid_max_val_eval_batches": 12,
        "full_num_train_epochs": 1.0,
        "full_max_steps": -1,
        "full_logging_steps": 5,
        "full_eval_steps": 100,
        "full_checkpoint_steps": 100,
        "full_max_train_eval_batches": 32,
        "full_max_val_eval_batches": -1,
    }

    if grid_json.strip():
        request["grid"] = parse_grid_json(grid_json)
    else:
        request["grid"] = DEFAULT_GRID

    if best_hparams_json.strip():
        request["best_hparams"] = json.loads(best_hparams_json)

    # If local JSONL files are passed, load them locally and forward raw rows
    # so the Modal worker can train without direct GCS credentials.
    local_train_paths = [path for path in train_paths if not str(path).startswith("gs://")]
    local_val_paths = [path for path in val_paths if not str(path).startswith("gs://")]
    if local_train_paths and len(local_train_paths) == len(train_paths):
        train_rows: list[dict[str, Any]] = []
        for path in local_train_paths:
            train_rows.extend(load_jsonl_rows(path, gcp_project))
        request["train_rows"] = train_rows
    if local_val_paths and len(local_val_paths) == len(val_paths):
        val_rows: list[dict[str, Any]] = []
        for path in local_val_paths:
            val_rows.extend(load_jsonl_rows(path, gcp_project))
        request["val_rows"] = val_rows

    # If rationale_source is a local file, pre-load it so the Modal worker
    # doesn't need GCS credentials or local filesystem access.
    if rationale_source and rationale_source != "solution" and not rationale_source.startswith("gs://"):
        request["rationale_by_problem"] = load_rationale_by_problem(rationale_source, gcp_project)
        print(f"Pre-loaded {len(request['rationale_by_problem'])} rationales from {rationale_source}")

    result = run_training_pipeline.remote(request)

    if output_dir.strip():
        local_results_root = Path(output_dir).expanduser().resolve()
    else:
        local_results_root = Path(__file__).parent / "results" / run_name
    local_results_root.mkdir(parents=True, exist_ok=True)

    download_from_modal_volume(
        volume_name=result["artifacts_volume_name"],
        remote_path=result["artifacts_volume_run_path"],
        local_destination=local_results_root,
    )

    write_json(local_results_root / "modal_return_summary.json", result)

    print(f"Run name: {run_name}")
    print(f"Stage: {result['stage']}")
    print(f"Training objective: {result.get('training_objective')}")
    print(f"Rationale source: {result.get('rationale_source')}")
    print(f"Downloaded artifacts to: {local_results_root}")
    print(f"Best hyperparameters: {result.get('best_hparams')}")
    if result.get("full_train_result"):
        print("Full training completed.")
        print(f"GCS checkpoint root: {gcs_checkpoint_root}/{run_name}")
    else:
        print("Full training was not run in this stage.")
