"""Evaluate stratified heldout splits on Modal GPU for all model variants.

Supports:
  - Pretrained baseline (no adapter, model_type="pretrained")
  - Answer-only SFT (LoRA checkpoint)
  - Single Task Distill (LoRA checkpoint, generates rationale then answer)
  - Multitask SFT human (LoRA checkpoint)
  - Multitask SFT synthetic (LoRA checkpoint)

All fine-tuned variants are evaluated with a problem-only prompt; the model
generates its own chain-of-thought if trained to do so.
Evaluates on heldout splits (easy/medium/hard) and optionally on test splits.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "distill-heldout-eval"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GCP_PROJECT = "cs224n-dapo-distill"

ARTIFACTS_VOLUME_NAME = "distill-multitask-artifacts"
ARTIFACTS_DIR = "/artifacts"

DEFAULT_NEW_MODEL_RUN_NAME = "qwen25_05b_lora_reasoning_run_local_20260304a"
DEFAULT_RESULTS_DIR_NAME = "heldout_eval_multimodel"

LOCAL_HELDOUT_SPLITS = {
    "easy": Path("dataset-cs129/stratified_heldout/easy_heldout.jsonl"),
    "hard": Path("dataset-cs129/stratified_heldout/hard_heldout.jsonl"),
    "medium": Path("dataset-cs129/stratified_heldout/medium_heldout.jsonl"),
}

LOCAL_TEST_SPLITS = {
    "easy": Path("dataset-cs129/stratified/easy_test.jsonl"),
    "hard": Path("dataset-cs129/stratified/hard_test.jsonl"),
    "medium": Path("dataset-cs129/stratified/medium_test.jsonl"),
}

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.49.0",
        "peft",
        "datasets",
        "google-cloud-storage",
        "tqdm",
        "matplotlib",
    )
)

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME_NAME, create_if_missing=True)


def load_local_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@app.function(
    image=eval_image,
    gpu="A100",
    timeout=60 * 60 * 12,
    memory=32768,
    volumes={ARTIFACTS_DIR: artifacts_volume},
)
def evaluate_models_remote(
    model_spec: dict[str, str],
    heldout_rows_by_split: dict[str, list[dict[str, Any]]],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    gcp_project: str = GCP_PROJECT,
) -> dict[str, Any]:
    import os
    import tempfile
    from pathlib import Path

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def parse_gcs_uri(uri: str) -> tuple[str, str]:
        no_scheme = uri[len("gs://") :]
        bucket, _, blob = no_scheme.partition("/")
        if not bucket:
            raise ValueError(f"Invalid GCS URI: {uri}")
        return bucket, blob

    def download_gcs_prefix_to_dir(gcs_prefix: str, local_dir: Path, project: str) -> None:
        from google.cloud import storage

        raw_json = os.environ.get("SERVICE_ACCOUNT_JSON", "").strip()
        service_account_info = None
        if raw_json:
            service_account_info = json.loads(raw_json)

        bucket_name, prefix = parse_gcs_uri(gcs_prefix)
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        if service_account_info:
            client = storage.Client.from_service_account_info(service_account_info, project=project)
        else:
            client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)
        blobs = client.list_blobs(bucket, prefix=prefix)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            rel = blob.name[len(prefix) :] if prefix else blob.name
            dst = local_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dst))

    def extract_boxed_payload(text: Any) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            if answer:
                return answer
        boxed_match = re.search(r"\\boxed\s*\{([^}]+)\}", text)
        if boxed_match:
            return boxed_match.group(1).strip()
        boxed_simple = re.search(r"\\boxed\s+([^\s$.,;]+)", text)
        if boxed_simple:
            return boxed_simple.group(1).strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[-1]
        return ""

    def normalize_answer(answer: str) -> str:
        answer = " ".join(str(answer).split())
        answer = answer.replace("$", "").replace("\\", "")
        answer = answer.replace("{", "").replace("}", "")
        return answer.lower().strip()

    def calculate_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
        if not predictions:
            return 0.0
        correct = 0
        for pred, gold in zip(predictions, ground_truth):
            if normalize_answer(pred) == normalize_answer(gold):
                correct += 1
        return correct / len(predictions)

    def build_prompt(problem: str, tokenizer) -> str:
        user_content = f"Solve the following math problem.\n\nProblem:\n{problem}\n"
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def load_model_and_tokenizer(model_path: str, base_model_name: str, project: str, model_type: str = "lora"):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        if model_type == "pretrained":
            # Pretrained baseline: load the base model directly, no adapter
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model.eval()
            return model, tokenizer

        if model_path.startswith("gs://"):
            model_temp = Path(tempfile.mkdtemp(prefix="heldout_model_dl_"))
            download_gcs_prefix_to_dir(model_path, model_temp, project=project)
            resolved_path = model_temp
        else:
            resolved_path = Path(model_path)

        adapter_config = resolved_path / "adapter_config.json"
        if adapter_config.exists():
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, str(resolved_path))
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(resolved_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
        model.eval()
        return model, tokenizer

    def evaluate_one_split(
        model: Any,
        tokenizer: Any,
        rows: list[dict[str, Any]],
        split_name: str,
    ) -> dict[str, Any]:
        prompts = [
            build_prompt(str(row.get("problem", "")), tokenizer)
            for row in rows
        ]
        gold = [extract_boxed_payload(row.get("solution", "")) for row in rows]
        preds: list[str] = []

        for start in tqdm(range(0, len(prompts), batch_size), desc=f"{split_name}"):
            batch_prompts = prompts[start : start + batch_size]
            tokenized = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            device = next(model.parameters()).device
            tokenized = {key: value.to(device) for key, value in tokenized.items()}

            with torch.inference_mode():
                output_ids = model.generate(
                    **tokenized,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            prompt_lens = tokenized["attention_mask"].sum(dim=1).tolist()
            for idx, generated in enumerate(output_ids):
                continuation = generated[prompt_lens[idx] :]
                text = tokenizer.decode(continuation, skip_special_tokens=True)
                preds.append(extract_boxed_payload(text))

        accuracy = calculate_accuracy(preds, gold)
        return {
            "split": split_name,
            "count": len(rows),
            "accuracy": accuracy,
        }

    split_names = list(heldout_rows_by_split.keys())
    name = model_spec["name"]
    model_path = model_spec.get("path", "")
    base_model_name = model_spec.get("base_model_name", BASE_MODEL_NAME)
    model_type = model_spec.get("model_type", "lora")
    model_result: dict[str, Any] = {
        "name": name,
        "path": model_path,
        "model_type": model_type,
        "status": "ok",
        "splits": [],
    }
    try:
        model, tokenizer = load_model_and_tokenizer(
            model_path=model_path,
            base_model_name=base_model_name,
            project=gcp_project,
            model_type=model_type,
        )
        for split_name in split_names:
            split_rows = heldout_rows_by_split[split_name]
            split_result = evaluate_one_split(
                model=model,
                tokenizer=tokenizer,
                rows=split_rows,
                split_name=split_name,
            )
            model_result["splits"].append(split_result)
        model_result["macro_accuracy"] = (
            sum(split["accuracy"] for split in model_result["splits"]) / len(model_result["splits"])
            if model_result["splits"]
            else None
        )
    except Exception as exc:
        model_result["status"] = "error"
        model_result["error"] = str(exc)

    return model_result


def plot_results(results: dict[str, Any], output_dir: Path, eval_test_splits: bool = False) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    split_order = ["easy", "hard", "medium"]
    model_rows = results.get("models", [])

    labels = [model["name"] for model in model_rows]
    x = list(range(len(labels)))
    width = 0.22
    split_offsets = {"easy": -width, "hard": 0.0, "medium": width}
    split_colors = {"easy": "#1f77b4", "hard": "#d62728", "medium": "#2ca02c"}

    plt.figure(figsize=(12, 6))
    for split in split_order:
        values = []
        for model in model_rows:
            if model.get("status") != "ok":
                values.append(0.0)
                continue
            split_map = {row["split"]: row["accuracy"] for row in model.get("splits", [])}
            values.append(float(split_map.get(split, 0.0)))
        bar_positions = [idx + split_offsets[split] for idx in x]
        split_label = "test" if eval_test_splits else "heldout"
        plt.bar(bar_positions, values, width=width, label=f"{split}_{split_label}", color=split_colors[split], alpha=0.8)

    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Exact Match Accuracy")
    plt.ylim(0.0, 1.0)
    title_label = "Test" if eval_test_splits else "Heldout"
    plt.title(f"{title_label} Accuracy by Split (easy / medium / hard)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()


@app.local_entrypoint()
def main(
    run_name: str = "",
    # Model run names / GCS paths for each variant (empty string = skip that model)
    run_multitask_human: str = "",
    run_multitask_synthetic: str = "",
    run_answer_only: str = "",
    run_single_task_distill: str = "",
    gcp_secret_name: str = "",
    max_new_tokens: int = 512,
    batch_size: int = 8,
    eval_test_splits: bool = False,
) -> None:
    """Evaluate all configured model variants on heldout (and optionally test) splits.

    Each run_* argument accepts either:
      - A Modal artifacts run name (used as {ARTIFACTS_DIR}/{run_name}/full_train/final_model)
      - A GCS path (gs://...)
    Leave empty to skip that variant.
    """
    if not run_name:
        run_name = f"{DEFAULT_RESULTS_DIR_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    heldout_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    splits_source = LOCAL_TEST_SPLITS if eval_test_splits else LOCAL_HELDOUT_SPLITS
    for split_name, path in splits_source.items():
        heldout_rows_by_split[split_name] = load_local_jsonl(path)

    def artifact_path(run: str) -> str:
        if run.startswith("gs://"):
            return run
        return f"{ARTIFACTS_DIR}/{run}/full_train/final_model"

    models: list[dict[str, Any]] = [
        {
            "name": "pretrained_baseline",
            "path": "",
            "model_type": "pretrained",
            "base_model_name": BASE_MODEL_NAME,
        },
    ]
    if run_multitask_human.strip():
        models.append({
            "name": "multitask_sft_human",
            "path": artifact_path(run_multitask_human),
            "model_type": "lora",
            "base_model_name": BASE_MODEL_NAME,
        })
    if run_multitask_synthetic.strip():
        models.append({
            "name": "multitask_sft_synthetic",
            "path": artifact_path(run_multitask_synthetic),
            "model_type": "lora",
            "base_model_name": BASE_MODEL_NAME,
        })
    if run_answer_only.strip():
        models.append({
            "name": "answer_only_sft",
            "path": artifact_path(run_answer_only),
            "model_type": "lora",
            "base_model_name": BASE_MODEL_NAME,
        })
    if run_single_task_distill.strip():
        models.append({
            "name": "single_task_distill",
            "path": artifact_path(run_single_task_distill),
            "model_type": "lora",
            "base_model_name": BASE_MODEL_NAME,
        })

    eval_fn = evaluate_models_remote
    if gcp_secret_name.strip():
        eval_fn = evaluate_models_remote.with_options(
            secrets=[modal.Secret.from_name(gcp_secret_name.strip())]
        )

    # Fan out one container per model, all running in parallel.
    model_results = list(eval_fn.map(
        models,
        kwargs={
            "heldout_rows_by_split": heldout_rows_by_split,
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "gcp_project": GCP_PROJECT,
        },
    ))

    remote_results = {"models": model_results}

    output_dir = Path("dataset-cs129/distill/results") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    split_label = "test" if eval_test_splits else "heldout"
    (output_dir / f"{split_label}_eval_summary.json").write_text(
        json.dumps(remote_results, indent=2),
        encoding="utf-8",
    )
    plot_results(remote_results, output_dir=output_dir, eval_test_splits=eval_test_splits)

    # Copy training curves from the multitask-human run if available.
    if run_multitask_human.strip() and not run_multitask_human.startswith("gs://"):
        src_metrics = (
            Path("dataset-cs129/distill/train/results")
            / run_multitask_human
            / run_multitask_human
            / "full_train"
            / "metrics"
        )
        dst_metrics = output_dir / "training_metrics_curves"
        if src_metrics.exists():
            if dst_metrics.exists():
                shutil.rmtree(dst_metrics)
            shutil.copytree(src_metrics, dst_metrics)
            print(f"Copied training curves to: {dst_metrics}")
        else:
            print("Training metric curves source folder was not found.")

    print(f"Saved {split_label} evaluation results to: {output_dir}")
    print(f"Summary JSON: {output_dir / f'{split_label}_eval_summary.json'}")
    print(f"Comparison plot: {output_dir / 'model_comparison.png'}")
