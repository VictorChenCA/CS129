"""Evaluate the out-of-the-box Qwen2.5-0.5B-Instruct baseline on Modal GPU.

Evaluates on:
  1. Stratified test splits (easy / medium / hard)
  2. Hendrycks MATH – Algebra test set

Each split runs on its own A100 in parallel (4 containers total).

Run (from repo root):
  modal run dataset-cs129/distill/eval/eval_baseline_modal.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import modal

APP_NAME = "distill-baseline-eval"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

LOCAL_TEST_SPLITS = {
    "easy_test": Path("dataset-cs129/stratified/easy_test.jsonl"),
    "medium_test": Path("dataset-cs129/stratified/medium_test.jsonl"),
    "hard_test": Path("dataset-cs129/stratified/hard_test.jsonl"),
}

LOCAL_ALGEBRA_TEST = Path("dataset-cs129/hendrycks_math/algebra_test.json")

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.49.0",
        "accelerate>=0.26.0",
        "tqdm",
    )
)

app = modal.App(APP_NAME)


def load_local_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_local_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@app.function(
    image=eval_image,
    gpu="A100",
    timeout=60 * 60 * 4,
    memory=32768,
)
def evaluate_split_remote(
    split_name: str,
    rows: list[dict[str, Any]],
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> dict[str, Any]:
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        return lines[-1] if lines else ""

    def normalize_answer(answer: str) -> str:
        answer = " ".join(str(answer).split())
        answer = answer.replace("$", "").replace("\\", "")
        answer = answer.replace("{", "").replace("}", "")
        return answer.lower().strip()

    def calculate_accuracy(predictions: list[str], ground_truth: list[str]) -> float:
        if not predictions:
            return 0.0
        correct = sum(
            normalize_answer(p) == normalize_answer(g)
            for p, g in zip(predictions, ground_truth)
        )
        return correct / len(predictions)

    def build_prompt(problem: str, tokenizer) -> str:
        messages = [{"role": "user", "content": f"Solve the following math problem.\n\nProblem:\n{problem}\n"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"[{split_name}] Loading {BASE_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    prompts = [build_prompt(str(row.get("problem", "")), tokenizer) for row in rows]
    gold = [extract_boxed_payload(row.get("solution", "")) for row in rows]
    preds: list[str] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc=split_name):
        batch_prompts = prompts[start: start + batch_size]
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        device = next(model.parameters()).device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

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
            continuation = generated[prompt_lens[idx]:]
            text = tokenizer.decode(continuation, skip_special_tokens=True)
            preds.append(extract_boxed_payload(text))

    accuracy = calculate_accuracy(preds, gold)
    print(f"[{split_name}] accuracy={accuracy:.4f} ({len(rows)} examples)")
    return {"split": split_name, "count": len(rows), "accuracy": accuracy}


@app.local_entrypoint()
def main(
    max_new_tokens: int = 512,
    batch_size: int = 8,
) -> None:
    splits: list[tuple[str, list[dict[str, Any]]]] = []

    for split_name, path in LOCAL_TEST_SPLITS.items():
        rows = load_local_jsonl(path)
        splits.append((split_name, rows))
        print(f"Loaded {split_name}: {len(rows)} examples")

    algebra_rows = load_local_json(LOCAL_ALGEBRA_TEST)
    splits.append(("algebra_test", algebra_rows))
    print(f"Loaded algebra_test: {len(algebra_rows)} examples")

    # Fan out: one A100 per split (4 containers in parallel)
    split_results = list(evaluate_split_remote.map(
        [s[0] for s in splits],
        [s[1] for s in splits],
        kwargs={"max_new_tokens": max_new_tokens, "batch_size": batch_size},
    ))

    output_dir = Path("dataset-cs129/distill/results/baseline_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"baseline_eval_{timestamp}.json"
    out_path.write_text(json.dumps({"model": BASE_MODEL_NAME, "splits": split_results}, indent=2))

    print("\n=== Baseline Evaluation Results ===")
    for info in split_results:
        print(f"  {info['split']:20s}  accuracy={info['accuracy']:.4f}  n={info['count']}")
    print(f"\nSaved to: {out_path}")
