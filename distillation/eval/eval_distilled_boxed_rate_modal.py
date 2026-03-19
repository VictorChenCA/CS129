"""Evaluate boxed{} usage rate for the knowledge-distilled baseline model.

Downloads the distilled model from GCS, runs greedy generation on the MATH
algebra test set (EleutherAI/hendrycks_math), and reports:
  - accuracy  (cross-check: should be ~20.3%)
  - boxed{} rate  (fraction of responses containing \\boxed{})

Usage:
    modal run distillation/eval/eval_distilled_boxed_rate_modal.py
"""

import json

import modal

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.49.0",
        "datasets",
        "peft",
        "accelerate",
        "numpy",
        "tqdm",
        "huggingface_hub",
        "google-cloud-storage",
    )
)

app = modal.App("eval-distilled-boxed-rate")

GCS_MODEL_PATH = "gs://checkpoints-cs224n/final_best/distilled_models/multi_task_algebra_distilled/"
SYSTEM_PROMPT = (
    "You are a helpful math tutor. Solve problems step by step, "
    "showing your reasoning clearly. Put your final answer in \\boxed{}."
)


@app.function(
    image=eval_image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("googlecloud-secret")],
)
def run_eval():
    import os
    import re
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ── GCS download ────────────────────────────────────────────────────────

    def parse_gcs_uri(uri: str):
        no_scheme = uri[len("gs://"):]
        bucket, _, blob = no_scheme.partition("/")
        return bucket, blob

    def download_gcs_prefix(gcs_prefix: str, local_dir: Path) -> None:
        from google.cloud import storage

        raw_json = os.environ.get("SERVICE_ACCOUNT_JSON", "").strip()
        if raw_json:
            client = storage.Client.from_service_account_info(json.loads(raw_json))
        else:
            client = storage.Client()

        bucket_name, prefix = parse_gcs_uri(gcs_prefix)
        if prefix and not prefix.endswith("/"):
            prefix = prefix + "/"
        bucket = client.bucket(bucket_name)
        blobs = client.list_blobs(bucket, prefix=prefix)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            rel = blob.name[len(prefix):] if prefix else blob.name
            dst = local_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dst))

    # ── Helper functions (copied verbatim from modal_evaluate.py) ────────────

    def extract_last_boxed(text):
        positions = []
        idx = 0
        while True:
            idx = text.find("\\boxed{", idx)
            if idx == -1:
                break
            positions.append(idx)
            idx += 1
        if not positions:
            return None
        last = positions[-1]
        start = last + len("\\boxed{")
        brace = 1
        pos = start
        while pos < len(text) and brace > 0:
            if text[pos] == "{":
                brace += 1
            elif text[pos] == "}":
                brace -= 1
            pos += 1
        return text[start: pos - 1].strip() if brace == 0 else None

    def normalize_ans(a):
        if not a:
            return ""
        a = a.strip().replace("\\$", "").replace("$", "")
        a = " ".join(a.split())
        prev = ""
        while prev != a:
            prev = a
            a = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", a)
        for lt, rp in [
            ("\\cdot", "*"), ("\\times", "*"), ("\\div", "/"),
            ("\\left", ""), ("\\right", ""),
        ]:
            a = a.replace(lt, rp)
        a = re.sub(r"\\([a-zA-Z]+)", r"\1", a)
        return a.lower().strip()

    def compare(extracted, gt):
        if not extracted or not gt:
            return False
        en, gn = normalize_ans(extracted), normalize_ans(gt)
        if en == gn:
            return True
        try:
            def ev(s):
                s = s.replace(" ", "")
                if "/" in s and "//" not in s:
                    p = s.split("/")
                    if len(p) == 2:
                        return float(p[0].strip("()")) / float(p[1].strip("()"))
                return float(s)
            if abs((ev(en) - ev(gn)) / (ev(gn) if ev(gn) != 0 else 1)) < 1e-4:
                return True
        except Exception:
            pass
        return False

    # ── Download model ───────────────────────────────────────────────────────

    model_dir = Path("/tmp/distilled_model")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model from {GCS_MODEL_PATH} …")
    download_gcs_prefix(GCS_MODEL_PATH, model_dir)
    print("Download complete.")

    # ── Load model & tokenizer ───────────────────────────────────────────────

    BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    adapter_config = model_dir / "adapter_config.json"
    if adapter_config.exists():
        print("LoRA adapter detected — loading with PeftModel and merging …")
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        model = model.merge_and_unload()
    else:
        print("No adapter_config.json found — treating as full merged model …")
        model = base_model

    model.eval()

    # ── Load algebra test set ────────────────────────────────────────────────

    print("Loading algebra test set …")
    ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test", trust_remote_code=True)
    problems = [ex["problem"] for ex in ds]
    solutions = [ex["solution"] for ex in ds]

    # Extract ground-truth answers from solution strings
    gt_answers = [extract_last_boxed(sol) for sol in solutions]

    print(f"Loaded {len(problems)} examples.")

    # ── Batched greedy generation ────────────────────────────────────────────

    BATCH_SIZE = 8
    MAX_NEW_TOKENS = 512

    all_responses = []
    for start in tqdm(range(0, len(problems), BATCH_SIZE), desc="Generating"):
        batch_problems = problems[start: start + BATCH_SIZE]
        prompts = []
        for prob in batch_problems:
            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Solve this problem:\n\n{prob}"},
            ]
            prompts.append(
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )

        tokenized = tokenizer(
            prompts,
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
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        prompt_lens = tokenized["attention_mask"].sum(dim=1).tolist()
        for idx, generated in enumerate(output_ids):
            continuation = generated[int(prompt_lens[idx]):]
            text = tokenizer.decode(continuation, skip_special_tokens=True)
            all_responses.append(text)

    # ── Compute metrics ──────────────────────────────────────────────────────

    n = len(all_responses)
    correct = 0
    used_boxed = 0

    per_level: dict = {}
    levels = [ex.get("level", "unknown") for ex in ds]

    for i, response in enumerate(all_responses):
        has_boxed = "\\boxed{" in response
        if has_boxed:
            used_boxed += 1

        extracted = extract_last_boxed(response)
        is_correct = compare(extracted, gt_answers[i])
        if is_correct:
            correct += 1

        level = levels[i]
        if level not in per_level:
            per_level[level] = {"total": 0, "correct": 0, "used_boxed": 0}
        per_level[level]["total"] += 1
        if is_correct:
            per_level[level]["correct"] += 1
        if has_boxed:
            per_level[level]["used_boxed"] += 1

    accuracy = correct / n
    boxed_rate = used_boxed / n

    for lvl in per_level:
        lvl_data = per_level[lvl]
        lvl_data["accuracy"] = lvl_data["correct"] / lvl_data["total"]
        lvl_data["boxed_rate"] = lvl_data["used_boxed"] / lvl_data["total"]

    result = {
        "model": GCS_MODEL_PATH,
        "num_examples": n,
        "accuracy": accuracy,
        "boxed_rate": boxed_rate,
        "correct": correct,
        "used_boxed": used_boxed,
        "per_level": per_level,
    }

    print(f"\n=== Results ===")
    print(f"Examples : {n}")
    print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Boxed{{}} rate: {boxed_rate:.4f} ({boxed_rate*100:.1f}%)")

    return result


@app.local_entrypoint()
def main():
    result = run_eval.remote()
    print(result)
    out_path = "distillation/eval/distilled_boxed_rate_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to {out_path}")
