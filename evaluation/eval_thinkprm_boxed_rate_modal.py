"""
Modal evaluation: ThinkPRM verifier + DAPO generator — boxed-format extraction rate.

Evaluates the full Hendrycks MATH algebra test set (~1187 examples) across 4 shards
(one A100 each).  For every example we record whether the ThinkPRM-selected candidate
contains a \\boxed{} token, allowing us to diagnose format compliance separately from
answer correctness.

Key parameters: N=4 candidates, max_new_tokens=2048, 4 shards.

Usage:
    modal run evaluation/eval_thinkprm_boxed_rate_modal.py
    modal run evaluation/eval_thinkprm_boxed_rate_modal.py --num-shards 4 --num-candidates 4

Download results:
    modal volume get targeted-training-artifacts /thinkprm_boxed_rate/ ./thinkprm_boxed_rate/
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import modal
from datasets import load_dataset

# ──────────────────────────── Modal Configuration ────────────────────────────

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "tqdm",
        "google-cloud-storage",
        "google-auth",
        "sympy",
    )
)

app = modal.App("CS224N-project")

model_volume = modal.Volume.from_name("trained-models", create_if_missing=True)
MODEL_DIR = "/models"

artifacts_volume = modal.Volume.from_name(
    "targeted-training-artifacts", create_if_missing=True
)
ARTIFACTS_DIR = "/artifacts"

# ──────────────────────────── Constants ──────────────────────────────────────

DAPO_GCS_PATH = "gs://checkpoints-cs224n/real_final_best/v3_optimized/targeted_dapo_qwen_math/"
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
HF_TOKEN = "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"

MODAL_SECRET_NAME = "googlecloud-secret"
SERVICE_ACCOUNT_ENV_KEYS = (
    "SERVICE_ACCOUNT_JSON",
    "CHECKPOINT_SERVICE_ACCOUNT_JSON",
    "JOSEPH_SERVICE_ACCOUNT_JSON",
    "SERVICE_ACCOUNT_JSON_JOSEPH",
    "GOOGLE_APPLICATION_CREDENTIALS_JSON",
    "GCP_SERVICE_ACCOUNT_JSON",
)

GPU_CONFIG = "A100"

THINKPRM_ADAPTER_PATH = f"{MODEL_DIR}/verifier_thinkprm"


# ──────────────────────────── Remote Shard Function ──────────────────────────

@app.function(
    image=eval_image,
    volumes={MODEL_DIR: model_volume, ARTIFACTS_DIR: artifacts_volume},
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 8,
    memory=32768,
    secrets=[modal.Secret.from_name(MODAL_SECRET_NAME)],
)
def run_thinkprm_boxed_eval_shard(
    limit: int,
    offset: int,
    shard_idx: int,
    num_candidates: int = 4,
    max_new_tokens: int = 2048,
    verifier_max_new_tokens: int = 2048,
    temperature: float = 0.4,
    top_p: float = 0.95,
    seed: int = 42,
    subject: str = "algebra",
):
    """Run ThinkPRM boxed-rate evaluation on one shard of the algebra test set."""
    import json
    import os
    import re
    import sys
    import tempfile
    import time
    from pathlib import Path

    import torch
    from datasets import load_dataset
    from huggingface_hub import login
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
        PEFT_AVAILABLE = True
    except ImportError:
        PeftModel = None
        PEFT_AVAILABLE = False

    from tqdm import tqdm

    try:
        import sympy
        SYMPY_AVAILABLE = True
    except ImportError:
        SYMPY_AVAILABLE = False

    login(token=HF_TOKEN)
    torch.manual_seed(seed + shard_idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + shard_idx)

    shard_start = time.time()

    # ────────────────── GCS Utilities ──────────────────

    def _build_gcs_client():
        from google.cloud import storage
        from google.oauth2 import service_account

        for env_key in SERVICE_ACCOUNT_ENV_KEYS:
            sa_json = os.environ.get(env_key)
            if not sa_json:
                continue
            sa_info = json.loads(sa_json)
            credentials = service_account.Credentials.from_service_account_info(sa_info)
            project_id = sa_info.get("project_id", "")
            client = storage.Client(
                credentials=credentials, project=project_id or None
            )
            print(f"  GCS auth via secret key: {env_key}")
            return client
        print("  GCS auth via ADC (application default credentials)")
        return storage.Client()

    def download_gcs_prefix(gcs_uri: str, local_dir: Path) -> int:
        if not gcs_uri.startswith("gs://"):
            raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")
        no_scheme = gcs_uri[len("gs://"):]
        bucket_name, _, prefix = no_scheme.partition("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        client = _build_gcs_client()
        local_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        total_bytes = 0
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            if blob.name.endswith("/"):
                continue
            rel = blob.name[len(prefix):] if prefix else blob.name
            if not rel:
                continue
            dst = local_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dst))
            count += 1
            total_bytes += int(blob.size or 0)
        print(f"  Downloaded {count} files ({total_bytes / 1e6:.1f} MB) from {gcs_uri}")
        return count

    # ────────────────── Model Loading ──────────────────

    def _get_dtype():
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def load_dapo_model(local_path: str):
        """Load DAPO generator (full model or LoRA adapter)."""
        dtype = _get_dtype()
        local = Path(local_path)

        if (local / "merged").is_dir():
            local = local / "merged"
            local_path = str(local)
            print(f"  Found merged/ subfolder, using: {local_path}")

        is_adapter = (local / "adapter_config.json").exists()
        if is_adapter and PEFT_AVAILABLE:
            print(f"  Loading DAPO as LoRA adapter from {local_path}")
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME, torch_dtype=dtype, device_map="auto"
            )
            model = PeftModel.from_pretrained(base, local_path)
            model = model.merge_and_unload()
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)
        else:
            print(f"  Loading DAPO as full model from {local_path}")
            model = AutoModelForCausalLM.from_pretrained(
                local_path, torch_dtype=dtype, device_map="auto"
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        return model, tokenizer

    def load_verifier_model(adapter_path: str, base_model_name: str = BASE_MODEL_NAME):
        """Load verifier model with LoRA adapter (r=16, alpha=32) on Qwen-0.5B."""
        dtype = _get_dtype()
        local = Path(adapter_path)

        if not local.exists():
            raise FileNotFoundError(
                f"Verifier adapter not found at {adapter_path}. "
                f"Train it first with victor-verify/train_thinkprm_modal.py"
            )

        if (local / "adapter_config.json").exists() and PEFT_AVAILABLE:
            print(f"  Loading verifier LoRA adapter from {adapter_path}")
            adapter_cfg = json.loads((local / "adapter_config.json").read_text())
            print(f"    LoRA r={adapter_cfg.get('r')}, alpha={adapter_cfg.get('lora_alpha')}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=dtype, device_map="auto"
            )
            model = PeftModel.from_pretrained(base, adapter_path)
            model = model.merge_and_unload()
        else:
            print(f"  Loading verifier as full model from {adapter_path}")
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                adapter_path, torch_dtype=dtype, device_map="auto"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        return model, tokenizer

    # ────────────────── Prompt Building ──────────────────

    def build_generation_prompt(problem: str) -> str:
        return (
            "Solve the following math problem step by step.\n\n"
            f"Problem:\n{problem}\n\n"
            "Instructions:\n"
            "- Show your reasoning and work step by step.\n"
            "- At the end, put your final answer in \\boxed{your_answer} format.\n"
            "- Example: If the answer is 42, end with \\boxed{42}.\n\n"
            "Solution:"
        )

    def build_verification_prompt(problem: str, candidate: str) -> str:
        return (
            "You are given a math problem and a proposed step-by-step solution:\n\n"
            "[Math Problem]\n\n"
            f"{problem}\n\n"
            "[Solution]\n\n"
            f"{candidate}\n\n"
            "Review and critique each step in the proposed solution to determine "
            "whether each step is correct. If the solution is incomplete, only "
            "verify the provided steps"
        )

    # ────────────────── Answer Extraction / Comparison ──────────────────

    def extract_gold_answer(example):
        if "answer" in example:
            return str(example["answer"]).strip()
        if "solution" in example:
            solution = str(example["solution"])
            boxed_match = re.search(r'\\boxed\s*\{([^}]+)\}', solution)
            if boxed_match:
                return boxed_match.group(1).strip()
            return solution.strip()
        return ""

    def clean_latex_answer(answer: str) -> str:
        if not answer:
            return ""
        if '\\color{' in answer:
            color_match = re.search(r'\\color\{[^}]*\}?', answer)
            if color_match:
                after_pos = color_match.end()
                if after_pos < len(answer):
                    after_color = answer[after_pos:].strip()
                    after_color = re.sub(r'[{}]*$', '', after_color)
                    if after_color and not after_color.startswith('\\'):
                        answer = after_color
                else:
                    return ""
        answer = re.sub(r'\\textcolor\{[^}]+\}\{([^}]+)\}', r'\1', answer)
        answer = re.sub(r'\\color\{[^}]+\}', '', answer)
        answer = re.sub(r'\\(?:textbf|textit|mathrm|text)\{([^}]+)\}', r'\1', answer)
        answer = re.sub(r'\\(?:textbf|textit|text|mathrm|emph)\s+', '', answer)
        answer = re.sub(r'\{\s*\}', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.strip('{}')
        return answer.strip()

    def extract_pred_answer(text: str) -> str:
        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
            if answer:
                return clean_latex_answer(answer)
        boxed_matches = list(re.finditer(r"\\boxed\s*\{", text))
        if boxed_matches:
            last_match = boxed_matches[-1]
            start_pos = last_match.end()
            brace_count = 1
            pos = start_pos
            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1
            if brace_count == 0:
                answer = text[start_pos:pos - 1].strip()
                if answer:
                    cleaned = clean_latex_answer(answer)
                    if cleaned:
                        return cleaned
            else:
                partial = text[start_pos:].strip()
                partial = re.sub(r'\\[a-zA-Z]+\{?$', '', partial)
                partial = re.sub(r'\{[^}]*$', '', partial)
                if partial.strip():
                    cleaned = clean_latex_answer(partial)
                    if cleaned:
                        return cleaned
        boxed_simple = re.search(r"\\boxed\s+([^\s$.,;]+)", text)
        if boxed_simple:
            return clean_latex_answer(boxed_simple.group(1).strip())
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if lines:
            for line in reversed(lines[-3:]):
                number_match = re.search(
                    r'(?:answer|is|equals?|=\s*)(?:the\s+)?(\d+(?:\.\d+)?)',
                    line, re.IGNORECASE,
                )
                if number_match:
                    return number_match.group(1).strip()
            return clean_latex_answer(lines[-1])
        return ""

    def normalize_answer(answer: str) -> str:
        answer = " ".join(answer.split())
        answer = answer.replace("$", "").replace("\\", "").replace("{", "").replace("}", "")
        return answer.lower().strip()

    def is_correct(pred: str, gold: str) -> bool:
        if not pred or not gold:
            return False
        pred_norm = normalize_answer(pred)
        gold_norm = normalize_answer(gold)
        if not pred_norm or not gold_norm:
            return False
        if pred_norm == gold_norm:
            return True
        try:
            def eval_numeric(s):
                s = s.replace(' ', '').replace(',', '')
                if '/' in s and '//' not in s:
                    parts = s.split('/')
                    if len(parts) == 2:
                        return float(parts[0].strip('()')) / float(parts[1].strip('()'))
                return float(s)
            pred_val = eval_numeric(pred_norm)
            gold_val = eval_numeric(gold_norm)
            if gold_val != 0:
                if abs((pred_val - gold_val) / gold_val) < 1e-4:
                    return True
            elif abs(pred_val - gold_val) < 1e-6:
                return True
        except (ValueError, ZeroDivisionError, TypeError):
            pass
        if SYMPY_AVAILABLE:
            try:
                pred_expr = sympy.simplify(pred_norm)
                gold_expr = sympy.simplify(gold_norm)
                if sympy.simplify(pred_expr - gold_expr) == 0:
                    return True
            except Exception:
                pass
        return False

    # ────────────────── Generation Helper ──────────────────

    def _model_ctx_len(tokenizer, fallback=8192):
        ml = getattr(tokenizer, "model_max_length", fallback)
        if ml is None or ml > 100_000:
            return fallback
        return int(ml)

    def generate_solution(model, tokenizer, problem, max_tokens, temp, tp, device,
                          do_sample: bool = True):
        prompt = build_generation_prompt(problem)
        try:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt_text = prompt
        inputs = tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=_model_ctx_len(tokenizer),
        ).to(device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = tp
        else:
            gen_kwargs["temperature"] = 0.0
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        gen_ids = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    # ────────────────── ThinkPRM Scoring ──────────────────

    def _token_ids(tokenizer, text):
        return tokenizer.encode(text, add_special_tokens=False)

    def _next_token_ratio(next_logits, yes_id, no_id):
        import torch.nn.functional as F
        log_probs = F.log_softmax(next_logits, dim=-1)
        log_p_yes = log_probs[yes_id]
        log_p_no = log_probs[no_id]
        log_sum = torch.logsumexp(torch.stack([log_p_yes, log_p_no]), dim=0)
        return torch.exp(log_p_yes - log_sum).item()

    def _sequence_logprob(model, tokenizer, context_text, continuation_text, device):
        import torch.nn.functional as F
        ctx_ids = tokenizer(context_text, return_tensors='pt', add_special_tokens=False).to(device)
        cont_ids = tokenizer(continuation_text, return_tensors='pt', add_special_tokens=False).to(device)
        full_ids = torch.cat([ctx_ids['input_ids'], cont_ids['input_ids']], dim=1)
        with torch.no_grad():
            logits = model(full_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)
        ctx_len = ctx_ids["input_ids"].shape[1]
        lps = []
        for i, tid in enumerate(cont_ids['input_ids'][0]):
            pos = ctx_len + i - 1
            if pos >= 0:
                lps.append(log_probs[0, pos, tid].item())
        return sum(lps)

    def score_with_thinkprm(model, tokenizer, problem, candidate_text,
                            max_tokens=2048, device="cuda"):
        """
        Score candidate with ThinkPRM: P(yes) / (P(yes) + P(no)).
        Returns (score, verification_chain_text). Score = -1.0 on failure.
        """
        prompt = build_verification_prompt(problem, candidate_text)
        try:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ) + "\nLet's verify step by step:"
        except Exception:
            prompt_text = prompt + "\nLet's verify step by step:"

        old_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        verification_chain = ""
        try:
            max_ctx = _model_ctx_len(tokenizer, fallback=16384)
            inputs = tokenizer(
                prompt_text, return_tensors="pt",
                truncation=True, max_length=max_ctx,
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            verification_chain = tokenizer.decode(gen_ids, skip_special_tokens=True)

            forced_question = 'Is the solution correct?'
            forced_context = prompt_text + verification_chain + '\n' + forced_question
            forced_inputs = tokenizer(
                forced_context, return_tensors='pt',
                truncation=True, max_length=max_ctx,
            ).to(device)
            with torch.no_grad():
                forced_out = model(**forced_inputs)
                next_logits = forced_out.logits[0, -1, :]

            yes_id = no_id = None
            for cand in ['yes', ' yes', 'Yes', ' Yes']:
                ids = _token_ids(tokenizer, cand)
                if len(ids) == 1:
                    yes_id = ids[0]
                    break
            for cand in ['no', ' no', 'No', ' No']:
                ids = _token_ids(tokenizer, cand)
                if len(ids) == 1:
                    no_id = ids[0]
                    break

            if yes_id is not None and no_id is not None:
                score = _next_token_ratio(next_logits, yes_id, no_id)
                return score, verification_chain
            else:
                lp_yes = _sequence_logprob(model, tokenizer, forced_context, " yes", device)
                lp_no = _sequence_logprob(model, tokenizer, forced_context, " no", device)
                log_sum = torch.logsumexp(torch.tensor([lp_yes, lp_no]), dim=0)
                score = torch.exp(torch.tensor(lp_yes) - log_sum).item()
                return score, verification_chain
        except Exception as e:
            print(f"  ThinkPRM scoring error: {e}", file=sys.stderr)
            return -1.0, verification_chain
        finally:
            tokenizer.truncation_side = old_side

    # ────────────────── Boxed-Rate Helper ──────────────────

    def has_boxed(text: str) -> bool:
        return bool(re.search(r'\\boxed\s*\{', text))

    # ==================================================================
    #  MAIN SHARD LOGIC
    # ==================================================================

    print("=" * 80)
    print(f"SHARD {shard_idx}: offset={offset}, limit={limit}")
    print("=" * 80)

    # ── Step 1: Download DAPO from GCS ──
    print(f"\n[Shard {shard_idx}] Downloading DAPO generator from GCS...")
    dapo_local = Path(tempfile.mkdtemp(prefix=f"dapo_model_shard{shard_idx}_"))
    n_files = download_gcs_prefix(DAPO_GCS_PATH, dapo_local)
    if n_files == 0:
        raise RuntimeError(f"No files downloaded from {DAPO_GCS_PATH}")

    # ── Step 2: Load dataset shard ──
    print(f"\n[Shard {shard_idx}] Loading dataset shard...")
    dataset = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
    dataset = dataset.select(range(offset, min(offset + limit, len(dataset))))
    print(f"  Shard size: {len(dataset)} examples")

    # ── Step 3: Load DAPO model ──
    print(f"\n[Shard {shard_idx}] Loading DAPO model...")
    dapo_model, dapo_tokenizer = load_dapo_model(str(dapo_local))
    dapo_device = str(next(dapo_model.parameters()).device)
    print(f"  DAPO on device: {dapo_device}")

    # ── Step 4: Load ThinkPRM verifier ──
    print(f"\n[Shard {shard_idx}] Loading ThinkPRM verifier from {THINKPRM_ADAPTER_PATH}...")
    thinkprm_model, thinkprm_tokenizer = load_verifier_model(THINKPRM_ADAPTER_PATH)
    thinkprm_device = str(next(thinkprm_model.parameters()).device)
    print(f"  ThinkPRM on device: {thinkprm_device}")

    # ── Step 5: Prepare output directory ──
    output_dir = Path(ARTIFACTS_DIR) / "thinkprm_boxed_rate" / str(shard_idx)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    # ── Step 6: Per-example loop ──
    results = []
    baseline_correct_count = 0
    selected_correct_count = 0
    baseline_boxed_count = 0
    selected_boxed_count = 0
    total_parse_failures = 0

    with open(results_path, "w", encoding="utf-8") as out_f:
        for local_idx, example in enumerate(tqdm(dataset, desc=f"Shard {shard_idx}")):
            global_idx = offset + local_idx
            problem = example["problem"]
            gold_answer = extract_gold_answer(example)
            level = example.get("level", "Unknown")

            try:
                # Baseline: greedy generation
                baseline_text = generate_solution(
                    dapo_model, dapo_tokenizer, problem,
                    max_new_tokens, temperature, top_p, dapo_device,
                    do_sample=False,
                )
                baseline_answer = extract_pred_answer(baseline_text)
                baseline_ok = is_correct(baseline_answer, gold_answer)
                baseline_boxed = has_boxed(baseline_text)
                if baseline_ok:
                    baseline_correct_count += 1
                if baseline_boxed:
                    baseline_boxed_count += 1

                # N sampled candidates + ThinkPRM scoring
                candidates_info = []
                valid_candidates = []
                example_parse_failures = 0

                for c_idx in range(num_candidates):
                    cand_text = generate_solution(
                        dapo_model, dapo_tokenizer, problem,
                        max_new_tokens, temperature, top_p, dapo_device,
                        do_sample=True,
                    )
                    cand_answer = extract_pred_answer(cand_text)

                    score, verif_chain = score_with_thinkprm(
                        thinkprm_model, thinkprm_tokenizer,
                        problem, cand_text,
                        max_tokens=verifier_max_new_tokens,
                        device=thinkprm_device,
                    )

                    candidates_info.append({
                        "candidate_idx": c_idx,
                        "reasoning": cand_text,
                        "answer": cand_answer,
                        "verifier_score": float(score),
                        "verification_chain": verif_chain[:500],
                    })

                    if score >= 0:
                        valid_candidates.append((c_idx, score, cand_text, cand_answer))
                    else:
                        example_parse_failures += 1
                        total_parse_failures += 1

                # Weighted majority voting
                best_text = baseline_text
                best_answer = baseline_answer
                best_score = None

                if valid_candidates:
                    vote_totals = {}
                    best_cand_for_answer = {}
                    best_score_for_answer = {}

                    for c_idx, score, c_text, c_answer in valid_candidates:
                        if not c_answer or not c_answer.strip():
                            continue
                        if c_answer not in vote_totals:
                            vote_totals[c_answer] = 0.0
                            best_cand_for_answer[c_answer] = (c_idx, c_text)
                            best_score_for_answer[c_answer] = score
                        vote_totals[c_answer] += score
                        if score > best_score_for_answer[c_answer]:
                            best_cand_for_answer[c_answer] = (c_idx, c_text)
                            best_score_for_answer[c_answer] = score

                    if vote_totals:
                        winning_answer = max(vote_totals, key=lambda a: vote_totals[a])
                        _, best_text = best_cand_for_answer[winning_answer]
                        best_score = best_score_for_answer[winning_answer]
                        best_answer = winning_answer

                selected_correct = is_correct(best_answer, gold_answer)
                selected_boxed = has_boxed(best_text)

                if selected_correct:
                    selected_correct_count += 1
                if selected_boxed:
                    selected_boxed_count += 1

                row = {
                    "id": global_idx,
                    "shard_idx": shard_idx,
                    "problem": problem,
                    "difficulty_level": level,
                    "gold_answer": gold_answer,
                    "baseline_reasoning": baseline_text,
                    "baseline_answer": baseline_answer,
                    "baseline_correct": baseline_ok,
                    "baseline_has_boxed": baseline_boxed,
                    "candidates": candidates_info,
                    "selected_reasoning": best_text,
                    "selected_answer": best_answer,
                    "selected_correct": selected_correct,
                    "selected_has_boxed": selected_boxed,
                    "best_verifier_score": float(best_score) if best_score is not None else None,
                    "thinkprm_parse_failures": example_parse_failures,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                results.append(row)

            except Exception as e:
                print(f"  [Shard {shard_idx}] Error on example {global_idx}: {e}", file=sys.stderr)
                continue

    # ── Step 7: Save shard summary ──
    n = len(results)
    elapsed_min = (time.time() - shard_start) / 60.0
    summary = {
        "shard_idx": shard_idx,
        "offset": offset,
        "limit": limit,
        "num_examples": n,
        "baseline_accuracy": round(baseline_correct_count / n, 4) if n else 0.0,
        "selected_accuracy": round(selected_correct_count / n, 4) if n else 0.0,
        "baseline_boxed_rate": round(baseline_boxed_count / n, 4) if n else 0.0,
        "selected_boxed_rate": round(selected_boxed_count / n, 4) if n else 0.0,
        "baseline_correct": baseline_correct_count,
        "selected_correct": selected_correct_count,
        "baseline_boxed": baseline_boxed_count,
        "selected_boxed": selected_boxed_count,
        "total_parse_failures": total_parse_failures,
        "elapsed_minutes": round(elapsed_min, 1),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    artifacts_volume.commit()

    print(f"\n[Shard {shard_idx}] Done in {elapsed_min:.1f} min")
    print(f"  Examples:           {n}")
    print(f"  Baseline accuracy:  {summary['baseline_accuracy']:.4f}")
    print(f"  Selected accuracy:  {summary['selected_accuracy']:.4f}")
    print(f"  Baseline boxed rate:{summary['baseline_boxed_rate']:.4f}")
    print(f"  Selected boxed rate:{summary['selected_boxed_rate']:.4f}")

    return summary


# ──────────────────────────── Local Entrypoint ───────────────────────────────

@app.local_entrypoint()
def main(
    num_shards: int = 4,
    num_candidates: int = 4,
    max_new_tokens: int = 2048,
    verifier_max_new_tokens: int = 2048,
    temperature: float = 0.4,
    top_p: float = 0.95,
    seed: int = 42,
    subject: str = "algebra",
    total_limit: int = 0,
):
    print("=" * 70)
    print("  ThinkPRM Boxed-Rate Evaluation (Modal, Sharded)")
    print("=" * 70)
    print(f"  DAPO generator:   {DAPO_GCS_PATH}")
    print(f"  ThinkPRM adapter: {THINKPRM_ADAPTER_PATH}")
    print(f"  Base model:       {BASE_MODEL_NAME}")
    print(f"  Dataset:          EleutherAI/hendrycks_math/{subject} (test)")
    print(f"  Num shards:       {num_shards}")
    print(f"  Candidates/ex:    {num_candidates}")
    print(f"  max_new_tokens:   {max_new_tokens}")
    print(f"  Temperature:      {temperature}")
    print(f"  total_limit:      {total_limit} (0 = full dataset)")
    print("=" * 70)

    # Compute total and per-shard limits
    ds = load_dataset("EleutherAI/hendrycks_math", subject, split="test")
    total = len(ds) if total_limit == 0 else min(total_limit, len(ds))
    print(f"\nTotal examples to evaluate: {total}")

    base = total // num_shards
    remainder = total % num_shards

    shard_configs = []
    current_offset = 0
    for i in range(num_shards):
        shard_limit = base + (1 if i < remainder else 0)
        shard_configs.append((i, shard_limit, current_offset))
        current_offset += shard_limit

    print("\nShard plan:")
    for i, lim, off in shard_configs:
        print(f"  Shard {i}: offset={off}, limit={lim} (examples {off}–{off+lim-1})")
    print()

    # Dispatch shards in parallel
    with ThreadPoolExecutor(max_workers=num_shards) as executor:
        futures = {
            i: executor.submit(
                run_thinkprm_boxed_eval_shard.remote,
                limit=lim,
                offset=off,
                shard_idx=i,
                num_candidates=num_candidates,
                max_new_tokens=max_new_tokens,
                verifier_max_new_tokens=verifier_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                subject=subject,
            )
            for i, lim, off in shard_configs
        }
        summaries = {i: futures[i].result() for i in futures}

    # Aggregate results
    total_examples = sum(s["num_examples"] for s in summaries.values())
    total_baseline_correct = sum(s["baseline_correct"] for s in summaries.values())
    total_selected_correct = sum(s["selected_correct"] for s in summaries.values())
    total_baseline_boxed = sum(s["baseline_boxed"] for s in summaries.values())
    total_selected_boxed = sum(s["selected_boxed"] for s in summaries.values())
    total_parse_failures = sum(s["total_parse_failures"] for s in summaries.values())

    baseline_accuracy = total_baseline_correct / total_examples if total_examples else 0.0
    selected_accuracy = total_selected_correct / total_examples if total_examples else 0.0
    baseline_boxed_rate = total_baseline_boxed / total_examples if total_examples else 0.0
    selected_boxed_rate = total_selected_boxed / total_examples if total_examples else 0.0

    aggregated = {
        "dataset": f"EleutherAI/hendrycks_math/{subject}",
        "dapo_generator": DAPO_GCS_PATH,
        "num_shards": num_shards,
        "num_candidates": num_candidates,
        "max_new_tokens": max_new_tokens,
        "verifier_max_new_tokens": verifier_max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "total_examples": total_examples,
        "baseline_accuracy": round(baseline_accuracy, 4),
        "selected_accuracy": round(selected_accuracy, 4),
        "accuracy_delta": round(selected_accuracy - baseline_accuracy, 4),
        "baseline_boxed_rate": round(baseline_boxed_rate, 4),
        "selected_boxed_rate": round(selected_boxed_rate, 4),
        "total_baseline_correct": total_baseline_correct,
        "total_selected_correct": total_selected_correct,
        "total_baseline_boxed": total_baseline_boxed,
        "total_selected_boxed": total_selected_boxed,
        "total_parse_failures": total_parse_failures,
        "per_shard_summaries": summaries,
    }

    print("\n" + "=" * 70)
    print("  AGGREGATED RESULTS")
    print("=" * 70)
    print(f"  Total examples:       {total_examples}")
    print(f"  Baseline accuracy:    {baseline_accuracy:.4f} ({total_baseline_correct}/{total_examples})")
    print(f"  Selected accuracy:    {selected_accuracy:.4f} ({total_selected_correct}/{total_examples})")
    print(f"  Accuracy delta:       {selected_accuracy - baseline_accuracy:+.4f}")
    print(f"  Baseline boxed rate:  {baseline_boxed_rate:.4f} ({total_baseline_boxed}/{total_examples})")
    print(f"  Selected boxed rate:  {selected_boxed_rate:.4f} ({total_selected_boxed}/{total_examples})")
    print(f"  Total parse failures: {total_parse_failures}")
    print("=" * 70)

    # Save aggregated results locally
    local_out = "evaluation/thinkprm_boxed_rate_results.json"
    with open(local_out, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"\nAggregated results saved to: {local_out}")
    print("\nTo download per-shard JSONL files:")
    print("  modal volume get targeted-training-artifacts /thinkprm_boxed_rate/ ./thinkprm_boxed_rate/")
