## Distilling Step-by-Step Pipeline

Four steps to reproduce all five paper model variants.

---

### Step 1: Upload dataset to GCS

Generate the stratified split locally, then push to GCS:

```bash
# From repo root
python dataset-cs129/create_stratified_split.py

gsutil -m cp dataset-cs129/stratified/*.jsonl \
  gs://cs224n-dapo-distill-data/stratified_math/stratified/

gsutil -m cp dataset-cs129/stratified_heldout/*.jsonl \
  gs://cs224n-dapo-distill-data/stratified_math/stratified_heldout/
```

---

### Step 2 (optional): Extract synthetic rationales locally

Uses Gemini on GCP Vertex AI to generate rationales for each training problem.
Required for **Rationale Prompting** and **Multitask SFT (synthetic)** variants.

Outputs a JSONL where each line is `{"problem": "...", "rationale": "..."}`.

Ensure you have Application Default Credentials set up:

```bash
gcloud auth application-default login
```

Install dependencies if not already in your environment:

```bash
pip install google-auth google-cloud-storage
```

Run extraction locally (processes all ~3780 train problems):

```bash
# From repo root
python dataset-cs129/distill/pseudo_rationales/extract_pseudo_rationales.py \
  --model_name google/gemini-2.5-flash \
  --gcp_project cs224n-dapo-distill \
  --vertex_location us-central1 \
  --input_jsonl dataset-cs129/stratified/easy_train.jsonl \
               dataset-cs129/stratified/medium_train.jsonl \
               dataset-cs129/stratified/hard_train.jsonl \
  --limit 3780 \
  --max_parallel_requests 8 \
  --out_jsonl dataset-cs129/distill/pseudo_rationales/gemini_rationales.jsonl
```

---

### Step 3: Train on Modal GPU

Single entrypoint: `train/train_model_multitask_stratified.py`.
See `train/README.md` for the full command table.

```bash
# Multitask SFT (human rationales) — default
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage search_then_full --training-objective multitask

# Multitask SFT (synthetic rationales)
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage search_then_full --training-objective multitask \
  --rationale-source dataset-cs129/distill/pseudo_rationales/gemini_rationales.jsonl

# Answer-only SFT
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage search_then_full --training-objective answer_only

# Rationale Prompting
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage search_then_full --training-objective answer_only \
  --rationale-in-prompt \
  --rationale-source dataset-cs129/distill/pseudo_rationales/gemini_rationales.jsonl
```

The **pretrained baseline** needs no training.

---

### Step 4: Evaluate on Modal GPU

```bash
modal run dataset-cs129/distill/eval/eval_stratified_heldout_modal.py \
  --run-answer-only <run_name> \
  --run-single-task-distill <run_name> \
  --run-multitask-human <run_name> \
  --run-multitask-synthetic <run_name>
```

Always evaluates the pretrained baseline. Leave any `--run_*` argument empty to skip that variant.
Results saved to `distilling_step_by_step-gemini/results/<run_name>/`.

---

### Notes

- `pseudo_rationales/` — Gemini rationale extraction via Vertex AI (runs locally)
- `train/build_train_set.py` — preprocessing helper for synthetic rationale JSONL
- All training/eval GPU work runs on Modal (`A10G`)
