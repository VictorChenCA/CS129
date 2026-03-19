# CS129 — Distilling Step-by-Step on Hendrycks MATH

Fine-tunes a small language model (Qwen2.5-0.5B) on stratified difficulty splits of the Hendrycks MATH benchmark using synthetic rationales from Gemini, then evaluates reasoning quality with the CaSE framework.

---

## Pipeline

```
dataset/          →   distillation/         →   evaluation/          →   results/
Stratified splits     Rationale generation      CaSE reasoning            Final CSVs,
(easy/medium/hard)    + SFT training            quality evaluation         plots, tables
                      + holdout eval
```

---

## Directory Structure

```
CS129/
├── environment.yml                    # Conda environment (Linux)
├── gcs_utils/                         # Shared GCS upload/download utilities
├── dataset/                           # Hendrycks MATH with stratified difficulty splits
│   ├── create_stratified_split.py     # Generate easy/medium/hard splits
│   ├── upload_to_gcs.py
│   ├── hendrycks_math/                # Raw JSON by subject
│   ├── stratified/                    # Train/test/valid splits per difficulty
│   └── stratified_heldout/            # Held-out eval sets
├── holdout/                           # 150-problem holdout (50 per difficulty)
│   └── combined_holdout_50_each.jsonl
├── distillation/                      # Full SFT pipeline
│   ├── README.md                      # Step-by-step reproduction guide
│   ├── pseudo_rationales/             # Gemini rationale extraction
│   ├── train/                         # Model training (multitask, answer-only, etc.)
│   └── eval/                          # Holdout + baseline evaluation on Modal
├── evaluation/                        # CaSE reasoning quality evaluation
│   ├── case_eval.py                   # Core CaSE implementation
│   ├── parallel_inference.py          # Batched model inference for CaSE
│   ├── merge_results.py               # Merge per-model CaSE outputs
│   ├── plot_comparison.py             # Plot model comparisons
│   ├── case_eval_results/             # Per-run CaSE scores and plots
│   └── jsonl_model_output_for_CaSE/   # Raw model outputs
├── results/                           # Final experiment results
│   ├── final_results.csv
│   ├── final_results.json
│   ├── manifest.json
│   └── plots/                         # PNG figures
└── scripts/                           # Utility shell scripts
    ├── monitor_training.sh            # Watch training progress via tmux
    └── monitor_stop.sh                # Stop inference after N examples
```

---

## Quickstart

### 1. Dataset

```bash
python dataset/create_stratified_split.py
python dataset/upload_to_gcs.py
```

### 2. Generate rationales (optional — needed for synthetic variants)

```bash
python distillation/pseudo_rationales/extract_pseudo_rationales.py \
  --input_jsonl dataset/stratified/easy_train.jsonl \
               dataset/stratified/medium_train.jsonl \
               dataset/stratified/hard_train.jsonl \
  --out_jsonl distillation/pseudo_rationales/gemini_rationales.jsonl
```

### 3. Train

```bash
# Multitask SFT with human rationales (default)
modal run distillation/train/train_model_multitask_stratified.py \
  --stage search_then_full --training-objective multitask
```

See `distillation/README.md` for all five training objective variants.

### 4. Evaluate (holdout)

```bash
modal run distillation/eval/eval_stratified_heldout_modal.py \
  --run-multitask-human <run_name>
```

### 5. CaSE reasoning quality evaluation

```bash
# Run inference for all models
python evaluation/parallel_inference.py

# Score with CaSE
python evaluation/case_eval.py

# Merge and plot
python evaluation/merge_results.py
python evaluation/plot_comparison.py
```

---

## Environment

```bash
conda env create -f environment.yml
conda activate cs129
```

Requires Modal and GCP credentials for training/eval jobs.
