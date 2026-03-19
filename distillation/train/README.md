# Training

Single entrypoint: `train_model_multitask_stratified.py` (Modal GPU).

---

## Model variants

| Variant | `--training-objective` | `--rationale-source` | `--rationale-in-prompt` | `--lambda-rationale` |
|---|---|---|---|---|
| Answer-only SFT | `answer_only` | `solution` | omit | — |
| Rationale Prompting | `answer_only` | `gemini_rationales.jsonl` | pass flag | — |
| Single Task Distill | `multitask` | `gemini_rationales.jsonl` | omit | `0` |
| Multitask SFT (human rationales) | `multitask` | `solution` | omit | `1` (default) |
| Multitask SFT (synthetic rationales) | `multitask` | `gemini_rationales.jsonl` | omit | `1` (default) |

The pretrained baseline requires no training — evaluate directly with `eval/eval_stratified_heldout_modal.py`.

---

## Commands

```bash
# Answer-only SFT
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage search_then_full \
  --training-objective answer_only

# Rationale Prompting
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage search_then_full \
  --training-objective answer_only \
  --rationale-in-prompt \
  --rationale-source dataset-cs129/distill/pseudo_rationales/gemini_rationales.jsonl

# Single Task Distill (Gemini rationale in output, answer loss only)
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage search_then_full \
  --training-objective multitask \
  --lambda-rationale 0 \
  --rationale-source dataset-cs129/distill/pseudo_rationales/gemini_rationales.jsonl

# Multitask SFT (human rationales from dataset solution field)
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage search_then_full \
  --training-objective multitask

# Multitask SFT (synthetic rationales from Gemini)
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage search_then_full \
  --training-objective multitask \
  --rationale-source dataset-cs129/distill/pseudo_rationales/gemini_rationales.jsonl

# Full train only (skip grid search, supply best hparams from a prior search)
modal run dataset-cs129/distill/train/train_model_multitask_stratified.py \
  --stage full_train \
  --training-objective multitask \
  --best-hparams-json '{"learning_rate":2e-5,"lora_r":16,"lora_alpha":32,"lora_dropout":0.05}'
```

---

## Rationale source format

`--rationale-source` accepts:
- `solution` (default) — use the `solution` field from each training row.
- A local path to a JSONL file where each line has `{"problem": "...", "rationale": "..."}`. Generate this with `pseudo_rationales/extract_pseudo_rationales.py`.
