# Held-Out MATH Examples

The three held-out files contain all examples from the Hendrycks MATH dataset that were **not** selected for the train, valid, or test splits. They serve as a separate evaluation reservoir — useful for few-shot pool construction, secondary evaluation, or ablation studies — without touching the primary train/valid/test data.

## Files

| File | Entries | Contents |
|---|---|---|
| `easy_heldout.jsonl` | 1,423 | Easy examples (Level 1–2) not in train/valid/test |
| `medium_heldout.jsonl` | 3,807 | Medium examples (Level 3–4) not in train/valid/test |
| `hard_heldout.jsonl` | 1,808 | Hard examples (Level 5) not in train/valid/test |

**Total held-out:** 7,038 entries.

## How Held-Out Examples Were Selected

For each *(subject × difficulty)* cell, the full pool (original train + test combined) was shuffled with `random.seed(42)`. The first 180 went to train, the next 30 to valid, the next 50 to test. Everything after position 260 became held-out.

```
pool (shuffled) = [train_0 … train_179 | valid_0 … valid_29 | test_0 … test_49 | heldout_0 … heldout_N]
```

This guarantees **zero overlap** between held-out and the primary train/valid/test splits.

## Per-Subject Held-Out Counts

### Easy (Level 1 + Level 2)

| Subject | Pool | Train | Valid | Test | Held-out |
|---|---|---|---|---|---|
| algebra | 854 | 180 | 30 | 50 | 594 |
| counting_and_probability | 309 | 180 | 30 | 50 | 49 |
| geometry | 261 | 180 | 30 | 50 | 1 |
| intermediate_algebra | 405 | 180 | 30 | 50 | 145 |
| number_theory | 300 | 180 | 30 | 50 | 40 |
| prealgebra | 718 | 180 | 30 | 50 | 458 |
| precalculus | 396 | 180 | 30 | 50 | 136 |
| **Total** | **3,243** | **1,260** | **210** | **350** | **1,423** |

### Medium (Level 3 + Level 4)

| Subject | Pool | Train | Valid | Test | Held-out |
|---|---|---|---|---|---|
| algebra | 1,334 | 180 | 30 | 50 | 1,074 |
| counting_and_probability | 537 | 180 | 30 | 50 | 277 |
| geometry | 533 | 180 | 30 | 50 | 273 |
| intermediate_algebra | 1,084 | 180 | 30 | 50 | 824 |
| number_theory | 642 | 180 | 30 | 50 | 382 |
| prealgebra | 913 | 180 | 30 | 50 | 653 |
| precalculus | 584 | 180 | 30 | 50 | 324 |
| **Total** | **5,627** | **1,260** | **210** | **350** | **3,807** |

### Hard (Level 5)

| Subject | Pool | Train | Valid | Test | Held-out |
|---|---|---|---|---|---|
| algebra | 743 | 180 | 30 | 50 | 483 |
| counting_and_probability | 399 | 180 | 30 | 50 | 139 |
| geometry | 553 | 180 | 30 | 50 | 293 |
| intermediate_algebra | 709 | 180 | 30 | 50 | 449 |
| number_theory | 467 | 180 | 30 | 50 | 207 |
| prealgebra | 445 | 180 | 30 | 50 | 185 |
| precalculus | 312 | 180 | 30 | 50 | 52 |
| **Total** | **3,628** | **1,260** | **210** | **350** | **1,808** |

## Intended Uses

- **Few-shot pool:** Sample demonstration examples from held-out without contaminating train, valid, or test.
- **Secondary evaluation:** Evaluate on held-out after primary results are finalized to check for overfitting to the test set.
- **Ablation studies:** Vary train size by sampling subsets; held-out provides a clean additional evaluation set.
- **Data augmentation:** Use held-out problems as additional supervision in a second training phase.

> **Note:** The `geometry` easy held-out is minimal (1 entry) because the full geometry easy pool is only 261 examples — just enough to fill the train + valid + test quota of 260.

## Entry Format

Same as the train, valid, and test files. Each entry has:

```json
{
  "problem": "...",
  "level": "Level 1",
  "type": "Algebra",
  "solution": "...",
  "subject": "algebra",
  "difficulty_group": "easy"
}
```

Each `.jsonl` file contains one JSON object per line. See `../stratified/README.md` for the full field reference.
