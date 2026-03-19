# Held-Out MATH Examples

The three held-out files contain all examples from the Hendrycks MATH dataset that were **not** selected for the train or test splits. They serve as a separate evaluation reservoir — useful for few-shot pool construction, secondary evaluation, or ablation studies — without touching the primary train/test data.

## Files

| File | Entries | Contents |
|---|---|---|
| `easy_heldout.jsonl` | 1,493 | Easy examples (Level 1–2) not in train/test |
| `medium_heldout.jsonl` | 3,877 | Medium examples (Level 3–4) not in train/test |
| `hard_heldout.jsonl` | 1,878 | Hard examples (Level 5) not in train/test |

**Total held-out:** 7,248 entries.

## How Held-Out Examples Were Selected

For each *(subject × difficulty)* cell, the full pool (original train + test combined) was shuffled with `random.seed(42)`. The first 200 went to train and the next 50 to test. Everything after position 250 became held-out.

```
pool (shuffled) = [train_0 … train_199 | test_0 … test_49 | heldout_0 … heldout_N]
```

This guarantees **zero overlap** between held-out and the primary train/test splits.

## Per-Subject Held-Out Counts

### Easy (Level 1 + Level 2)

| Subject | Pool | Train | Test | Held-out |
|---|---|---|---|---|
| algebra | 854 | 200 | 50 | 604 |
| counting_and_probability | 309 | 200 | 50 | 59 |
| geometry | 261 | 200 | 50 | 11 |
| intermediate_algebra | 405 | 200 | 50 | 155 |
| number_theory | 300 | 200 | 50 | 50 |
| prealgebra | 718 | 200 | 50 | 468 |
| precalculus | 396 | 200 | 50 | 146 |
| **Total** | **3,243** | **1,400** | **350** | **1,493** |

### Medium (Level 3 + Level 4)

| Subject | Pool | Train | Test | Held-out |
|---|---|---|---|---|
| algebra | 1,334 | 200 | 50 | 1,084 |
| counting_and_probability | 537 | 200 | 50 | 287 |
| geometry | 533 | 200 | 50 | 283 |
| intermediate_algebra | 1,084 | 200 | 50 | 834 |
| number_theory | 642 | 200 | 50 | 392 |
| prealgebra | 913 | 200 | 50 | 663 |
| precalculus | 584 | 200 | 50 | 334 |
| **Total** | **5,627** | **1,400** | **350** | **3,877** |

### Hard (Level 5)

| Subject | Pool | Train | Test | Held-out |
|---|---|---|---|---|
| algebra | 743 | 200 | 50 | 493 |
| counting_and_probability | 399 | 200 | 50 | 149 |
| geometry | 553 | 200 | 50 | 303 |
| intermediate_algebra | 709 | 200 | 50 | 459 |
| number_theory | 467 | 200 | 50 | 217 |
| prealgebra | 445 | 200 | 50 | 195 |
| precalculus | 312 | 200 | 50 | 62 |
| **Total** | **3,628** | **1,400** | **350** | **1,878** |

## Intended Uses

- **Few-shot pool:** Sample demonstration examples from held-out without contaminating train or test.
- **Secondary evaluation:** Evaluate on held-out after primary results are finalized to check for overfitting to the test set.
- **Ablation studies:** Vary train size by sampling subsets; held-out provides a clean additional evaluation set.
- **Data augmentation:** Use held-out problems as additional supervision in a second training phase.

> **Note:** The `geometry` easy held-out is small (11 entries) because the full geometry easy pool is only 261 examples — just enough to fill the train + test quota with 11 left over.

## Entry Format

Same as the train and test files. Each entry has:

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
