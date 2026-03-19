# Stratified MATH Dataset Split

A balanced, reproducible re-split of the [Hendrycks MATH](https://github.com/hendrycks/math/) dataset with fixed per-cell quotas across all seven subjects and three difficulty tiers. The original train and test sets are **combined** per subject, then re-split with `random.seed(42)`: the first 200 per _(subject × difficulty)_ cell go to train, the next 50 to test, and the remainder to held-out. Two anomalous `Level ?` entries in Geometry are excluded.

## Files

| File                 | Entries | Description                     |
| -------------------- | ------- | ------------------------------- |
| `easy_train.jsonl`   | 1,400   | 200 easy examples per subject   |
| `easy_test.jsonl`    | 350     | 50 easy examples per subject    |
| `medium_train.jsonl` | 1,400   | 200 medium examples per subject |
| `medium_test.jsonl`  | 350     | 50 medium examples per subject  |
| `hard_train.jsonl`   | 1,400   | 200 hard examples per subject   |
| `hard_test.jsonl`    | 350     | 50 hard examples per subject    |

**Total:** 4,200 entries across 6 files. Held-out files are in `../stratified_heldout/`.

## Difficulty Groups

| Group    | Original Levels  | Rationale                 |
| -------- | ---------------- | ------------------------- |
| `easy`   | Level 1, Level 2 | Foundational problems     |
| `medium` | Level 3, Level 4 | Intermediate problems     |
| `hard`   | Level 5          | Most challenging problems |

**Subjects (7):** `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus`

## Entry Format

Each entry is a JSON object with the following fields:

```json
{
  "problem": "Find the value of ...",
  "level": "Level 2",
  "type": "Algebra",
  "solution": "We begin by ...",
  "subject": "algebra",
  "difficulty_group": "easy"
}
```

| Field              | Source   | Description                                                                        |
| ------------------ | -------- | ---------------------------------------------------------------------------------- |
| `problem`          | Original | Problem statement                                                                  |
| `level`            | Original | Original Hendrycks level: `Level 1`, `Level 2`, `Level 3`, `Level 4`, `Level 5`   |
| `type`             | Original | Subject category: `Algebra`, `Counting & Probability`, `Geometry`, `Intermediate Algebra`, `Number Theory`, `Prealgebra`, `Precalculus` |
| `solution`         | Original | Full worked solution                                                               |
| `subject`          | Added    | Snake-case subject name: `algebra`, `counting_and_probability`, `geometry`, `intermediate_algebra`, `number_theory`, `prealgebra`, `precalculus` |
| `difficulty_group` | Added    | Difficulty tier: `easy`, `medium`, or `hard`                                       |

## Reproducing

```bash
cd dataset/
python3 create_stratified_split.py
```

Each `.jsonl` file contains one JSON object per line.

Requires only the Python standard library and the 14 source files in `dataset/hendrycks_math/`.
