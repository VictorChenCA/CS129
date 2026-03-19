"""
Create a stratified split of the Hendrycks MATH dataset.

Quotas per (subject × difficulty) cell:
  - train: 200
  - test:  50
  - held-out: remainder

Difficulty groups:
  - easy:   Level 1, Level 2
  - medium: Level 3, Level 4
  - hard:   Level 5
"""

import json
import os
import random
from collections import defaultdict

SEED = 42
TRAIN_QUOTA = 200
TEST_QUOTA = 50

SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

DIFFICULTY_MAP = {
    "easy": {"Level 1", "Level 2"},
    "medium": {"Level 3", "Level 4"},
    "hard": {"Level 5"},
}

INPUT_DIR = os.path.join(os.path.dirname(__file__), "hendrycks_math")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "stratified")
HELDOUT_DIR = os.path.join(os.path.dirname(__file__), "stratified_heldout")


def load_subject(subject: str) -> list[dict]:
    """Load and combine train + test JSON files for a subject."""
    pool = []
    for split in ("train", "test"):
        path = os.path.join(INPUT_DIR, f"{subject}_{split}.json")
        with open(path) as f:
            data = json.load(f)
        # Add subject field for traceability
        for entry in data:
            entry["subject"] = subject
        pool.extend(data)
    return pool


def main():
    random.seed(SEED)

    # Accumulators: difficulty -> split -> list of entries
    buckets: dict[str, dict[str, list]] = {
        diff: {"train": [], "test": [], "heldout": []}
        for diff in DIFFICULTY_MAP
    }

    # Summary table: subject -> difficulty -> split -> count
    summary: dict[str, dict[str, dict[str, int]]] = {}

    for subject in SUBJECTS:
        pool = load_subject(subject)

        # Filter out anomalous "Level ?" entries
        pool = [e for e in pool if e.get("level", "").startswith("Level ")]
        filtered_out = sum(
            1 for e in load_subject(subject)
            if not e.get("level", "").startswith("Level ")
        )
        if filtered_out:
            print(f"  [{subject}] filtered out {filtered_out} entries with invalid level")

        summary[subject] = {}

        for diff, levels in DIFFICULTY_MAP.items():
            cell = [e for e in pool if e["level"] in levels]

            # Shuffle deterministically
            random.shuffle(cell)

            train = cell[:TRAIN_QUOTA]
            test = cell[TRAIN_QUOTA: TRAIN_QUOTA + TEST_QUOTA]
            heldout = cell[TRAIN_QUOTA + TEST_QUOTA:]

            # Tag entries
            for e in cell:
                e["difficulty_group"] = diff

            buckets[diff]["train"].extend(train)
            buckets[diff]["test"].extend(test)
            buckets[diff]["heldout"].extend(heldout)

            summary[subject][diff] = {
                "pool": len(cell),
                "train": len(train),
                "test": len(test),
                "heldout": len(heldout),
            }

    # Shuffle final combined lists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(HELDOUT_DIR, exist_ok=True)
    for diff in DIFFICULTY_MAP:
        for split in ("train", "test", "heldout"):
            random.shuffle(buckets[diff][split])
            dir_ = HELDOUT_DIR if split == "heldout" else OUTPUT_DIR
            path = os.path.join(dir_, f"{diff}_{split}.jsonl")
            with open(path, "w") as f:
                for entry in buckets[diff][split]:
                    f.write(json.dumps(entry) + "\n")

    # Print summary table
    print()
    header = f"{'Subject':<35} {'Diff':<8} {'Pool':>6} {'Train':>6} {'Test':>6} {'Heldout':>8}"
    print(header)
    print("-" * len(header))

    totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for subject in SUBJECTS:
        for diff in DIFFICULTY_MAP:
            s = summary[subject][diff]
            print(
                f"{subject:<35} {diff:<8} {s['pool']:>6} {s['train']:>6} "
                f"{s['test']:>6} {s['heldout']:>8}"
            )
            totals[diff]["pool"] += s["pool"]
            totals[diff]["train"] += s["train"]
            totals[diff]["test"] += s["test"]
            totals[diff]["heldout"] += s["heldout"]

    print("-" * len(header))
    for diff in DIFFICULTY_MAP:
        t = totals[diff]
        print(
            f"{'TOTAL':<35} {diff:<8} {t['pool']:>6} {t['train']:>6} "
            f"{t['test']:>6} {t['heldout']:>8}"
        )

    print()
    print("Output files:")
    for diff in DIFFICULTY_MAP:
        for split in ("train", "test", "heldout"):
            dir_ = HELDOUT_DIR if split == "heldout" else OUTPUT_DIR
            path = os.path.join(dir_, f"{diff}_{split}.jsonl")
            count = len(buckets[diff][split])
            print(f"  {path}  ({count} entries)")


if __name__ == "__main__":
    main()
