from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from case_eval import ExampleResult, StepEvaluation, aggregate_dataset_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple CaSE result JSONs into one combined result.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CaSE result JSON files to merge.")
    parser.add_argument("--output", required=True, help="Path to save merged result JSON.")
    parser.add_argument("--force_total_loaded", type=int, default=None, help="Override merged total_examples_loaded.")
    parser.add_argument("--force_total_skipped", type=int, default=None, help="Override merged total_examples_skipped.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def step_eval_from_dict(raw: dict[str, Any]) -> StepEvaluation:
    return StepEvaluation(
        step_index=int(raw["step_index"]),
        step_text=str(raw["step_text"]),
        prior_context=list(raw.get("prior_context", [])),
        relevance_score=int(raw["relevance_score"]),
        relevance_reason=str(raw["relevance_reason"]),
        coherence_score=int(raw["coherence_score"]),
        coherence_reason=str(raw["coherence_reason"]),
        judge_model=str(raw["judge_model"]),
        judge_backend=str(raw["judge_backend"]),
    )


def example_from_dict(raw: dict[str, Any]) -> ExampleResult:
    return ExampleResult(
        id=str(raw["id"]),
        question=str(raw["question"]),
        answer=raw.get("answer"),
        rationale=str(raw["rationale"]),
        steps=list(raw.get("steps", [])),
        step_evaluations=[step_eval_from_dict(item) for item in raw.get("step_evaluations", [])],
        example_metrics=raw.get("example_metrics"),
        error=raw.get("error"),
    )


def main() -> None:
    args = parse_args()
    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    output_path = Path(args.output).expanduser().resolve()

    payloads = [load_json(path) for path in input_paths]
    combined_examples: list[ExampleResult] = []
    total_loaded = 0
    total_skipped = 0

    for payload in payloads:
        summary = payload["dataset_summary"]
        total_loaded += int(summary["total_examples_loaded"])
        total_skipped += int(summary["total_examples_skipped"])
        combined_examples.extend(example_from_dict(raw) for raw in payload.get("examples", []))

    combined_examples.sort(key=lambda ex: ex.id)

    failed_examples = sum(1 for ex in combined_examples if ex.error is not None)
    effective_total_loaded = total_loaded if args.force_total_loaded is None else int(args.force_total_loaded)
    effective_total_skipped = total_skipped if args.force_total_skipped is None else int(args.force_total_skipped)
    skipped_load = max(0, effective_total_skipped - failed_examples)
    merged_summary = aggregate_dataset_metrics(combined_examples, effective_total_loaded, skipped_load)

    config = dict(payloads[0].get("config", {}))
    config["merged_from"] = [str(path) for path in input_paths]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "config": config,
                "dataset_summary": merged_summary,
                "examples": [ex.__dict__ | {"step_evaluations": [step.__dict__ for step in ex.step_evaluations]} for ex in combined_examples],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved merged result: {output_path}")


if __name__ == "__main__":
    main()
