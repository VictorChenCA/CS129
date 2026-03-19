from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODEL_ORDER = [
    ("sft", "SFT"),
    ("grpo", "GRPO"),
    ("targeted_dapo", "Targeted DAPO"),
    ("distilled_multitask", "Distilled Multitask"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a matplotlib comparison table for CaSE eval results.")
    parser.add_argument("--results_dir", required=True, help="Directory containing per-model CaSE eval JSON outputs.")
    parser.add_argument("--plots_dir", default=None, help="Directory to save generated plots. Defaults to <results_dir>/plots.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_result_path(results_dir: Path, model_key: str) -> Path:
    direct = results_dir / f"{model_key}_case_eval_results.json"
    if direct.exists():
        return direct

    alt = list(results_dir.glob(f"*{model_key}*.json"))
    if len(alt) == 1:
        return alt[0]
    if not alt:
        raise FileNotFoundError(f"Could not find result JSON for model_key={model_key} in {results_dir}")
    raise FileExistsError(f"Multiple candidate result JSONs found for model_key={model_key}: {alt}")


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def format_float(value: float) -> str:
    return f"{value:.3f}"


def build_summary_rows(results_dir: Path) -> tuple[list[str], list[list[str]], list[list[float]], str]:
    metric_specs = [
        ("mean_step_relevance", "Step Rel"),
        ("mean_step_coherence", "Step Coh"),
        ("mean_example_relevance", "Ex Rel"),
        ("mean_example_coherence", "Ex Coh"),
        ("fraction_examples_all_steps_relevant", "All Rel"),
        ("fraction_examples_all_steps_coherent", "All Coh"),
    ]

    headers = ["Model", "Examples", "Steps"] + [label for _, label in metric_specs]
    table_rows: list[list[str]] = []
    numeric_rows: list[list[float]] = []
    judge_model = ""

    for model_key, display_name in DEFAULT_MODEL_ORDER:
        payload = load_json(find_result_path(results_dir, model_key))
        summary = payload["dataset_summary"]
        config = payload.get("config", {})
        judge_model = judge_model or str(config.get("model_name", ""))

        row = [
            display_name,
            str(summary["total_examples_evaluated"]),
            str(summary["total_steps_evaluated"]),
        ]
        numeric = [float(summary["total_examples_evaluated"]), float(summary["total_steps_evaluated"])]

        for metric_key, _ in metric_specs:
            value = float(summary[metric_key])
            numeric.append(value)
            if metric_key.startswith("fraction_"):
                row.append(format_pct(value))
            else:
                row.append(format_float(value))

        table_rows.append(row)
        numeric_rows.append(numeric)

    return headers, table_rows, numeric_rows, judge_model


def save_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(headers)]
    for row in rows:
        escaped = [f"\"{cell}\"" if "," in cell else cell for cell in row]
        lines.append(",".join(escaped))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_table(
    headers: list[str],
    rows: list[list[str]],
    numeric_rows: list[list[float]],
    judge_model: str,
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    value_matrix = np.array([row[2:] for row in numeric_rows], dtype=float)
    metric_max = value_matrix.max(axis=0)
    metric_min = value_matrix.min(axis=0)
    metric_spread = np.where(metric_max - metric_min == 0, 1.0, metric_max - metric_min)

    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.85)

    header_color = "#243b53"
    header_text_color = "white"
    zebra_a = "#f7fafc"
    zebra_b = "#edf2f7"
    accent = plt.get_cmap("YlGn")

    for col_idx in range(len(headers)):
        cell = table[(0, col_idx)]
        cell.set_facecolor(header_color)
        cell.set_text_props(color=header_text_color, weight="bold")
        cell.set_edgecolor("#d9e2ec")

    for row_idx in range(1, len(rows) + 1):
        base = zebra_a if row_idx % 2 else zebra_b
        for col_idx in range(len(headers)):
            cell = table[(row_idx, col_idx)]
            cell.set_edgecolor("#d9e2ec")
            if col_idx < 3:
                cell.set_facecolor(base)
                if col_idx == 0:
                    cell.set_text_props(weight="bold")
                continue

            metric_idx = col_idx - 3
            metric_value = numeric_rows[row_idx - 1][col_idx - 1]
            norm = (metric_value - metric_min[metric_idx]) / metric_spread[metric_idx]
            blend = accent(0.18 + 0.62 * norm)
            cell.set_facecolor(blend)
            if metric_value == metric_max[metric_idx]:
                cell.set_text_props(weight="bold")

    title = "CaSE Reasoning Quality Comparison"
    subtitle = "Higher is better across all displayed metrics"
    if judge_model:
        subtitle = f"{subtitle} | Judge: {judge_model}"

    fig.text(0.5, 0.95, title, ha="center", va="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.91, subtitle, ha="center", va="center", fontsize=10.5, color="#486581")
    fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.88])
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_metric_bars(
    headers: list[str],
    rows: list[list[str]],
    numeric_rows: list[list[float]],
    judge_model: str,
    output_png: Path,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    model_names = [row[0] for row in rows]
    metric_labels = headers[3:]
    metric_values = np.array([row[2:] for row in numeric_rows], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    axes = axes.flatten()
    colors = ["#33658A", "#86BBD8", "#758E4F", "#F26419"]

    for idx, (ax, metric_name) in enumerate(zip(axes, metric_labels)):
        values = metric_values[:, idx]
        bars = ax.bar(model_names, values, color=colors, edgecolor="#1F2933", linewidth=0.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(metric_name, fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.22, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=18)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(0.98, value + 0.03),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    title = "CaSE Metric Comparison"
    subtitle = "Per-metric view across all four models"
    if judge_model:
        subtitle = f"{subtitle} | Judge: {judge_model}"

    fig.text(0.5, 0.97, title, ha="center", va="center", fontsize=18, weight="bold")
    fig.text(0.5, 0.94, subtitle, ha="center", va="center", fontsize=10.5, color="#486581")
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.91])
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()
    plots_dir = (results_dir / "plots") if args.plots_dir is None else Path(args.plots_dir).expanduser().resolve()
    table_png = plots_dir / "case_eval_comparison_table.png"
    bars_png = plots_dir / "case_eval_comparison_bars.png"
    output_csv = plots_dir / "case_eval_comparison.csv"

    headers, rows, numeric_rows, judge_model = build_summary_rows(results_dir)
    render_table(headers, rows, numeric_rows, judge_model, table_png)
    render_metric_bars(headers, rows, numeric_rows, judge_model, bars_png)
    save_csv(output_csv, headers, rows)

    print(f"Saved PNG: {table_png}")
    print(f"Saved PNG: {bars_png}")
    print(f"Saved CSV: {output_csv}")


if __name__ == "__main__":
    main()
