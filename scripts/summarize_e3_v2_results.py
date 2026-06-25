from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PRIMARY_VARIANTS = [
    "extra_training",
    "current_gate_only",
    "stabilized_unnormalized",
    "stabilized_renormalized",
    "uniform_gate",
    "shuffled_gate",
    "boundary_focused",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_nested(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_root.glob("*/*/metrics.json")):
        variant = metrics_path.parent.parent.name
        sample_id = metrics_path.parent.name
        metrics = read_json(metrics_path)

        diagnostics_path = metrics_path.parent / "gate_diagnostics.json"
        diagnostics: dict[str, Any] = {}
        if diagnostics_path.exists():
            diagnostics = read_json(diagnostics_path)

        gate_summary = diagnostics.get("gate_summary", {})
        low_5 = get_nested(diagnostics, ["low_gate", "bottom_5pct"], {})
        low_10 = get_nested(diagnostics, ["low_gate", "bottom_10pct"], {})

        rows.append(
            {
                "sample_id": sample_id,
                "variant": metrics.get("variant", variant),
                "requested_variant": metrics.get("requested_variant"),
                "seed": metrics.get("seed"),
                "ari": metrics.get("ari"),
                "delta_vs_original": metrics.get("delta_vs_original"),
                "delta_vs_baseline": metrics.get("delta_vs_baseline"),
                "baseline_ari": metrics.get("baseline_ari"),
                "original_reencoding_ari": metrics.get("original_reencoding_ari"),
                "renormalize_gate": metrics.get("renormalize_gate"),
                "mean_gate": metrics.get("mean_gate", gate_summary.get("mean_gate")),
                "std_gate": metrics.get("std_gate", gate_summary.get("std_gate")),
                "gate_p1": metrics.get("gate_p1", gate_summary.get("gate_p1")),
                "gate_p5": metrics.get("gate_p5", gate_summary.get("gate_p5")),
                "gate_p10": metrics.get("gate_p10", gate_summary.get("gate_p10")),
                "gate_p50": metrics.get("gate_p50", gate_summary.get("gate_p50")),
                "gate_p90": metrics.get("gate_p90", gate_summary.get("gate_p90")),
                "gate_p95": metrics.get("gate_p95", gate_summary.get("gate_p95")),
                "gate_p99": metrics.get("gate_p99", gate_summary.get("gate_p99")),
                "mean_effective_degree": metrics.get(
                    "mean_effective_degree",
                    gate_summary.get("mean_effective_degree"),
                ),
                "minimum_effective_degree": metrics.get(
                    "minimum_effective_degree",
                    gate_summary.get("minimum_effective_degree"),
                ),
                "gate_auc_for_cross_domain": diagnostics.get(
                    "gate_auc_for_cross_domain"
                ),
                "original_cross_domain_ratio": diagnostics.get(
                    "original_cross_domain_ratio"
                ),
                "low5_precision": low_5.get("low_gate_precision"),
                "low5_enrichment": low_5.get("low_gate_enrichment"),
                "low5_cdr": low_5.get("low_gate_cdr"),
                "low5_sdr": low_5.get("low_gate_sdr"),
                "low5_cdr_over_sdr": low_5.get("cdr_over_sdr"),
                "low10_precision": low_10.get("low_gate_precision"),
                "low10_enrichment": low_10.get("low_gate_enrichment"),
                "low10_cdr": low_10.get("low_gate_cdr"),
                "low10_sdr": low_10.get("low_gate_sdr"),
                "low10_cdr_over_sdr": low_10.get("cdr_over_sdr"),
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for variant in sorted(grouped, key=variant_sort_key):
        items = grouped[variant]
        summary: dict[str, Any] = {
            "variant": variant,
            "n_runs": len(items),
        }
        for key in [
            "ari",
            "delta_vs_original",
            "delta_vs_baseline",
            "mean_gate",
            "std_gate",
            "minimum_effective_degree",
            "gate_auc_for_cross_domain",
            "low5_enrichment",
            "low5_cdr_over_sdr",
            "low10_enrichment",
            "low10_cdr_over_sdr",
        ]:
            values = [as_float(item.get(key)) for item in items]
            clean = [value for value in values if value is not None]
            summary[f"{key}_mean"] = (
                sum(clean) / len(clean) if clean else None
            )
        summary_rows.append(summary)
    return summary_rows


def variant_sort_key(variant: str) -> tuple[int, str]:
    if variant in PRIMARY_VARIANTS:
        return (PRIMARY_VARIANTS.index(variant), variant)
    return (len(PRIMARY_VARIANTS), variant)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows found._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            elif value is None:
                cells.append("")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_rows = sorted(
        rows,
        key=lambda row: (
            str(row.get("sample_id")),
            variant_sort_key(str(row.get("variant"))),
            str(row.get("seed")),
        ),
    )
    text = [
        "# E3-v2 Result Summary",
        "",
        "## Variant-level summary",
        "",
        markdown_table(
            summary_rows,
            [
                "variant",
                "n_runs",
                "ari_mean",
                "delta_vs_original_mean",
                "gate_auc_for_cross_domain_mean",
                "low5_enrichment_mean",
                "low5_cdr_over_sdr_mean",
            ],
        ),
        "",
        "## Run-level summary",
        "",
        markdown_table(
            key_rows,
            [
                "sample_id",
                "variant",
                "seed",
                "ari",
                "delta_vs_original",
                "mean_gate",
                "gate_auc_for_cross_domain",
                "low5_enrichment",
                "low5_cdr_over_sdr",
            ],
        ),
        "",
        "## Minimal decision checks",
        "",
        "- Main method should be `stabilized_renormalized`.",
        "- It should outperform `extra_training` on mean ARI.",
        "- It should not be matched by `uniform_gate` or `shuffled_gate`.",
        "- Mechanism is weak if low-gate enrichment <= 1 or Gate AUC <= 0.5.",
        "",
    ]
    path.write_text("\n".join(text), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize E3-v2 ASG-STAGATE result folders."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/soft_gate_v2"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/soft_gate_v2_summary"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.results_root)
    summary_rows = group_summary(rows)

    write_csv(args.output_dir / "e3_v2_runs.csv", rows)
    write_csv(args.output_dir / "e3_v2_variant_summary.csv", summary_rows)
    write_markdown(args.output_dir / "e3_v2_summary.md", rows, summary_rows)

    print(f"Found {len(rows)} E3-v2 runs under {args.results_root}")
    print(f"Wrote summary files to {args.output_dir}")


if __name__ == "__main__":
    main()
