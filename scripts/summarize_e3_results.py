from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = [
    "extra_training",
    "gate_only",
    "gate_distribution",
    "full",
]

DEFAULT_SAMPLES = [
    "151674",
    "151676",
    "HBC",
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def pick(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[Any]) -> float | None:
    numbers = [safe_float(value) for value in values]
    numbers = [value for value in numbers if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def fmt(value: Any) -> str:
    number = safe_float(value)
    return "" if number is None else f"{number:.4f}"


def flatten_low_gate(diagnostics: dict[str, Any], bucket: str) -> dict[str, Any]:
    low_gate = diagnostics.get("low_gate", {})
    item = low_gate.get(bucket, {})
    return {
        f"{bucket}_precision": item.get("low_gate_precision"),
        f"{bucket}_enrichment": item.get("low_gate_enrichment"),
        f"{bucket}_cdr": item.get("low_gate_cdr"),
        f"{bucket}_sdr": item.get("low_gate_sdr"),
        f"{bucket}_cdr_over_sdr": item.get("cdr_over_sdr"),
        f"{bucket}_same_domain_high_gate_retention": item.get(
            "same_domain_high_gate_retention"
        ),
    }


def collect_soft_gate(
    results_root: Path,
    variants: list[str],
    samples: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for variant in variants:
        for sample in samples:
            run_dir = results_root / "soft_gate" / variant / sample
            metrics = read_json(run_dir / "metrics.json")
            diagnostics = read_json(run_dir / "gate_diagnostics.json")

            if not metrics:
                rows.append(
                    {
                        "method_group": "soft_gate",
                        "variant": variant,
                        "sample_id": sample,
                        "status": "missing",
                        "run_dir": str(run_dir),
                    }
                )
                continue

            gate_summary = diagnostics.get("gate_summary", {})
            row: dict[str, Any] = {
                "method_group": "soft_gate",
                "variant": variant,
                "sample_id": sample,
                "status": "ok",
                "run_dir": str(run_dir),
                "ari": pick(metrics, "ari", "soft_gate_ari"),
                "baseline_ari": metrics.get("baseline_ari"),
                "original_reencoding_ari": metrics.get("original_reencoding_ari"),
                "delta_vs_baseline": metrics.get("delta_vs_baseline"),
                "delta_vs_original": metrics.get("delta_vs_original"),
                "mean_gate": pick(
                    metrics,
                    "mean_gate",
                    default=gate_summary.get("mean_gate"),
                ),
                "std_gate": pick(
                    metrics,
                    "std_gate",
                    default=gate_summary.get("std_gate"),
                ),
                "min_gate": pick(
                    metrics,
                    "minimum_gate",
                    default=gate_summary.get("minimum_gate"),
                ),
                "max_gate": pick(
                    metrics,
                    "maximum_gate",
                    default=gate_summary.get("maximum_gate"),
                ),
                "mean_effective_degree": pick(
                    metrics,
                    "mean_effective_degree",
                    default=gate_summary.get("mean_effective_degree"),
                ),
                "min_effective_degree": pick(
                    metrics,
                    "minimum_effective_degree",
                    default=gate_summary.get("minimum_effective_degree"),
                ),
                "max_effective_degree": pick(
                    metrics,
                    "maximum_effective_degree",
                    default=gate_summary.get("maximum_effective_degree"),
                ),
                "learned_beta": metrics.get("learned_beta"),
                "learned_eta": metrics.get("learned_eta"),
                "final_total_loss": metrics.get("final_total_loss"),
                "final_reconstruction_loss": metrics.get("final_reconstruction_loss"),
                "final_budget_loss": metrics.get("final_budget_loss"),
                "final_degree_loss": metrics.get("final_degree_loss"),
                "final_preserve_loss": metrics.get("final_preserve_loss"),
                "original_cross_domain_ratio": diagnostics.get(
                    "original_cross_domain_ratio"
                ),
                "gate_auc_for_cross_domain": diagnostics.get(
                    "gate_auc_for_cross_domain"
                ),
                "n_preserve_edges": pick(
                    diagnostics,
                    "n_preserve_edges",
                    default=metrics.get("n_preserve_edges"),
                ),
            }
            row.update(flatten_low_gate(diagnostics, "bottom_5pct"))
            row.update(flatten_low_gate(diagnostics, "bottom_10pct"))
            rows.append(row)

    return rows


def collect_reference_methods(
    results_root: Path,
    samples: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    references = [
        ("baseline", results_root / "stagate"),
        ("original", results_root / "rule_based" / "original"),
        ("expression", results_root / "rule_based" / "expression"),
        ("embedding", results_root / "rule_based" / "embedding"),
        ("oracle", results_root / "oracle"),
        ("cluster_free", results_root / "cluster_free"),
    ]

    for method, root in references:
        for sample in samples:
            metrics = read_json(root / sample / "metrics.json")
            if not metrics:
                rows.append(
                    {
                        "method_group": "reference",
                        "variant": method,
                        "sample_id": sample,
                        "status": "missing",
                        "run_dir": str(root / sample),
                    }
                )
                continue

            rows.append(
                {
                    "method_group": "reference",
                    "variant": method,
                    "sample_id": sample,
                    "status": "ok",
                    "run_dir": str(root / sample),
                    "ari": pick(
                        metrics,
                        "ari",
                        "rule_based_ari",
                        "oracle_ari",
                        "cluster_free_ari",
                        "soft_gate_ari",
                    ),
                    "baseline_ari": metrics.get("baseline_ari"),
                    "original_reencoding_ari": metrics.get(
                        "original_reencoding_ari"
                    ),
                    "delta_vs_baseline": pick(
                        metrics,
                        "delta_vs_baseline",
                        "delta_ari",
                    ),
                    "delta_vs_original": metrics.get("delta_vs_original"),
                }
            )

    return rows


def build_ari_table(rows: list[dict[str, Any]], variants: list[str], samples: list[str]) -> str:
    lines = [
        "## ARI by method",
        "",
        "| Method | " + " | ".join(samples) + " | Mean ARI | Mean Δ vs Original |",
        "|---|" + "|".join(["---:"] * (len(samples) + 2)) + "|",
    ]

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    ordered_methods = [
        ("baseline", "baseline"),
        ("original", "original"),
        ("oracle", "oracle"),
        ("expression", "expression"),
        ("embedding", "embedding"),
        ("cluster_free", "cluster_free"),
        *[(variant, variant) for variant in variants],
    ]

    for method_key, method_label in ordered_methods:
        subset = [row for row in ok_rows if row.get("variant") == method_key]
        by_sample = {row["sample_id"]: row for row in subset}
        ari_values = [by_sample.get(sample, {}).get("ari") for sample in samples]
        delta_values = [
            by_sample.get(sample, {}).get("delta_vs_original")
            for sample in samples
        ]
        if not any(value is not None for value in ari_values):
            continue
        sample_cells = " | ".join(
            fmt(by_sample.get(sample, {}).get("ari")) for sample in samples
        )
        lines.append(
            f"| {method_label} | {sample_cells} | "
            f"{fmt(mean(ari_values))} | {fmt(mean(delta_values))} |"
        )

    return "\n".join(lines)


def build_full_gate_table(rows: list[dict[str, Any]], samples: list[str]) -> str:
    soft_rows = [
        row
        for row in rows
        if row.get("method_group") == "soft_gate" and row.get("status") == "ok"
    ]
    lines = [
        "## Gate diagnosis: full variant",
        "",
        (
            "| Sample | Mean gate | Min eff. degree | Gate AUC | "
            "Low-5 enrich | Low-5 CDR | Low-5 SDR |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for sample in samples:
        row = next(
            (
                item
                for item in soft_rows
                if item.get("variant") == "full" and item.get("sample_id") == sample
            ),
            {},
        )
        lines.append(
            f"| {sample} | "
            f"{fmt(row.get('mean_gate'))} | "
            f"{fmt(row.get('min_effective_degree'))} | "
            f"{fmt(row.get('gate_auc_for_cross_domain'))} | "
            f"{fmt(row.get('bottom_5pct_enrichment'))} | "
            f"{fmt(row.get('bottom_5pct_cdr'))} | "
            f"{fmt(row.get('bottom_5pct_sdr'))} |"
        )

    return "\n".join(lines)


def build_interpretation(rows: list[dict[str, Any]], samples: list[str]) -> str:
    soft_rows = [
        row
        for row in rows
        if row.get("method_group") == "soft_gate" and row.get("status") == "ok"
    ]
    full_rows = [row for row in soft_rows if row.get("variant") == "full"]
    extra_rows = [
        row for row in soft_rows if row.get("variant") == "extra_training"
    ]
    full_by_sample = {row["sample_id"]: row for row in full_rows}
    extra_by_sample = {row["sample_id"]: row for row in extra_rows}

    full_deltas = [
        full_by_sample.get(sample, {}).get("delta_vs_original")
        for sample in samples
    ]
    full_minus_extra = []
    for sample in samples:
        full_ari = safe_float(full_by_sample.get(sample, {}).get("ari"))
        extra_ari = safe_float(extra_by_sample.get(sample, {}).get("ari"))
        if full_ari is not None and extra_ari is not None:
            full_minus_extra.append(full_ari - extra_ari)

    low5_enrich = [
        full_by_sample.get(sample, {}).get("bottom_5pct_enrichment")
        for sample in samples
    ]
    low5_cdr_minus_sdr = []
    for sample in samples:
        cdr = safe_float(full_by_sample.get(sample, {}).get("bottom_5pct_cdr"))
        sdr = safe_float(full_by_sample.get(sample, {}).get("bottom_5pct_sdr"))
        if cdr is not None and sdr is not None:
            low5_cdr_minus_sdr.append(cdr - sdr)

    return "\n".join(
        [
            "## Quick interpretation",
            "",
            f"- Full mean Δ vs original: `{fmt(mean(full_deltas))}`.",
            f"- Full mean ARI advantage over extra_training: `{fmt(mean(full_minus_extra))}`.",
            f"- Full mean bottom-5 low-gate enrichment: `{fmt(mean(low5_enrich))}`.",
            f"- Full mean bottom-5 CDR minus SDR: `{fmt(mean(low5_cdr_minus_sdr))}`.",
            "",
            "Decision rules:",
            "",
            "- If `full` does not exceed `extra_training`, the gate mechanism is not yet supported.",
            "- If bottom-5 enrichment is not greater than 1, low-gate edges are not enriched for cross-domain edges.",
            "- If bottom-5 CDR is not greater than SDR, do not proceed to hard pruning.",
            "- If gate AUC is <= 0.5, treat the gate as mechanistically invalid even if ARI improves.",
            "",
        ]
    )


def build_markdown(
    rows: list[dict[str, Any]],
    variants: list[str],
    samples: list[str],
) -> str:
    parts = [
        "# E3 Soft Gate Summary",
        "",
        build_ari_table(rows, variants, samples),
        "",
        build_full_gate_table(rows, samples),
        "",
        build_interpretation(rows, samples),
    ]
    return "\n".join(parts)


def write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    variants: list[str],
    samples: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "e3_summary.json"
    csv_path = output_dir / "e3_summary.csv"
    md_path = output_dir / "e3_summary.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with md_path.open("w", encoding="utf-8") as file:
        file.write(build_markdown(rows, variants, samples))

    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {md_path}")


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize E3 soft-gate JSON metrics on the server so only small "
            "analysis files need to be copied back locally."
        )
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/e3_summary"),
    )
    parser.add_argument(
        "--samples",
        default=",".join(DEFAULT_SAMPLES),
        help="Comma-separated sample IDs.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated soft-gate variants.",
    )
    args = parser.parse_args()

    samples = parse_csv_list(args.samples)
    variants = parse_csv_list(args.variants)

    rows: list[dict[str, Any]] = []
    rows.extend(collect_reference_methods(args.results_root, samples))
    rows.extend(collect_soft_gate(args.results_root, variants, samples))
    write_outputs(rows, args.output_dir, variants, samples)


if __name__ == "__main__":
    main()
