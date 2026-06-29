"""Summarize baseline/refined BAGR results across seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DLPFC_FIVE_DOMAIN_SAMPLES = {"151669", "151670", "151671", "151672"}
DLPFC_SEVEN_DOMAIN_SAMPLES = {
    "151507",
    "151508",
    "151509",
    "151510",
    "151673",
    "151674",
    "151675",
    "151676",
}

METRICS = [
    "global_ari",
    "boundary_ari",
    "interior_ari",
    "boundary_error_rate",
    "interior_error_rate",
    "cross_gt_domain_edge_ratio",
    "cross_pred_domain_edge_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare multi-seed baseline and refined boundary summaries."
    )
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--refined-summary", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/bagr/multiseed"),
    )
    return parser.parse_args()


def ensure_seed(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if "seed" not in frame.columns or frame["seed"].isna().all():
        frame["seed"] = frame["input_dir"].astype(str).str.extract(r"seed_(\d+)")[0]
    frame["seed"] = frame["seed"].astype(int)
    return frame


def infer_n_domains(sample_id: str) -> int | None:
    sample_id = str(sample_id)
    if sample_id in DLPFC_FIVE_DOMAIN_SAMPLES:
        return 5
    if sample_id in DLPFC_SEVEN_DOMAIN_SAMPLES:
        return 7
    return None


def infer_domain_group(sample_id: str) -> str:
    n_domains = infer_n_domains(sample_id)
    if n_domains is None:
        return "unknown"
    return f"{n_domains}_domain"


def metric_columns() -> list[str]:
    columns: list[str] = []
    for metric in METRICS:
        columns.extend([
            f"{metric}_baseline",
            f"{metric}_refined",
            f"delta_{metric}",
        ])
    return columns


def summarize(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    value_cols = metric_columns()
    summary = (
        frame.groupby(by, dropna=False)
        .agg(
            **{f"{col}_mean": (col, "mean") for col in value_cols},
            **{f"{col}_std": (col, "std") for col in value_cols},
            n_samples=("sample_id", "nunique"),
            n_runs=("seed", "count"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    args = parse_args()
    baseline = ensure_seed(pd.read_csv(args.baseline_summary))
    refined = ensure_seed(pd.read_csv(args.refined_summary))
    keep = ["sample_id", "seed", "input_dir", *METRICS]
    merged = baseline[keep].merge(
        refined[keep],
        on=["sample_id", "seed"],
        suffixes=("_baseline", "_refined"),
        validate="one_to_one",
    )
    merged["n_domains"] = merged["sample_id"].map(infer_n_domains)
    merged["domain_group"] = merged["sample_id"].map(infer_domain_group)
    for metric in METRICS:
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_refined"] - merged[f"{metric}_baseline"]
        )

    sample_summary = summarize(merged, ["sample_id", "domain_group", "n_domains"])
    group_summary = summarize(merged, ["domain_group", "n_domains"])
    overall_summary = summarize(merged.assign(domain_group="all", n_domains=pd.NA), [
        "domain_group",
        "n_domains",
    ])
    group_summary = pd.concat([overall_summary, group_summary], ignore_index=True)

    per_seed_path = args.output_prefix.with_name(args.output_prefix.name + "_per_seed.csv")
    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.csv")
    group_path = args.output_prefix.with_name(args.output_prefix.name + "_group_summary.csv")
    per_seed_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(per_seed_path, index=False)
    sample_summary.to_csv(summary_path, index=False)
    group_summary.to_csv(group_path, index=False)
    print(f"Saved: {per_seed_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {group_path}")
    print(group_summary.to_string(index=False))


if __name__ == "__main__":
    main()



