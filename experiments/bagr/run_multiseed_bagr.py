"""Run BAGR pilot experiments across samples and random seeds.

This is a server-side orchestration helper. It calls the existing baseline,
PCA preparation, graph refinement, refined retraining, and fixed-boundary
diagnosis scripts in a reproducible directory layout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Setting:
    prune_ratio: str
    min_confidence: str
    expr_quantile: str
    cluster_quantile: str

    @property
    def tag(self) -> str:
        return (
            f"prune_{self.prune_ratio}_conf_{self.min_confidence}"
            f"_exprq_{self.expr_quantile}"
        )


DEFAULT_SETTINGS = {
    "151674": Setting("0.03", "0.85", "0.80", "0.80"),
    "151675": Setting("0.02", "0.75", "0.80", "0.80"),
}

DLPFC_ALL_SAMPLES = (
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
)
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-sample, multi-seed BAGR pilot experiments."
    )
    parser.add_argument("--samples", nargs="+", default=["151674", "151675"])
    parser.add_argument(
        "--all-dlpfc",
        action="store_true",
        help="Run all 12 DLPFC slices, overriding --samples.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        help=(
            "Sample-specific setting as sample:prune_ratio:min_conf:expr_q[:cluster_q]. "
            "Example: 151674:0.03:0.85:0.80:0.80"
        ),
    )
    parser.add_argument(
        "--default-setting",
        default=None,
        help="Fallback setting as prune_ratio:min_conf:expr_q[:cluster_q].",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help=(
            "Override mclust cluster count for all samples. By default DLPFC uses "
            "5 clusters for 151669-151672 and 7 clusters for the other slices."
        ),
    )
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--max-prune-per-node", type=int, default=1)
    parser.add_argument("--alpha-uncertainty", type=float, default=0.5)
    parser.add_argument("--sim-metric", choices=["cosine", "pearson"], default="cosine")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--skip-build-graph", action="store_true")
    parser.add_argument("--skip-refined", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    return parser.parse_args()


def parse_setting_value(value: str) -> Setting:
    parts = value.split(":")
    if len(parts) == 3:
        prune_ratio, min_confidence, expr_quantile = parts
        cluster_quantile = "0.80"
    elif len(parts) == 4:
        prune_ratio, min_confidence, expr_quantile, cluster_quantile = parts
    else:
        raise ValueError(f"Invalid setting: {value!r}")
    return Setting(prune_ratio, min_confidence, expr_quantile, cluster_quantile)


def parse_settings(args: argparse.Namespace) -> dict[str, Setting]:
    settings = dict(DEFAULT_SETTINGS)
    for item in args.setting:
        parts = item.split(":")
        if len(parts) not in (4, 5):
            raise ValueError(f"Invalid --setting: {item!r}")
        settings[parts[0]] = parse_setting_value(":".join(parts[1:]))
    if args.default_setting is not None:
        default = parse_setting_value(args.default_setting)
        for sample in args.samples:
            settings.setdefault(sample, default)
    missing = [sample for sample in args.samples if sample not in settings]
    if missing:
        raise ValueError("Missing settings for samples: " + ", ".join(missing))
    return settings


def run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def maybe_overwrite(args: argparse.Namespace) -> list[str]:
    return ["--overwrite"] if args.overwrite else []


def infer_dlpfc_clusters(sample_id: str) -> int:
    sample_id = str(sample_id)
    if sample_id in DLPFC_FIVE_DOMAIN_SAMPLES:
        return 5
    if sample_id in DLPFC_SEVEN_DOMAIN_SAMPLES:
        return 7
    raise ValueError(
        f"Unknown DLPFC sample_id {sample_id!r}; pass --clusters to override."
    )


def clusters_for_sample(args: argparse.Namespace, sample_id: str) -> int:
    if args.clusters is not None:
        return int(args.clusters)
    return infer_dlpfc_clusters(sample_id)

def main() -> None:
    args = parse_args()
    if args.all_dlpfc:
        args.samples = list(DLPFC_ALL_SAMPLES)
    settings = parse_settings(args)
    baseline_dirs: list[Path] = []
    refined_dirs: list[Path] = []
    boundary_edges: list[Path] = []
    eval_edges: list[Path] = []

    baseline_root = args.output_root / "stagate_baseline"
    bagr_root = args.output_root / "bagr"

    for sample in args.samples:
        setting = settings[sample]
        clusters = clusters_for_sample(args, sample)
        for seed in args.seeds:
            baseline_dir = baseline_root / sample / f"seed_{seed}"
            bagr_dir = bagr_root / sample / f"seed_{seed}" / setting.tag
            refined_dir = bagr_dir / "refined_stagate"
            baseline_dirs.append(baseline_dir)
            refined_dirs.append(refined_dir)
            boundary_edges.append(baseline_dir / "spatial_edges.csv")
            eval_edges.append(refined_dir / "spatial_edges.csv")

            if not args.skip_baseline:
                run([
                    args.python, "experiments/baseline/run_stagate_baseline.py",
                    "--dataset", "dlpfc", "--data-root", str(args.data_root),
                    "--sample-id", sample, "--clusters", str(clusters),
                    "--seed", str(seed), "--device", args.device,
                    "--output-dir", str(baseline_dir), *maybe_overwrite(args),
                ], args.dry_run)

            if not args.skip_pca:
                run([
                    args.python, "experiments/bagr/prepare_pca_expression.py",
                    "--input-dir", str(baseline_dir), "--data-root", str(args.data_root),
                    "--sample-id", sample, "--n-top-genes", str(args.n_top_genes),
                    "--n-components", str(args.pca_components), *maybe_overwrite(args),
                ], args.dry_run)

            if not args.skip_build_graph:
                run([
                    args.python, "experiments/bagr/build_bagr_graph.py",
                    "--input-dir", str(baseline_dir),
                    "--prune-ratio", setting.prune_ratio,
                    "--max-prune-per-node", str(args.max_prune_per_node),
                    "--alpha-uncertainty", str(args.alpha_uncertainty),
                    "--sim-metric", args.sim_metric,
                    "--min-confidence", setting.min_confidence,
                    "--expr-dissim-quantile", setting.expr_quantile,
                    "--cluster-pair-sep-quantile", setting.cluster_quantile,
                    "--output-dir", str(bagr_dir), *maybe_overwrite(args),
                ], args.dry_run)

            if not args.skip_refined:
                run([
                    args.python, "experiments/bagr/run_stagate_refined.py",
                    "--baseline-dir", str(baseline_dir),
                    "--refined-edges", str(bagr_dir / "refined_edges.csv"),
                    "--dataset", "dlpfc", "--data-root", str(args.data_root),
                    "--sample-id", sample, "--clusters", str(clusters),
                    "--seed", str(seed), "--device", args.device,
                    "--output-dir", str(refined_dir), *maybe_overwrite(args),
                ], args.dry_run)

    if args.skip_analysis:
        return

    baseline_summary = bagr_root / "multiseed_baseline_boundary_summary.csv"
    refined_summary = bagr_root / "multiseed_refined_boundary_summary.csv"
    run([
        args.python, "experiments/baseline/analyze_boundary_errors.py",
        "--input-dir", *[str(path) for path in baseline_dirs],
        "--boundary-edges-path", *[str(path / "spatial_edges.csv") for path in baseline_dirs],
        "--eval-edges-path", *[str(path / "spatial_edges.csv") for path in baseline_dirs],
        "--summary-output", str(baseline_summary), *maybe_overwrite(args),
    ], args.dry_run)
    run([
        args.python, "experiments/baseline/analyze_boundary_errors.py",
        "--input-dir", *[str(path) for path in refined_dirs],
        "--boundary-edges-path", *[str(path) for path in boundary_edges],
        "--eval-edges-path", *[str(path) for path in eval_edges],
        "--summary-output", str(refined_summary), *maybe_overwrite(args),
    ], args.dry_run)
    run([
        args.python, "experiments/bagr/summarize_multiseed_results.py",
        "--baseline-summary", str(baseline_summary),
        "--refined-summary", str(refined_summary),
        "--output-prefix", str(bagr_root / "multiseed"),
    ], args.dry_run)


if __name__ == "__main__":
    main()







