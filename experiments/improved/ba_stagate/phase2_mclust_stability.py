"""Phase 2 diagnostic: mclust stability under tiny embedding jitter.

This script tests whether full mclust reclustering is sensitive to tiny
embedding perturbations. It does not train any model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from baseline_reporting import get_runtime_metadata, write_json
from mclust_posterior import mclust_with_posterior
from phase1_boundary_diagnostics import (
    build_neighbor_lists,
    encode_ground_truth,
    matched_prediction_errors,
    neighbor_disagreement,
)
from phase2_adapter_aligned import align_labels_to_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check mclust stability under tiny STAGATE embedding jitter."
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ba_stagate/phase2_diagnostics/mclust_stability"),
    )
    parser.add_argument("--embedding-key", default="STAGATE")
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--mclust-model", default="EEE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epsilons",
        default="0,0.001,0.003,0.005",
        help="Comma-separated jitter scales relative to per-dimension std.",
    )
    parser.add_argument("--r-home", default=None)
    parser.add_argument("--r-user", default=None)
    return parser.parse_args()


def parse_epsilons(value: str) -> list[float]:
    epsilons = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not epsilons:
        raise ValueError("--epsilons must contain at least one value")
    return epsilons


def clustering_metrics(
    labels: np.ndarray,
    gt_labels: np.ndarray,
    evaluated: np.ndarray,
    gt_boundary: np.ndarray,
) -> dict[str, float | None]:
    gt_interior = (~gt_boundary) & evaluated
    metrics: dict[str, float | None] = {
        "ari": float(adjusted_rand_score(gt_labels[evaluated], labels[evaluated])),
        "nmi": float(normalized_mutual_info_score(gt_labels[evaluated], labels[evaluated])),
        "ami": float(adjusted_mutual_info_score(gt_labels[evaluated], labels[evaluated])),
        "gt_boundary_ari": None,
        "gt_interior_ari": None,
    }
    if gt_boundary.sum() > 1:
        metrics["gt_boundary_ari"] = float(
            adjusted_rand_score(gt_labels[gt_boundary], labels[gt_boundary])
        )
    if gt_interior.sum() > 1:
        metrics["gt_interior_ari"] = float(
            adjusted_rand_score(gt_labels[gt_interior], labels[gt_interior])
        )
    return metrics


def run_stability(args: argparse.Namespace) -> dict[str, object]:
    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        os.environ["R_USER"] = args.r_user
    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Input h5ad not found: {args.input_h5ad}")

    adata = sc.read_h5ad(args.input_h5ad)
    if args.embedding_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm['{args.embedding_key}']")
    if "Spatial_Net" not in adata.uns:
        raise KeyError("Missing adata.uns['Spatial_Net']")
    if args.ground_truth_key not in adata.obs:
        raise KeyError(f"Missing adata.obs['{args.ground_truth_key}']")

    embedding = np.asarray(adata.obsm[args.embedding_key], dtype=np.float64)
    gt_labels, evaluated = encode_ground_truth(adata.obs[args.ground_truth_key])
    neighbors = build_neighbor_lists(adata.uns["Spatial_Net"], adata.obs_names)
    gt_boundary_score = neighbor_disagreement(gt_labels, neighbors, valid_mask=evaluated)
    gt_boundary = (gt_boundary_score > 0) & evaluated

    rng = np.random.default_rng(args.seed)
    dim_std = embedding.std(axis=0, keepdims=True)
    dim_std[dim_std == 0] = 1.0
    epsilons = parse_epsilons(args.epsilons)

    baseline_labels: np.ndarray | None = None
    rows = []
    spot_rows = pd.DataFrame({"spot_id": adata.obs_names})
    for epsilon in epsilons:
        jitter = rng.normal(size=embedding.shape) * dim_std * epsilon
        labels = mclust_with_posterior(
            embedding + jitter,
            num_cluster=args.clusters,
            model_names=args.mclust_model,
            random_seed=args.seed,
        )["labels"]
        if baseline_labels is None:
            baseline_labels = labels
            aligned = labels
        else:
            aligned = align_labels_to_reference(baseline_labels, labels)

        errors = matched_prediction_errors(gt_labels, labels, evaluated)
        changed = (
            np.zeros(labels.shape[0], dtype=bool)
            if baseline_labels is labels
            else aligned != baseline_labels
        )
        gt_interior = (~gt_boundary) & evaluated
        row = {
            "epsilon": epsilon,
            **clustering_metrics(labels, gt_labels, evaluated, gt_boundary),
            "overall_error_rate": float(errors[evaluated].mean()),
            "gt_boundary_error_rate": (
                float(errors[gt_boundary].mean()) if gt_boundary.any() else None
            ),
            "gt_interior_error_rate": (
                float(errors[gt_interior].mean()) if gt_interior.any() else None
            ),
            "changed_label_ratio": float(changed[evaluated].mean()),
            "gt_boundary_changed_label_ratio": (
                float(changed[gt_boundary].mean()) if gt_boundary.any() else None
            ),
            "gt_interior_changed_label_ratio": (
                float(changed[gt_interior].mean()) if gt_interior.any() else None
            ),
        }
        rows.append(row)
        spot_rows[f"label_epsilon_{epsilon}"] = labels
        spot_rows[f"changed_epsilon_{epsilon}"] = changed

    output_dir = args.output_dir / args.sample_id / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "mclust_stability.csv", index=False)
    spot_rows.to_csv(output_dir / "mclust_stability_spot_labels.csv", index=False)

    result = {
        "sample_id": args.sample_id,
        "input_h5ad": str(args.input_h5ad),
        "embedding_key": args.embedding_key,
        "mclust": {
            "num_cluster": args.clusters,
            "modelNames": args.mclust_model,
            "random_seed": args.seed,
        },
        "epsilons": epsilons,
        "rows": rows,
        "runtime": get_runtime_metadata(REPO_ROOT),
    }
    write_json(output_dir / "mclust_stability.json", result)
    print(f"mclust stability diagnostics saved to {output_dir.resolve()}")
    return result


def main() -> None:
    run_stability(parse_args())


if __name__ == "__main__":
    main()
