"""Phase 2 diagnostic: fixed-prototype assignment evaluation.

This evaluates STAGATE and BA-STAGATE embeddings with the same fixed core
prototypes instead of rerunning full mclust. It helps separate embedding
calibration effects from global mclust instability.
"""

from __future__ import annotations

import argparse
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
    silhouette_score,
)

from baseline_reporting import get_runtime_metadata, write_json
from phase1_boundary_diagnostics import (
    build_neighbor_lists,
    encode_ground_truth,
    matched_prediction_errors,
    neighbor_disagreement,
)
from phase2_adapter import encode_pseudo_labels, select_core_masks
from phase2_adapter_aligned import align_labels_to_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate embeddings by fixed prototype assignment."
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--boundary-scores", type=Path, required=True)
    parser.add_argument("--ba-embedding", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ba_stagate/phase2_diagnostics/fixed_prototype"),
    )
    parser.add_argument("--embedding-key", default="STAGATE")
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument("--tau-core", type=float, default=0.8)
    parser.add_argument("--core-bottom-quantile", type=float, default=0.5)
    parser.add_argument("--fallback-core-ratio", type=float, default=0.2)
    parser.add_argument("--min-core-spots", type=int, default=5)
    return parser.parse_args()


def load_boundary_scores(path: Path, obs_names: pd.Index) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Boundary score file not found: {path}")
    scores = pd.read_csv(path)
    required = {
        "spot_id",
        "pseudo_label",
        "pseudo_confidence",
        "combined_boundary_score",
        "pseudo_boundary",
        "gt_boundary",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"boundary_scores.csv is missing columns: {missing}")
    scores = scores.set_index("spot_id")
    missing_spots = obs_names.difference(scores.index)
    if len(missing_spots) > 0:
        raise ValueError(f"boundary_scores.csv is missing spot: {missing_spots[0]}")
    return scores.loc[obs_names].reset_index()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def build_prototypes(
    embedding: np.ndarray,
    core_masks: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    labels = sorted(core_masks)
    prototypes = []
    for label in labels:
        mask = core_masks[label]
        if not mask.any():
            raise ValueError(f"No core spots for pseudo label {label}")
        prototypes.append(embedding[mask].mean(axis=0))
    return np.vstack(prototypes), labels


def assign_by_prototype(embedding: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    scores = normalize_rows(embedding) @ normalize_rows(prototypes).T
    return scores.argmax(axis=1)


def metric_block(
    embedding: np.ndarray,
    assigned: np.ndarray,
    gt_labels: np.ndarray,
    evaluated: np.ndarray,
    gt_boundary: np.ndarray,
    pseudo_boundary: np.ndarray,
) -> dict[str, float | None]:
    gt_interior = (~gt_boundary) & evaluated
    pseudo_interior = (~pseudo_boundary) & evaluated
    metrics: dict[str, float | None] = {
        "ari": float(adjusted_rand_score(gt_labels[evaluated], assigned[evaluated])),
        "nmi": float(normalized_mutual_info_score(gt_labels[evaluated], assigned[evaluated])),
        "ami": float(adjusted_mutual_info_score(gt_labels[evaluated], assigned[evaluated])),
        "silhouette": None,
        "gt_boundary_ari": None,
        "gt_interior_ari": None,
        "pseudo_boundary_ari": None,
        "pseudo_interior_ari": None,
    }
    unique_labels = np.unique(assigned[evaluated])
    if 1 < unique_labels.size < evaluated.sum():
        metrics["silhouette"] = float(silhouette_score(embedding[evaluated], assigned[evaluated]))
    for key, mask in [
        ("gt_boundary_ari", gt_boundary),
        ("gt_interior_ari", gt_interior),
        ("pseudo_boundary_ari", pseudo_boundary & evaluated),
        ("pseudo_interior_ari", pseudo_interior),
    ]:
        if mask.sum() > 1:
            metrics[key] = float(adjusted_rand_score(gt_labels[mask], assigned[mask]))
    return metrics


def run_fixed_prototype_eval(args: argparse.Namespace) -> dict[str, object]:
    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Input h5ad not found: {args.input_h5ad}")
    if not args.ba_embedding.exists():
        raise FileNotFoundError(f"BA embedding file not found: {args.ba_embedding}")

    adata = sc.read_h5ad(args.input_h5ad)
    if args.embedding_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm['{args.embedding_key}']")
    if "Spatial_Net" not in adata.uns:
        raise KeyError("Missing adata.uns['Spatial_Net']")
    if args.ground_truth_key not in adata.obs:
        raise KeyError(f"Missing adata.obs['{args.ground_truth_key}']")

    stagate_embedding = np.asarray(adata.obsm[args.embedding_key], dtype=np.float64)
    ba_embedding = np.load(args.ba_embedding).astype(np.float64)
    if ba_embedding.shape != stagate_embedding.shape:
        raise ValueError(
            f"BA embedding shape {ba_embedding.shape} does not match "
            f"STAGATE shape {stagate_embedding.shape}"
        )

    scores = load_boundary_scores(args.boundary_scores, adata.obs_names)
    pseudo_labels_raw = scores["pseudo_label"].to_numpy()
    pseudo_labels, original_label_values = encode_pseudo_labels(pseudo_labels_raw)
    confidence = scores["pseudo_confidence"].to_numpy(dtype=np.float64)
    boundary_score = scores["combined_boundary_score"].to_numpy(dtype=np.float64)
    pseudo_boundary = scores["pseudo_boundary"].astype(bool).to_numpy()
    gt_boundary_from_phase1 = scores["gt_boundary"].astype(bool).to_numpy()

    gt_labels, evaluated = encode_ground_truth(adata.obs[args.ground_truth_key])
    neighbors = build_neighbor_lists(adata.uns["Spatial_Net"], adata.obs_names)
    gt_boundary_score = neighbor_disagreement(gt_labels, neighbors, valid_mask=evaluated)
    gt_boundary = ((gt_boundary_score > 0) & evaluated) | gt_boundary_from_phase1

    core_args = argparse.Namespace(
        tau_core=args.tau_core,
        core_bottom_quantile=args.core_bottom_quantile,
        fallback_core_ratio=args.fallback_core_ratio,
        min_core_spots=args.min_core_spots,
    )
    core_masks = select_core_masks(
        pseudo_labels,
        confidence,
        boundary_score,
        core_args,
    )
    prototypes, prototype_labels = build_prototypes(stagate_embedding, core_masks)
    stagate_assigned = assign_by_prototype(stagate_embedding, prototypes)
    ba_assigned = assign_by_prototype(ba_embedding, prototypes)
    aligned_ba_assigned = align_labels_to_reference(stagate_assigned, ba_assigned)

    stagate_metrics = metric_block(
        stagate_embedding,
        stagate_assigned,
        gt_labels,
        evaluated,
        gt_boundary,
        pseudo_boundary,
    )
    ba_metrics = metric_block(
        ba_embedding,
        ba_assigned,
        gt_labels,
        evaluated,
        gt_boundary,
        pseudo_boundary,
    )

    gt_interior = (~gt_boundary) & evaluated
    pseudo_interior = (~pseudo_boundary) & evaluated
    changed = aligned_ba_assigned != stagate_assigned
    stagate_error = matched_prediction_errors(gt_labels, stagate_assigned, evaluated)
    ba_error = matched_prediction_errors(gt_labels, ba_assigned, evaluated)

    output_dir = args.output_dir / args.sample_id / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    spot_table = pd.DataFrame(
        {
            "spot_id": adata.obs_names,
            "pseudo_label": pseudo_labels_raw,
            "fixed_proto_stagate_label": stagate_assigned,
            "fixed_proto_ba_label": ba_assigned,
            "fixed_proto_ba_label_aligned": aligned_ba_assigned,
            "changed_after_alignment": changed,
            "pseudo_boundary": pseudo_boundary,
            "gt_boundary": gt_boundary,
            "stagate_error_after_hungarian": stagate_error,
            "ba_error_after_hungarian": ba_error,
            "embedding_l2_perturbation": np.linalg.norm(ba_embedding - stagate_embedding, axis=1),
        }
    )
    spot_table.to_csv(output_dir / "fixed_prototype_spot_results.csv", index=False)

    result = {
        "sample_id": args.sample_id,
        "experiment": args.experiment,
        "input_h5ad": str(args.input_h5ad),
        "boundary_scores": str(args.boundary_scores),
        "ba_embedding": str(args.ba_embedding),
        "prototype_source": "STAGATE high-confidence interior core spots",
        "prototype_labels": [int(label) for label in prototype_labels],
        "prototype_original_labels": [
            str(original_label_values[label]) for label in prototype_labels
        ],
        "core_spots_per_cluster": {
            str(original_label_values[label]): int(mask.sum())
            for label, mask in core_masks.items()
        },
        "stagate_fixed_prototype_metrics": stagate_metrics,
        "ba_fixed_prototype_metrics": ba_metrics,
        "metric_delta": {
            key: (
                ba_metrics[key] - stagate_metrics[key]
                if ba_metrics.get(key) is not None and stagate_metrics.get(key) is not None
                else None
            )
            for key in ba_metrics
        },
        "changed_label_ratio_after_alignment": {
            "overall": float(changed[evaluated].mean()) if evaluated.any() else None,
            "gt_boundary": float(changed[gt_boundary].mean()) if gt_boundary.any() else None,
            "gt_interior": float(changed[gt_interior].mean()) if gt_interior.any() else None,
            "pseudo_boundary": (
                float(changed[(pseudo_boundary & evaluated)].mean())
                if (pseudo_boundary & evaluated).any()
                else None
            ),
            "pseudo_interior": (
                float(changed[pseudo_interior].mean()) if pseudo_interior.any() else None
            ),
        },
        "error_rates_after_hungarian": {
            "stagate_overall": float(stagate_error[evaluated].mean()) if evaluated.any() else None,
            "ba_overall": float(ba_error[evaluated].mean()) if evaluated.any() else None,
            "stagate_gt_boundary": (
                float(stagate_error[gt_boundary].mean()) if gt_boundary.any() else None
            ),
            "ba_gt_boundary": float(ba_error[gt_boundary].mean()) if gt_boundary.any() else None,
            "stagate_gt_interior": (
                float(stagate_error[gt_interior].mean()) if gt_interior.any() else None
            ),
            "ba_gt_interior": (
                float(ba_error[gt_interior].mean()) if gt_interior.any() else None
            ),
        },
        "runtime": get_runtime_metadata(REPO_ROOT),
    }
    write_json(output_dir / "fixed_prototype_metrics.json", result)
    print(f"fixed-prototype diagnostics saved to {output_dir.resolve()}")
    return result


def main() -> None:
    run_fixed_prototype_eval(parse_args())


if __name__ == "__main__":
    main()
