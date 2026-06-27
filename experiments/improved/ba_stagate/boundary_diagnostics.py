"""Phase 1 boundary diagnostics for BA-STAGATE.

The script reads a Phase 0 baseline ``.h5ad`` artifact and computes boundary
scores and diagnostic metrics. It does not train STAGATE or BA-STAGATE.
Ground-truth labels are used only for diagnostics.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from baseline_reporting import get_runtime_metadata, write_json
from mclust_posterior import mclust_with_posterior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Phase 1 BA-STAGATE boundary diagnostics."
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ba_stagate/phase1_boundary_diagnostics"),
    )
    parser.add_argument("--embedding-key", default="STAGATE")
    parser.add_argument("--baseline-label-key", default="mclust")
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--mclust-model", default="EEE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embedding-neighbors", type=int, default=10)
    parser.add_argument("--boundary-ratio", type=float, default=0.2)
    parser.add_argument("--lambda-spatial", type=float, default=0.5)
    parser.add_argument("--lambda-embedding", type=float, default=0.3)
    parser.add_argument("--lambda-confidence", type=float, default=0.2)
    parser.add_argument("--r-home", default=None)
    parser.add_argument("--r-user", default=None)
    return parser.parse_args()


def validate_inputs(adata: sc.AnnData, args: argparse.Namespace) -> None:
    missing = []
    if args.embedding_key not in adata.obsm:
        missing.append(f"adata.obsm['{args.embedding_key}']")
    if "Spatial_Net" not in adata.uns:
        missing.append("adata.uns['Spatial_Net']")
    if args.ground_truth_key not in adata.obs:
        missing.append(f"adata.obs['{args.ground_truth_key}']")
    if missing:
        raise KeyError("Missing required baseline fields: " + ", ".join(missing))

    graph = adata.uns["Spatial_Net"]
    if not isinstance(graph, pd.DataFrame):
        raise TypeError("adata.uns['Spatial_Net'] must be a pandas DataFrame")
    required_columns = {"Cell1", "Cell2"}
    missing_columns = sorted(required_columns.difference(graph.columns))
    if missing_columns:
        raise ValueError(f"Spatial_Net is missing columns: {missing_columns}")


def build_neighbor_lists(graph: pd.DataFrame, obs_names: pd.Index) -> list[np.ndarray]:
    spot_to_index = {str(spot): index for index, spot in enumerate(obs_names)}
    neighbors: list[set[int]] = [set() for _ in range(len(obs_names))]

    for cell1, cell2 in graph.loc[:, ["Cell1", "Cell2"]].itertuples(
        index=False,
        name=None,
    ):
        source = spot_to_index.get(str(cell1))
        target = spot_to_index.get(str(cell2))
        if source is None or target is None or source == target:
            continue
        neighbors[source].add(target)
        neighbors[target].add(source)

    return [np.asarray(sorted(items), dtype=np.int64) for items in neighbors]


def neighbor_disagreement(
    labels: np.ndarray,
    neighbors: list[np.ndarray],
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    scores = np.zeros(labels.shape[0], dtype=np.float64)
    for index, neighbor_index in enumerate(neighbors):
        if neighbor_index.size == 0:
            scores[index] = 0.0
            continue
        used_neighbors = neighbor_index
        if valid_mask is not None:
            used_neighbors = used_neighbors[valid_mask[used_neighbors]]
        if used_neighbors.size == 0:
            scores[index] = 0.0
            continue
        scores[index] = np.mean(labels[used_neighbors] != labels[index])
    return scores


def embedding_neighbor_lists(
    embedding: np.ndarray,
    n_neighbors: int,
) -> list[np.ndarray]:
    n_spots = embedding.shape[0]
    if n_spots <= 1:
        return [np.asarray([], dtype=np.int64) for _ in range(n_spots)]
    n_query_neighbors = min(n_neighbors + 1, n_spots)
    nearest = NearestNeighbors(n_neighbors=n_query_neighbors)
    nearest.fit(embedding)
    indices = nearest.kneighbors(embedding, return_distance=False)
    return [
        row[row != index][:n_neighbors].astype(np.int64)
        for index, row in enumerate(indices)
    ]


def top_ratio_mask(scores: np.ndarray, ratio: float) -> np.ndarray:
    if not 0 < ratio < 1:
        raise ValueError("--boundary-ratio must be between 0 and 1")
    n_selected = max(1, int(np.ceil(scores.shape[0] * ratio)))
    order = np.argsort(scores, kind="mergesort")
    selected = order[-n_selected:]
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[selected] = True
    return mask


def encode_ground_truth(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    valid = values.notna().to_numpy()
    labels = np.full(values.shape[0], -1, dtype=np.int64)
    if valid.any():
        encoder = LabelEncoder()
        labels[valid] = encoder.fit_transform(values.loc[valid].astype(str))
    return labels, valid


def safe_metric(metric_name: str, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if np.unique(y_true).size < 2:
        return None
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_score))
    if metric_name == "average_precision":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(f"Unsupported metric: {metric_name}")


def summarize_binary_overlap(
    predicted_mask: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, float | int | None]:
    if predicted_mask.shape != target_mask.shape:
        raise ValueError("Mask shapes do not match")
    y_true = target_mask.astype(int)
    y_pred = predicted_mask.astype(int)
    return {
        "predicted_count": int(predicted_mask.sum()),
        "target_count": int(target_mask.sum()),
        "overlap_count": int((predicted_mask & target_mask).sum()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": (
            float(matthews_corrcoef(y_true, y_pred))
            if np.unique(y_true).size > 1 and np.unique(y_pred).size > 1
            else None
        ),
    }


def entropy_from_neighbors(labels: np.ndarray, neighbors: list[np.ndarray]) -> np.ndarray:
    entropies = np.zeros(labels.shape[0], dtype=np.float64)
    for index, neighbor_index in enumerate(neighbors):
        if neighbor_index.size == 0:
            continue
        observed = labels[neighbor_index]
        _, counts = np.unique(observed, return_counts=True)
        probabilities = counts / counts.sum()
        entropies[index] = float(-(probabilities * np.log(probabilities)).sum())
    return entropies


def plot_spatial_score(
    adata: sc.AnnData,
    output_path: Path,
    column: str,
    title: str,
) -> None:
    sc.pl.spatial(
        adata,
        img_key="hires",
        color=column,
        title=title,
        show=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close("all")


def run_diagnostics(args: argparse.Namespace) -> dict[str, object]:
    warnings.filterwarnings("ignore")

    if args.r_home:
        import os

        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        import os

        os.environ["R_USER"] = args.r_user

    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Input h5ad not found: {args.input_h5ad}")

    adata = sc.read_h5ad(args.input_h5ad)
    validate_inputs(adata, args)

    embedding = np.asarray(adata.obsm[args.embedding_key], dtype=np.float64)
    mclust = mclust_with_posterior(
        embedding,
        num_cluster=args.clusters,
        model_names=args.mclust_model,
        random_seed=args.seed,
    )
    pseudo_labels = mclust["labels"]
    confidence = mclust["confidence"]

    spatial_neighbors = build_neighbor_lists(adata.uns["Spatial_Net"], adata.obs_names)
    embedding_neighbors = embedding_neighbor_lists(embedding, args.embedding_neighbors)

    spatial_boundary_score = neighbor_disagreement(pseudo_labels, spatial_neighbors)
    embedding_boundary_score = neighbor_disagreement(pseudo_labels, embedding_neighbors)
    confidence_boundary_score = 1.0 - confidence
    combined_boundary_score = (
        args.lambda_spatial * spatial_boundary_score
        + args.lambda_embedding * embedding_boundary_score
        + args.lambda_confidence * confidence_boundary_score
    )

    pseudo_boundary_mask = top_ratio_mask(combined_boundary_score, args.boundary_ratio)

    gt_labels, gt_valid = encode_ground_truth(adata.obs[args.ground_truth_key])
    gt_boundary_score = neighbor_disagreement(
        gt_labels,
        spatial_neighbors,
        valid_mask=gt_valid,
    )
    gt_boundary_mask = (gt_boundary_score > 0) & gt_valid

    evaluated = gt_valid
    predicted_error = np.zeros(adata.n_obs, dtype=bool)
    if evaluated.any():
        predicted_error[evaluated] = pseudo_labels[evaluated] != gt_labels[evaluated]

    pseudo_entropy = entropy_from_neighbors(pseudo_labels, spatial_neighbors)
    gt_entropy = entropy_from_neighbors(gt_labels, spatial_neighbors)

    adata.obs["phase1_pseudo_label"] = pd.Categorical(pseudo_labels.astype(str))
    adata.obs["phase1_pseudo_confidence"] = confidence
    adata.obs["phase1_spatial_boundary_score"] = spatial_boundary_score
    adata.obs["phase1_embedding_boundary_score"] = embedding_boundary_score
    adata.obs["phase1_combined_boundary_score"] = combined_boundary_score
    adata.obs["phase1_pseudo_boundary"] = pseudo_boundary_mask
    adata.obs["phase1_gt_boundary"] = gt_boundary_mask
    adata.obs["phase1_pseudo_error"] = predicted_error
    adata.obs["phase1_pseudo_neighbor_entropy"] = pseudo_entropy
    adata.obs["phase1_gt_neighbor_entropy"] = gt_entropy

    output_dir = args.output_dir / args.sample_id
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    score_table = pd.DataFrame(
        {
            "spot_id": adata.obs_names,
            "pseudo_label": pseudo_labels,
            "pseudo_confidence": confidence,
            "spatial_boundary_score": spatial_boundary_score,
            "embedding_boundary_score": embedding_boundary_score,
            "combined_boundary_score": combined_boundary_score,
            "pseudo_boundary": pseudo_boundary_mask,
            "ground_truth": adata.obs[args.ground_truth_key].astype(str).to_numpy(),
            "gt_boundary": gt_boundary_mask,
            "pseudo_error": predicted_error,
            "pseudo_neighbor_entropy": pseudo_entropy,
            "gt_neighbor_entropy": gt_entropy,
        }
    )
    score_table.to_csv(output_dir / "boundary_scores.csv", index=False)
    np.save(output_dir / "mclust_posterior.npy", mclust["posterior"])

    valid_scores = combined_boundary_score[evaluated]
    valid_errors = predicted_error[evaluated]
    high_boundary_error_rate = float(predicted_error[pseudo_boundary_mask & evaluated].mean())
    interior_mask = (~pseudo_boundary_mask) & evaluated
    interior_error_rate = (
        float(predicted_error[interior_mask].mean()) if interior_mask.any() else None
    )
    gt_boundary_error_rate = (
        float(predicted_error[gt_boundary_mask].mean()) if gt_boundary_mask.any() else None
    )
    gt_interior_mask = (~gt_boundary_mask) & evaluated
    gt_interior_error_rate = (
        float(predicted_error[gt_interior_mask].mean())
        if gt_interior_mask.any()
        else None
    )

    diagnostics = {
        "sample_id": args.sample_id,
        "input_h5ad": str(args.input_h5ad),
        "n_spots": int(adata.n_obs),
        "n_evaluated_spots": int(evaluated.sum()),
        "embedding_key": args.embedding_key,
        "ground_truth_key": args.ground_truth_key,
        "mclust": {
            "num_cluster": args.clusters,
            "modelNames": args.mclust_model,
            "random_seed": args.seed,
        },
        "score_weights": {
            "spatial": args.lambda_spatial,
            "embedding": args.lambda_embedding,
            "confidence": args.lambda_confidence,
        },
        "boundary_ratio": args.boundary_ratio,
        "embedding_neighbors": args.embedding_neighbors,
        "pseudo_boundary_vs_gt_boundary": summarize_binary_overlap(
            pseudo_boundary_mask[evaluated],
            gt_boundary_mask[evaluated],
        ),
        "combined_score_predicts_gt_boundary": {
            "roc_auc": safe_metric(
                "roc_auc",
                gt_boundary_mask[evaluated].astype(int),
                valid_scores,
            ),
            "average_precision": safe_metric(
                "average_precision",
                gt_boundary_mask[evaluated].astype(int),
                valid_scores,
            ),
        },
        "combined_score_predicts_pseudo_error": {
            "roc_auc": safe_metric("roc_auc", valid_errors.astype(int), valid_scores),
            "average_precision": safe_metric(
                "average_precision",
                valid_errors.astype(int),
                valid_scores,
            ),
        },
        "error_rates": {
            "overall": float(valid_errors.mean()) if evaluated.any() else None,
            "pseudo_boundary_top_ratio": high_boundary_error_rate,
            "pseudo_interior": interior_error_rate,
            "gt_boundary": gt_boundary_error_rate,
            "gt_interior": gt_interior_error_rate,
        },
        "ari": {
            "global": float(adjusted_rand_score(gt_labels[evaluated], pseudo_labels[evaluated])),
            "gt_boundary": (
                float(adjusted_rand_score(gt_labels[gt_boundary_mask], pseudo_labels[gt_boundary_mask]))
                if gt_boundary_mask.sum() > 1
                else None
            ),
            "gt_interior": (
                float(adjusted_rand_score(gt_labels[gt_interior_mask], pseudo_labels[gt_interior_mask]))
                if gt_interior_mask.sum() > 1
                else None
            ),
        },
        "nmi": {
            "global": float(normalized_mutual_info_score(gt_labels[evaluated], pseudo_labels[evaluated])),
            "gt_boundary": (
                float(normalized_mutual_info_score(gt_labels[gt_boundary_mask], pseudo_labels[gt_boundary_mask]))
                if gt_boundary_mask.sum() > 1
                else None
            ),
            "gt_interior": (
                float(normalized_mutual_info_score(gt_labels[gt_interior_mask], pseudo_labels[gt_interior_mask]))
                if gt_interior_mask.sum() > 1
                else None
            ),
        },
        "score_summary": {
            "combined_mean": float(combined_boundary_score.mean()),
            "combined_std": float(combined_boundary_score.std()),
            "confidence_mean": float(confidence.mean()),
            "confidence_std": float(confidence.std()),
        },
        "runtime": get_runtime_metadata(REPO_ROOT),
    }
    write_json(output_dir / "diagnostics_metrics.json", diagnostics)

    plot_spatial_score(
        adata,
        figures_dir / "spatial_boundary_score.png",
        "phase1_spatial_boundary_score",
        "Spatial boundary score",
    )
    plot_spatial_score(
        adata,
        figures_dir / "embedding_boundary_score.png",
        "phase1_embedding_boundary_score",
        "Embedding boundary score",
    )
    plot_spatial_score(
        adata,
        figures_dir / "combined_boundary_score.png",
        "phase1_combined_boundary_score",
        "Combined boundary score",
    )
    plot_spatial_score(
        adata,
        figures_dir / "confidence_spatial.png",
        "phase1_pseudo_confidence",
        "mclust posterior confidence",
    )
    plot_spatial_score(
        adata,
        figures_dir / "error_spatial.png",
        "phase1_pseudo_error",
        "Pseudo-label error",
    )
    plot_spatial_score(
        adata,
        figures_dir / "gt_boundary_spatial.png",
        "phase1_gt_boundary",
        "Ground-truth boundary",
    )
    plot_spatial_score(
        adata,
        figures_dir / "pseudo_boundary_spatial.png",
        "phase1_pseudo_boundary",
        "Pseudo boundary top ratio",
    )

    adata.write_h5ad(output_dir / f"{args.sample_id}_phase1_diagnostics.h5ad")
    print(f"Boundary diagnostics saved to {output_dir.resolve()}")
    return diagnostics


def main() -> None:
    run_diagnostics(parse_args())


if __name__ == "__main__":
    main()
