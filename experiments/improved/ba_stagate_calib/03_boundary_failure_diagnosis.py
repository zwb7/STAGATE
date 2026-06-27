"""Boundary failure diagnosis for STAGATE spatial domain predictions.

This script implements Experiment 1 for BA-STAGATE-Calib. It does not train
STAGATE or rerun clustering. It consumes an AnnData file that already contains:

  - STAGATE embeddings in ``adata.obsm[embedding_key]``
  - a spatial graph in ``adata.uns[spatial_net_key]`` with Cell1/Cell2 columns
  - predicted domain labels in ``adata.obs[pred_key]``
  - ground-truth domain labels in ``adata.obs[truth_key]``

It writes per-spot boundary diagnostics and aggregate boundary-vs-interior
failure metrics for the STAGATE baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors


DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)
DEFAULT_TOP_RATIOS = (0.10, 0.20, 0.30)


def _as_str_array(values: Iterable[object]) -> np.ndarray:
    """Return labels as strings while preserving missing values as empty tags."""

    series = pd.Series(values)
    return series.astype("string").fillna("__missing__").to_numpy(dtype=str)


def _read_adata(path: Path):
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required to read .h5ad files. Install it on the server "
            "environment before running this diagnostic script."
        ) from exc

    return ad.read_h5ad(path)


def _check_required_keys(
    adata,
    embedding_key: str,
    pred_key: str,
    truth_key: str,
    spatial_net_key: str,
) -> None:
    missing = []
    if embedding_key not in adata.obsm:
        missing.append(f"adata.obsm[{embedding_key!r}]")
    if pred_key not in adata.obs:
        missing.append(f"adata.obs[{pred_key!r}]")
    if truth_key not in adata.obs:
        missing.append(f"adata.obs[{truth_key!r}]")
    if spatial_net_key not in adata.uns:
        missing.append(f"adata.uns[{spatial_net_key!r}]")
    if missing:
        raise KeyError("Missing required fields: " + ", ".join(missing))


def build_neighbor_lists(adata, spatial_net_key: str = "Spatial_Net") -> list[list[int]]:
    """Build undirected spatial neighbor lists from STAGATE-style Spatial_Net."""

    graph = adata.uns[spatial_net_key]
    if not isinstance(graph, pd.DataFrame):
        graph = pd.DataFrame(graph)
    required = {"Cell1", "Cell2"}
    if not required.issubset(graph.columns):
        raise ValueError(
            f"adata.uns[{spatial_net_key!r}] must contain Cell1 and Cell2 columns."
        )

    obs_to_idx = {obs_name: idx for idx, obs_name in enumerate(adata.obs_names)}
    neighbors: list[set[int]] = [set() for _ in range(adata.n_obs)]

    for cell1, cell2 in graph.loc[:, ["Cell1", "Cell2"]].itertuples(index=False):
        if cell1 not in obs_to_idx or cell2 not in obs_to_idx:
            continue
        i = obs_to_idx[cell1]
        j = obs_to_idx[cell2]
        if i == j:
            continue
        neighbors[i].add(j)
        neighbors[j].add(i)

    return [sorted(items) for items in neighbors]


def label_discordance(labels: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    """Fraction of neighbors whose label differs from each spot label."""

    scores = np.zeros(labels.shape[0], dtype=float)
    for idx, neigh in enumerate(neighbors):
        if not neigh:
            scores[idx] = 0.0
            continue
        neigh_labels = labels[np.asarray(neigh, dtype=int)]
        scores[idx] = float(np.mean(neigh_labels != labels[idx]))
    return scores


def embedding_label_discordance(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int = 10,
) -> np.ndarray:
    """Label discordance among kNN neighbors in STAGATE embedding space."""

    if embedding.ndim != 2:
        raise ValueError(f"Embedding must be 2D, got shape {embedding.shape}.")
    if embedding.shape[0] != labels.shape[0]:
        raise ValueError("Embedding row count does not match label count.")
    if embedding.shape[0] < 2:
        return np.zeros(embedding.shape[0], dtype=float)

    k = min(n_neighbors + 1, embedding.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nbrs.fit(embedding)
    indices = nbrs.kneighbors(embedding, return_distance=False)

    scores = np.zeros(embedding.shape[0], dtype=float)
    for idx in range(embedding.shape[0]):
        neigh = [int(j) for j in indices[idx] if int(j) != idx]
        if not neigh:
            scores[idx] = 0.0
            continue
        neigh = neigh[:n_neighbors]
        scores[idx] = float(np.mean(labels[np.asarray(neigh, dtype=int)] != labels[idx]))
    return scores


def read_confidence(adata, confidence_key: str | None) -> np.ndarray:
    """Read optional pseudo-label confidence; default to 1.0 when unavailable."""

    if confidence_key is None:
        return np.ones(adata.n_obs, dtype=float)
    if confidence_key not in adata.obs:
        raise KeyError(f"adata.obs[{confidence_key!r}] not found.")

    confidence = pd.to_numeric(adata.obs[confidence_key], errors="coerce")
    confidence = confidence.fillna(1.0).clip(lower=0.0, upper=1.0)
    return confidence.to_numpy(dtype=float)


def combined_boundary_score(
    spatial_score: np.ndarray,
    embedding_score: np.ndarray,
    confidence: np.ndarray,
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> np.ndarray:
    """Combine spatial discordance, embedding discordance, and uncertainty."""

    w_spatial, w_embedding, w_uncertainty = weights
    total = w_spatial + w_embedding + w_uncertainty
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Boundary weights must sum to 1.0, got {total}.")
    uncertainty = 1.0 - np.clip(confidence, 0.0, 1.0)
    return (
        w_spatial * spatial_score
        + w_embedding * embedding_score
        + w_uncertainty * uncertainty
    )


def hungarian_match_correct(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    """Return per-spot correctness after optimal cluster-to-label matching."""

    contingency = contingency_matrix(true_labels, pred_labels, sparse=False)
    if contingency.size == 0:
        return np.zeros(true_labels.shape[0], dtype=bool)

    row_ind, col_ind = linear_sum_assignment(-contingency)
    true_classes = np.unique(true_labels)
    pred_classes = np.unique(pred_labels)
    pred_to_true = {
        pred_classes[col]: true_classes[row]
        for row, col in zip(row_ind, col_ind)
        if row < len(true_classes) and col < len(pred_classes)
    }
    mapped = np.asarray([pred_to_true.get(label, "__unmatched__") for label in pred_labels])
    return mapped == true_labels


def safe_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """ARI with NaN for subsets that are too small or single-class."""

    if true_labels.shape[0] < 2:
        return float("nan")
    if np.unique(true_labels).shape[0] < 2 or np.unique(pred_labels).shape[0] < 2:
        return float("nan")
    return float(adjusted_rand_score(true_labels, pred_labels))


def safe_error_rate(errors: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float(np.mean(errors[mask]))


def safe_auc(errors: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(errors).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(errors.astype(int), scores))


def safe_spearman(errors: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(errors).shape[0] < 2 or np.unique(scores).shape[0] < 2:
        return float("nan")
    value = spearmanr(scores, errors.astype(float), nan_policy="omit").correlation
    return float(value) if value is not None else float("nan")


def compute_top_ratio_rows(
    truth: np.ndarray,
    pred: np.ndarray,
    errors: np.ndarray,
    gt_boundary: np.ndarray,
    boundary_score: np.ndarray,
    ratios: Iterable[float],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    global_error = float(np.mean(errors))
    order = np.argsort(-boundary_score)

    for ratio in ratios:
        count = max(1, int(round(truth.shape[0] * ratio)))
        mask = np.zeros(truth.shape[0], dtype=bool)
        mask[order[:count]] = True
        error_rate = float(np.mean(errors[mask]))
        rows.append(
            {
                "top_boundary_ratio": float(ratio),
                "n_spots": int(mask.sum()),
                "ari": safe_ari(truth[mask], pred[mask]),
                "error_rate": error_rate,
                "error_enrichment": (
                    float(error_rate / global_error) if global_error > 0 else float("nan")
                ),
                "gt_boundary_overlap": float(np.mean(gt_boundary[mask])),
            }
        )
    return rows


def run_boundary_diagnosis(
    adata,
    embedding_key: str,
    pred_key: str,
    truth_key: str,
    confidence_key: str | None,
    spatial_net_key: str,
    embedding_knn: int,
    weights: tuple[float, float, float],
    top_ratios: Iterable[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _check_required_keys(adata, embedding_key, pred_key, truth_key, spatial_net_key)

    embedding = np.asarray(adata.obsm[embedding_key], dtype=float)
    pred = _as_str_array(adata.obs[pred_key])
    truth = _as_str_array(adata.obs[truth_key])
    confidence = read_confidence(adata, confidence_key)
    neighbors = build_neighbor_lists(adata, spatial_net_key=spatial_net_key)

    spatial_score = label_discordance(pred, neighbors)
    embedding_score = embedding_label_discordance(
        embedding,
        pred,
        n_neighbors=embedding_knn,
    )
    boundary_score = combined_boundary_score(
        spatial_score,
        embedding_score,
        confidence,
        weights=weights,
    )
    gt_boundary_score = label_discordance(truth, neighbors)
    gt_boundary = gt_boundary_score > 0.0

    correct = hungarian_match_correct(truth, pred)
    errors = ~correct
    interior = ~gt_boundary

    global_error = float(np.mean(errors))
    boundary_error = safe_error_rate(errors, gt_boundary)
    interior_error = safe_error_rate(errors, interior)
    metrics = {
        "n_spots": int(adata.n_obs),
        "n_spatial_edges_undirected": int(sum(len(items) for items in neighbors) / 2),
        "global_ari": safe_ari(truth, pred),
        "global_nmi": float(normalized_mutual_info_score(truth, pred)),
        "global_ami": float(adjusted_mutual_info_score(truth, pred)),
        "global_error_rate": global_error,
        "gt_boundary_spots": int(gt_boundary.sum()),
        "gt_boundary_fraction": float(np.mean(gt_boundary)),
        "boundary_ari": safe_ari(truth[gt_boundary], pred[gt_boundary]),
        "interior_ari": safe_ari(truth[interior], pred[interior]),
        "boundary_error_rate": boundary_error,
        "interior_error_rate": interior_error,
        "boundary_interior_error_ratio": (
            float(boundary_error / interior_error)
            if interior_error and not math.isnan(interior_error)
            else float("nan")
        ),
        "boundary_score_error_auc": safe_auc(errors, boundary_score),
        "boundary_score_error_spearman": safe_spearman(errors, boundary_score),
        "mean_boundary_score_correct": float(np.mean(boundary_score[correct]))
        if correct.any()
        else float("nan"),
        "mean_boundary_score_error": float(np.mean(boundary_score[errors]))
        if errors.any()
        else float("nan"),
    }

    spot_df = pd.DataFrame(
        {
            "spot_id": np.asarray(adata.obs_names, dtype=str),
            "pred_label": pred,
            "truth_label": truth,
            "is_correct_after_hungarian": correct,
            "is_error_after_hungarian": errors,
            "confidence": confidence,
            "spatial_boundary_score": spatial_score,
            "embedding_boundary_score": embedding_score,
            "boundary_score": boundary_score,
            "gt_boundary_score": gt_boundary_score,
            "is_gt_boundary": gt_boundary,
            "spatial_degree": np.asarray([len(items) for items in neighbors], dtype=int),
        }
    )
    metric_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )
    top_ratio_df = pd.DataFrame(
        compute_top_ratio_rows(truth, pred, errors, gt_boundary, boundary_score, top_ratios)
    )
    return spot_df, metric_df, top_ratio_df


def parse_float_triplet(value: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats.")
    return (parts[0], parts[1], parts[2])


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run STAGATE boundary failure diagnosis from a precomputed h5ad."
    )
    parser.add_argument("--adata", required=True, type=Path, help="Input .h5ad path.")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for diagnostic CSV/JSON outputs.",
    )
    parser.add_argument("--embedding-key", default="STAGATE")
    parser.add_argument("--pred-key", default="mclust")
    parser.add_argument("--truth-key", default="Ground Truth")
    parser.add_argument(
        "--confidence-key",
        default=None,
        help="Optional adata.obs key with mclust/GMM confidence in [0, 1].",
    )
    parser.add_argument("--spatial-net-key", default="Spatial_Net")
    parser.add_argument("--embedding-knn", default=10, type=int)
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        type=parse_float_triplet,
        help="Boundary score weights as spatial,embedding,uncertainty.",
    )
    parser.add_argument(
        "--top-ratios",
        default=list(DEFAULT_TOP_RATIOS),
        type=parse_float_list,
        help="Comma-separated top boundary score ratios for enrichment analysis.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    adata = _read_adata(args.adata)
    spot_df, metric_df, top_ratio_df = run_boundary_diagnosis(
        adata=adata,
        embedding_key=args.embedding_key,
        pred_key=args.pred_key,
        truth_key=args.truth_key,
        confidence_key=args.confidence_key,
        spatial_net_key=args.spatial_net_key,
        embedding_knn=args.embedding_knn,
        weights=args.weights,
        top_ratios=args.top_ratios,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spot_df.to_csv(args.output_dir / "spot_boundary_diagnostics.csv", index=False)
    metric_df.to_csv(args.output_dir / "boundary_failure_metrics.csv", index=False)
    top_ratio_df.to_csv(args.output_dir / "top_boundary_enrichment.csv", index=False)

    run_config = {
        "adata": str(args.adata),
        "embedding_key": args.embedding_key,
        "pred_key": args.pred_key,
        "truth_key": args.truth_key,
        "confidence_key": args.confidence_key,
        "spatial_net_key": args.spatial_net_key,
        "embedding_knn": args.embedding_knn,
        "weights": list(args.weights),
        "top_ratios": list(args.top_ratios),
    }
    with (args.output_dir / "boundary_failure_config.json").open("w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    print("Boundary failure diagnosis completed.")
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
