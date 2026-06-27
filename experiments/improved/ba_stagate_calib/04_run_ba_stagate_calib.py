"""Run deterministic BA-STAGATE-Calib on precomputed STAGATE results.

This script implements Experiment 2: BA-STAGATE-Calib Main Result.

It does not train STAGATE and does not modify STAGATE_pyG. It consumes an AnnData
file with precomputed STAGATE embeddings, STAGATE/mclust labels, ground-truth
labels, and Spatial_Net, then performs conservative boundary-only label
calibration:

  1. Compute boundary structural entanglement scores.
  2. Select top boundary_ratio spots as boundary candidates.
  3. Build high-confidence, low-boundary domain-core prototypes.
  4. Reassign only boundary candidates if a local candidate domain exceeds the
     original label score by a fixed margin and satisfies conservative evidence
     constraints on spatial support and prototype similarity.
  5. Report STAGATE vs BA-STAGATE-Calib metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import NearestNeighbors


DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)


def _read_adata(path: Path):
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required to read .h5ad files. Install it on the server "
            "environment before running this script."
        ) from exc

    return ad.read_h5ad(path)


def _as_str_array(values: Iterable[object]) -> np.ndarray:
    series = pd.Series(values)
    return series.astype("string").fillna("__missing__").to_numpy(dtype=str)


def _check_required_keys(
    adata,
    embedding_key: str,
    pred_key: str,
    spatial_net_key: str,
    truth_key: Optional[str],
) -> None:
    missing = []
    if embedding_key not in adata.obsm:
        missing.append(f"adata.obsm[{embedding_key!r}]")
    if pred_key not in adata.obs:
        missing.append(f"adata.obs[{pred_key!r}]")
    if spatial_net_key not in adata.uns:
        missing.append(f"adata.uns[{spatial_net_key!r}]")
    if truth_key and truth_key not in adata.obs:
        missing.append(f"adata.obs[{truth_key!r}]")
    if missing:
        raise KeyError("Missing required fields: " + ", ".join(missing))


def build_neighbor_lists(adata, spatial_net_key: str = "Spatial_Net") -> List[List[int]]:
    graph = adata.uns[spatial_net_key]
    if not isinstance(graph, pd.DataFrame):
        graph = pd.DataFrame(graph)
    if not {"Cell1", "Cell2"}.issubset(graph.columns):
        raise ValueError(
            f"adata.uns[{spatial_net_key!r}] must contain Cell1 and Cell2 columns."
        )

    obs_to_idx = {obs_name: idx for idx, obs_name in enumerate(adata.obs_names)}
    neighbors: List[set] = [set() for _ in range(adata.n_obs)]
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


def label_discordance(labels: np.ndarray, neighbors: List[List[int]]) -> np.ndarray:
    scores = np.zeros(labels.shape[0], dtype=float)
    for idx, neigh in enumerate(neighbors):
        if not neigh:
            continue
        neigh_labels = labels[np.asarray(neigh, dtype=int)]
        scores[idx] = float(np.mean(neigh_labels != labels[idx]))
    return scores


def embedding_label_discordance(
    embedding: np.ndarray,
    labels: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
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
        neigh = [int(j) for j in indices[idx] if int(j) != idx][:n_neighbors]
        if neigh:
            scores[idx] = float(np.mean(labels[np.asarray(neigh, dtype=int)] != labels[idx]))
    return scores


def read_confidence(adata, confidence_key: Optional[str]) -> np.ndarray:
    if confidence_key is None:
        return np.ones(adata.n_obs, dtype=float)
    if confidence_key not in adata.obs:
        raise KeyError(f"adata.obs[{confidence_key!r}] not found.")
    confidence = pd.to_numeric(adata.obs[confidence_key], errors="coerce")
    return confidence.fillna(1.0).clip(lower=0.0, upper=1.0).to_numpy(dtype=float)


def combined_boundary_score(
    spatial_score: np.ndarray,
    embedding_score: np.ndarray,
    confidence: np.ndarray,
    weights: Tuple[float, float, float],
) -> np.ndarray:
    total = sum(weights)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Boundary weights must sum to 1.0, got {total}.")
    return (
        weights[0] * spatial_score
        + weights[1] * embedding_score
        + weights[2] * (1.0 - np.clip(confidence, 0.0, 1.0))
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return matrix / denom


def select_boundary_candidates(boundary_score: np.ndarray, boundary_ratio: float) -> np.ndarray:
    if not 0.0 < boundary_ratio <= 1.0:
        raise ValueError("boundary_ratio must be in (0, 1].")
    n_candidates = max(1, int(round(boundary_score.shape[0] * boundary_ratio)))
    order = np.argsort(-boundary_score)
    mask = np.zeros(boundary_score.shape[0], dtype=bool)
    mask[order[:n_candidates]] = True
    return mask


def build_domain_core_prototypes(
    embedding: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    boundary_score: np.ndarray,
    tau_core: float,
    core_quantile: float,
    fallback_ratio: float = 0.20,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    prototypes: Dict[str, np.ndarray] = {}
    core_masks: Dict[str, np.ndarray] = {}

    for label in np.unique(labels):
        cluster_mask = labels == label
        cluster_indices = np.flatnonzero(cluster_mask)
        if cluster_indices.shape[0] == 0:
            continue

        threshold = float(np.quantile(boundary_score[cluster_mask], core_quantile))
        core_mask = cluster_mask & (confidence >= tau_core) & (boundary_score <= threshold)

        if int(core_mask.sum()) == 0:
            fallback_count = max(1, int(math.ceil(cluster_indices.shape[0] * fallback_ratio)))
            cluster_conf = confidence[cluster_indices]
            cluster_boundary = boundary_score[cluster_indices]
            order = np.lexsort((cluster_boundary, -cluster_conf))
            fallback_indices = cluster_indices[order[:fallback_count]]
            core_mask = np.zeros(labels.shape[0], dtype=bool)
            core_mask[fallback_indices] = True

        prototypes[str(label)] = embedding[core_mask].mean(axis=0)
        core_masks[str(label)] = core_mask

    return prototypes, core_masks


def make_posterior_lookup(
    adata,
    labels: np.ndarray,
    confidence: np.ndarray,
    posterior_key: Optional[str],
) -> Tuple[Dict[str, int], Optional[np.ndarray], List[str]]:
    unique_labels = [str(label) for label in np.unique(labels)]
    label_to_col = {label: idx for idx, label in enumerate(unique_labels)}

    if posterior_key is None:
        return label_to_col, None, unique_labels
    if posterior_key not in adata.obsm:
        raise KeyError(f"adata.obsm[{posterior_key!r}] not found.")

    posterior = np.asarray(adata.obsm[posterior_key], dtype=float)
    if posterior.shape[0] != labels.shape[0]:
        raise ValueError("Posterior row count does not match number of spots.")
    if posterior.shape[1] != len(unique_labels):
        raise ValueError(
            "Posterior column count must match the number of predicted labels "
            "when using the default sorted-label mapping."
        )
    return label_to_col, posterior, unique_labels


def posterior_value(
    spot_idx: int,
    label: str,
    original_label: str,
    confidence: np.ndarray,
    label_to_col: Dict[str, int],
    posterior: Optional[np.ndarray],
) -> float:
    if posterior is not None:
        col = label_to_col.get(label)
        return float(posterior[spot_idx, col]) if col is not None else 0.0
    return 0.0


def spatial_support(label: str, labels: np.ndarray, neigh: Sequence[int]) -> float:
    if not neigh:
        return 0.0
    neigh_labels = labels[np.asarray(neigh, dtype=int)]
    return float(np.mean(neigh_labels == label))


def run_calibration(
    adata,
    embedding_key: str,
    pred_key: str,
    truth_key: Optional[str],
    confidence_key: Optional[str],
    posterior_key: Optional[str],
    spatial_net_key: str,
    embedding_knn: int,
    weights: Tuple[float, float, float],
    boundary_ratio: float,
    tau_core: float,
    core_quantile: float,
    top_k_proto: int,
    margin: float,
    alpha: float,
    beta: float,
    gamma: float,
    require_spatial_gain: float,
    require_proto_gain: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _check_required_keys(adata, embedding_key, pred_key, spatial_net_key, truth_key)

    embedding = np.asarray(adata.obsm[embedding_key], dtype=float)
    labels = _as_str_array(adata.obs[pred_key])
    confidence = read_confidence(adata, confidence_key)
    neighbors = build_neighbor_lists(adata, spatial_net_key)

    spatial_score = label_discordance(labels, neighbors)
    embedding_score = embedding_label_discordance(embedding, labels, embedding_knn)
    boundary_score = combined_boundary_score(spatial_score, embedding_score, confidence, weights)
    boundary_candidates = select_boundary_candidates(boundary_score, boundary_ratio)

    prototypes, core_masks = build_domain_core_prototypes(
        embedding=embedding,
        labels=labels,
        confidence=confidence,
        boundary_score=boundary_score,
        tau_core=tau_core,
        core_quantile=core_quantile,
    )
    if not prototypes:
        raise RuntimeError("No domain-core prototypes were constructed.")

    prototype_labels = sorted(prototypes.keys())
    prototype_matrix = np.vstack([prototypes[label] for label in prototype_labels])
    norm_embedding = l2_normalize(embedding)
    norm_prototypes = l2_normalize(prototype_matrix)
    prototype_label_to_row = {label: idx for idx, label in enumerate(prototype_labels)}

    label_to_col, posterior, posterior_labels = make_posterior_lookup(
        adata=adata,
        labels=labels,
        confidence=confidence,
        posterior_key=posterior_key,
    )

    calibrated = labels.copy()
    rows: List[Dict[str, object]] = []

    for idx in range(labels.shape[0]):
        original = str(labels[idx])
        candidate_labels = {original}
        for neigh_idx in neighbors[idx]:
            candidate_labels.add(str(labels[neigh_idx]))

        proto_scores = norm_prototypes @ norm_embedding[idx]
        top_count = min(top_k_proto, len(prototype_labels))
        for proto_idx in np.argsort(-proto_scores)[:top_count]:
            candidate_labels.add(prototype_labels[int(proto_idx)])

        candidate_labels = sorted(label for label in candidate_labels if label in prototypes)
        if original not in candidate_labels and original in prototypes:
            candidate_labels.append(original)

        score_parts: Dict[str, Tuple[float, float, float, float]] = {}
        for label in candidate_labels:
            proto_row = prototype_label_to_row[label]
            proto_similarity = float(norm_embedding[idx] @ norm_prototypes[proto_row])
            local_support = spatial_support(label, labels, neighbors[idx])
            posterior_score = posterior_value(
                spot_idx=idx,
                label=label,
                original_label=original,
                confidence=confidence,
                label_to_col=label_to_col,
                posterior=posterior,
            )
            total_score = (
                alpha * proto_similarity
                + beta * local_support
                + gamma * posterior_score
            )
            score_parts[label] = (
                total_score,
                proto_similarity,
                local_support,
                posterior_score,
            )

        best_label = max(score_parts, key=lambda item: score_parts[item][0])
        original_parts = score_parts[original]
        original_score = original_parts[0]
        best_score = score_parts[best_label][0]
        proto_gain = score_parts[best_label][1] - original_parts[1]
        spatial_gain = score_parts[best_label][2] - original_parts[2]
        score_margin = best_score - original_score
        passes_spatial_gain = spatial_gain >= require_spatial_gain
        passes_proto_gain = proto_gain > require_proto_gain
        changed = False

        if (
            boundary_candidates[idx]
            and best_label != original
            and score_margin > margin
            and passes_spatial_gain
            and passes_proto_gain
        ):
            calibrated[idx] = best_label
            changed = True

        best_parts = score_parts[best_label]
        rows.append(
            {
                "spot_id": str(adata.obs_names[idx]),
                "original_label": original,
                "calibrated_label": str(calibrated[idx]),
                "is_boundary_candidate": bool(boundary_candidates[idx]),
                "changed": bool(changed),
                "boundary_score": float(boundary_score[idx]),
                "spatial_boundary_score": float(spatial_score[idx]),
                "embedding_boundary_score": float(embedding_score[idx]),
                "confidence": float(confidence[idx]),
                "candidate_labels": ";".join(candidate_labels),
                "best_label": best_label,
                "original_score": float(original_score),
                "best_score": float(best_score),
                "score_margin": float(score_margin),
                "original_proto_similarity": float(original_parts[1]),
                "best_proto_similarity": float(best_parts[1]),
                "proto_gain": float(proto_gain),
                "original_spatial_support": float(original_parts[2]),
                "best_spatial_support": float(best_parts[2]),
                "spatial_gain": float(spatial_gain),
                "passes_spatial_gain": bool(passes_spatial_gain),
                "passes_proto_gain": bool(passes_proto_gain),
                "original_posterior": float(original_parts[3]),
                "best_posterior": float(best_parts[3]),
            }
        )

    detail_df = pd.DataFrame(rows)
    core_df = pd.DataFrame(
        [
            {
                "label": label,
                "n_cluster_spots": int(np.sum(labels == label)),
                "n_core_spots": int(core_masks[label].sum()),
            }
            for label in prototype_labels
        ]
    )

    metrics_df = build_metrics_table(
        adata=adata,
        truth_key=truth_key,
        original_labels=labels,
        calibrated_labels=calibrated,
        neighbors=neighbors,
        boundary_candidates=boundary_candidates,
    )
    return detail_df, metrics_df, core_df


def hungarian_correct(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
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
    if true_labels.shape[0] < 2:
        return float("nan")
    if np.unique(true_labels).shape[0] < 2 or np.unique(pred_labels).shape[0] < 2:
        return float("nan")
    return float(adjusted_rand_score(true_labels, pred_labels))


def method_metrics(
    method: str,
    truth: np.ndarray,
    labels: np.ndarray,
    gt_boundary: np.ndarray,
) -> Dict[str, object]:
    correct = hungarian_correct(truth, labels)
    errors = ~correct
    gt_interior = ~gt_boundary
    return {
        "method": method,
        "global_ari": safe_ari(truth, labels),
        "global_nmi": float(normalized_mutual_info_score(truth, labels)),
        "global_ami": float(adjusted_mutual_info_score(truth, labels)),
        "global_error_rate": float(np.mean(errors)),
        "boundary_ari": safe_ari(truth[gt_boundary], labels[gt_boundary]),
        "interior_ari": safe_ari(truth[gt_interior], labels[gt_interior]),
        "boundary_error_rate": float(np.mean(errors[gt_boundary]))
        if gt_boundary.any()
        else float("nan"),
        "interior_error_rate": float(np.mean(errors[gt_interior]))
        if gt_interior.any()
        else float("nan"),
    }


def build_metrics_table(
    adata,
    truth_key: Optional[str],
    original_labels: np.ndarray,
    calibrated_labels: np.ndarray,
    neighbors: List[List[int]],
    boundary_candidates: np.ndarray,
) -> pd.DataFrame:
    changed = calibrated_labels != original_labels
    rows: List[Dict[str, object]] = []

    if truth_key:
        truth = _as_str_array(adata.obs[truth_key])
        gt_boundary = label_discordance(truth, neighbors) > 0.0
        rows.append(method_metrics("STAGATE", truth, original_labels, gt_boundary))
        rows.append(method_metrics("BA_STAGATE_Calib", truth, calibrated_labels, gt_boundary))

        gt_interior = ~gt_boundary
        rows.append(
            {
                "method": "label_change",
                "n_changed": int(changed.sum()),
                "changed_ratio": float(np.mean(changed)),
                "candidate_changed_ratio": float(np.mean(changed[boundary_candidates]))
                if boundary_candidates.any()
                else float("nan"),
                "noncandidate_changed_ratio": float(np.mean(changed[~boundary_candidates]))
                if (~boundary_candidates).any()
                else float("nan"),
                "gt_boundary_changed_ratio": float(np.mean(changed[gt_boundary]))
                if gt_boundary.any()
                else float("nan"),
                "gt_interior_changed_ratio": float(np.mean(changed[gt_interior]))
                if gt_interior.any()
                else float("nan"),
                "n_boundary_candidates": int(boundary_candidates.sum()),
            }
        )
    else:
        rows.append(
            {
                "method": "label_change",
                "n_changed": int(changed.sum()),
                "changed_ratio": float(np.mean(changed)),
                "candidate_changed_ratio": float(np.mean(changed[boundary_candidates]))
                if boundary_candidates.any()
                else float("nan"),
                "noncandidate_changed_ratio": float(np.mean(changed[~boundary_candidates]))
                if (~boundary_candidates).any()
                else float("nan"),
                "n_boundary_candidates": int(boundary_candidates.sum()),
            }
        )

    return pd.DataFrame(rows)


def parse_float_triplet(value: str) -> Tuple[float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated floats.")
    return (parts[0], parts[1], parts[2])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic BA-STAGATE-Calib on precomputed STAGATE outputs."
    )
    parser.add_argument("--adata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--embedding-key", default="STAGATE")
    parser.add_argument("--pred-key", default="mclust")
    parser.add_argument("--truth-key", default="Ground Truth")
    parser.add_argument("--confidence-key", default=None)
    parser.add_argument("--posterior-key", default=None)
    parser.add_argument("--spatial-net-key", default="Spatial_Net")
    parser.add_argument("--embedding-knn", default=10, type=int)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, type=parse_float_triplet)
    parser.add_argument("--boundary-ratio", default=0.15, type=float)
    parser.add_argument("--tau-core", default=0.90, type=float)
    parser.add_argument("--core-quantile", default=0.50, type=float)
    parser.add_argument("--top-k-proto", default=2, type=int)
    parser.add_argument("--margin", default=0.0, type=float)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--gamma", default=0.25, type=float)
    parser.add_argument(
        "--require-spatial-gain",
        default=0.10,
        type=float,
        help="Minimum spatial support gain required before relabeling.",
    )
    parser.add_argument(
        "--require-proto-gain",
        default=0.00,
        type=float,
        help="Minimum prototype similarity gain required before relabeling.",
    )
    parser.add_argument(
        "--save-h5ad",
        action="store_true",
        help="Also save a copy of AnnData with BA-STAGATE-Calib labels in obs.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    adata = _read_adata(args.adata)
    detail_df, metrics_df, core_df = run_calibration(
        adata=adata,
        embedding_key=args.embedding_key,
        pred_key=args.pred_key,
        truth_key=args.truth_key,
        confidence_key=args.confidence_key,
        posterior_key=args.posterior_key,
        spatial_net_key=args.spatial_net_key,
        embedding_knn=args.embedding_knn,
        weights=args.weights,
        boundary_ratio=args.boundary_ratio,
        tau_core=args.tau_core,
        core_quantile=args.core_quantile,
        top_k_proto=args.top_k_proto,
        margin=args.margin,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        require_spatial_gain=args.require_spatial_gain,
        require_proto_gain=args.require_proto_gain,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(args.output_dir / "ba_stagate_calib_labels.csv", index=False)
    metrics_df.to_csv(args.output_dir / "ba_stagate_calib_metrics.csv", index=False)
    core_df.to_csv(args.output_dir / "ba_stagate_calib_core_prototypes.csv", index=False)

    config = {
        "adata": str(args.adata),
        "embedding_key": args.embedding_key,
        "pred_key": args.pred_key,
        "truth_key": args.truth_key,
        "confidence_key": args.confidence_key,
        "posterior_key": args.posterior_key,
        "spatial_net_key": args.spatial_net_key,
        "embedding_knn": args.embedding_knn,
        "weights": list(args.weights),
        "boundary_ratio": args.boundary_ratio,
        "tau_core": args.tau_core,
        "core_quantile": args.core_quantile,
        "top_k_proto": args.top_k_proto,
        "margin": args.margin,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "require_spatial_gain": args.require_spatial_gain,
        "require_proto_gain": args.require_proto_gain,
    }
    with (args.output_dir / "ba_stagate_calib_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    if args.save_h5ad:
        adata.obs["ba_stagate_calib"] = detail_df["calibrated_label"].to_numpy()
        adata.obs["ba_stagate_calib_changed"] = detail_df["changed"].to_numpy()
        adata.obs["ba_stagate_calib_boundary_candidate"] = detail_df[
            "is_boundary_candidate"
        ].to_numpy()
        adata.obs["ba_stagate_calib_boundary_score"] = detail_df[
            "boundary_score"
        ].to_numpy()
        adata.write_h5ad(args.output_dir / "adata_ba_stagate_calib.h5ad")

    print("BA-STAGATE-Calib completed.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
