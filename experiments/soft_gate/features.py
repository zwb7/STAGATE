from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.mixture import GaussianMixture


@dataclass
class EdgePriorResult:
    table: pd.DataFrame
    soft_assignments: np.ndarray
    preserve_c_threshold: float
    preserve_z_threshold: float


def validate_warmup_embedding(
    adata: sc.AnnData,
    embedding_key: str,
) -> np.ndarray:
    if embedding_key not in adata.obsm:
        raise KeyError(f"Warm-up embedding not found in adata.obsm: {embedding_key}")
    embedding = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    if embedding.ndim != 2 or embedding.shape[0] != adata.n_obs:
        raise ValueError(
            f"{embedding_key} must have shape ({adata.n_obs}, latent_dim), "
            f"got {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError(f"{embedding_key} contains NaN or infinite values")
    return embedding


def fit_soft_assignments(
    embedding: np.ndarray,
    n_components: int,
    seed: int,
) -> np.ndarray:
    if n_components <= 1:
        raise ValueError("n_components must be greater than 1")
    if n_components >= embedding.shape[0]:
        raise ValueError("n_components must be smaller than the number of spots")
    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=seed,
    )
    model.fit(embedding)
    probabilities = model.predict_proba(embedding)
    if probabilities.shape != (embedding.shape[0], n_components):
        raise RuntimeError(
            "Unexpected soft-assignment shape: "
            f"{probabilities.shape}, expected {(embedding.shape[0], n_components)}"
        )
    return probabilities.astype(np.float64, copy=False)


def cosine_pair_similarity(
    embedding: np.ndarray,
    pairs: pd.DataFrame,
) -> np.ndarray:
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = np.divide(
        embedding,
        norm,
        out=np.zeros_like(embedding),
        where=norm > 0,
    )
    node_a = pairs["node_a_index"].to_numpy(dtype=int)
    node_b = pairs["node_b_index"].to_numpy(dtype=int)
    return np.einsum("ij,ij->i", normalized[node_a], normalized[node_b])


def build_adjacency_lists(
    n_nodes: int,
    pairs: pd.DataFrame,
) -> list[np.ndarray]:
    neighbors: list[list[int]] = [[] for _ in range(n_nodes)]
    node_a = pairs["node_a_index"].to_numpy(dtype=int)
    node_b = pairs["node_b_index"].to_numpy(dtype=int)
    for left, right in zip(node_a, node_b, strict=True):
        neighbors[int(left)].append(int(right))
        neighbors[int(right)].append(int(left))
    return [
        np.asarray(sorted(set(node_neighbors)), dtype=int)
        for node_neighbors in neighbors
    ]


def energy_distance(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    if sample_a.ndim != 2 or sample_b.ndim != 2:
        raise ValueError("Energy distance inputs must be two-dimensional")
    if sample_a.shape[1] != sample_b.shape[1]:
        raise ValueError("Energy distance inputs have different feature dimensions")
    cross = np.linalg.norm(sample_a[:, None, :] - sample_b[None, :, :], axis=2)
    within_a = np.linalg.norm(sample_a[:, None, :] - sample_a[None, :, :], axis=2)
    within_b = np.linalg.norm(sample_b[:, None, :] - sample_b[None, :, :], axis=2)
    value = 2.0 * cross.mean() - within_a.mean() - within_b.mean()
    return float(max(value, 0.0))


def compute_local_energy_distances(
    embedding: np.ndarray,
    pairs: pd.DataFrame,
    adjacency_lists: list[np.ndarray],
) -> np.ndarray:
    distances = np.empty(pairs.shape[0], dtype=np.float64)
    node_a = pairs["node_a_index"].to_numpy(dtype=int)
    node_b = pairs["node_b_index"].to_numpy(dtype=int)
    for row, (left, right) in enumerate(zip(node_a, node_b, strict=True)):
        left_indices = np.concatenate(
            [np.asarray([left], dtype=int), adjacency_lists[int(left)]]
        )
        right_indices = np.concatenate(
            [np.asarray([right], dtype=int), adjacency_lists[int(right)]]
        )
        distances[row] = energy_distance(
            embedding[left_indices],
            embedding[right_indices],
        )
    return distances


def zscore(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std() + eps)


def build_edge_priors(
    adata: sc.AnnData,
    pairs: pd.DataFrame,
    *,
    clusters: int,
    seed: int,
    embedding_key: str = "STAGATE",
    preserve_c_quantile: float = 0.75,
    preserve_z_quantile: float = 0.70,
    eps: float = 1e-8,
) -> EdgePriorResult:
    if not 0.0 < preserve_c_quantile < 1.0:
        raise ValueError("preserve_c_quantile must be in (0, 1)")
    if not 0.0 < preserve_z_quantile < 1.0:
        raise ValueError("preserve_z_quantile must be in (0, 1)")

    embedding = validate_warmup_embedding(adata, embedding_key)
    soft_assignments = fit_soft_assignments(embedding, clusters, seed)
    node_a = pairs["node_a_index"].to_numpy(dtype=int)
    node_b = pairs["node_b_index"].to_numpy(dtype=int)

    consistency = np.einsum(
        "ij,ij->i",
        soft_assignments[node_a],
        soft_assignments[node_b],
    )
    consistency = np.clip(consistency, eps, 1.0)
    embedding_similarity = cosine_pair_similarity(embedding, pairs)
    adjacency_lists = build_adjacency_lists(adata.n_obs, pairs)
    energy = compute_local_energy_distances(embedding, pairs, adjacency_lists)
    energy_z = zscore(energy, eps=eps)

    c_threshold = float(np.quantile(consistency, preserve_c_quantile))
    z_threshold = float(np.quantile(embedding_similarity, preserve_z_quantile))
    preserve_edge = (consistency > c_threshold) & (
        embedding_similarity > z_threshold
    )

    table = pairs.copy()
    table["soft_domain_consistency"] = consistency
    table["log_soft_domain_consistency"] = np.log(consistency + eps)
    table["embedding_similarity"] = embedding_similarity
    table["energy_distance"] = energy
    table["energy_distance_z"] = energy_z
    table["preserve_edge"] = preserve_edge
    return EdgePriorResult(
        table=table,
        soft_assignments=soft_assignments,
        preserve_c_threshold=c_threshold,
        preserve_z_threshold=z_threshold,
    )
