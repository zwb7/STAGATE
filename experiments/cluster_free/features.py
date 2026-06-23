from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc

from examples.rule_based_graph_refinement import (
    score_embedding_edges,
    score_expression_edges,
)

FEATURE_COLUMNS = [
    "expression_similarity",
    "embedding_similarity",
    "spatial_distance",
    "neighborhood_jaccard",
    "degree_min",
    "degree_max",
]


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum <= minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def build_neighbor_sets(
    pairs: pd.DataFrame,
    n_nodes: int,
) -> tuple[list[set[int]], np.ndarray]:
    neighbors = [set() for _ in range(n_nodes)]
    for node_a, node_b in pairs[
        ["node_a_index", "node_b_index"]
    ].itertuples(index=False, name=None):
        node_a = int(node_a)
        node_b = int(node_b)
        neighbors[node_a].add(node_b)
        neighbors[node_b].add(node_a)
    degrees = np.fromiter(
        (len(node_neighbors) for node_neighbors in neighbors),
        dtype=np.int64,
        count=n_nodes,
    )
    return neighbors, degrees


def neighborhood_jaccard(
    pairs: pd.DataFrame,
    neighbors: list[set[int]],
) -> np.ndarray:
    scores = np.empty(pairs.shape[0], dtype=np.float64)
    for row_index, (node_a, node_b) in enumerate(
        pairs[["node_a_index", "node_b_index"]].itertuples(
            index=False,
            name=None,
        )
    ):
        set_a = neighbors[int(node_a)]
        set_b = neighbors[int(node_b)]
        union_size = len(set_a | set_b)
        scores[row_index] = (
            len(set_a & set_b) / union_size if union_size else 0.0
        )
    return scores


def build_edge_features(
    adata: sc.AnnData,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    neighbors, degrees = build_neighbor_sets(pairs, adata.n_obs)
    node_a = pairs["node_a_index"].to_numpy(dtype=int)
    node_b = pairs["node_b_index"].to_numpy(dtype=int)
    degree_a = degrees[node_a]
    degree_b = degrees[node_b]

    features = pairs.loc[
        :,
        [
            "pair_id",
            "node_a_index",
            "node_b_index",
            "node_a",
            "node_b",
            "distance",
        ],
    ].copy()
    features["expression_similarity"] = score_expression_edges(adata, pairs)
    features["embedding_similarity"] = score_embedding_edges(adata, pairs)
    features["spatial_distance"] = pairs["distance"].to_numpy(dtype=float)
    features["neighborhood_jaccard"] = neighborhood_jaccard(
        pairs,
        neighbors,
    )
    features["degree_min"] = np.minimum(degree_a, degree_b)
    features["degree_max"] = np.maximum(degree_a, degree_b)

    feature_values = features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(feature_values).all():
        raise ValueError("Edge features contain NaN or infinite values")
    return features


def build_soft_targets(
    features: pd.DataFrame,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
) -> np.ndarray:
    if not np.isclose(alpha + beta + gamma, 1.0):
        raise ValueError("Soft-target weights must sum to one")
    expression = minmax_scale(
        features["expression_similarity"].to_numpy(dtype=float)
    )
    embedding = minmax_scale(
        features["embedding_similarity"].to_numpy(dtype=float)
    )
    neighborhood = minmax_scale(
        features["neighborhood_jaccard"].to_numpy(dtype=float)
    )
    targets = alpha * expression + beta * embedding + gamma * neighborhood
    return np.clip(targets, 0.0, 1.0)


def standardize_features(
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    safe_standard_deviation = np.where(
        standard_deviation > 0,
        standard_deviation,
        1.0,
    )
    standardized = (values - mean) / safe_standard_deviation
    return (
        standardized.astype(np.float32),
        mean,
        safe_standard_deviation,
    )

