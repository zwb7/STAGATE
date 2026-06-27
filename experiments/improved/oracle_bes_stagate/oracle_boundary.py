"""Ground-truth boundary utilities for the Oracle-BES-STAGATE experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


UNKNOWN_LABELS = {"", "NA", "N/A", "nan", "None", "Unknown", "unknown"}


@dataclass(frozen=True)
class OracleBoundaryData:
    labels: np.ndarray
    label_values: np.ndarray
    valid_mask: np.ndarray
    gt_boundary_score: np.ndarray
    gt_boundary_mask: np.ndarray
    gt_interior_mask: np.ndarray
    neighbors: list[np.ndarray]
    core_masks: dict[int, np.ndarray]
    adjacent_negative_labels: list[list[int]]


def validate_spatial_graph(graph: pd.DataFrame) -> None:
    if not isinstance(graph, pd.DataFrame):
        raise TypeError("adata.uns['Spatial_Net'] must be a pandas DataFrame")
    missing = sorted({"Cell1", "Cell2"}.difference(graph.columns))
    if missing:
        raise ValueError(f"Spatial_Net is missing columns: {missing}")


def build_neighbor_lists(graph: pd.DataFrame, obs_names: pd.Index) -> list[np.ndarray]:
    validate_spatial_graph(graph)
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


def encode_labels(values: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = values.astype(object)
    valid = raw.notna().to_numpy()
    as_text = raw.astype(str).to_numpy()
    valid &= ~np.isin(as_text, list(UNKNOWN_LABELS))

    labels = np.full(values.shape[0], -1, dtype=np.int64)
    label_values = np.asarray(sorted(pd.unique(as_text[valid])), dtype=object)
    label_to_index = {label: index for index, label in enumerate(label_values)}
    for index, label in enumerate(as_text):
        if valid[index]:
            labels[index] = label_to_index[label]
    return labels, label_values, valid


def neighbor_disagreement_score(
    labels: np.ndarray,
    neighbors: list[np.ndarray],
    valid_mask: np.ndarray,
) -> np.ndarray:
    scores = np.zeros(labels.shape[0], dtype=np.float64)
    for index, neighbor_index in enumerate(neighbors):
        if not valid_mask[index] or neighbor_index.size == 0:
            continue
        used = neighbor_index[valid_mask[neighbor_index]]
        if used.size == 0:
            continue
        scores[index] = float(np.mean(labels[used] != labels[index]))
    return scores


def adjacent_negative_labels(
    labels: np.ndarray,
    neighbors: list[np.ndarray],
    valid_mask: np.ndarray,
) -> list[list[int]]:
    result: list[list[int]] = []
    for index, neighbor_index in enumerate(neighbors):
        if not valid_mask[index] or neighbor_index.size == 0:
            result.append([])
            continue
        used = neighbor_index[valid_mask[neighbor_index]]
        own_label = int(labels[index])
        result.append(
            sorted(
                {
                    int(labels[neighbor])
                    for neighbor in used
                    if int(labels[neighbor]) != own_label
                }
            )
        )
    return result


def select_core_masks(
    labels: np.ndarray,
    valid_mask: np.ndarray,
    interior_mask: np.ndarray,
    min_core_spots: int,
) -> dict[int, np.ndarray]:
    core_masks: dict[int, np.ndarray] = {}
    for label in sorted(np.unique(labels[valid_mask])):
        label_mask = (labels == label) & valid_mask
        core_mask = label_mask & interior_mask
        if int(core_mask.sum()) < min_core_spots:
            core_mask = label_mask
        core_masks[int(label)] = core_mask
    return core_masks


def random_boundary_mask(
    valid_mask: np.ndarray,
    target_count: int,
    seed: int,
) -> np.ndarray:
    valid_indices = np.flatnonzero(valid_mask)
    if target_count > valid_indices.size:
        raise ValueError(
            "Cannot sample more random boundary spots than valid labeled spots: "
            f"{target_count} > {valid_indices.size}"
        )
    rng = np.random.default_rng(seed)
    selected = rng.choice(valid_indices, size=target_count, replace=False)
    mask = np.zeros(valid_mask.shape[0], dtype=bool)
    mask[selected] = True
    return mask


def build_oracle_boundary_data(
    adata: object,
    ground_truth_key: str,
    min_core_spots: int = 5,
) -> OracleBoundaryData:
    if ground_truth_key not in adata.obs:
        raise KeyError(f"Missing adata.obs['{ground_truth_key}']")
    if "Spatial_Net" not in adata.uns:
        raise KeyError("Missing adata.uns['Spatial_Net']")

    neighbors = build_neighbor_lists(adata.uns["Spatial_Net"], adata.obs_names)
    labels, label_values, valid_mask = encode_labels(adata.obs[ground_truth_key])
    gt_boundary_score = neighbor_disagreement_score(labels, neighbors, valid_mask)
    gt_boundary_mask = (gt_boundary_score > 0.0) & valid_mask
    gt_interior_mask = (gt_boundary_score == 0.0) & valid_mask
    core_masks = select_core_masks(
        labels,
        valid_mask,
        gt_interior_mask,
        min_core_spots=min_core_spots,
    )
    negatives = adjacent_negative_labels(labels, neighbors, valid_mask)
    return OracleBoundaryData(
        labels=labels,
        label_values=label_values,
        valid_mask=valid_mask,
        gt_boundary_score=gt_boundary_score,
        gt_boundary_mask=gt_boundary_mask,
        gt_interior_mask=gt_interior_mask,
        neighbors=neighbors,
        core_masks=core_masks,
        adjacent_negative_labels=negatives,
    )


def attach_boundary_obs(
    adata: object,
    boundary_data: OracleBoundaryData,
    ground_truth_key: str,
) -> None:
    adata.obs["gt_label"] = adata.obs[ground_truth_key].astype(object)
    adata.obs["gt_boundary_score"] = boundary_data.gt_boundary_score
    adata.obs["is_gt_boundary"] = boundary_data.gt_boundary_mask
    adata.obs["is_gt_interior"] = boundary_data.gt_interior_mask
