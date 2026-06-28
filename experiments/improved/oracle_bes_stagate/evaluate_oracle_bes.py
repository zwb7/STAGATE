"""Evaluation helpers for Oracle-BES-STAGATE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def mclust_with_posterior(
    embedding: np.ndarray,
    num_cluster: int,
    model_names: str = "EEE",
    random_seed: int = 0,
) -> dict[str, np.ndarray]:
    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2:
        raise ValueError(f"embedding must be 2D, got shape {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("embedding contains NaN or infinite values")

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    from rpy2.robjects.vectors import FloatVector, IntVector, StrVector

    robjects.r.library("mclust")
    robjects.r["set.seed"](random_seed)
    r_embedding = robjects.r["matrix"](
        FloatVector(embedding.ravel(order="C")),
        nrow=embedding.shape[0],
        ncol=embedding.shape[1],
        byrow=True,
    )
    r_embedding = robjects.r["colnames<-"](
        r_embedding,
        StrVector([f"STAGATE_{index + 1}" for index in range(embedding.shape[1])]),
    )
    result = robjects.r["Mclust"](
        r_embedding,
        G=IntVector([num_cluster]),
        modelNames=StrVector([model_names]),
    )
    labels = np.asarray(list(result.rx2("classification")), dtype=int)
    posterior = np.asarray(result.rx2("z"), dtype=np.float64)
    return {
        "labels": labels,
        "posterior": posterior,
        "confidence": posterior.max(axis=1),
    }


def matched_prediction_errors(
    truth_labels: np.ndarray,
    predicted_labels: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    errors = np.zeros(predicted_labels.shape[0], dtype=bool)
    if not valid_mask.any():
        return errors

    truth_valid = truth_labels[valid_mask]
    predicted_valid = predicted_labels[valid_mask]
    truth_classes = np.unique(truth_valid)
    predicted_classes = np.unique(predicted_valid)
    truth_to_col = {label: index for index, label in enumerate(truth_classes)}
    predicted_to_row = {label: index for index, label in enumerate(predicted_classes)}
    contingency = np.zeros((len(predicted_classes), len(truth_classes)), dtype=np.int64)
    for truth_label, predicted_label in zip(truth_valid, predicted_valid):
        contingency[predicted_to_row[predicted_label], truth_to_col[truth_label]] += 1

    rows, cols = linear_sum_assignment(-contingency)
    predicted_to_truth = {
        predicted_classes[row]: truth_classes[col] for row, col in zip(rows, cols)
    }
    mapped = np.asarray(
        [predicted_to_truth.get(label, -1) for label in predicted_valid],
        dtype=np.int64,
    )
    errors[valid_mask] = mapped != truth_valid
    return errors


def _safe_subset_metrics(
    embedding: np.ndarray,
    truth_labels: np.ndarray,
    predicted_labels: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        f"{prefix}_ari": None,
        f"{prefix}_nmi": None,
        f"{prefix}_ami": None,
    }
    if int(mask.sum()) <= 1:
        return result
    result[f"{prefix}_ari"] = float(
        adjusted_rand_score(truth_labels[mask], predicted_labels[mask])
    )
    result[f"{prefix}_nmi"] = float(
        normalized_mutual_info_score(truth_labels[mask], predicted_labels[mask])
    )
    result[f"{prefix}_ami"] = float(
        adjusted_mutual_info_score(truth_labels[mask], predicted_labels[mask])
    )
    return result


def compute_metrics(
    embedding: np.ndarray,
    truth_labels: np.ndarray,
    predicted_labels: np.ndarray,
    valid_mask: np.ndarray,
    boundary_mask: np.ndarray,
    interior_mask: np.ndarray,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "ari": None,
        "nmi": None,
        "ami": None,
        "silhouette": None,
        "gt_boundary_error_rate": None,
        "gt_interior_error_rate": None,
        "boundary_interior_error_ratio": None,
    }
    if not valid_mask.any():
        return metrics
    metrics["ari"] = float(
        adjusted_rand_score(truth_labels[valid_mask], predicted_labels[valid_mask])
    )
    metrics["nmi"] = float(
        normalized_mutual_info_score(
            truth_labels[valid_mask],
            predicted_labels[valid_mask],
        )
    )
    metrics["ami"] = float(
        adjusted_mutual_info_score(truth_labels[valid_mask], predicted_labels[valid_mask])
    )
    n_predicted = np.unique(predicted_labels[valid_mask]).size
    if 1 < n_predicted < int(valid_mask.sum()):
        metrics["silhouette"] = float(
            silhouette_score(embedding[valid_mask], predicted_labels[valid_mask])
        )

    metrics.update(
        _safe_subset_metrics(
            embedding,
            truth_labels,
            predicted_labels,
            boundary_mask,
            "gt_boundary",
        )
    )
    metrics.update(
        _safe_subset_metrics(
            embedding,
            truth_labels,
            predicted_labels,
            interior_mask,
            "gt_interior",
        )
    )

    errors = matched_prediction_errors(truth_labels, predicted_labels, valid_mask)
    boundary_error = float(errors[boundary_mask].mean()) if boundary_mask.any() else None
    interior_error = float(errors[interior_mask].mean()) if interior_mask.any() else None
    metrics["gt_boundary_error_rate"] = boundary_error
    metrics["gt_interior_error_rate"] = interior_error
    metrics["boundary_interior_error_ratio"] = (
        float(boundary_error / interior_error)
        if boundary_error is not None and interior_error not in (None, 0.0)
        else None
    )
    return metrics


def perturbation_metrics(
    original_embedding: np.ndarray,
    refined_embedding: np.ndarray,
    boundary_mask: np.ndarray,
    interior_mask: np.ndarray,
) -> dict[str, float | None]:
    perturbation = np.linalg.norm(refined_embedding - original_embedding, axis=1)
    boundary_mean = float(perturbation[boundary_mask].mean()) if boundary_mask.any() else None
    interior_mean = float(perturbation[interior_mask].mean()) if interior_mask.any() else None
    return {
        "mean_l2_perturbation": float(perturbation.mean()),
        "boundary_mean_l2_perturbation": boundary_mean,
        "interior_mean_l2_perturbation": interior_mean,
        "boundary_interior_perturbation_ratio": (
            float(boundary_mean / interior_mean)
            if boundary_mean is not None and interior_mean not in (None, 0.0)
            else None
        ),
    }



def fixed_prototype_boundary_relabel(
    embedding: np.ndarray,
    baseline_labels: np.ndarray,
    boundary_mask: np.ndarray,
    interior_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Relabel only boundary spots by nearest fixed O0 interior cluster prototype."""
    relabeled = baseline_labels.copy()
    prototype_mask = interior_mask & valid_mask
    boundary_eval = boundary_mask & valid_mask
    prototype_labels = np.asarray(sorted(np.unique(baseline_labels[prototype_mask])))
    if prototype_labels.size == 0 or not boundary_eval.any():
        return relabeled

    prototypes = []
    for label in prototype_labels:
        members = prototype_mask & (baseline_labels == label)
        if members.any():
            prototypes.append(embedding[members].mean(axis=0))
    if not prototypes:
        return relabeled

    prototypes_array = np.vstack(prototypes)
    boundary_indices = np.flatnonzero(boundary_eval)
    distances = np.sum(
        (embedding[boundary_indices, None, :] - prototypes_array[None, :, :]) ** 2,
        axis=2,
    )
    relabeled[boundary_indices] = prototype_labels[np.argmin(distances, axis=1)]
    return relabeled

def label_change_metrics(
    baseline_labels: np.ndarray,
    refined_labels: np.ndarray,
    valid_mask: np.ndarray,
    boundary_mask: np.ndarray,
    interior_mask: np.ndarray,
) -> dict[str, float | None]:
    changed = baseline_labels != refined_labels
    return {
        "overall_changed_label_ratio": (
            float(changed[valid_mask].mean()) if valid_mask.any() else None
        ),
        "boundary_changed_label_ratio": (
            float(changed[boundary_mask].mean()) if boundary_mask.any() else None
        ),
        "interior_changed_label_ratio": (
            float(changed[interior_mask].mean()) if interior_mask.any() else None
        ),
    }


def correction_stats(
    truth_labels: np.ndarray,
    baseline_labels: np.ndarray,
    refined_labels: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float | int | None]:
    before_error = matched_prediction_errors(truth_labels, baseline_labels, valid_mask)
    after_error = matched_prediction_errors(truth_labels, refined_labels, valid_mask)
    changed = (baseline_labels != refined_labels) & valid_mask
    wrong_to_correct = int((before_error & ~after_error & valid_mask).sum())
    correct_to_wrong = int((~before_error & after_error & valid_mask).sum())
    wrong_to_wrong = int((before_error & after_error & valid_mask).sum())
    correct_to_correct = int((~before_error & ~after_error & valid_mask).sum())
    all_changed = int(changed.sum())
    return {
        "wrong_to_correct": wrong_to_correct,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_wrong": wrong_to_wrong,
        "correct_to_correct": correct_to_correct,
        "all_changed": all_changed,
        "correction_precision": (
            float(wrong_to_correct / all_changed) if all_changed else None
        ),
        "damage_rate": float(correct_to_wrong / all_changed) if all_changed else None,
        "net_correction": int(wrong_to_correct - correct_to_wrong),
    }


def summarize_runs_to_markdown(results_root: Path, output_path: Path) -> None:
    rows = []
    for metrics_path in sorted(results_root.glob("*/seed_*/*/metrics.csv")):
        rows.append(pd.read_csv(metrics_path).iloc[0].to_dict())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text(
            "# Oracle-BES-STAGATE summary\n\nNo metrics.csv files were found.\n",
            encoding="utf-8",
        )
        return

    table = pd.DataFrame(rows)
    lines: list[str] = ["# Oracle-BES-STAGATE summary", ""]
    selected = [
        column
        for column in [
            "sample_id",
            "experiment",
            "run_tag",
            "ari",
            "gt_boundary_ari",
            "gt_interior_ari",
            "nmi",
            "ami",
            "mean_l2_perturbation",
            "boundary_relabel_ari",
            "boundary_relabel_gt_boundary_ari",
            "boundary_relabel_gt_interior_ari",
        ]
        if column in table.columns
    ]
    lines.append("## Metrics")
    lines.append("")
    lines.append(dataframe_to_markdown(table[selected]))
    lines.append("")
    lines.append("Go / No-Go should be judged after the planned server runs finish.")
    output_path.write_text("\n".join(lines), encoding="utf-8")

def dataframe_to_markdown(table: pd.DataFrame) -> str:
    """Render a small markdown table without pandas' optional tabulate dependency."""
    if table.empty:
        return ""
    columns = [str(column) for column in table.columns]
    rows = []
    for _, row in table.iterrows():
        rows.append([_format_markdown_cell(row[column]) for column in table.columns])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _format_markdown_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")

def flatten_dict(payload: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            for child_key, child_value in flatten_dict(value).items():
                flat[f"{key}_{child_key}"] = child_value
        else:
            flat[key] = value
    return flat
