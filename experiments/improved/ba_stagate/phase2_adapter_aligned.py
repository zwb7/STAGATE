"""Recommended Phase 2 entrypoint with label-aligned change metrics.

It reuses ``phase2_adapter.py`` for training and evaluation, but replaces the
changed-label perturbation summary with a Hungarian-aligned version. Use this
entrypoint for smoke tests and reported Phase 2 runs.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

import phase2_adapter as base


def align_labels_to_reference(reference: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    reference_classes = np.unique(reference)
    predicted_classes = np.unique(predicted)
    reference_to_col = {label: index for index, label in enumerate(reference_classes)}
    predicted_to_row = {
        label: index for index, label in enumerate(predicted_classes)
    }
    contingency = np.zeros(
        (len(predicted_classes), len(reference_classes)),
        dtype=np.int64,
    )
    for reference_label, predicted_label in zip(reference, predicted):
        contingency[
            predicted_to_row[predicted_label],
            reference_to_col[reference_label],
        ] += 1

    rows, cols = linear_sum_assignment(-contingency)
    predicted_to_reference = {
        predicted_classes[row]: reference_classes[col]
        for row, col in zip(rows, cols)
    }
    return np.asarray(
        [predicted_to_reference.get(label, -1) for label in predicted],
        dtype=reference.dtype,
    )


def perturbation_summary(
    original: np.ndarray,
    shaped: np.ndarray,
    gt_boundary: np.ndarray,
    pseudo_boundary: np.ndarray,
    baseline_labels: np.ndarray,
    ba_labels: np.ndarray,
    evaluated: np.ndarray,
) -> dict[str, float | None]:
    perturbation = np.linalg.norm(shaped - original, axis=1)
    gt_interior = (~gt_boundary) & evaluated
    pseudo_interior = (~pseudo_boundary) & evaluated
    aligned_ba_labels = align_labels_to_reference(baseline_labels, ba_labels)
    return {
        "mean_l2": float(perturbation.mean()),
        "gt_boundary_mean_l2": (
            float(perturbation[gt_boundary].mean()) if gt_boundary.any() else None
        ),
        "gt_interior_mean_l2": (
            float(perturbation[gt_interior].mean()) if gt_interior.any() else None
        ),
        "pseudo_boundary_mean_l2": (
            float(perturbation[pseudo_boundary].mean()) if pseudo_boundary.any() else None
        ),
        "pseudo_interior_mean_l2": (
            float(perturbation[pseudo_interior].mean()) if pseudo_interior.any() else None
        ),
        "interior_changed_label_ratio": (
            float(np.mean(baseline_labels[gt_interior] != aligned_ba_labels[gt_interior]))
            if gt_interior.any()
            else None
        ),
        "pseudo_interior_changed_label_ratio": (
            float(
                np.mean(
                    baseline_labels[pseudo_interior]
                    != aligned_ba_labels[pseudo_interior]
                )
            )
            if pseudo_interior.any()
            else None
        ),
    }


base.perturbation_summary = perturbation_summary


if __name__ == "__main__":
    base.main()
