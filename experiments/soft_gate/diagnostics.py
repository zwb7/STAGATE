from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import roc_auc_score


def _edge_truth(
    adata: sc.AnnData,
    pairs: pd.DataFrame,
    ground_truth_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    labels = adata.obs[ground_truth_key]
    node_a = pairs["node_a_index"].to_numpy(dtype=int)
    node_b = pairs["node_b_index"].to_numpy(dtype=int)
    left = labels.iloc[node_a]
    right = labels.iloc[node_b]
    valid = left.notna().to_numpy() & right.notna().to_numpy()
    cross_domain = (
        left.astype(str).to_numpy() != right.astype(str).to_numpy()
    )
    return valid, cross_domain


def summarize_gates(
    pairs: pd.DataFrame,
    pair_gates: np.ndarray,
    effective_degree: np.ndarray,
) -> dict[str, float]:
    if pair_gates.shape[0] != pairs.shape[0]:
        raise ValueError(
            "pair_gates length does not match pairs: "
            f"{pair_gates.shape[0]} vs {pairs.shape[0]}"
        )
    return {
        "mean_gate": float(np.mean(pair_gates)),
        "std_gate": float(np.std(pair_gates)),
        "minimum_gate": float(np.min(pair_gates)),
        "maximum_gate": float(np.max(pair_gates)),
        "mean_effective_degree": float(np.mean(effective_degree)),
        "minimum_effective_degree": float(np.min(effective_degree)),
        "maximum_effective_degree": float(np.max(effective_degree)),
    }


def gate_diagnostics(
    adata: sc.AnnData,
    pairs: pd.DataFrame,
    pair_gates: np.ndarray,
    effective_degree: np.ndarray,
    *,
    ground_truth_key: str,
    low_gate_quantiles: tuple[float, ...] = (0.05, 0.10),
) -> dict[str, object]:
    report: dict[str, object] = {
        "gate_summary": summarize_gates(pairs, pair_gates, effective_degree),
        "low_gate": {},
    }
    if ground_truth_key not in adata.obs:
        report["has_ground_truth"] = False
        return report

    valid, cross_domain = _edge_truth(adata, pairs, ground_truth_key)
    if not bool(valid.any()):
        report["has_ground_truth"] = False
        return report

    valid_gates = pair_gates[valid]
    valid_cross = cross_domain[valid]
    original_cross_ratio = float(valid_cross.mean())
    report["has_ground_truth"] = True
    report["original_cross_domain_ratio"] = original_cross_ratio
    report["n_valid_edges"] = int(valid.sum())
    report["n_cross_domain_edges"] = int(valid_cross.sum())
    report["n_same_domain_edges"] = int((~valid_cross).sum())

    if np.unique(valid_cross).size == 2:
        report["gate_auc_for_cross_domain"] = float(
            roc_auc_score(valid_cross.astype(int), 1.0 - valid_gates)
        )
    else:
        report["gate_auc_for_cross_domain"] = None

    low_gate_report: dict[str, object] = {}
    for quantile in low_gate_quantiles:
        if not 0.0 < quantile < 1.0:
            raise ValueError("low_gate_quantiles must be in (0, 1)")
        cutoff = float(np.quantile(valid_gates, quantile))
        low = valid_gates <= cutoff
        low_cross = valid_cross[low]
        same = ~valid_cross
        cross_total = int(valid_cross.sum())
        same_total = int(same.sum())
        low_cross_count = int(low_cross.sum())
        low_same_count = int((~low_cross).sum())
        low_precision = float(low_cross.mean()) if low_cross.size else None
        enrichment = (
            float(low_precision / original_cross_ratio)
            if low_precision is not None and original_cross_ratio > 0
            else None
        )
        low_cdr = (
            float(low_cross_count / cross_total)
            if cross_total > 0
            else None
        )
        low_sdr = (
            float(low_same_count / same_total)
            if same_total > 0
            else None
        )
        low_gate_report[f"bottom_{int(round(quantile * 100))}pct"] = {
            "quantile": quantile,
            "gate_cutoff": cutoff,
            "n_low_gate_edges": int(low.sum()),
            "low_gate_precision": low_precision,
            "low_gate_enrichment": enrichment,
            "low_gate_cdr": low_cdr,
            "low_gate_sdr": low_sdr,
            "cdr_over_sdr": (
                float(low_cdr / low_sdr)
                if low_cdr is not None and low_sdr not in (None, 0.0)
                else None
            ),
            "same_domain_high_gate_retention": (
                float(1.0 - low_sdr) if low_sdr is not None else None
            ),
        }
    report["low_gate"] = low_gate_report
    return report
