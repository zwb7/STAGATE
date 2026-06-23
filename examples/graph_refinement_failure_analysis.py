"""Post-hoc deleted-edge composition analysis for graph refinement.

Ground-truth labels are used only after training to explain whether a refined
graph removed same-domain or cross-domain edges. This script does not train a
model and does not modify the input artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze deleted-edge composition after graph refinement."
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        required=True,
        help="Baseline h5ad containing Spatial_Net and ground-truth labels.",
    )
    parser.add_argument(
        "--refined-graph",
        type=Path,
        required=True,
        help="Refined graph CSV with Cell1 and Cell2 columns.",
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Optional final clustering metrics.json.",
    )
    parser.add_argument(
        "--reference-metrics",
        type=Path,
        default=None,
        help="Optional matched Original-control metrics.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/failure_analysis"),
    )
    parser.add_argument(
        "--high-edge-retention",
        type=float,
        default=0.9,
        help="Threshold used only for the high-retention diagnostic flag.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_graph(graph: pd.DataFrame, name: str) -> None:
    missing = sorted({"Cell1", "Cell2"}.difference(graph.columns))
    if missing:
        raise ValueError(f"{name} graph is missing columns: {missing}")


def canonical_edges(graph: pd.DataFrame) -> pd.DataFrame:
    source = graph["Cell1"].astype(str).to_numpy()
    target = graph["Cell2"].astype(str).to_numpy()
    if np.any(source == target):
        raise ValueError("Graph unexpectedly contains self-edges")
    node_a = np.minimum(source, target)
    node_b = np.maximum(source, target)
    return (
        pd.DataFrame({"node_a": node_a, "node_b": node_b})
        .drop_duplicates()
        .sort_values(["node_a", "node_b"])
        .reset_index(drop=True)
    )


def edge_key_frame(edges: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(edges[["node_a", "node_b"]])


def connectivity_stats(
    edges: pd.DataFrame,
    obs_names: pd.Index,
) -> dict[str, int | float]:
    node_to_index = {
        str(node): index for index, node in enumerate(obs_names.astype(str))
    }
    source = edges["node_a"].map(node_to_index)
    target = edges["node_b"].map(node_to_index)
    if source.isna().any() or target.isna().any():
        raise ValueError("Graph contains nodes absent from adata.obs_names")

    n_nodes = len(obs_names)
    adjacency = sp.coo_matrix(
        (
            np.ones(edges.shape[0] * 2, dtype=np.uint8),
            (
                np.concatenate([source.to_numpy(), target.to_numpy()]),
                np.concatenate([target.to_numpy(), source.to_numpy()]),
            ),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    n_components, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    sizes = np.bincount(labels, minlength=n_components)
    largest = int(sizes.max()) if n_nodes else 0
    return {
        "isolated_node_count": int((degree == 0).sum()),
        "isolated_node_ratio": float((degree == 0).mean()) if n_nodes else 0.0,
        "connected_component_count": int(n_components),
        "largest_connected_component_ratio": (
            float(largest / n_nodes) if n_nodes else 0.0
        ),
    }


def read_metrics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_ari(metrics: dict[str, Any]) -> float | None:
    for key in ("rule_based_ari", "oracle_ari", "ari", "reencoding_ari"):
        if metrics.get(key) is not None:
            return float(metrics[key])
    return None


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def classify_edges(
    edges: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    label_map = labels.to_dict()
    result = edges.copy()
    result["label_a"] = result["node_a"].map(label_map)
    result["label_b"] = result["node_b"].map(label_map)
    known = result["label_a"].notna() & result["label_b"].notna()
    result["edge_type"] = "unknown-label"
    result.loc[
        known & (result["label_a"] == result["label_b"]),
        "edge_type",
    ] = "same-domain"
    result.loc[
        known & (result["label_a"] != result["label_b"]),
        "edge_type",
    ] = "cross-domain"
    return result


def build_findings(
    *,
    same_deletion_rate: float | None,
    cross_deletion_rate: float | None,
    edge_retention_rate: float,
    isolated_ratio_change: float,
    ari_delta: float | None,
    high_edge_retention: float,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    ari_improved = ari_delta is not None and ari_delta > 0
    ari_declined = ari_delta is not None and ari_delta < 0

    if (
        cross_deletion_rate is not None
        and same_deletion_rate is not None
        and cross_deletion_rate > same_deletion_rate
        and ari_improved
    ):
        findings.append(
            {
                "code": "effective_refinement",
                "interpretation": (
                    "Cross-domain deletion exceeds same-domain deletion and "
                    "ARI improves."
                ),
            }
        )
    if (
        cross_deletion_rate is not None
        and same_deletion_rate is not None
        and same_deletion_rate > cross_deletion_rate
        and ari_declined
    ):
        findings.append(
            {
                "code": "same_domain_overdeletion",
                "interpretation": (
                    "Same-domain deletion exceeds cross-domain deletion while "
                    "ARI declines; the scorer direction is likely incorrect."
                ),
            }
        )
    if edge_retention_rate >= high_edge_retention and ari_declined:
        findings.append(
            {
                "code": "high_retention_but_ari_declines",
                "interpretation": (
                    "Few edges were removed, but ARI declined; deleted edges "
                    "may be locally important or bridge critical structure."
                ),
            }
        )
    if isolated_ratio_change > 0 and ari_declined:
        findings.append(
            {
                "code": "graph_fragmentation",
                "interpretation": (
                    "The isolated-node ratio increased together with an ARI "
                    "decline."
                ),
            }
        )
    if not findings:
        findings.append(
            {
                "code": "inconclusive",
                "interpretation": (
                    "The observed metric combination does not uniquely identify "
                    "one failure mode."
                ),
            }
        )
    return findings


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 <= args.high_edge_retention <= 1.0:
        raise ValueError("--high-edge-retention must be in [0, 1]")
    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Input h5ad not found: {args.input_h5ad}")
    if not args.refined_graph.exists():
        raise FileNotFoundError(
            f"Refined graph not found: {args.refined_graph}"
        )

    output_dir = args.output_dir / args.method / args.sample_id
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input_h5ad)
    if args.ground_truth_key not in adata.obs:
        raise KeyError(
            f"Ground-truth key not found: {args.ground_truth_key}"
        )
    if "Spatial_Net" not in adata.uns:
        raise KeyError("Original Spatial_Net not found in input h5ad")

    original_raw = adata.uns["Spatial_Net"]
    refined_raw = pd.read_csv(args.refined_graph)
    validate_graph(original_raw, "Original")
    validate_graph(refined_raw, "Refined")
    original = canonical_edges(original_raw)
    refined = canonical_edges(refined_raw)

    original_index = edge_key_frame(original)
    refined_index = edge_key_frame(refined)
    added_index = refined_index.difference(original_index)
    if len(added_index):
        raise ValueError(
            f"Refined graph contains {len(added_index)} edges absent from "
            "the original graph; E1.5 currently analyzes pruning only."
        )

    retained_mask = original_index.isin(refined_index)
    composition = classify_edges(
        original,
        adata.obs[args.ground_truth_key],
    )
    composition["retained"] = retained_mask
    composition["deleted"] = ~retained_mask
    composition.to_csv(
        output_dir / "deleted_edge_composition.csv",
        index=False,
    )

    summary_rows = []
    for edge_type in ("same-domain", "cross-domain", "unknown-label"):
        subset = composition["edge_type"] == edge_type
        original_count = int(subset.sum())
        retained_count = int((subset & composition["retained"]).sum())
        deleted_count = original_count - retained_count
        summary_rows.append(
            {
                "edge_type": edge_type,
                "original_edge_count": original_count,
                "retained_edge_count": retained_count,
                "deleted_edge_count": deleted_count,
                "deletion_rate": safe_ratio(deleted_count, original_count),
                "retention_rate": safe_ratio(retained_count, original_count),
            }
        )
    summary_table = pd.DataFrame.from_records(summary_rows)
    summary_table.to_csv(output_dir / "edge_type_summary.csv", index=False)

    same_row = summary_table.set_index("edge_type").loc["same-domain"]
    cross_row = summary_table.set_index("edge_type").loc["cross-domain"]
    total_deleted = int(composition["deleted"].sum())
    deleted_cross = int(
        (
            composition["deleted"]
            & (composition["edge_type"] == "cross-domain")
        ).sum()
    )
    deleted_precision = safe_ratio(deleted_cross, total_deleted)

    same_domain = composition[
        composition["edge_type"] == "same-domain"
    ].copy()
    per_domain_rows = []
    for domain, group in same_domain.groupby("label_a", observed=True):
        original_count = int(group.shape[0])
        retained_count = int(group["retained"].sum())
        per_domain_rows.append(
            {
                "domain": str(domain),
                "original_same_domain_edge_count": original_count,
                "retained_same_domain_edge_count": retained_count,
                "deleted_same_domain_edge_count": (
                    original_count - retained_count
                ),
                "same_domain_edge_retention_rate": safe_ratio(
                    retained_count,
                    original_count,
                ),
            }
        )
    per_domain = pd.DataFrame.from_records(per_domain_rows).sort_values(
        "domain"
    )
    per_domain.to_csv(output_dir / "per_domain_retention.csv", index=False)

    original_graph_stats = connectivity_stats(original, adata.obs_names)
    refined_graph_stats = connectivity_stats(refined, adata.obs_names)
    metrics = read_metrics(args.metrics)
    reference_metrics = read_metrics(args.reference_metrics)
    final_ari = extract_ari(metrics)
    reference_ari = extract_ari(reference_metrics)
    ari_delta = (
        float(final_ari - reference_ari)
        if final_ari is not None and reference_ari is not None
        else None
    )
    edge_retention_rate = float(refined.shape[0] / original.shape[0])
    isolated_ratio_change = float(
        refined_graph_stats["isolated_node_ratio"]
        - original_graph_stats["isolated_node_ratio"]
    )

    result: dict[str, Any] = {
        "sample_id": args.sample_id,
        "method": args.method,
        "analysis_only_uses_ground_truth": True,
        "input_h5ad": str(args.input_h5ad),
        "refined_graph": str(args.refined_graph),
        "ground_truth_key": args.ground_truth_key,
        "original_undirected_edge_count": int(original.shape[0]),
        "refined_undirected_edge_count": int(refined.shape[0]),
        "deleted_undirected_edge_count": total_deleted,
        "edge_retention_rate": edge_retention_rate,
        "same_domain_edge_deletion_rate": same_row["deletion_rate"],
        "cross_domain_edge_deletion_rate": cross_row["deletion_rate"],
        "deleted_edge_precision": deleted_precision,
        "same_domain_edge_retention_rate": same_row["retention_rate"],
        "unknown_label_deleted_edge_count": int(
            summary_table.set_index("edge_type")
            .loc["unknown-label", "deleted_edge_count"]
        ),
        "original_graph": original_graph_stats,
        "refined_graph": refined_graph_stats,
        "isolated_node_ratio_change": isolated_ratio_change,
        "final_ari": final_ari,
        "reference_ari": reference_ari,
        "ari_delta_vs_reference": ari_delta,
        "findings": build_findings(
            same_deletion_rate=same_row["deletion_rate"],
            cross_deletion_rate=cross_row["deletion_rate"],
            edge_retention_rate=edge_retention_rate,
            isolated_ratio_change=isolated_ratio_change,
            ari_delta=ari_delta,
            high_edge_retention=args.high_edge_retention,
        ),
    }
    with (output_dir / "failure_analysis.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    result = analyze(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
