"""Oracle GT-label filtering for a BAGR refined graph.

This script removes a user-specified fraction of remaining ground-truth
cross-domain edges from an already generated refined graph. It is intended for
oracle analysis, not for the formal label-free BAGR-STAGATE method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove a fraction of GT cross-domain edges from a refined edge list."
        )
    )
    parser.add_argument(
        "--refined-edges",
        required=True,
        type=Path,
        help="CSV file containing the refined graph edge list.",
    )
    parser.add_argument(
        "--labels-gt",
        required=True,
        type=Path,
        help="CSV file containing ground-truth labels.",
    )
    parser.add_argument(
        "--output-edges",
        required=True,
        type=Path,
        help="Output CSV path for the filtered refined graph.",
    )
    parser.add_argument(
        "--output-stats",
        type=Path,
        default=None,
        help="Optional JSON path for filtering statistics.",
    )
    parser.add_argument(
        "--drop-ratio",
        type=float,
        required=True,
        help="Fraction of remaining GT cross-domain edges to remove, in [0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling cross-domain edges to remove.",
    )
    parser.add_argument(
        "--source-col",
        default="source",
        help="Source node column in the edge CSV.",
    )
    parser.add_argument(
        "--target-col",
        default="target",
        help="Target node column in the edge CSV.",
    )
    parser.add_argument(
        "--id-col",
        default="spot_id",
        help="Spot/node id column in the ground-truth label CSV.",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Ground-truth label column in the label CSV.",
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        help=(
            "Keep edges as directed rows. By default, edges are normalized as "
            "undirected pairs and duplicates are removed."
        ),
    )
    return parser.parse_args()


def validate_drop_ratio(drop_ratio: float) -> None:
    if not 0.0 <= drop_ratio <= 1.0:
        raise ValueError(f"--drop-ratio must be in [0, 1], got {drop_ratio}")


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def normalize_undirected_edges(
    edges: pd.DataFrame, source_col: str, target_col: str
) -> pd.DataFrame:
    edges = edges.copy()
    source = edges[source_col].astype(str).to_numpy()
    target = edges[target_col].astype(str).to_numpy()
    edges[source_col] = np.minimum(source, target)
    edges[target_col] = np.maximum(source, target)
    return edges.drop_duplicates([source_col, target_col]).reset_index(drop=True)


def filter_refined_edges(
    refined_edges: pd.DataFrame,
    labels_gt: pd.DataFrame,
    drop_ratio: float,
    seed: int,
    source_col: str,
    target_col: str,
    id_col: str,
    label_col: str,
    directed: bool,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    validate_drop_ratio(drop_ratio)
    require_columns(refined_edges, [source_col, target_col], Path("refined_edges"))
    require_columns(labels_gt, [id_col, label_col], Path("labels_gt"))

    edges = refined_edges.copy()
    if not directed:
        edges = normalize_undirected_edges(edges, source_col, target_col)

    edges[source_col] = edges[source_col].astype(str)
    edges[target_col] = edges[target_col].astype(str)
    label_map = dict(zip(labels_gt[id_col].astype(str), labels_gt[label_col]))

    edges["_src_gt_label"] = edges[source_col].map(label_map)
    edges["_dst_gt_label"] = edges[target_col].map(label_map)

    missing_label_mask = edges["_src_gt_label"].isna() | edges["_dst_gt_label"].isna()
    missing_label_edges = int(missing_label_mask.sum())
    if missing_label_edges:
        missing_nodes = sorted(
            set(edges.loc[edges["_src_gt_label"].isna(), source_col])
            | set(edges.loc[edges["_dst_gt_label"].isna(), target_col])
        )
        preview = missing_nodes[:10]
        raise ValueError(
            f"{missing_label_edges} edges contain nodes without GT labels. "
            f"First missing node ids: {preview}"
        )

    cross_mask = edges["_src_gt_label"] != edges["_dst_gt_label"]
    cross_indices = edges.index[cross_mask].to_numpy()
    n_cross_before = int(len(cross_indices))
    n_drop = int(round(n_cross_before * drop_ratio))

    if n_drop:
        rng = np.random.default_rng(seed)
        drop_indices = rng.choice(cross_indices, size=n_drop, replace=False)
    else:
        drop_indices = np.array([], dtype=edges.index.dtype)

    filtered = edges.drop(index=drop_indices).copy()
    n_cross_after = int(
        (filtered["_src_gt_label"] != filtered["_dst_gt_label"]).sum()
    )

    stats: dict[str, int | float] = {
        "input_edges": int(len(refined_edges)),
        "normalized_input_edges": int(len(edges)),
        "gt_cross_edges_before": n_cross_before,
        "drop_ratio": float(drop_ratio),
        "dropped_gt_cross_edges": int(n_drop),
        "output_edges": int(len(filtered)),
        "gt_cross_edges_after": n_cross_after,
        "edge_retention_rate": float(len(filtered) / len(edges)) if len(edges) else 0.0,
        "gt_cross_edge_retention_rate": (
            float(n_cross_after / n_cross_before) if n_cross_before else 0.0
        ),
        "seed": int(seed),
        "directed": bool(directed),
    }

    filtered = filtered.drop(columns=["_src_gt_label", "_dst_gt_label"])
    return filtered, stats


def main() -> None:
    args = parse_args()

    refined_edges = pd.read_csv(args.refined_edges)
    labels_gt = pd.read_csv(args.labels_gt)

    filtered_edges, stats = filter_refined_edges(
        refined_edges=refined_edges,
        labels_gt=labels_gt,
        drop_ratio=args.drop_ratio,
        seed=args.seed,
        source_col=args.source_col,
        target_col=args.target_col,
        id_col=args.id_col,
        label_col=args.label_col,
        directed=args.directed,
    )

    args.output_edges.parent.mkdir(parents=True, exist_ok=True)
    filtered_edges.to_csv(args.output_edges, index=False)

    if args.output_stats is not None:
        args.output_stats.parent.mkdir(parents=True, exist_ok=True)
        args.output_stats.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
