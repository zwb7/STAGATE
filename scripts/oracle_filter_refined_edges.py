"""Oracle GT-label filtering for BAGR edge lists.

This script removes a user-specified fraction of remaining ground-truth
cross-domain edges from an already generated refined graph. It is intended for
oracle analysis, not for the formal label-free BAGR-STAGATE method.

It can also run an oracle swap analysis that edits refined_edges and
pruned_edges together: same-domain edges currently in pruned_edges are moved
back to refined_edges, and GT cross-domain edges currently in refined_edges are
moved into pruned_edges until a target pruned-edge precision is reached or no
more valid swaps are available. This keeps the number of pruned edges fixed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GT-label oracle filtering/swap analysis for BAGR edge lists."
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
        "--pruned-edges",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing the pruned edge list. When provided, the "
            "script swaps edges between refined_edges and pruned_edges to reach "
            "the requested --target-pruned-precision while keeping the pruned "
            "edge count fixed."
        ),
    )
    parser.add_argument(
        "--output-edges",
        default=None,
        type=Path,
        help=(
            "Output CSV path for the filtered/refined graph. In swap mode this "
            "is the output refined edge list unless --output-refined-edges is "
            "provided."
        ),
    )
    parser.add_argument(
        "--output-refined-edges",
        type=Path,
        default=None,
        help="Output CSV path for the refined edge list in swap mode.",
    )
    parser.add_argument(
        "--output-pruned-edges",
        type=Path,
        default=None,
        help="Output CSV path for the pruned edge list in swap mode.",
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
        default=None,
        help="Fraction of remaining GT cross-domain edges to remove, in [0, 1].",
    )
    parser.add_argument(
        "--target-pruned-precision",
        type=float,
        default=None,
        help=(
            "Target fraction of pruned edges that should be GT cross-domain "
            "edges in swap mode, e.g. 0.80."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling oracle edges.",
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


def validate_precision(precision: float) -> None:
    if not 0.0 <= precision <= 1.0:
        raise ValueError(
            f"--target-pruned-precision must be in [0, 1], got {precision}"
        )


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


def prepare_edges(
    edges: pd.DataFrame,
    source_col: str,
    target_col: str,
    path: Path,
    directed: bool,
) -> pd.DataFrame:
    require_columns(edges, [source_col, target_col], path)
    prepared = edges.copy()
    if not directed:
        prepared = normalize_undirected_edges(prepared, source_col, target_col)
    prepared[source_col] = prepared[source_col].astype(str)
    prepared[target_col] = prepared[target_col].astype(str)
    return prepared


def add_gt_labels(
    edges: pd.DataFrame,
    labels_gt: pd.DataFrame,
    source_col: str,
    target_col: str,
    id_col: str,
    label_col: str,
    path: Path,
) -> pd.DataFrame:
    require_columns(labels_gt, [id_col, label_col], Path("labels_gt"))
    labelled = edges.copy()
    label_map = dict(zip(labels_gt[id_col].astype(str), labels_gt[label_col]))
    labelled["_src_gt_label"] = labelled[source_col].map(label_map)
    labelled["_dst_gt_label"] = labelled[target_col].map(label_map)

    missing_label_mask = (
        labelled["_src_gt_label"].isna() | labelled["_dst_gt_label"].isna()
    )
    missing_label_edges = int(missing_label_mask.sum())
    if missing_label_edges:
        missing_nodes = sorted(
            set(labelled.loc[labelled["_src_gt_label"].isna(), source_col])
            | set(labelled.loc[labelled["_dst_gt_label"].isna(), target_col])
        )
        preview = missing_nodes[:10]
        raise ValueError(
            f"{path} has {missing_label_edges} edges containing nodes without "
            f"GT labels. First missing node ids: {preview}"
        )
    return labelled


def drop_internal_columns(edges: pd.DataFrame) -> pd.DataFrame:
    internal_cols = [col for col in edges.columns if col.startswith("_")]
    return edges.drop(columns=internal_cols)


def align_columns_for_concat(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(left.columns)
    for col in right.columns:
        if col not in columns:
            columns.append(col)
    return left.reindex(columns=columns), right.reindex(columns=columns)


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
    edges = prepare_edges(
        refined_edges, source_col, target_col, Path("refined_edges"), directed
    )
    edges = add_gt_labels(
        edges, labels_gt, source_col, target_col, id_col, label_col, Path("refined_edges")
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

    return drop_internal_columns(filtered), stats


def oracle_swap_edges(
    refined_edges: pd.DataFrame,
    pruned_edges: pd.DataFrame,
    labels_gt: pd.DataFrame,
    target_pruned_precision: float,
    seed: int,
    source_col: str,
    target_col: str,
    id_col: str,
    label_col: str,
    directed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float | bool]]:
    validate_precision(target_pruned_precision)

    refined = prepare_edges(
        refined_edges, source_col, target_col, Path("refined_edges"), directed
    )
    pruned = prepare_edges(
        pruned_edges, source_col, target_col, Path("pruned_edges"), directed
    )
    refined = add_gt_labels(
        refined, labels_gt, source_col, target_col, id_col, label_col, Path("refined_edges")
    )
    pruned = add_gt_labels(
        pruned, labels_gt, source_col, target_col, id_col, label_col, Path("pruned_edges")
    )

    pruned_cross_mask = pruned["_src_gt_label"] != pruned["_dst_gt_label"]
    refined_cross_mask = refined["_src_gt_label"] != refined["_dst_gt_label"]
    pruned_same_indices = pruned.index[~pruned_cross_mask].to_numpy()
    refined_cross_indices = refined.index[refined_cross_mask].to_numpy()

    n_pruned = int(len(pruned))
    current_tp = int(pruned_cross_mask.sum())
    target_tp = int(math.ceil(target_pruned_precision * n_pruned))
    needed_swaps = max(0, target_tp - current_tp)
    feasible_swaps = min(
        needed_swaps,
        int(len(pruned_same_indices)),
        int(len(refined_cross_indices)),
    )

    if feasible_swaps:
        rng = np.random.default_rng(seed)
        restore_indices = rng.choice(
            pruned_same_indices, size=feasible_swaps, replace=False
        )
        promote_indices = rng.choice(
            refined_cross_indices, size=feasible_swaps, replace=False
        )
    else:
        restore_indices = np.array([], dtype=pruned.index.dtype)
        promote_indices = np.array([], dtype=refined.index.dtype)

    restored_edges = pruned.loc[restore_indices].copy()
    promoted_edges = refined.loc[promote_indices].copy()
    pruned_remaining = pruned.drop(index=restore_indices)
    refined_remaining = refined.drop(index=promote_indices)

    pruned_remaining, promoted_edges = align_columns_for_concat(
        pruned_remaining, promoted_edges
    )
    refined_remaining, restored_edges = align_columns_for_concat(
        refined_remaining, restored_edges
    )
    new_pruned = pd.concat([pruned_remaining, promoted_edges], ignore_index=True)
    new_refined = pd.concat([refined_remaining, restored_edges], ignore_index=True)

    final_pruned_cross = int(
        (new_pruned["_src_gt_label"] != new_pruned["_dst_gt_label"]).sum()
    )
    final_refined_cross = int(
        (new_refined["_src_gt_label"] != new_refined["_dst_gt_label"]).sum()
    )
    final_precision = (
        float(final_pruned_cross / len(new_pruned)) if len(new_pruned) else 0.0
    )

    stats: dict[str, int | float | bool] = {
        "input_refined_edges": int(len(refined_edges)),
        "input_pruned_edges": int(len(pruned_edges)),
        "normalized_refined_edges": int(len(refined)),
        "normalized_pruned_edges": int(len(pruned)),
        "current_pruned_gt_cross_edges": current_tp,
        "current_pruned_precision": float(current_tp / n_pruned) if n_pruned else 0.0,
        "target_pruned_precision": float(target_pruned_precision),
        "target_pruned_gt_cross_edges": target_tp,
        "needed_swaps": int(needed_swaps),
        "performed_swaps": int(feasible_swaps),
        "available_same_domain_pruned_edges": int(len(pruned_same_indices)),
        "available_gt_cross_refined_edges": int(len(refined_cross_indices)),
        "final_pruned_gt_cross_edges": final_pruned_cross,
        "final_pruned_precision": final_precision,
        "final_refined_gt_cross_edges": final_refined_cross,
        "achieved_target": bool(final_precision >= target_pruned_precision),
        "output_refined_edges": int(len(new_refined)),
        "output_pruned_edges": int(len(new_pruned)),
        "seed": int(seed),
        "directed": bool(directed),
    }

    return drop_internal_columns(new_refined), drop_internal_columns(new_pruned), stats


def main() -> None:
    args = parse_args()

    refined_edges = pd.read_csv(args.refined_edges)
    labels_gt = pd.read_csv(args.labels_gt)

    if args.pruned_edges is not None:
        if args.target_pruned_precision is None:
            raise ValueError(
                "--target-pruned-precision is required when --pruned-edges is used"
            )
        output_refined_edges = args.output_refined_edges or args.output_edges
        if output_refined_edges is None:
            raise ValueError(
                "--output-edges or --output-refined-edges is required in swap mode"
            )
        if args.output_pruned_edges is None:
            raise ValueError("--output-pruned-edges is required in swap mode")

        pruned_edges = pd.read_csv(args.pruned_edges)
        refined_out, pruned_out, stats = oracle_swap_edges(
            refined_edges=refined_edges,
            pruned_edges=pruned_edges,
            labels_gt=labels_gt,
            target_pruned_precision=args.target_pruned_precision,
            seed=args.seed,
            source_col=args.source_col,
            target_col=args.target_col,
            id_col=args.id_col,
            label_col=args.label_col,
            directed=args.directed,
        )
        output_refined_edges.parent.mkdir(parents=True, exist_ok=True)
        args.output_pruned_edges.parent.mkdir(parents=True, exist_ok=True)
        refined_out.to_csv(output_refined_edges, index=False)
        pruned_out.to_csv(args.output_pruned_edges, index=False)
    else:
        if args.drop_ratio is None:
            raise ValueError("--drop-ratio is required unless --pruned-edges is used")
        if args.output_edges is None:
            raise ValueError("--output-edges is required unless swap outputs are used")
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
