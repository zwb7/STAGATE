"""Build oracle GT cross-domain pruning graphs for diagnostic upper bounds.

This script removes the highest-risk ground-truth cross-domain spatial edges at
several fractions, for example 25%, 50%, 75%, and 100%. It is an oracle
diagnostic: ground-truth labels are used only to estimate an upper bound and
must not be presented as the deployable BAGR-STAGATE method.

The output ``refined_edges.csv`` files can be passed to
``experiments/bagr/run_stagate_refined.py`` for remote-server retraining.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MISSING_LABELS = {"", "nan", "none", "null", "na", "n/a", "pd.na", "<na>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build oracle refined graphs by pruning top-risk GT cross-domain edges."
        )
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help=(
            "Baseline directory containing labels_gt.csv and spatial_edges.csv. "
            "The spatial_edges.csv is the graph to prune."
        ),
    )
    parser.add_argument(
        "--edge-scores",
        type=Path,
        required=True,
        help=(
            "BAGR edge_scores.csv containing node_a, node_b, and edge_risk. "
            "GT labels are not read from this file."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Directory for oracle graph outputs. Defaults to "
            "<baseline-dir>/oracle_gt_cross_domain_risk_prune."
        ),
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help=(
            "Fractions of GT cross-domain undirected edges to prune. Values may "
            "be proportions (0.25) or percentages (25)."
        ),
    )
    parser.add_argument(
        "--gt-label-column",
        default="ground_truth",
        help="Ground-truth label column in labels_gt.csv.",
    )
    parser.add_argument(
        "--edge-risk-column",
        default="edge_risk",
        help="Risk score column in edge_scores.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into non-empty output directories.",
    )
    return parser.parse_args()


def normalize_fraction(value: float) -> float:
    if value <= 0:
        raise ValueError("Pruning fractions must be positive")
    if value > 1:
        if value > 100:
            raise ValueError("Pruning fractions cannot exceed 100 percent")
        value = value / 100.0
    if value > 1:
        raise ValueError("Pruning fractions must be <= 1.0 or <= 100 percent")
    return float(value)


def format_fraction(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def normalize_label_series(series: pd.Series) -> pd.Series:
    values = series.astype("object").where(series.notna(), other=pd.NA)
    as_text = values.astype("string")
    missing = as_text.str.strip().str.lower().isin(MISSING_LABELS)
    return as_text.mask(missing, pd.NA)


def load_gt_labels(
    baseline_dir: Path,
    gt_label_column: str,
) -> pd.DataFrame:
    path = baseline_dir / "labels_gt.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing ground-truth labels: {path}")
    labels = pd.read_csv(path, dtype={"spot_id": str})
    missing = {"spot_id", gt_label_column}.difference(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if labels["spot_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated spot_id values")
    labels = labels[["spot_id", gt_label_column]].rename(
        columns={gt_label_column: "ground_truth"}
    )
    labels["ground_truth"] = normalize_label_series(labels["ground_truth"])
    return labels


def load_spatial_edges(baseline_dir: Path, known_spots: set[str]) -> pd.DataFrame:
    path = baseline_dir / "spatial_edges.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing spatial edges: {path}")
    edges = pd.read_csv(path, dtype={"Cell1": str, "Cell2": str})
    missing = {"Cell1", "Cell2"}.difference(edges.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    edges = edges.loc[edges["Cell1"] != edges["Cell2"]].copy()
    valid = edges["Cell1"].isin(known_spots) & edges["Cell2"].isin(known_spots)
    dropped = int((~valid).sum())
    if dropped:
        print(f"Dropping {dropped} edges with endpoints absent from labels_gt.csv")
    return edges.loc[valid].reset_index(drop=True)


def undirected_key(source: str, target: str) -> tuple[str, str]:
    return (source, target) if source <= target else (target, source)


def add_edge_keys(edges: pd.DataFrame) -> pd.DataFrame:
    keyed = edges.copy()
    pairs = [
        undirected_key(str(source), str(target))
        for source, target in keyed[["Cell1", "Cell2"]].itertuples(index=False)
    ]
    keyed["node_a"] = [source for source, _ in pairs]
    keyed["node_b"] = [target for _, target in pairs]
    return keyed


def build_undirected_edges(edges: pd.DataFrame) -> pd.DataFrame:
    keyed = add_edge_keys(edges)
    return keyed[["node_a", "node_b"]].drop_duplicates().reset_index(drop=True)


def load_edge_scores(path: Path, edge_risk_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing edge scores: {path}")
    scores = pd.read_csv(path, dtype={"node_a": str, "node_b": str})
    missing = {"node_a", "node_b", edge_risk_column}.difference(scores.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    scores = scores[["node_a", "node_b", edge_risk_column]].rename(
        columns={edge_risk_column: "edge_risk"}
    )
    pairs = [
        undirected_key(str(source), str(target))
        for source, target in scores[["node_a", "node_b"]].itertuples(index=False)
    ]
    scores["node_a"] = [source for source, _ in pairs]
    scores["node_b"] = [target for _, target in pairs]
    scores["edge_risk"] = pd.to_numeric(scores["edge_risk"], errors="raise")
    if not np.isfinite(scores["edge_risk"]).all():
        raise ValueError(f"{path} contains non-finite edge risk values")
    if scores.duplicated(["node_a", "node_b"]).any():
        duplicates = scores.loc[
            scores.duplicated(["node_a", "node_b"], keep=False),
            ["node_a", "node_b"],
        ].head()
        raise ValueError(
            "edge_scores.csv contains duplicated undirected edges, for example: "
            f"{duplicates.to_dict(orient='records')}"
        )
    return scores


def annotate_gt_cross_domain_edges(
    edge_pairs: pd.DataFrame,
    gt_labels: pd.DataFrame,
    edge_scores: pd.DataFrame,
) -> pd.DataFrame:
    gt_by_spot = gt_labels.set_index("spot_id")["ground_truth"].to_dict()
    annotated = edge_pairs.copy()
    annotated["gt_label_a"] = annotated["node_a"].map(gt_by_spot)
    annotated["gt_label_b"] = annotated["node_b"].map(gt_by_spot)
    annotated["gt_label_a_missing"] = annotated["gt_label_a"].isna()
    annotated["gt_label_b_missing"] = annotated["gt_label_b"].isna()
    annotated["gt_cross_domain"] = (
        annotated["gt_label_a"].notna()
        & annotated["gt_label_b"].notna()
        & (annotated["gt_label_a"] != annotated["gt_label_b"])
    )
    annotated = annotated.merge(
        edge_scores,
        on=["node_a", "node_b"],
        how="left",
        validate="one_to_one",
    )
    missing_scores = annotated["edge_risk"].isna()
    if missing_scores.any():
        examples = annotated.loc[
            missing_scores,
            ["node_a", "node_b"],
        ].head()
        raise ValueError(
            "edge_scores.csv does not cover all spatial edges, for example: "
            f"{examples.to_dict(orient='records')}"
        )
    return annotated


def select_oracle_pruned_edges(
    annotated_edges: pd.DataFrame,
    fraction: float,
) -> tuple[set[tuple[str, str]], pd.DataFrame]:
    candidates = annotated_edges.loc[annotated_edges["gt_cross_domain"]].copy()
    target = int(math.ceil(fraction * candidates.shape[0])) if not candidates.empty else 0
    selected_rows = candidates.sort_values(
        ["edge_risk", "node_a", "node_b"],
        ascending=[False, True, True],
    ).head(target)
    selected = {
        (str(row.node_a), str(row.node_b))
        for row in selected_rows.itertuples(index=False)
    }
    out = annotated_edges.copy()
    selected_index = pd.MultiIndex.from_tuples(selected, names=["node_a", "node_b"])
    current_index = pd.MultiIndex.from_frame(out[["node_a", "node_b"]])
    out["oracle_pruned"] = current_index.isin(selected_index)
    return selected, out


def apply_pruning(
    directed_edges: pd.DataFrame,
    pruned: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keyed = add_edge_keys(directed_edges)
    pruned_index = pd.MultiIndex.from_tuples(pruned, names=["node_a", "node_b"])
    current_index = pd.MultiIndex.from_frame(keyed[["node_a", "node_b"]])
    remove_mask = current_index.isin(pruned_index)
    pruned_edges = directed_edges.loc[remove_mask].copy().reset_index(drop=True)
    refined_edges = directed_edges.loc[~remove_mask].copy().reset_index(drop=True)
    return refined_edges, pruned_edges


def graph_stats(directed_edges: pd.DataFrame, spot_ids: pd.Series) -> dict[str, Any]:
    edge_pairs = build_undirected_edges(directed_edges)
    degree = {str(spot_id): 0 for spot_id in spot_ids.astype(str)}
    for node_a, node_b in edge_pairs[["node_a", "node_b"]].itertuples(index=False):
        degree[node_a] = degree.get(node_a, 0) + 1
        degree[node_b] = degree.get(node_b, 0) + 1
    degree_values = np.asarray(list(degree.values()), dtype=float)
    return {
        "directed_edge_count": int(directed_edges.shape[0]),
        "undirected_edge_count": int(edge_pairs.shape[0]),
        "mean_undirected_degree": float(degree_values.mean()) if degree_values.size else 0.0,
        "min_undirected_degree": int(degree_values.min()) if degree_values.size else 0,
        "max_undirected_degree": int(degree_values.max()) if degree_values.size else 0,
        "isolated_node_count": int((degree_values == 0).sum()),
    }


def build_for_fraction(
    *,
    baseline_dir: Path,
    edge_scores_path: Path,
    output_root: Path,
    gt_labels: pd.DataFrame,
    directed_edges: pd.DataFrame,
    annotated_edges: pd.DataFrame,
    fraction: float,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir = output_root / f"oracle_gt_cross_domain_risk_top_{format_fraction(fraction)}"
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected, scored_edges = select_oracle_pruned_edges(annotated_edges, fraction)
    refined_edges, pruned_directed_edges = apply_pruning(directed_edges, selected)

    scored_edges.to_csv(output_dir / "oracle_edge_scores.csv", index=False)
    refined_edges.to_csv(output_dir / "refined_edges.csv", index=False)
    pruned_directed_edges.to_csv(output_dir / "pruned_edges.csv", index=False)

    gt_cross = scored_edges.loc[scored_edges["gt_cross_domain"]]
    pruned_gt_cross = scored_edges.loc[scored_edges["oracle_pruned"]]
    report: dict[str, Any] = {
        "method": "oracle_gt_cross_domain_risk_pruning",
        "warning": (
            "Ground-truth labels were used to prune edges. This is an oracle "
            "upper-bound diagnostic, not a valid deployable method."
        ),
        "baseline_dir": str(baseline_dir),
        "edge_scores": str(edge_scores_path),
        "output_dir": str(output_dir),
        "fraction_of_gt_cross_domain_edges": float(fraction),
        "n_spots": int(gt_labels.shape[0]),
        "n_spots_with_gt": int(gt_labels["ground_truth"].notna().sum()),
        "original_graph": graph_stats(directed_edges, gt_labels["spot_id"]),
        "refined_graph": graph_stats(refined_edges, gt_labels["spot_id"]),
        "gt_cross_domain_undirected_edges": int(gt_cross.shape[0]),
        "target_pruned_undirected_edges": int(math.ceil(fraction * gt_cross.shape[0])),
        "actual_pruned_undirected_edges": int(len(selected)),
        "actual_pruned_directed_edge_rows": int(pruned_directed_edges.shape[0]),
        "mean_pruned_edge_risk": (
            float(pruned_gt_cross["edge_risk"].mean())
            if not pruned_gt_cross.empty
            else 0.0
        ),
        "min_pruned_edge_risk": (
            float(pruned_gt_cross["edge_risk"].min())
            if not pruned_gt_cross.empty
            else 0.0
        ),
        "max_pruned_edge_risk": (
            float(pruned_gt_cross["edge_risk"].max())
            if not pruned_gt_cross.empty
            else 0.0
        ),
    }
    write_json(output_dir / "graph_refinement_stats.json", report)
    print(
        f"Oracle GT prune {fraction:.2%}: saved {output_dir.resolve()} | "
        f"pruned {report['actual_pruned_undirected_edges']} undirected GT-cross edges"
    )
    return report


def build_oracle_graphs(args: argparse.Namespace) -> list[dict[str, Any]]:
    fractions = [normalize_fraction(value) for value in args.fractions]
    output_root = (
        args.output_root
        if args.output_root is not None
        else args.baseline_dir / "oracle_gt_cross_domain_risk_prune"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    gt_labels = load_gt_labels(args.baseline_dir, args.gt_label_column)
    known_spots = set(gt_labels["spot_id"].astype(str))
    directed_edges = load_spatial_edges(args.baseline_dir, known_spots)
    edge_pairs = build_undirected_edges(directed_edges)
    edge_scores = load_edge_scores(args.edge_scores, args.edge_risk_column)
    annotated_edges = annotate_gt_cross_domain_edges(edge_pairs, gt_labels, edge_scores)

    reports = [
        build_for_fraction(
            baseline_dir=args.baseline_dir,
            edge_scores_path=args.edge_scores,
            output_root=output_root,
            gt_labels=gt_labels,
            directed_edges=directed_edges,
            annotated_edges=annotated_edges,
            fraction=fraction,
            overwrite=args.overwrite,
        )
        for fraction in fractions
    ]
    write_json(
        output_root / "oracle_gt_cross_domain_risk_prune_summary.json",
        {
            "method": "oracle_gt_cross_domain_risk_pruning",
            "warning": (
                "Ground-truth labels were used to prune edges. Use only as an "
                "oracle diagnostic upper bound."
            ),
            "baseline_dir": str(args.baseline_dir),
            "edge_scores": str(args.edge_scores),
            "fractions": fractions,
            "runs": reports,
        },
    )
    return reports


def main() -> None:
    build_oracle_graphs(parse_args())


if __name__ == "__main__":
    main()
