"""Analyze BAGR graph-refinement quality before retraining.

This script evaluates whether BAGR pruning preferentially removed predicted
risky edges and, when ground-truth labels are available, whether those pruned
edges are enriched for GT cross-domain edges. It is an analysis-only step and
must not be used to build the graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MISSING_LABELS = {"", "nan", "none", "null", "na", "n/a", "pd.na", "<na>"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze BAGR-pruned graph quality using baseline artifacts."
    )
    parser.add_argument(
        "--bagr-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more BAGR output directories containing refined_edges.csv.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "Matching Step 1 baseline directories. Provide one directory for all "
            "BAGR dirs, or the same number as --bagr-dir."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for a single BAGR dir. Defaults to --bagr-dir.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional CSV path for aggregate graph-refinement diagnosis.",
    )
    parser.add_argument(
        "--ground-truth-column",
        default="ground_truth",
        help="Ground-truth column in labels_gt.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing diagnosis outputs.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def normalize_label_series(series: pd.Series) -> pd.Series:
    values = series.astype("object").where(series.notna(), other=pd.NA)
    as_text = values.astype("string")
    missing = as_text.str.strip().str.lower().isin(MISSING_LABELS)
    return as_text.mask(missing, pd.NA)


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


def undirected_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["node_a", "node_b"])
    return add_edge_keys(edges)[["node_a", "node_b"]].drop_duplicates().reset_index(drop=True)


def load_edges(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing edge file: {path}")
    edges = pd.read_csv(path, dtype={"Cell1": str, "Cell2": str})
    missing = {"Cell1", "Cell2"}.difference(edges.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return edges.loc[edges["Cell1"] != edges["Cell2"]].copy().reset_index(drop=True)


def load_labels(baseline_dir: Path, ground_truth_column: str) -> pd.DataFrame:
    pred_path = baseline_dir / "pred_labels.csv"
    gt_path = baseline_dir / "labels_gt.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing pred_labels.csv: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing labels_gt.csv: {gt_path}")
    pred = pd.read_csv(pred_path, dtype={"spot_id": str})
    gt = pd.read_csv(gt_path, dtype={"spot_id": str})
    if "spot_id" not in pred or "pred_label" not in pred:
        raise ValueError(f"{pred_path} must contain spot_id and pred_label columns")
    if "spot_id" not in gt or ground_truth_column not in gt:
        raise ValueError(f"{gt_path} must contain spot_id and {ground_truth_column} columns")
    labels = pred[["spot_id", "pred_label"]].merge(
        gt[["spot_id", ground_truth_column]],
        on="spot_id",
        how="outer",
        validate="one_to_one",
    )
    labels = labels.rename(columns={ground_truth_column: "ground_truth"})
    labels["pred_label"] = normalize_label_series(labels["pred_label"])
    labels["ground_truth"] = normalize_label_series(labels["ground_truth"])
    return labels


def edge_label_diagnosis(
    edge_pairs: pd.DataFrame,
    labels: pd.DataFrame,
    label_column: str,
    prefix: str,
) -> dict[str, Any]:
    label_by_spot = labels.set_index("spot_id")[label_column].to_dict()
    known = 0
    cross = 0
    same = 0
    for node_a, node_b in edge_pairs[["node_a", "node_b"]].itertuples(index=False):
        label_a = label_by_spot.get(node_a)
        label_b = label_by_spot.get(node_b)
        if pd.isna(label_a) or pd.isna(label_b):
            continue
        known += 1
        if label_a != label_b:
            cross += 1
        else:
            same += 1
    return {
        f"{prefix}_known_undirected_edges": int(known),
        f"{prefix}_same_domain_undirected_edges": int(same),
        f"{prefix}_cross_domain_undirected_edges": int(cross),
        f"{prefix}_cross_domain_edge_ratio": float(cross / known) if known else None,
        f"{prefix}_edge_homophily": float(same / known) if known else None,
    }


def build_neighbors(edge_pairs: pd.DataFrame, spot_ids: pd.Series) -> dict[str, set[str]]:
    neighbors = {str(spot_id): set() for spot_id in spot_ids.astype(str)}
    for node_a, node_b in edge_pairs[["node_a", "node_b"]].itertuples(index=False):
        if node_a in neighbors:
            neighbors[node_a].add(node_b)
        if node_b in neighbors:
            neighbors[node_b].add(node_a)
    return neighbors


def gt_boundary_spots(edge_pairs: pd.DataFrame, labels: pd.DataFrame) -> set[str]:
    label_by_spot = labels.set_index("spot_id")["ground_truth"].to_dict()
    neighbors = build_neighbors(edge_pairs, labels["spot_id"])
    boundary: set[str] = set()
    for spot_id, spot_neighbors in neighbors.items():
        label = label_by_spot.get(spot_id)
        if pd.isna(label):
            continue
        for neighbor in spot_neighbors:
            neighbor_label = label_by_spot.get(neighbor)
            if pd.notna(neighbor_label) and neighbor_label != label:
                boundary.add(spot_id)
                break
    return boundary


def edge_touch_ratio(edge_pairs: pd.DataFrame, spots: set[str]) -> float | None:
    if edge_pairs.empty:
        return None
    touches = edge_pairs["node_a"].isin(spots) | edge_pairs["node_b"].isin(spots)
    return float(touches.mean())


def graph_stats(edge_pairs: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    degree = {str(spot_id): 0 for spot_id in labels["spot_id"].astype(str)}
    for node_a, node_b in edge_pairs[["node_a", "node_b"]].itertuples(index=False):
        if node_a in degree:
            degree[node_a] += 1
        if node_b in degree:
            degree[node_b] += 1
    values = np.asarray(list(degree.values()), dtype=float)
    return {
        "undirected_edge_count": int(edge_pairs.shape[0]),
        "mean_undirected_degree": float(values.mean()) if values.size else 0.0,
        "min_undirected_degree": int(values.min()) if values.size else 0,
        "max_undirected_degree": int(values.max()) if values.size else 0,
        "isolated_node_count": int((values == 0).sum()),
    }


def load_edge_scores(bagr_dir: Path) -> pd.DataFrame | None:
    path = bagr_dir / "edge_scores.csv"
    if not path.exists():
        return None
    scores = pd.read_csv(path, dtype={"node_a": str, "node_b": str})
    required = {"node_a", "node_b", "edge_risk"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return scores


def annotate_pruned_edges(
    pruned_pairs: pd.DataFrame,
    labels: pd.DataFrame,
    original_boundary: set[str],
    scores: pd.DataFrame | None,
) -> pd.DataFrame:
    label_by_spot = labels.set_index("spot_id").to_dict(orient="index")
    out = pruned_pairs.copy()
    out["gt_label_a"] = out["node_a"].map(lambda node: label_by_spot.get(node, {}).get("ground_truth"))
    out["gt_label_b"] = out["node_b"].map(lambda node: label_by_spot.get(node, {}).get("ground_truth"))
    out["pred_label_a"] = out["node_a"].map(lambda node: label_by_spot.get(node, {}).get("pred_label"))
    out["pred_label_b"] = out["node_b"].map(lambda node: label_by_spot.get(node, {}).get("pred_label"))
    out["gt_both_known"] = out["gt_label_a"].notna() & out["gt_label_b"].notna()
    out["pred_both_known"] = out["pred_label_a"].notna() & out["pred_label_b"].notna()
    out["is_gt_cross_domain"] = out["gt_both_known"] & (out["gt_label_a"] != out["gt_label_b"])
    out["is_pred_cross_domain"] = out["pred_both_known"] & (out["pred_label_a"] != out["pred_label_b"])
    out["touches_gt_boundary"] = out["node_a"].isin(original_boundary) | out["node_b"].isin(original_boundary)
    if scores is not None:
        score_cols = [
            col
            for col in [
                "edge_risk",
                "labels_discordant",
                "confidence_pass",
                "min_endpoint_confidence",
                "similarity",
                "expression_dissimilarity",
                "max_boundary_score",
                "pruned",
            ]
            if col in scores.columns
        ]
        out = out.merge(
            scores[["node_a", "node_b", *score_cols]],
            on=["node_a", "node_b"],
            how="left",
            validate="one_to_one",
        )
    return out


def analyze_one(
    bagr_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    outputs = [
        output_dir / "graph_diagnosis.json",
        output_dir / "pruned_edge_gt_diagnosis.csv",
    ]
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Diagnosis outputs already exist: "
                + ", ".join(str(path) for path in existing)
                + ". Use --overwrite to replace them."
            )

    labels = load_labels(baseline_dir, args.ground_truth_column)
    original_edges = load_edges(baseline_dir / "spatial_edges.csv")
    refined_edges = load_edges(bagr_dir / "refined_edges.csv")
    pruned_edges_path = bagr_dir / "pruned_edges.csv"
    if pruned_edges_path.exists():
        pruned_edges = load_edges(pruned_edges_path)
    else:
        original_keyed = add_edge_keys(original_edges)
        refined_set = set(
            undirected_edges(refined_edges)[["node_a", "node_b"]].itertuples(index=False, name=None)
        )
        original_tuples = original_keyed[["node_a", "node_b"]].apply(tuple, axis=1)
        pruned_edges = original_keyed.loc[~original_tuples.isin(refined_set)].copy()

    original_pairs = undirected_edges(original_edges)
    refined_pairs = undirected_edges(refined_edges)
    pruned_pairs = undirected_edges(pruned_edges)
    original_boundary = gt_boundary_spots(original_pairs, labels)
    scores = load_edge_scores(bagr_dir)
    pruned_annotated = annotate_pruned_edges(pruned_pairs, labels, original_boundary, scores)

    original_gt = edge_label_diagnosis(original_pairs, labels, "ground_truth", "original_gt")
    pruned_gt = edge_label_diagnosis(pruned_pairs, labels, "ground_truth", "pruned_gt")
    refined_gt = edge_label_diagnosis(refined_pairs, labels, "ground_truth", "refined_gt")
    original_pred = edge_label_diagnosis(original_pairs, labels, "pred_label", "original_pred")
    pruned_pred = edge_label_diagnosis(pruned_pairs, labels, "pred_label", "pruned_pred")
    refined_pred = edge_label_diagnosis(refined_pairs, labels, "pred_label", "refined_pred")

    original_cross = original_gt["original_gt_cross_domain_edge_ratio"]
    pruned_precision = pruned_gt["pruned_gt_cross_domain_edge_ratio"]
    enrichment = (
        float(pruned_precision / original_cross)
        if pruned_precision is not None and original_cross not in (None, 0)
        else None
    )
    original_stats = graph_stats(original_pairs, labels)
    refined_stats = graph_stats(refined_pairs, labels)
    bagr_stats = read_json(bagr_dir / "graph_refinement_stats.json")

    risk_stats: dict[str, Any] = {}
    if scores is not None:
        risk = scores["edge_risk"]
        if "pruned" in scores:
            pruned_scores = scores.loc[scores["pruned"].astype(bool)]
        else:
            pruned_scores = pd.DataFrame()
        risk_stats = {
            "eligible_undirected_edges_from_scores": int(scores["eligible"].astype(bool).sum()) if "eligible" in scores else None,
            "positive_risk_undirected_edges_from_scores": int((risk > 0).sum()),
            "mean_edge_risk_all": float(risk.mean()) if not risk.empty else None,
            "mean_edge_risk_pruned": float(pruned_scores["edge_risk"].mean()) if not pruned_scores.empty else None,
            "min_edge_risk_pruned": float(pruned_scores["edge_risk"].min()) if not pruned_scores.empty else None,
        }

    diagnosis: dict[str, Any] = {
        "sample_id": bagr_stats.get("sample_id") or read_json(baseline_dir / "metrics_global.json").get("sample_id"),
        "baseline_dir": str(baseline_dir),
        "bagr_dir": str(bagr_dir),
        "n_spots": int(labels.shape[0]),
        "original_graph": original_stats,
        "refined_graph": refined_stats,
        "mean_degree_delta": float(refined_stats["mean_undirected_degree"] - original_stats["mean_undirected_degree"]),
        "isolated_node_count_delta": int(refined_stats["isolated_node_count"] - original_stats["isolated_node_count"]),
        "actual_pruned_undirected_edges": int(pruned_pairs.shape[0]),
        "actual_pruned_directed_rows": int(pruned_edges.shape[0]),
        "pruned_edge_touching_gt_boundary_ratio": edge_touch_ratio(pruned_pairs, original_boundary),
        "gt_boundary_spot_count": int(len(original_boundary)),
        "original_cross_gt_edge_ratio": original_cross,
        "pruned_cross_gt_edge_ratio": pruned_precision,
        "refined_cross_gt_edge_ratio": refined_gt["refined_gt_cross_domain_edge_ratio"],
        "pruned_edge_precision": pruned_precision,
        "pruned_edge_enrichment": enrichment,
        "original_cross_pred_edge_ratio": original_pred["original_pred_cross_domain_edge_ratio"],
        "pruned_cross_pred_edge_ratio": pruned_pred["pruned_pred_cross_domain_edge_ratio"],
        "refined_cross_pred_edge_ratio": refined_pred["refined_pred_cross_domain_edge_ratio"],
        **original_gt,
        **pruned_gt,
        **refined_gt,
        **original_pred,
        **pruned_pred,
        **refined_pred,
        **risk_stats,
        "bagr_stats": bagr_stats,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pruned_annotated.to_csv(output_dir / "pruned_edge_gt_diagnosis.csv", index=False)
    write_json(output_dir / "graph_diagnosis.json", diagnosis)
    print(
        f"{diagnosis['sample_id'] or bagr_dir.name}: "
        f"pruned_edge_precision={pruned_precision}, "
        f"original_cross_gt={original_cross}, "
        f"enrichment={enrichment}, "
        f"isolated_delta={diagnosis['isolated_node_count_delta']}"
    )
    return diagnosis


def flatten_summary(diagnosis: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_id",
        "baseline_dir",
        "bagr_dir",
        "actual_pruned_undirected_edges",
        "actual_pruned_directed_rows",
        "original_cross_gt_edge_ratio",
        "pruned_edge_precision",
        "pruned_edge_enrichment",
        "refined_cross_gt_edge_ratio",
        "pruned_edge_touching_gt_boundary_ratio",
        "original_cross_pred_edge_ratio",
        "pruned_cross_pred_edge_ratio",
        "refined_cross_pred_edge_ratio",
        "mean_degree_delta",
        "isolated_node_count_delta",
        "eligible_undirected_edges_from_scores",
        "positive_risk_undirected_edges_from_scores",
        "mean_edge_risk_pruned",
        "min_edge_risk_pruned",
    ]
    return {key: diagnosis.get(key) for key in keys}


def pair_dirs(bagr_dirs: list[Path], baseline_dirs: list[Path]) -> list[tuple[Path, Path]]:
    if len(baseline_dirs) == 1:
        return [(bagr_dir, baseline_dirs[0]) for bagr_dir in bagr_dirs]
    if len(bagr_dirs) != len(baseline_dirs):
        raise ValueError(
            "Provide either one --baseline-dir for all --bagr-dir values, or matching counts."
        )
    return list(zip(bagr_dirs, baseline_dirs))


def main() -> None:
    args = parse_args()
    pairs = pair_dirs(args.bagr_dir, args.baseline_dir)
    if args.output_dir is not None and len(pairs) > 1:
        raise ValueError("--output-dir is only supported with one --bagr-dir")

    summaries = []
    for bagr_dir, baseline_dir in pairs:
        output_dir = args.output_dir if args.output_dir is not None else bagr_dir
        diagnosis = analyze_one(bagr_dir, baseline_dir, output_dir, args)
        summaries.append(flatten_summary(diagnosis))

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(args.summary_output, index=False)
        print(f"Summary saved to {args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
