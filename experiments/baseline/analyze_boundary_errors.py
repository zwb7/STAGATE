"""Diagnose boundary-localized errors in vanilla STAGATE outputs.

This script consumes the lightweight artifacts produced by
``run_stagate_baseline.py``. It does not train models and does not require the
full AnnData h5ad artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MISSING_LABELS = {"", "nan", "none", "null", "na", "n/a", "pd.na", "<na>"}


def comb2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values * (values - 1.0) / 2.0


def contingency_from_labels(truth: pd.Series, pred: pd.Series) -> np.ndarray:
    truth_codes, _ = pd.factorize(truth.astype(str), sort=True)
    pred_codes, _ = pd.factorize(pred.astype(str), sort=True)
    table = np.zeros((truth_codes.max() + 1, pred_codes.max() + 1), dtype=float)
    for truth_code, pred_code in zip(truth_codes, pred_codes):
        table[truth_code, pred_code] += 1.0
    return table


def adjusted_rand_score_local(truth: pd.Series, pred: pd.Series) -> float:
    table = contingency_from_labels(truth, pred)
    n = table.sum()
    if n < 2:
        return 0.0
    sum_comb = comb2(table).sum()
    row_comb = comb2(table.sum(axis=1)).sum()
    col_comb = comb2(table.sum(axis=0)).sum()
    total_comb = comb2(np.array([n]))[0]
    expected = row_comb * col_comb / total_comb if total_comb else 0.0
    maximum = 0.5 * (row_comb + col_comb)
    denominator = maximum - expected
    if denominator == 0:
        return 1.0 if sum_comb == maximum else 0.0
    return float((sum_comb - expected) / denominator)


def normalized_mutual_info_score_local(truth: pd.Series, pred: pd.Series) -> float:
    table = contingency_from_labels(truth, pred)
    n = table.sum()
    if n == 0:
        return 0.0
    pij = table / n
    pi = pij.sum(axis=1)
    pj = pij.sum(axis=0)
    nz = pij > 0
    expected = np.outer(pi, pj)
    mi = float((pij[nz] * np.log(pij[nz] / expected[nz])).sum())
    h_truth = float(-(pi[pi > 0] * np.log(pi[pi > 0])).sum())
    h_pred = float(-(pj[pj > 0] * np.log(pj[pj > 0])).sum())
    denom = (h_truth + h_pred) / 2.0
    return float(mi / denom) if denom else 1.0


def maximize_assignment(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column assignments maximizing the contingency total.

    Uses SciPy when available. The fallback is exact for the small cluster counts
    expected in DLPFC-style experiments and avoids adding a hard SciPy runtime
    dependency for this analysis-only script.
    """
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(-matrix)
    except Exception:
        pass

    if matrix.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    transposed = False
    work = np.asarray(matrix, dtype=float)
    if work.shape[0] > work.shape[1]:
        work = work.T
        transposed = True

    n_rows, n_cols = work.shape
    states: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for row in range(n_rows):
        next_states: dict[int, tuple[float, list[tuple[int, int]]]] = {}
        for mask, (score, pairs) in states.items():
            for col in range(n_cols):
                bit = 1 << col
                if mask & bit:
                    continue
                new_mask = mask | bit
                new_score = score + work[row, col]
                previous = next_states.get(new_mask)
                if previous is None or new_score > previous[0]:
                    next_states[new_mask] = (new_score, pairs + [(row, col)])
        states = next_states
    best_pairs = max(states.values(), key=lambda item: item[0])[1]
    rows = np.array([row for row, _ in best_pairs], dtype=int)
    cols = np.array([col for _, col in best_pairs], dtype=int)
    if transposed:
        return cols, rows
    return rows, cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GT-boundary and pred-boundary errors for STAGATE."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more Step 1 output directories containing pred_labels.csv, "
            "labels_gt.csv, spatial_edges.csv, and optionally metrics_global.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for a single --input-dir. Defaults to the input "
            "directory. For multiple input dirs, outputs are written beside each input."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional CSV path for an aggregate summary over all input dirs.",
    )
    parser.add_argument(
        "--boundary-edges-path",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional edge CSV(s) used only to define GT-boundary/interior masks. "
            "Defaults to each input directory's spatial_edges.csv. For refined "
            "runs, pass the matching baseline original spatial_edges.csv."
        ),
    )
    parser.add_argument(
        "--eval-edges-path",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Optional edge CSV(s) used only for graph-level metrics. Defaults to "
            "each input directory's spatial_edges.csv."
        ),
    )
    parser.add_argument(
        "--pred-label-column",
        default="pred_label",
        help="Predicted-label column in pred_labels.csv.",
    )
    parser.add_argument(
        "--ground-truth-column",
        default="ground_truth",
        help="Ground-truth column in labels_gt.csv.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Optional sample id override. Only valid for a single input dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing diagnosis outputs.",
    )
    return parser.parse_args()


def normalize_label_series(series: pd.Series) -> pd.Series:
    values = series.astype("object").where(series.notna(), other=pd.NA)
    as_text = values.astype("string")
    missing = as_text.str.strip().str.lower().isin(MISSING_LABELS)
    return as_text.mask(missing, pd.NA)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def infer_sample_id(input_dir: Path, metrics: dict[str, Any], override: str | None) -> str:
    if override is not None:
        return override
    sample_id = metrics.get("sample_id")
    if sample_id:
        return str(sample_id)
    if input_dir.name.startswith("seed_") and input_dir.parent.name:
        return input_dir.parent.name
    return input_dir.name


def load_labels(
    input_dir: Path,
    pred_label_column: str,
    ground_truth_column: str,
) -> pd.DataFrame:
    pred_path = input_dir / "pred_labels.csv"
    gt_path = input_dir / "labels_gt.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing ground-truth file: {gt_path}")

    pred = pd.read_csv(pred_path, dtype={"spot_id": str})
    gt = pd.read_csv(gt_path, dtype={"spot_id": str})
    for path, frame, column in [
        (pred_path, pred, pred_label_column),
        (gt_path, gt, ground_truth_column),
    ]:
        missing = {"spot_id", column}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["spot_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicated spot_id values")

    labels = pred[["spot_id", pred_label_column]].merge(
        gt[["spot_id", ground_truth_column]],
        on="spot_id",
        how="outer",
        validate="one_to_one",
    )
    labels = labels.rename(
        columns={pred_label_column: "pred_label", ground_truth_column: "ground_truth"}
    )
    labels["pred_label"] = normalize_label_series(labels["pred_label"])
    labels["ground_truth"] = normalize_label_series(labels["ground_truth"])
    return labels


def load_edges_from_path(edge_path: Path, known_spots: set[str]) -> pd.DataFrame:
    if not edge_path.exists():
        raise FileNotFoundError(f"Missing spatial edge file: {edge_path}")
    edges = pd.read_csv(edge_path, dtype={"Cell1": str, "Cell2": str})
    required = {"Cell1", "Cell2"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"{edge_path} is missing columns: {sorted(missing)}")
    edges = edges.loc[edges["Cell1"] != edges["Cell2"]].copy()
    source_known = edges["Cell1"].isin(known_spots)
    target_known = edges["Cell2"].isin(known_spots)
    return edges.loc[source_known & target_known].reset_index(drop=True)


def load_edges(input_dir: Path, known_spots: set[str]) -> pd.DataFrame:
    return load_edges_from_path(input_dir / "spatial_edges.csv", known_spots)


def undirected_edges(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(columns=["node_a", "node_b"])
    pairs = pd.DataFrame(
        {
            "node_a": np.minimum(edges["Cell1"].astype(str), edges["Cell2"].astype(str)),
            "node_b": np.maximum(edges["Cell1"].astype(str), edges["Cell2"].astype(str)),
        }
    )
    return pairs.drop_duplicates().reset_index(drop=True)


def build_neighbors(edge_pairs: pd.DataFrame, spot_ids: pd.Series) -> dict[str, set[str]]:
    neighbors = {str(spot_id): set() for spot_id in spot_ids.astype(str)}
    for node_a, node_b in edge_pairs[["node_a", "node_b"]].itertuples(index=False):
        if node_a in neighbors:
            neighbors[node_a].add(node_b)
        if node_b in neighbors:
            neighbors[node_b].add(node_a)
    return neighbors


def boundary_table(
    labels: pd.DataFrame,
    neighbors: dict[str, set[str]],
) -> pd.DataFrame:
    gt_by_spot = labels.set_index("spot_id")["ground_truth"].to_dict()
    pred_by_spot = labels.set_index("spot_id")["pred_label"].to_dict()
    rows: list[dict[str, Any]] = []
    for spot_id in labels["spot_id"].astype(str):
        spot_neighbors = sorted(neighbors.get(spot_id, set()))
        gt_label = gt_by_spot.get(spot_id)
        pred_label = pred_by_spot.get(spot_id)

        gt_neighbors = [
            neighbor
            for neighbor in spot_neighbors
            if pd.notna(gt_label) and pd.notna(gt_by_spot.get(neighbor))
        ]
        pred_neighbors = [
            neighbor
            for neighbor in spot_neighbors
            if pd.notna(pred_label) and pd.notna(pred_by_spot.get(neighbor))
        ]
        gt_cross = sum(gt_by_spot[neighbor] != gt_label for neighbor in gt_neighbors)
        pred_cross = sum(pred_by_spot[neighbor] != pred_label for neighbor in pred_neighbors)
        rows.append(
            {
                "spot_id": spot_id,
                "ground_truth": gt_label,
                "pred_label": pred_label,
                "n_neighbors": len(spot_neighbors),
                "n_gt_labeled_neighbors": len(gt_neighbors),
                "n_gt_cross_domain_neighbors": int(gt_cross),
                "gt_cross_domain_neighbor_ratio": (
                    float(gt_cross / len(gt_neighbors)) if gt_neighbors else np.nan
                ),
                "is_gt_boundary": bool(gt_cross > 0),
                "n_pred_labeled_neighbors": len(pred_neighbors),
                "n_pred_cross_domain_neighbors": int(pred_cross),
                "pred_cross_domain_neighbor_ratio": (
                    float(pred_cross / len(pred_neighbors)) if pred_neighbors else np.nan
                ),
                "is_pred_boundary": bool(pred_cross > 0),
            }
        )
    return pd.DataFrame(rows)


def ari_for_mask(labels: pd.DataFrame, mask: pd.Series) -> float | None:
    subset = labels.loc[mask & labels["ground_truth"].notna() & labels["pred_label"].notna()]
    if subset.shape[0] < 2:
        return None
    return float(
        adjusted_rand_score_local(
            subset["ground_truth"].astype(str),
            subset["pred_label"].astype(str),
        )
    )


def nmi_for_mask(labels: pd.DataFrame, mask: pd.Series) -> float | None:
    subset = labels.loc[mask & labels["ground_truth"].notna() & labels["pred_label"].notna()]
    if subset.shape[0] < 2:
        return None
    return float(
        normalized_mutual_info_score_local(
            subset["ground_truth"].astype(str),
            subset["pred_label"].astype(str),
        )
    )


def best_label_match_correctness(labels: pd.DataFrame) -> tuple[pd.Series, dict[str, str]]:
    valid = labels["ground_truth"].notna() & labels["pred_label"].notna()
    correctness = pd.Series(False, index=labels.index, dtype=bool)
    if not valid.any():
        return correctness, {}

    truth_valid = labels.loc[valid, "ground_truth"].astype(str)
    pred_valid = labels.loc[valid, "pred_label"].astype(str)
    truth_values = sorted(truth_valid.unique())
    pred_values = sorted(pred_valid.unique())
    truth_to_index = {value: index for index, value in enumerate(truth_values)}
    pred_to_index = {value: index for index, value in enumerate(pred_values)}
    contingency = np.zeros((len(pred_values), len(truth_values)), dtype=int)
    for pred_label, truth_label in zip(pred_valid, truth_valid):
        contingency[pred_to_index[pred_label], truth_to_index[truth_label]] += 1

    row_ind, col_ind = maximize_assignment(contingency)
    mapping = {
        pred_values[row]: truth_values[col]
        for row, col in zip(row_ind, col_ind)
        if contingency[row, col] > 0
    }
    correctness.loc[valid] = pred_valid.map(mapping) == truth_valid
    return correctness, mapping


def edge_homophily(
    edge_pairs: pd.DataFrame,
    labels: pd.DataFrame,
    label_column: str,
) -> dict[str, Any]:
    label_by_spot = labels.set_index("spot_id")[label_column].to_dict()
    known = []
    cross = []
    for node_a, node_b in edge_pairs[["node_a", "node_b"]].itertuples(index=False):
        label_a = label_by_spot.get(node_a)
        label_b = label_by_spot.get(node_b)
        both_known = pd.notna(label_a) and pd.notna(label_b)
        known.append(both_known)
        if both_known:
            cross.append(label_a != label_b)
    known_count = int(sum(known))
    cross_count = int(sum(cross))
    same_count = int(known_count - cross_count)
    return {
        "known_edge_count": known_count,
        "same_domain_edge_count": same_count,
        "cross_domain_edge_count": cross_count,
        "edge_homophily": float(same_count / known_count) if known_count else None,
        "cross_domain_edge_ratio": float(cross_count / known_count) if known_count else None,
    }


def rate(mask: pd.Series, denominator_mask: pd.Series) -> float | None:
    denom = int(denominator_mask.sum())
    if denom == 0:
        return None
    return float((mask & denominator_mask).sum() / denom)


def analyze_one(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    boundary_edges_path: Path | None = None,
    eval_edges_path: Path | None = None,
) -> dict[str, Any]:
    output_files = [
        output_dir / "boundary_mask.csv",
        output_dir / "interior_mask.csv",
        output_dir / "boundary_metrics.json",
        output_dir / "edge_homophily.json",
        output_dir / "spot_boundary_diagnostics.csv",
    ]
    if not args.overwrite:
        existing = [path for path in output_files if path.exists()]
        if existing:
            raise FileExistsError(
                "Diagnosis outputs already exist: "
                + ", ".join(str(path) for path in existing)
                + ". Use --overwrite to replace them."
            )

    metrics_global = read_json(input_dir / "metrics_global.json")
    sample_id = infer_sample_id(input_dir, metrics_global, args.sample_id)
    labels = load_labels(input_dir, args.pred_label_column, args.ground_truth_column)
    known_spots = set(labels["spot_id"].astype(str))
    boundary_edges_path = boundary_edges_path or input_dir / "spatial_edges.csv"
    eval_edges_path = eval_edges_path or input_dir / "spatial_edges.csv"
    boundary_edges = load_edges_from_path(boundary_edges_path, known_spots)
    eval_edges = load_edges_from_path(eval_edges_path, known_spots)
    boundary_edge_pairs = undirected_edges(boundary_edges)
    eval_edge_pairs = undirected_edges(eval_edges)
    neighbors = build_neighbors(boundary_edge_pairs, labels["spot_id"])
    boundary = boundary_table(labels, neighbors)

    correctness, mapping = best_label_match_correctness(labels)
    evaluated = labels["ground_truth"].notna() & labels["pred_label"].notna()
    labels = labels.copy()
    labels["is_evaluated"] = evaluated
    labels["is_correct"] = correctness.where(evaluated, other=pd.NA)
    labels["is_wrong"] = (~correctness).where(evaluated, other=False)

    diagnostics = boundary.merge(
        labels[["spot_id", "is_evaluated", "is_correct", "is_wrong"]],
        on="spot_id",
        how="left",
        validate="one_to_one",
    )
    gt_boundary = diagnostics["is_gt_boundary"] & diagnostics["is_evaluated"]
    gt_interior = (~diagnostics["is_gt_boundary"]) & diagnostics["is_evaluated"]
    wrong = diagnostics["is_wrong"].fillna(False).astype(bool) & diagnostics["is_evaluated"]
    correct = diagnostics["is_correct"].fillna(False).astype(bool) & diagnostics["is_evaluated"]

    boundary_error_rate = rate(wrong, gt_boundary)
    interior_error_rate = rate(wrong, gt_interior)
    enrichment = (
        float(boundary_error_rate / interior_error_rate)
        if boundary_error_rate is not None
        and interior_error_rate not in (None, 0)
        else None
    )

    wrong_neighbor_ratio = diagnostics.loc[
        wrong, "gt_cross_domain_neighbor_ratio"
    ].dropna()
    correct_neighbor_ratio = diagnostics.loc[
        correct, "gt_cross_domain_neighbor_ratio"
    ].dropna()

    gt_edge = edge_homophily(eval_edge_pairs, labels, "ground_truth")
    pred_edge = edge_homophily(eval_edge_pairs, labels, "pred_label")
    edge_report = {
        "sample_id": sample_id,
        "input_dir": str(input_dir),
        "boundary_edges_path": str(boundary_edges_path),
        "eval_edges_path": str(eval_edges_path),
        "n_boundary_undirected_edges": int(boundary_edge_pairs.shape[0]),
        "n_eval_undirected_edges": int(eval_edge_pairs.shape[0]),
        "ground_truth": gt_edge,
        "predicted": pred_edge,
    }

    boundary_metrics: dict[str, Any] = {
        "sample_id": sample_id,
        "input_dir": str(input_dir),
        "boundary_definition": "fixed_reference_graph" if boundary_edges_path != input_dir / "spatial_edges.csv" else "input_spatial_graph",
        "boundary_edges_path": str(boundary_edges_path),
        "eval_edges_path": str(eval_edges_path),
        "n_boundary_undirected_edges": int(boundary_edge_pairs.shape[0]),
        "n_eval_undirected_edges": int(eval_edge_pairs.shape[0]),
        "n_spots": int(labels.shape[0]),
        "n_evaluated_spots": int(evaluated.sum()),
        "global_ari": safe_float(metrics_global.get("ari")),
        "global_nmi": safe_float(metrics_global.get("nmi")),
        "computed_global_ari": ari_for_mask(labels, evaluated),
        "computed_global_nmi": nmi_for_mask(labels, evaluated),
        "gt_boundary_spot_count": int(gt_boundary.sum()),
        "gt_interior_spot_count": int(gt_interior.sum()),
        "pred_boundary_spot_count": int(
            (diagnostics["is_pred_boundary"] & diagnostics["is_evaluated"]).sum()
        ),
        "boundary_ari": ari_for_mask(labels, gt_boundary),
        "interior_ari": ari_for_mask(labels, gt_interior),
        "boundary_nmi": nmi_for_mask(labels, gt_boundary),
        "interior_nmi": nmi_for_mask(labels, gt_interior),
        "boundary_error_count": int((wrong & gt_boundary).sum()),
        "interior_error_count": int((wrong & gt_interior).sum()),
        "boundary_error_rate": boundary_error_rate,
        "interior_error_rate": interior_error_rate,
        "boundary_enrichment_ratio": enrichment,
        "wrong_spot_mean_gt_cross_domain_neighbor_ratio": (
            float(wrong_neighbor_ratio.mean()) if not wrong_neighbor_ratio.empty else None
        ),
        "correct_spot_mean_gt_cross_domain_neighbor_ratio": (
            float(correct_neighbor_ratio.mean())
            if not correct_neighbor_ratio.empty
            else None
        ),
        "wrong_spot_count_with_gt_neighbors": int(wrong_neighbor_ratio.shape[0]),
        "correct_spot_count_with_gt_neighbors": int(correct_neighbor_ratio.shape[0]),
        "cross_gt_domain_edge_ratio": gt_edge["cross_domain_edge_ratio"],
        "edge_homophily_gt": gt_edge["edge_homophily"],
        "cross_pred_domain_edge_ratio": pred_edge["cross_domain_edge_ratio"],
        "edge_homophily_pred": pred_edge["edge_homophily"],
        "label_mapping_for_correctness": mapping,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_dir / "spot_boundary_diagnostics.csv", index=False)
    diagnostics.loc[diagnostics["is_gt_boundary"]].to_csv(
        output_dir / "boundary_mask.csv",
        index=False,
    )
    diagnostics.loc[~diagnostics["is_gt_boundary"]].to_csv(
        output_dir / "interior_mask.csv",
        index=False,
    )
    write_json(output_dir / "boundary_metrics.json", boundary_metrics)
    write_json(output_dir / "edge_homophily.json", edge_report)
    global_ari = boundary_metrics["computed_global_ari"]
    boundary_ari = boundary_metrics["boundary_ari"]
    interior_ari = boundary_metrics["interior_ari"]
    global_ari_text = "NA" if global_ari is None else f"{global_ari:.4f}"
    boundary_ari_text = "NA" if boundary_ari is None else f"{boundary_ari:.4f}"
    interior_ari_text = "NA" if interior_ari is None else f"{interior_ari:.4f}"
    print(
        f"{sample_id}: global_ari={global_ari_text}, "
        f"boundary_ari={boundary_ari_text}, "
        f"interior_ari={interior_ari_text}, "
        f"boundary_error_rate={boundary_error_rate}, "
        f"interior_error_rate={interior_error_rate}"
    )
    return boundary_metrics


def match_optional_paths(
    paths: list[Path] | None,
    expected_count: int,
    option_name: str,
) -> list[Path | None]:
    if not paths:
        return [None] * expected_count
    if len(paths) == 1:
        return list(paths) * expected_count
    if len(paths) != expected_count:
        raise ValueError(
            f"{option_name} expects either one path or {expected_count} paths, "
            f"got {len(paths)}"
        )
    return list(paths)


def main() -> None:
    args = parse_args()
    if args.output_dir is not None and len(args.input_dir) > 1:
        raise ValueError("--output-dir is only supported with a single --input-dir")
    if args.sample_id is not None and len(args.input_dir) > 1:
        raise ValueError("--sample-id override is only supported with one --input-dir")

    boundary_paths = match_optional_paths(args.boundary_edges_path, len(args.input_dir), "--boundary-edges-path")
    eval_paths = match_optional_paths(args.eval_edges_path, len(args.input_dir), "--eval-edges-path")

    summaries = []
    for index, input_dir in enumerate(args.input_dir):
        output_dir = args.output_dir if args.output_dir is not None else input_dir
        summaries.append(
            analyze_one(
                input_dir,
                output_dir,
                args,
                boundary_edges_path=boundary_paths[index],
                eval_edges_path=eval_paths[index],
            )
        )

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summaries).to_csv(args.summary_output, index=False)
        print(f"Summary saved to {args.summary_output.resolve()}")


if __name__ == "__main__":
    main()
