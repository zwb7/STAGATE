"""Build a BAGR-pruned spatial graph from vanilla STAGATE artifacts.

This is the pruning-only MVP for BAGR-STAGATE. It consumes lightweight Step 1
outputs and writes a refined edge list plus edge-level diagnostics. It does not
train STAGATE and does not use ground-truth labels.
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
        description="Build BAGR risky-edge scores and a pruned spatial graph."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Step 1 baseline directory containing pred_labels.csv, posterior.npy, "
            "embedding.npy, and spatial_edges.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to <input-dir>/bagr_prune_<ratio>."
        ),
    )
    parser.add_argument(
        "--pca-expression",
        type=Path,
        default=None,
        help=(
            "PCA expression feature matrix (.npy) in pred_labels.csv spot order. "
            "Defaults to <input-dir>/pca_expression.npy."
        ),
    )
    parser.add_argument(
        "--allow-embedding-similarity",
        action="store_true",
        help=(
            "Use embedding.npy for expression-similarity scoring when "
            "pca_expression.npy is unavailable. This is a fallback/debug mode, "
            "not the preferred BAGR protocol."
        ),
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        required=True,
        help="Fraction of undirected spatial edges to prune, e.g. 0.05 or 5 for 5%.",
    )
    parser.add_argument(
        "--max-prune-per-node",
        type=int,
        default=1,
        help="Maximum number of undirected edges pruned incident to any spot.",
    )
    parser.add_argument(
        "--alpha-uncertainty",
        type=float,
        default=0.5,
        help="Weight for posterior uncertainty in b_i = alpha*u_i + (1-alpha)*c_i.",
    )
    parser.add_argument(
        "--sim-metric",
        choices=["cosine", "pearson"],
        default="cosine",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Require both edge endpoints to have max posterior >= this value.",
    )
    parser.add_argument(
        "--expr-dissim-quantile",
        type=float,
        default=None,
        help=(
            "Optional hard gate: require expression dissimilarity to be at or "
            "above this quantile over all spatial edges, e.g. 0.8 or 0.9."
        ),
    )
    parser.add_argument(
        "--cluster-pair-sep-quantile",
        type=float,
        default=None,
        help=(
            "Optional hard gate: require predicted-cluster pair separation to be "
            "at or above this quantile over discordant spatial edges."
        ),
    )
    parser.add_argument(
        "--ratio-denominator",
        choices=["all", "eligible"],
        default="all",
        help=(
            "Whether --prune-ratio is applied to all undirected edges or only "
            "eligible risky edges. The paper MVP uses all."
        ),
    )
    parser.add_argument(
        "--pred-label-column",
        default="pred_label",
        help="Predicted-label column in pred_labels.csv.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def normalize_prune_ratio(value: float) -> float:
    if value <= 0:
        raise ValueError("--prune-ratio must be positive")
    if value > 1:
        if value > 100:
            raise ValueError("--prune-ratio cannot exceed 100 when given as percent")
        value = value / 100.0
    if value > 1:
        raise ValueError("--prune-ratio must be <= 1.0 or <= 100 percent")
    return float(value)


def format_ratio(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def resolve_output_dir(args: argparse.Namespace, prune_ratio: float) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    return args.input_dir / f"bagr_prune_{format_ratio(prune_ratio)}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_label_series(series: pd.Series) -> pd.Series:
    values = series.astype("object").where(series.notna(), other=pd.NA)
    as_text = values.astype("string")
    missing = as_text.str.strip().str.lower().isin(MISSING_LABELS)
    return as_text.mask(missing, pd.NA)


def load_predictions(input_dir: Path, pred_label_column: str) -> pd.DataFrame:
    path = input_dir / "pred_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction labels: {path}")
    labels = pd.read_csv(path, dtype={"spot_id": str})
    missing = {"spot_id", pred_label_column}.difference(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if labels["spot_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated spot_id values")
    labels = labels[["spot_id", pred_label_column]].rename(
        columns={pred_label_column: "pred_label"}
    )
    labels["pred_label"] = normalize_label_series(labels["pred_label"])
    if labels["pred_label"].isna().any():
        missing_count = int(labels["pred_label"].isna().sum())
        raise ValueError(f"pred_labels.csv contains {missing_count} missing labels")
    return labels


def load_edges(input_dir: Path, known_spots: set[str]) -> pd.DataFrame:
    path = input_dir / "spatial_edges.csv"
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
        print(f"Dropping {dropped} edges with endpoints absent from pred_labels.csv")
    return edges.loc[valid].reset_index(drop=True)


def undirected_key(source: str, target: str) -> tuple[str, str]:
    return (source, target) if source <= target else (target, source)


def add_edge_keys(edges: pd.DataFrame) -> pd.DataFrame:
    keyed = edges.copy()
    pairs = [undirected_key(str(a), str(b)) for a, b in keyed[["Cell1", "Cell2"]].itertuples(index=False)]
    keyed["node_a"] = [a for a, _ in pairs]
    keyed["node_b"] = [b for _, b in pairs]
    return keyed


def build_undirected_edges(edges: pd.DataFrame) -> pd.DataFrame:
    keyed = add_edge_keys(edges)
    return keyed[["node_a", "node_b"]].drop_duplicates().reset_index(drop=True)


def build_neighbors(edge_pairs: pd.DataFrame, spot_ids: pd.Series) -> dict[str, set[str]]:
    neighbors = {str(spot_id): set() for spot_id in spot_ids.astype(str)}
    for node_a, node_b in edge_pairs[["node_a", "node_b"]].itertuples(index=False):
        if node_a in neighbors:
            neighbors[node_a].add(node_b)
        if node_b in neighbors:
            neighbors[node_b].add(node_a)
    return neighbors


def load_matrix(path: Path, expected_rows: int, label: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    matrix = np.load(path)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be 2D, got shape {matrix.shape}")
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            f"{label} row count {matrix.shape[0]} does not match pred_labels rows {expected_rows}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    return np.asarray(matrix, dtype=float)


def load_similarity_features(args: argparse.Namespace, n_spots: int) -> tuple[np.ndarray, str, Path]:
    pca_path = args.pca_expression or args.input_dir / "pca_expression.npy"
    if pca_path.exists():
        return load_matrix(pca_path, n_spots, "pca expression"), "pca_expression", pca_path
    if args.allow_embedding_similarity:
        embedding_path = args.input_dir / "embedding.npy"
        return load_matrix(embedding_path, n_spots, "embedding"), "embedding_fallback", embedding_path
    raise FileNotFoundError(
        f"Missing PCA expression matrix: {pca_path}. Generate pca_expression.npy "
        "in Step 1 spot order, or pass --allow-embedding-similarity for a fallback."
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    value = float(np.dot(a, b) / denom)
    return max(-1.0, min(1.0, value))


def pearson_similarity(a: np.ndarray, b: np.ndarray) -> float:
    centered_a = a - a.mean()
    centered_b = b - b.mean()
    return cosine_similarity(centered_a, centered_b)


def pair_similarity(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        return cosine_similarity(a, b)
    if metric == "pearson":
        return pearson_similarity(a, b)
    raise ValueError(f"Unsupported similarity metric: {metric}")


def validate_optional_quantile(value: float | None, name: str) -> None:
    if value is None:
        return
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def quantile_threshold(values: pd.Series, quantile: float | None) -> float | None:
    if quantile is None:
        return None
    clean = values.dropna()
    if clean.empty:
        return None
    return float(clean.quantile(quantile))


def cluster_pair_separation_map(
    labels: pd.DataFrame,
    features: np.ndarray,
) -> dict[tuple[str, str], float]:
    label_values = labels["pred_label"].astype(str).to_numpy()
    result: dict[tuple[str, str], float] = {}
    stats: dict[str, tuple[np.ndarray, float]] = {}
    for label in sorted(set(label_values)):
        subset = features[label_values == label]
        centroid = subset.mean(axis=0)
        if subset.shape[0] <= 1:
            sigma = 0.0
        else:
            sigma = float(np.linalg.norm(subset - centroid, axis=1).mean())
        stats[label] = (centroid, sigma)
    labels_sorted = sorted(stats)
    for i, label_a in enumerate(labels_sorted):
        centroid_a, sigma_a = stats[label_a]
        for label_b in labels_sorted[i + 1 :]:
            centroid_b, sigma_b = stats[label_b]
            denom = sigma_a + sigma_b + 1e-12
            separation = float(np.linalg.norm(centroid_a - centroid_b) / denom)
            result[(label_a, label_b)] = separation
    return result


def cluster_pair_key(label_a: Any, label_b: Any) -> tuple[str, str]:
    a = str(label_a)
    b = str(label_b)
    return (a, b) if a <= b else (b, a)


def compute_spot_scores(
    labels: pd.DataFrame,
    posterior: np.ndarray,
    neighbors: dict[str, set[str]],
    alpha: float,
) -> pd.DataFrame:
    if not 0 <= alpha <= 1:
        raise ValueError("--alpha-uncertainty must be between 0 and 1")
    label_by_spot = labels.set_index("spot_id")["pred_label"].to_dict()
    confidence = posterior.max(axis=1)
    rows: list[dict[str, Any]] = []
    for index, spot_id in enumerate(labels["spot_id"].astype(str)):
        spot_neighbors = sorted(neighbors.get(spot_id, set()))
        label = label_by_spot[spot_id]
        comparable_neighbors = [
            neighbor
            for neighbor in spot_neighbors
            if pd.notna(label_by_spot.get(neighbor))
        ]
        conflicts = sum(label_by_spot[neighbor] != label for neighbor in comparable_neighbors)
        conflict_score = (
            float(conflicts / len(comparable_neighbors)) if comparable_neighbors else 0.0
        )
        uncertainty = float(1.0 - confidence[index])
        boundary_score = float(alpha * uncertainty + (1.0 - alpha) * conflict_score)
        rows.append(
            {
                "spot_id": spot_id,
                "pred_label": label,
                "confidence": float(confidence[index]),
                "posterior_uncertainty": uncertainty,
                "local_label_conflict": conflict_score,
                "boundary_score": boundary_score,
                "n_pred_neighbors": len(comparable_neighbors),
                "n_conflicting_pred_neighbors": int(conflicts),
            }
        )
    return pd.DataFrame(rows)


def compute_edge_scores(
    edge_pairs: pd.DataFrame,
    labels: pd.DataFrame,
    spot_scores: pd.DataFrame,
    features: np.ndarray,
    sim_metric: str,
    min_confidence: float,
    expr_dissim_quantile: float | None,
    cluster_pair_sep_quantile: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_optional_quantile(expr_dissim_quantile, "--expr-dissim-quantile")
    validate_optional_quantile(
        cluster_pair_sep_quantile,
        "--cluster-pair-sep-quantile",
    )
    spot_order = {
        spot_id: index for index, spot_id in enumerate(labels["spot_id"].astype(str))
    }
    label_by_spot = labels.set_index("spot_id")["pred_label"].to_dict()
    score_by_spot = spot_scores.set_index("spot_id").to_dict(orient="index")
    separation_by_pair = cluster_pair_separation_map(labels, features)
    rows: list[dict[str, Any]] = []
    for node_a, node_b in edge_pairs[["node_a", "node_b"]].itertuples(index=False):
        idx_a = spot_order[node_a]
        idx_b = spot_order[node_b]
        label_a = label_by_spot[node_a]
        label_b = label_by_spot[node_b]
        score_a = score_by_spot[node_a]
        score_b = score_by_spot[node_b]
        confidence_a = float(score_a["confidence"])
        confidence_b = float(score_b["confidence"])
        labels_discordant = bool(label_a != label_b)
        confidence_pass = bool(
            confidence_a >= min_confidence and confidence_b >= min_confidence
        )
        pred_boundary_endpoint = bool(
            score_a["local_label_conflict"] > 0 or score_b["local_label_conflict"] > 0
        )
        sim = pair_similarity(features[idx_a], features[idx_b], sim_metric)
        expression_dissimilarity = float(1.0 - sim)
        max_boundary_score = float(
            max(score_a["boundary_score"], score_b["boundary_score"])
        )
        pair_sep = separation_by_pair.get(cluster_pair_key(label_a, label_b), 0.0)
        risk = (
            expression_dissimilarity
            * confidence_a
            * confidence_b
            * max_boundary_score
            if labels_discordant and confidence_pass
            else 0.0
        )
        rows.append(
            {
                "node_a": node_a,
                "node_b": node_b,
                "label_a": label_a,
                "label_b": label_b,
                "labels_discordant": labels_discordant,
                "confidence_a": confidence_a,
                "confidence_b": confidence_b,
                "min_endpoint_confidence": min(confidence_a, confidence_b),
                "confidence_pass": confidence_pass,
                "pred_boundary_endpoint": pred_boundary_endpoint,
                "uncertainty_a": float(score_a["posterior_uncertainty"]),
                "uncertainty_b": float(score_b["posterior_uncertainty"]),
                "label_conflict_a": float(score_a["local_label_conflict"]),
                "label_conflict_b": float(score_b["local_label_conflict"]),
                "boundary_score_a": float(score_a["boundary_score"]),
                "boundary_score_b": float(score_b["boundary_score"]),
                "max_boundary_score": max_boundary_score,
                "similarity": sim,
                "expression_dissimilarity": expression_dissimilarity,
                "cluster_pair_separation": pair_sep,
                "edge_risk": float(risk),
            }
        )
    scored = pd.DataFrame(rows)
    expr_threshold = quantile_threshold(
        scored["expression_dissimilarity"],
        expr_dissim_quantile,
    )
    discordant_sep = scored.loc[
        scored["labels_discordant"],
        "cluster_pair_separation",
    ]
    cluster_threshold = quantile_threshold(discordant_sep, cluster_pair_sep_quantile)
    scored["expr_dissim_threshold"] = expr_threshold
    scored["cluster_pair_sep_threshold"] = cluster_threshold
    scored["expr_dissim_pass"] = (
        True
        if expr_threshold is None
        else scored["expression_dissimilarity"] >= expr_threshold
    )
    scored["cluster_pair_sep_pass"] = (
        True
        if cluster_threshold is None
        else scored["cluster_pair_separation"] >= cluster_threshold
    )
    scored["eligible"] = (
        scored["labels_discordant"]
        & scored["confidence_pass"]
        & scored["expr_dissim_pass"]
        & scored["cluster_pair_sep_pass"]
        & scored["pred_boundary_endpoint"]
        & (scored["edge_risk"] > 0)
    )
    gate_stats = {
        "expr_dissim_quantile": expr_dissim_quantile,
        "expr_dissim_threshold": expr_threshold,
        "cluster_pair_sep_quantile": cluster_pair_sep_quantile,
        "cluster_pair_sep_threshold": cluster_threshold,
        "expr_dissim_pass_undirected_edges": int(scored["expr_dissim_pass"].sum()),
        "cluster_pair_sep_pass_undirected_edges": int(scored["cluster_pair_sep_pass"].sum()),
        "pred_boundary_endpoint_undirected_edges": int(
            scored["pred_boundary_endpoint"].sum()
        ),
    }
    return scored, gate_stats


def select_pruned_edges(
    edge_scores: pd.DataFrame,
    prune_ratio: float,
    max_prune_per_node: int,
    ratio_denominator: str,
) -> tuple[set[tuple[str, str]], pd.DataFrame, dict[str, Any]]:
    if max_prune_per_node < 0:
        raise ValueError("--max-prune-per-node must be non-negative")
    if edge_scores.empty or max_prune_per_node == 0:
        selected = set()
        out = edge_scores.copy()
        out["pruned"] = False
        return selected, out, {"target_pruned_undirected_edges": 0}

    eligible = edge_scores.loc[edge_scores["eligible"]].copy()
    denominator = edge_scores.shape[0] if ratio_denominator == "all" else eligible.shape[0]
    target = int(math.ceil(prune_ratio * denominator)) if denominator else 0
    target = min(target, eligible.shape[0])
    counts: dict[str, int] = {}
    selected: set[tuple[str, str]] = set()
    sorted_edges = eligible.sort_values(
        ["edge_risk", "node_a", "node_b"],
        ascending=[False, True, True],
    )
    for row in sorted_edges.itertuples(index=False):
        if len(selected) >= target:
            break
        node_a = str(row.node_a)
        node_b = str(row.node_b)
        if counts.get(node_a, 0) >= max_prune_per_node:
            continue
        if counts.get(node_b, 0) >= max_prune_per_node:
            continue
        selected.add((node_a, node_b))
        counts[node_a] = counts.get(node_a, 0) + 1
        counts[node_b] = counts.get(node_b, 0) + 1

    out = edge_scores.copy()
    selected_index = pd.MultiIndex.from_tuples(selected, names=["node_a", "node_b"])
    current_index = pd.MultiIndex.from_frame(out[["node_a", "node_b"]])
    out["pruned"] = current_index.isin(selected_index)
    stats = {
        "target_pruned_undirected_edges": int(target),
        "actual_pruned_undirected_edges": int(len(selected)),
        "prune_limited_by_per_node_cap": bool(len(selected) < target),
        "eligible_undirected_edges": int(eligible.shape[0]),
    }
    return selected, out, stats


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


def build_bagr_graph(args: argparse.Namespace) -> dict[str, Any]:
    prune_ratio = normalize_prune_ratio(args.prune_ratio)
    output_dir = resolve_output_dir(args, prune_ratio)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_predictions(args.input_dir, args.pred_label_column)
    posterior = load_matrix(args.input_dir / "posterior.npy", labels.shape[0], "posterior")
    features, feature_source, feature_path = load_similarity_features(args, labels.shape[0])
    directed_edges = load_edges(args.input_dir, set(labels["spot_id"].astype(str)))
    edge_pairs = build_undirected_edges(directed_edges)
    neighbors = build_neighbors(edge_pairs, labels["spot_id"])

    spot_scores = compute_spot_scores(
        labels=labels,
        posterior=posterior,
        neighbors=neighbors,
        alpha=args.alpha_uncertainty,
    )
    edge_scores, gate_stats = compute_edge_scores(
        edge_pairs=edge_pairs,
        labels=labels,
        spot_scores=spot_scores,
        features=features,
        sim_metric=args.sim_metric,
        min_confidence=args.min_confidence,
        expr_dissim_quantile=args.expr_dissim_quantile,
        cluster_pair_sep_quantile=args.cluster_pair_sep_quantile,
    )
    pruned, scored_edges, prune_stats = select_pruned_edges(
        edge_scores=edge_scores,
        prune_ratio=prune_ratio,
        max_prune_per_node=args.max_prune_per_node,
        ratio_denominator=args.ratio_denominator,
    )
    refined_edges, pruned_directed_edges = apply_pruning(directed_edges, pruned)

    spot_scores.to_csv(output_dir / "spot_scores.csv", index=False)
    scored_edges.to_csv(output_dir / "edge_scores.csv", index=False)
    refined_edges.to_csv(output_dir / "refined_edges.csv", index=False)
    pruned_directed_edges.to_csv(output_dir / "pruned_edges.csv", index=False)

    metrics_global = read_json(args.input_dir / "metrics_global.json")
    original_stats = graph_stats(directed_edges, labels["spot_id"])
    refined_stats = graph_stats(refined_edges, labels["spot_id"])
    risk_positive = scored_edges.loc[scored_edges["edge_risk"] > 0, "edge_risk"]
    report: dict[str, Any] = {
        "sample_id": metrics_global.get("sample_id"),
        "method": "bagr_static_edge_pruning",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "feature_source": feature_source,
        "feature_path": str(feature_path),
        "prune_ratio": prune_ratio,
        "ratio_denominator": args.ratio_denominator,
        "max_prune_per_node": int(args.max_prune_per_node),
        "alpha_uncertainty": float(args.alpha_uncertainty),
        "sim_metric": args.sim_metric,
        "min_confidence": float(args.min_confidence),
        **gate_stats,
        "n_spots": int(labels.shape[0]),
        "original_graph": original_stats,
        "refined_graph": refined_stats,
        "pruned_directed_edge_count": int(pruned_directed_edges.shape[0]),
        "edge_retention_ratio_directed": (
            float(refined_edges.shape[0] / directed_edges.shape[0])
            if directed_edges.shape[0]
            else None
        ),
        "risk_positive_undirected_edges": int(risk_positive.shape[0]),
        "max_edge_risk": float(risk_positive.max()) if not risk_positive.empty else 0.0,
        "mean_positive_edge_risk": (
            float(risk_positive.mean()) if not risk_positive.empty else 0.0
        ),
        **prune_stats,
    }
    write_json(output_dir / "graph_refinement_stats.json", report)
    print(
        "BAGR graph refinement saved to "
        f"{output_dir.resolve()} | pruned {report['actual_pruned_undirected_edges']} "
        f"undirected edges ({report['pruned_directed_edge_count']} directed rows)"
    )
    return report


def main() -> None:
    build_bagr_graph(parse_args())


if __name__ == "__main__":
    main()
