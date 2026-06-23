"""Evaluate STAGATE with original or rule-refined spatial graphs.

The script reads a baseline ``.h5ad`` so preprocessing, highly variable genes,
node order, warm-up embeddings, and the original spatial graph are identical
to the official baseline. The ``original`` method performs an unchanged-graph
re-encoding control. Other methods score only existing spatial edges, apply
node-local top-ratio pruning with minimum-degree protection, and reinitialize
the official STAGATE model on the refined graph.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

import STAGATE_pyG as STAGATE

Method = Literal["original", "expression", "embedding"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run original-graph control or similarity graph refinement."
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        required=True,
        help="Baseline h5ad with Spatial_Net, HVG flags, and STAGATE embedding.",
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument(
        "--method",
        choices=["original", "expression", "embedding"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/rule_based"),
        help="Results are saved under <output-dir>/<method>/<sample-id>/.",
    )
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=None,
        help="Defaults to metrics.json beside --input-h5ad.",
    )
    parser.add_argument(
        "--retain-ratio",
        type=float,
        default=0.9,
        help="Node-local fraction of highest-scoring incident edges to retain.",
    )
    parser.add_argument(
        "--minimum-degree",
        type=int,
        default=1,
        help="Minimum retained degree for every node that was not isolated.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda:7",
        help="PyTorch device, for example cpu, cuda:0, or auto.",
    )
    parser.add_argument("--r-home", default=None)
    parser.add_argument("--r-user", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files in an existing method/sample output directory.",
    )
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.retain_ratio <= 1.0:
        raise ValueError("--retain-ratio must be in the interval (0, 1]")
    if args.minimum_degree < 0:
        raise ValueError("--minimum-degree must be non-negative")


def validate_baseline_adata(
    adata: sc.AnnData,
    ground_truth_key: str,
    method: Method,
) -> None:
    if ground_truth_key not in adata.obs:
        raise KeyError(
            f"Ground-truth key not found in adata.obs: {ground_truth_key}"
        )
    if "Spatial_Net" not in adata.uns:
        raise KeyError("Original Spatial_Net not found in adata.uns")
    if "highly_variable" not in adata.var:
        raise KeyError(
            "The baseline h5ad does not contain highly_variable gene flags"
        )
    if method == "embedding" and "STAGATE" not in adata.obsm:
        raise KeyError("Baseline STAGATE embedding not found in adata.obsm")

    graph = adata.uns["Spatial_Net"]
    if not isinstance(graph, pd.DataFrame):
        raise TypeError("adata.uns['Spatial_Net'] must be a pandas DataFrame")
    required_columns = {"Cell1", "Cell2", "Distance"}
    missing = sorted(required_columns.difference(graph.columns))
    if missing:
        raise ValueError(f"Spatial_Net is missing columns: {missing}")


def canonicalize_spatial_graph(
    graph: pd.DataFrame,
    obs_names: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the original directed graph with pair IDs and unique pairs."""
    node_to_index = {str(node): index for index, node in enumerate(obs_names)}
    directed = graph.loc[:, ["Cell1", "Cell2", "Distance"]].copy()
    directed["Cell1"] = directed["Cell1"].astype(str)
    directed["Cell2"] = directed["Cell2"].astype(str)
    source_index = directed["Cell1"].map(node_to_index)
    target_index = directed["Cell2"].map(node_to_index)
    if source_index.isna().any() or target_index.isna().any():
        raise ValueError("Spatial_Net contains spot IDs absent from adata.obs_names")
    if (source_index == target_index).any():
        raise ValueError("Spatial_Net unexpectedly contains self-edges")

    directed["_source_index"] = source_index.astype(int)
    directed["_target_index"] = target_index.astype(int)
    directed["_node_a"] = np.minimum(
        directed["_source_index"],
        directed["_target_index"],
    )
    directed["_node_b"] = np.maximum(
        directed["_source_index"],
        directed["_target_index"],
    )

    pair_keys = pd.MultiIndex.from_frame(directed[["_node_a", "_node_b"]])
    unique_keys = pair_keys.drop_duplicates()
    pair_lookup = pd.Series(
        np.arange(len(unique_keys), dtype=int),
        index=unique_keys,
    )
    directed["_pair_id"] = pair_lookup.reindex(pair_keys).to_numpy()

    pair_distance = directed.groupby("_pair_id", sort=True)["Distance"].min()
    pairs = pd.DataFrame(
        {
            "pair_id": np.arange(len(unique_keys), dtype=int),
            "node_a_index": unique_keys.get_level_values(0).astype(int),
            "node_b_index": unique_keys.get_level_values(1).astype(int),
        }
    )
    names = np.asarray(obs_names.astype(str))
    pairs["node_a"] = names[pairs["node_a_index"].to_numpy()]
    pairs["node_b"] = names[pairs["node_b_index"].to_numpy()]
    pairs["distance"] = pair_distance.reindex(pairs["pair_id"]).to_numpy()
    return directed, pairs


def row_normalized_sparse(matrix: sp.spmatrix) -> sp.csr_matrix:
    matrix = matrix.tocsr().astype(np.float64)
    norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    inverse_norm = np.zeros_like(norm)
    nonzero = norm > 0
    inverse_norm[nonzero] = 1.0 / norm[nonzero]
    return sp.diags(inverse_norm).dot(matrix).tocsr()


def score_expression_edges(
    adata: sc.AnnData,
    pairs: pd.DataFrame,
) -> np.ndarray:
    hvg_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
    if not hvg_mask.any():
        raise ValueError("No highly variable genes are selected")
    expression = adata[:, hvg_mask].X
    source = pairs["node_a_index"].to_numpy(dtype=int)
    target = pairs["node_b_index"].to_numpy(dtype=int)

    if sp.issparse(expression):
        normalized = row_normalized_sparse(expression)
        scores = normalized[source].multiply(normalized[target]).sum(axis=1)
        return np.asarray(scores).ravel()

    expression_array = np.asarray(expression, dtype=np.float64)
    norm = np.linalg.norm(expression_array, axis=1, keepdims=True)
    normalized = np.divide(
        expression_array,
        norm,
        out=np.zeros_like(expression_array),
        where=norm > 0,
    )
    return np.einsum(
        "ij,ij->i",
        normalized[source],
        normalized[target],
    )


def score_embedding_edges(
    adata: sc.AnnData,
    pairs: pd.DataFrame,
) -> np.ndarray:
    embedding = np.asarray(adata.obsm["STAGATE"], dtype=np.float64)
    if embedding.ndim != 2 or embedding.shape[0] != adata.n_obs:
        raise ValueError(
            "STAGATE embedding must have shape "
            f"({adata.n_obs}, latent_dim), got {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("STAGATE embedding contains NaN or infinite values")

    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = np.divide(
        embedding,
        norm,
        out=np.zeros_like(embedding),
        where=norm > 0,
    )
    source = pairs["node_a_index"].to_numpy(dtype=int)
    target = pairs["node_b_index"].to_numpy(dtype=int)
    return np.einsum(
        "ij,ij->i",
        normalized[source],
        normalized[target],
    )


def select_node_local_edges(
    pairs: pd.DataFrame,
    n_nodes: int,
    retain_ratio: float,
    minimum_degree: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Select each node's best incident edges, then take the symmetric union."""
    pair_ids = pairs["pair_id"].to_numpy(dtype=int)
    node_a = pairs["node_a_index"].to_numpy(dtype=int)
    node_b = pairs["node_b_index"].to_numpy(dtype=int)
    scores = pairs["score"].to_numpy(dtype=float)

    incident = pd.DataFrame(
        {
            "node_index": np.concatenate([node_a, node_b]),
            "pair_id": np.concatenate([pair_ids, pair_ids]),
            "score": np.concatenate([scores, scores]),
        }
    )
    selected = np.zeros(pairs.shape[0], dtype=bool)
    node_rows: list[dict[str, int | float]] = []

    grouped = incident.groupby("node_index", sort=True)
    for node_index in range(n_nodes):
        if node_index not in grouped.groups:
            node_rows.append(
                {
                    "node_index": node_index,
                    "original_degree": 0,
                    "requested_keep_count": 0,
                    "selected_by_node_count": 0,
                }
            )
            continue

        node_edges = grouped.get_group(node_index).sort_values(
            ["score", "pair_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        degree = int(node_edges.shape[0])
        ratio_count = int(math.floor(degree * retain_ratio))
        keep_count = min(
            degree,
            max(minimum_degree, ratio_count),
        )
        chosen_pair_ids = node_edges.iloc[:keep_count]["pair_id"].to_numpy(
            dtype=int
        )
        selected[chosen_pair_ids] = True
        node_rows.append(
            {
                "node_index": node_index,
                "original_degree": degree,
                "requested_keep_count": keep_count,
                "selected_by_node_count": int(chosen_pair_ids.size),
            }
        )

    return selected, pd.DataFrame.from_records(node_rows)


def graph_connectivity_stats(
    graph: pd.DataFrame,
    obs_names: pd.Index,
) -> dict[str, int | float]:
    node_to_index = {str(node): index for index, node in enumerate(obs_names)}
    source = graph["Cell1"].astype(str).map(node_to_index)
    target = graph["Cell2"].astype(str).map(node_to_index)
    if source.isna().any() or target.isna().any():
        raise ValueError("Spatial_Net contains spot IDs absent from adata.obs_names")

    n_nodes = len(obs_names)
    adjacency = sp.coo_matrix(
        (
            np.ones(graph.shape[0], dtype=np.uint8),
            (source.astype(int), target.astype(int)),
        ),
        shape=(n_nodes, n_nodes),
    )
    adjacency = (adjacency + adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    degree = np.asarray((adjacency > 0).sum(axis=1)).ravel()
    n_components, component_labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(component_labels, minlength=n_components)
    largest_component = int(component_sizes.max()) if n_nodes else 0
    return {
        "n_nodes": n_nodes,
        "n_directed_edges": int(graph.shape[0]),
        "n_undirected_edges": int(adjacency.nnz // 2),
        "mean_undirected_degree": float(degree.mean()) if n_nodes else 0.0,
        "minimum_undirected_degree": int(degree.min()) if n_nodes else 0,
        "isolated_node_count": int((degree == 0).sum()),
        "connected_component_count": int(n_components),
        "largest_component_size": largest_component,
        "largest_component_ratio": (
            float(largest_component / n_nodes) if n_nodes else 0.0
        ),
    }


def build_refined_graph(
    adata: sc.AnnData,
    method: Method,
    retain_ratio: float,
    minimum_degree: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    original_graph = adata.uns["Spatial_Net"].copy()
    directed, pairs = canonicalize_spatial_graph(
        original_graph,
        adata.obs_names,
    )
    if method == "expression":
        scores = score_expression_edges(adata, pairs)
    elif method == "embedding":
        scores = score_embedding_edges(adata, pairs)
    else:
        raise ValueError(f"Unsupported method: {method}")
    if not np.isfinite(scores).all():
        raise ValueError("Edge scores contain NaN or infinite values")

    pairs["score"] = scores
    selected, node_selection = select_node_local_edges(
        pairs,
        adata.n_obs,
        retain_ratio,
        minimum_degree,
    )
    pairs["selected"] = selected

    selected_pair_ids = set(
        pairs.loc[pairs["selected"], "pair_id"].astype(int).tolist()
    )
    refined_graph = directed.loc[
        directed["_pair_id"].isin(selected_pair_ids),
        ["Cell1", "Cell2", "Distance"],
    ].copy()
    refined_graph.reset_index(drop=True, inplace=True)

    original_stats = graph_connectivity_stats(
        original_graph,
        adata.obs_names,
    )
    refined_stats = graph_connectivity_stats(
        refined_graph,
        adata.obs_names,
    )
    if (
        refined_stats["isolated_node_count"]
        > original_stats["isolated_node_count"]
    ):
        raise RuntimeError(
            "Graph refinement introduced new isolated nodes despite "
            "minimum-degree protection"
        )

    graph_report: dict[str, object] = {
        "method": method,
        "requested_retain_ratio": retain_ratio,
        "minimum_degree": minimum_degree,
        "selection_policy": "node_local_top_ratio_symmetric_union",
        "original_directed_edge_count": original_stats["n_directed_edges"],
        "retained_directed_edge_count": refined_stats["n_directed_edges"],
        "original_undirected_edge_count": original_stats["n_undirected_edges"],
        "retained_undirected_edge_count": refined_stats["n_undirected_edges"],
        "directed_edge_retention_ratio": float(
            refined_stats["n_directed_edges"]
            / original_stats["n_directed_edges"]
        ),
        "undirected_edge_retention_ratio": float(
            refined_stats["n_undirected_edges"]
            / original_stats["n_undirected_edges"]
        ),
        "original_isolated_node_count": original_stats[
            "isolated_node_count"
        ],
        "refined_isolated_node_count": refined_stats[
            "isolated_node_count"
        ],
        "original_connected_component_count": original_stats[
            "connected_component_count"
        ],
        "refined_connected_component_count": refined_stats[
            "connected_component_count"
        ],
        "original_largest_component_ratio": original_stats[
            "largest_component_ratio"
        ],
        "refined_largest_component_ratio": refined_stats[
            "largest_component_ratio"
        ],
        "original_mean_undirected_degree": original_stats[
            "mean_undirected_degree"
        ],
        "refined_mean_undirected_degree": refined_stats[
            "mean_undirected_degree"
        ],
        "original_minimum_undirected_degree": original_stats[
            "minimum_undirected_degree"
        ],
        "refined_minimum_undirected_degree": refined_stats[
            "minimum_undirected_degree"
        ],
        "score_minimum": float(scores.min()),
        "score_maximum": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_standard_deviation": float(scores.std()),
    }
    return refined_graph, pairs, node_selection, graph_report


def build_original_graph_control(
    adata: sc.AnnData,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    original_graph = adata.uns["Spatial_Net"].loc[
        :,
        ["Cell1", "Cell2", "Distance"],
    ].copy()
    stats = graph_connectivity_stats(original_graph, adata.obs_names)
    graph_report: dict[str, object] = {
        "method": "original",
        "requested_retain_ratio": 1.0,
        "minimum_degree": None,
        "selection_policy": "unchanged_original_graph",
        "original_directed_edge_count": stats["n_directed_edges"],
        "retained_directed_edge_count": stats["n_directed_edges"],
        "original_undirected_edge_count": stats["n_undirected_edges"],
        "retained_undirected_edge_count": stats["n_undirected_edges"],
        "directed_edge_retention_ratio": 1.0,
        "undirected_edge_retention_ratio": 1.0,
        "original_isolated_node_count": stats["isolated_node_count"],
        "refined_isolated_node_count": stats["isolated_node_count"],
        "original_connected_component_count": stats["connected_component_count"],
        "refined_connected_component_count": stats["connected_component_count"],
        "original_largest_component_ratio": stats["largest_component_ratio"],
        "refined_largest_component_ratio": stats["largest_component_ratio"],
        "original_mean_undirected_degree": stats["mean_undirected_degree"],
        "refined_mean_undirected_degree": stats["mean_undirected_degree"],
        "original_minimum_undirected_degree": stats["minimum_undirected_degree"],
        "refined_minimum_undirected_degree": stats["minimum_undirected_degree"],
    }
    return original_graph, pd.DataFrame(), pd.DataFrame(), graph_report


def clear_baseline_outputs(adata: sc.AnnData) -> None:
    if "mclust" in adata.obs:
        del adata.obs["mclust"]
    for key in ["STAGATE", "X_umap"]:
        if key in adata.obsm:
            del adata.obsm[key]
    for key in ["connectivities", "distances"]:
        if key in adata.obsp:
            del adata.obsp[key]
    for key in ["neighbors", "paga", "STAGATE_loss"]:
        if key in adata.uns:
            del adata.uns[key]


def load_baseline_ari(
    input_h5ad: Path,
    baseline_metrics: Path | None,
) -> tuple[float | None, str | None]:
    metrics_path = (
        baseline_metrics
        if baseline_metrics is not None
        else input_h5ad.parent / "metrics.json"
    )
    if not metrics_path.exists():
        return None, None
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    value = metrics.get("ari")
    if value is None:
        return None, str(metrics_path)
    return float(value), str(metrics_path)


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    warnings.filterwarnings("ignore")
    validate_args(args)

    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        os.environ["R_USER"] = args.r_user
    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Baseline h5ad not found: {args.input_h5ad}")

    output_dir = args.output_dir / args.method / args.sample_id
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite to replace its files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline artifact from {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)
    validate_baseline_adata(
        adata,
        args.ground_truth_key,
        args.method,
    )

    if args.method == "original":
        refined_graph, edge_scores, node_selection, graph_report = (
            build_original_graph_control(adata)
        )
    else:
        refined_graph, edge_scores, node_selection, graph_report = (
            build_refined_graph(
                adata,
                args.method,
                args.retain_ratio,
                args.minimum_degree,
            )
        )
    print(
        f"{args.method} graph: "
        f"{graph_report['original_undirected_edge_count']} -> "
        f"{graph_report['retained_undirected_edge_count']} undirected edges "
        f"(actual retention="
        f"{graph_report['undirected_edge_retention_ratio']:.4f})"
    )

    if args.method != "original":
        edge_scores.loc[
            :,
            ["pair_id", "node_a", "node_b", "distance", "score", "selected"],
        ].to_csv(output_dir / "edge_scores.csv", index=False)
        node_selection["spot_id"] = np.asarray(adata.obs_names.astype(str))[
            node_selection["node_index"].to_numpy(dtype=int)
        ]
        node_selection.to_csv(output_dir / "node_selection.csv", index=False)
    refined_graph.to_csv(output_dir / "refined_spatial_net.csv", index=False)
    with (output_dir / "graph_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(graph_report, file, indent=2, ensure_ascii=False)

    clear_baseline_outputs(adata)
    adata.uns["Spatial_Net"] = refined_graph

    STAGATE.Stats_Spatial_Net(adata)
    save_current_figure(output_dir / "refined_spatial_network_stats.png")

    device = resolve_device(args.device)
    adata = STAGATE.train_STAGATE(
        adata,
        hidden_dims=[args.hidden_dim, args.latent_dim],
        n_epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        random_seed=args.seed,
        save_loss=True,
        save_reconstrction=False,
        device=device,
    )

    sc.pp.neighbors(adata, use_rep="STAGATE", random_state=args.seed)
    sc.tl.umap(adata, random_state=args.seed)
    adata = STAGATE.mclust_R(
        adata,
        used_obsm="STAGATE",
        num_cluster=args.clusters,
        random_seed=args.seed,
    )

    evaluation = adata.obs[["mclust", args.ground_truth_key]].dropna()
    rule_based_ari = adjusted_rand_score(
        evaluation[args.ground_truth_key].astype(str),
        evaluation["mclust"].astype(str),
    )
    baseline_ari, baseline_metrics_path = load_baseline_ari(
        args.input_h5ad,
        args.baseline_metrics,
    )
    delta_ari = (
        float(rule_based_ari - baseline_ari)
        if baseline_ari is not None
        else None
    )
    print(f"Rule-based Adjusted Rand Index: {rule_based_ari:.4f}")
    if baseline_ari is not None:
        print(f"Baseline ARI: {baseline_ari:.4f}; Delta ARI: {delta_ari:+.4f}")

    method_title = {
        "original": "Original graph re-encoding",
        "expression": "Expression similarity",
        "embedding": "Embedding similarity",
    }[args.method]
    sc.pl.umap(
        adata,
        color=["mclust", args.ground_truth_key],
        title=[f"{method_title} (ARI={rule_based_ari:.2f})", "Ground Truth"],
        show=False,
    )
    save_current_figure(output_dir / "umap_clusters.png")

    sc.pl.spatial(
        adata,
        img_key="hires",
        color=["mclust", args.ground_truth_key],
        title=[f"{method_title} (ARI={rule_based_ari:.2f})", "Ground Truth"],
        show=False,
    )
    save_current_figure(output_dir / "spatial_clusters.png")

    final_loss = adata.uns.get("STAGATE_loss")
    if torch.is_tensor(final_loss):
        final_loss = float(final_loss.detach().cpu())
        adata.uns["STAGATE_loss"] = final_loss

    result = {
        "sample_id": args.sample_id,
        "method": (
            "original_graph_reencoding"
            if args.method == "original"
            else f"{args.method}_similarity_refinement"
        ),
        "uses_ground_truth_for_refinement": False,
        "input_h5ad": str(args.input_h5ad),
        "ground_truth_key": args.ground_truth_key,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_evaluated_spots": int(evaluation.shape[0]),
        "n_clusters": args.clusters,
        "baseline_ari": baseline_ari,
        "baseline_metrics": baseline_metrics_path,
        "rule_based_ari": float(rule_based_ari),
        "delta_ari": delta_ari,
        "requested_retain_ratio": (
            1.0 if args.method == "original" else args.retain_ratio
        ),
        "actual_edge_retention_ratio": graph_report[
            "undirected_edge_retention_ratio"
        ],
        "minimum_degree": (
            None if args.method == "original" else args.minimum_degree
        ),
        "refined_isolated_node_count": graph_report[
            "refined_isolated_node_count"
        ],
        "final_reconstruction_loss": final_loss,
        "device": str(device),
        "seed": args.seed,
        "hidden_dims": [args.hidden_dim, args.latent_dim],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }

    output_suffix = (
        "original_reencoding"
        if args.method == "original"
        else f"{args.method}_similarity"
    )
    output_h5ad = output_dir / f"{args.sample_id}_{output_suffix}_stagate.h5ad"
    adata.write_h5ad(output_h5ad)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_dir.resolve()}")
    return result


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
