"""Evaluate STAGATE after oracle removal of cross-domain spatial edges.

This is an analysis-only experiment. It uses ground-truth annotations to
remove spatial edges whose two endpoints have different known labels, then
reinitializes and retrains the official STAGATE model on the refined graph.

The input must be a baseline ``.h5ad`` produced by the existing DLPFC or HBC
baseline scripts. Reusing that artifact keeps preprocessing, highly variable
genes, node order, and the original spatial graph identical to the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the analysis-only oracle graph refinement experiment."
    )
    parser.add_argument(
        "--input-h5ad",
        type=Path,
        required=True,
        help="Baseline h5ad containing Ground Truth and the original Spatial_Net.",
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/oracle"),
    )
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=None,
        help="Defaults to metrics.json beside --input-h5ad when available.",
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
        help="Replace an existing output directory for this sample.",
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


def validate_baseline_adata(
    adata: sc.AnnData,
    ground_truth_key: str,
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

    graph = adata.uns["Spatial_Net"]
    if not isinstance(graph, pd.DataFrame):
        raise TypeError("adata.uns['Spatial_Net'] must be a pandas DataFrame")
    required_columns = {"Cell1", "Cell2", "Distance"}
    missing = sorted(required_columns.difference(graph.columns))
    if missing:
        raise ValueError(f"Spatial_Net is missing columns: {missing}")


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
        "mean_undirected_degree": float(degree.mean()) if n_nodes else 0.0,
        "minimum_undirected_degree": int(degree.min()) if n_nodes else 0,
        "isolated_node_count": int((degree == 0).sum()),
        "connected_component_count": int(n_components),
        "largest_component_size": largest_component,
        "largest_component_ratio": (
            float(largest_component / n_nodes) if n_nodes else 0.0
        ),
    }


def build_oracle_graph(
    adata: sc.AnnData,
    ground_truth_key: str,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    original_graph = adata.uns["Spatial_Net"].copy()
    labels = adata.obs[ground_truth_key]
    label_by_spot = labels.to_dict()

    source_labels = original_graph["Cell1"].map(label_by_spot)
    target_labels = original_graph["Cell2"].map(label_by_spot)
    source_known = source_labels.notna()
    target_known = target_labels.notna()
    both_known = source_known & target_known
    cross_domain = both_known & (source_labels != target_labels)
    unknown_endpoint = ~(both_known)

    refined_graph = original_graph.loc[~cross_domain].copy().reset_index(drop=True)
    original_stats = graph_connectivity_stats(
        original_graph,
        adata.obs_names,
    )
    refined_stats = graph_connectivity_stats(
        refined_graph,
        adata.obs_names,
    )

    original_edges = int(original_graph.shape[0])
    retained_edges = int(refined_graph.shape[0])
    removed_edges = int(cross_domain.sum())
    report: dict[str, int | float] = {
        "original_directed_edge_count": original_edges,
        "retained_directed_edge_count": retained_edges,
        "removed_cross_domain_directed_edge_count": removed_edges,
        "unknown_endpoint_directed_edge_count": int(unknown_endpoint.sum()),
        "edge_retention_ratio": (
            float(retained_edges / original_edges) if original_edges else 0.0
        ),
        "original_isolated_node_count": original_stats["isolated_node_count"],
        "refined_isolated_node_count": refined_stats["isolated_node_count"],
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
        "refined_minimum_undirected_degree": refined_stats[
            "minimum_undirected_degree"
        ],
    }
    return refined_graph, report


def clear_baseline_outputs(adata: sc.AnnData) -> None:
    for key in ["mclust"]:
        if key in adata.obs:
            del adata.obs[key]
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
    return (float(value), str(metrics_path)) if value is not None else (None, str(metrics_path))


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    warnings.filterwarnings("ignore")

    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        os.environ["R_USER"] = args.r_user
    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Baseline h5ad not found: {args.input_h5ad}")

    output_dir = args.output_dir / args.sample_id
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --overwrite to replace its files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline artifact from {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)
    validate_baseline_adata(adata, args.ground_truth_key)

    refined_graph, graph_report = build_oracle_graph(
        adata,
        args.ground_truth_key,
    )
    print(
        "Oracle graph: "
        f"{graph_report['original_directed_edge_count']} -> "
        f"{graph_report['retained_directed_edge_count']} directed edges "
        f"(retention={graph_report['edge_retention_ratio']:.4f})"
    )
    refined_graph.to_csv(output_dir / "oracle_spatial_net.csv", index=False)
    with (output_dir / "graph_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(graph_report, file, indent=2, ensure_ascii=False)

    clear_baseline_outputs(adata)
    adata.uns["Spatial_Net"] = refined_graph

    STAGATE.Stats_Spatial_Net(adata)
    save_current_figure(output_dir / "oracle_spatial_network_stats.png")

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
    oracle_ari = adjusted_rand_score(
        evaluation[args.ground_truth_key].astype(str),
        evaluation["mclust"].astype(str),
    )
    baseline_ari, baseline_metrics_path = load_baseline_ari(
        args.input_h5ad,
        args.baseline_metrics,
    )
    delta_ari = (
        float(oracle_ari - baseline_ari)
        if baseline_ari is not None
        else None
    )
    print(f"Oracle Adjusted Rand Index: {oracle_ari:.4f}")
    if baseline_ari is not None:
        print(f"Baseline ARI: {baseline_ari:.4f}; Delta ARI: {delta_ari:+.4f}")

    sc.pl.umap(
        adata,
        color=["mclust", args.ground_truth_key],
        title=[f"Oracle STAGATE (ARI={oracle_ari:.2f})", "Ground Truth"],
        show=False,
    )
    save_current_figure(output_dir / "umap_clusters.png")

    sc.pl.spatial(
        adata,
        img_key="hires",
        color=["mclust", args.ground_truth_key],
        title=[f"Oracle STAGATE (ARI={oracle_ari:.2f})", "Ground Truth"],
        show=False,
    )
    save_current_figure(output_dir / "spatial_clusters.png")

    final_loss = adata.uns.get("STAGATE_loss")
    if torch.is_tensor(final_loss):
        final_loss = float(final_loss.detach().cpu())
        adata.uns["STAGATE_loss"] = final_loss

    result = {
        "sample_id": args.sample_id,
        "method": "oracle_graph_refinement",
        "analysis_only_uses_ground_truth": True,
        "input_h5ad": str(args.input_h5ad),
        "ground_truth_key": args.ground_truth_key,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_evaluated_spots": int(evaluation.shape[0]),
        "n_clusters": args.clusters,
        "baseline_ari": baseline_ari,
        "baseline_metrics": baseline_metrics_path,
        "oracle_ari": float(oracle_ari),
        "delta_ari": delta_ari,
        "edge_retention_ratio": graph_report["edge_retention_ratio"],
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

    adata.write_h5ad(output_dir / f"{args.sample_id}_oracle_stagate.h5ad")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_dir.resolve()}")
    return result


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
