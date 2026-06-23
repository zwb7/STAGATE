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
import torch
from sklearn.metrics import adjusted_rand_score

import STAGATE_pyG as STAGATE
from examples.rule_based_graph_refinement import (
    canonicalize_spatial_graph,
    clear_baseline_outputs,
    graph_connectivity_stats,
    load_baseline_ari,
    select_node_local_edges,
    validate_baseline_adata,
)
from experiments.cluster_free.features import (
    FEATURE_COLUMNS,
    build_edge_features,
    build_soft_targets,
    standardize_features,
)
from experiments.cluster_free.scorer import train_edge_scorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cluster-free learned graph refinement."
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cluster_free"),
    )
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--original-metrics", type=Path, default=None)
    parser.add_argument("--retain-ratio", type=float, default=0.9)
    parser.add_argument("--minimum-degree", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--scorer-epochs", type=int, default=300)
    parser.add_argument("--scorer-learning-rate", type=float, default=1e-3)
    parser.add_argument("--scorer-weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--scorer-device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--r-home", default=None)
    parser.add_argument("--r-user", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
    return device


def save_current_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")


def read_metric(path: Path | None, key: str) -> float | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file).get(key)
    return float(value) if value is not None else None


def default_original_metrics(sample_id: str) -> Path:
    return Path("results/rule_based/original") / sample_id / "metrics.json"


def build_refined_graph(
    adata: sc.AnnData,
    pairs: pd.DataFrame,
    scores: np.ndarray,
    retain_ratio: float,
    minimum_degree: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    original_graph = adata.uns["Spatial_Net"].copy()
    directed, canonical_pairs = canonicalize_spatial_graph(
        original_graph,
        adata.obs_names,
    )
    if not canonical_pairs[
        ["pair_id", "node_a_index", "node_b_index"]
    ].equals(pairs[["pair_id", "node_a_index", "node_b_index"]]):
        raise ValueError("Edge feature order no longer matches the spatial graph")

    scored_pairs = pairs.copy()
    scored_pairs["score"] = scores
    selected, node_selection = select_node_local_edges(
        scored_pairs,
        adata.n_obs,
        retain_ratio,
        minimum_degree,
    )
    scored_pairs["selected"] = selected
    selected_pair_ids = set(
        scored_pairs.loc[selected, "pair_id"].astype(int).tolist()
    )
    refined_graph = directed.loc[
        directed["_pair_id"].isin(selected_pair_ids),
        ["Cell1", "Cell2", "Distance"],
    ].copy()
    refined_graph.reset_index(drop=True, inplace=True)

    original_stats = graph_connectivity_stats(original_graph, adata.obs_names)
    refined_stats = graph_connectivity_stats(refined_graph, adata.obs_names)
    if (
        refined_stats["isolated_node_count"]
        > original_stats["isolated_node_count"]
    ):
        raise RuntimeError("Cluster-free refinement introduced isolated nodes")

    graph_report: dict[str, object] = {
        "method": "cluster_free_learned_refinement",
        "selection_policy": "node_local_top_ratio_symmetric_union",
        "requested_retain_ratio": retain_ratio,
        "minimum_degree": minimum_degree,
        "original_directed_edge_count": original_stats["n_directed_edges"],
        "retained_directed_edge_count": refined_stats["n_directed_edges"],
        "original_undirected_edge_count": original_stats["n_undirected_edges"],
        "retained_undirected_edge_count": refined_stats["n_undirected_edges"],
        "undirected_edge_retention_ratio": float(
            refined_stats["n_undirected_edges"]
            / original_stats["n_undirected_edges"]
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
        "score_minimum": float(scores.min()),
        "score_maximum": float(scores.max()),
        "score_mean": float(scores.mean()),
        "score_standard_deviation": float(scores.std()),
    }
    return refined_graph, node_selection, graph_report


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    warnings.filterwarnings("ignore")
    if not 0.0 < args.retain_ratio <= 1.0:
        raise ValueError("--retain-ratio must be in (0, 1]")
    if args.minimum_degree < 0:
        raise ValueError("--minimum-degree must be non-negative")
    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Baseline h5ad not found: {args.input_h5ad}")
    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        os.environ["R_USER"] = args.r_user

    output_dir = args.output_dir / args.sample_id
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input_h5ad)
    validate_baseline_adata(adata, args.ground_truth_key, "embedding")
    _, pairs = canonicalize_spatial_graph(
        adata.uns["Spatial_Net"],
        adata.obs_names,
    )
    edge_features = build_edge_features(adata, pairs)
    targets = build_soft_targets(
        edge_features,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    standardized, feature_mean, feature_std = standardize_features(
        edge_features
    )

    scorer_device = resolve_device(args.scorer_device)
    scorer_result = train_edge_scorer(
        standardized,
        targets.astype(np.float32),
        epochs=args.scorer_epochs,
        learning_rate=args.scorer_learning_rate,
        weight_decay=args.scorer_weight_decay,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        device=scorer_device,
    )
    edge_features["soft_target"] = targets
    edge_features.to_csv(output_dir / "edge_features.csv", index=False)
    pd.DataFrame(
        {
            "pair_id": edge_features["pair_id"],
            "soft_target": targets,
        }
    ).to_csv(output_dir / "pseudo_targets.csv", index=False)
    pd.DataFrame(
        {
            "pair_id": edge_features["pair_id"],
            "node_a": edge_features["node_a"],
            "node_b": edge_features["node_b"],
            "score": scorer_result.scores,
        }
    ).to_csv(output_dir / "edge_scores.csv", index=False)
    with (output_dir / "training_history.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(scorer_result.history, file, indent=2)
    torch.save(
        {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in scorer_result.model.state_dict().items()
            },
            "feature_columns": FEATURE_COLUMNS,
            "feature_mean": feature_mean,
            "feature_standard_deviation": feature_std,
            "soft_target_weights": {
                "alpha": args.alpha,
                "beta": args.beta,
                "gamma": args.gamma,
            },
            "seed": args.seed,
        },
        output_dir / "scorer.pt",
    )

    refined_graph, node_selection, graph_report = build_refined_graph(
        adata,
        edge_features,
        scorer_result.scores,
        args.retain_ratio,
        args.minimum_degree,
    )
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
    ari = adjusted_rand_score(
        evaluation[args.ground_truth_key].astype(str),
        evaluation["mclust"].astype(str),
    )
    baseline_ari, baseline_metrics_path = load_baseline_ari(
        args.input_h5ad,
        args.baseline_metrics,
    )
    original_metrics_path = (
        args.original_metrics
        if args.original_metrics is not None
        else default_original_metrics(args.sample_id)
    )
    original_ari = read_metric(original_metrics_path, "rule_based_ari")

    sc.pl.umap(
        adata,
        color=["mclust", args.ground_truth_key],
        title=[f"Cluster-free E3 (ARI={ari:.2f})", "Ground Truth"],
        show=False,
    )
    save_current_figure(output_dir / "umap_clusters.png")
    sc.pl.spatial(
        adata,
        img_key="hires",
        color=["mclust", args.ground_truth_key],
        title=[f"Cluster-free E3 (ARI={ari:.2f})", "Ground Truth"],
        show=False,
    )
    save_current_figure(output_dir / "spatial_clusters.png")

    final_loss = adata.uns.get("STAGATE_loss")
    if torch.is_tensor(final_loss):
        final_loss = float(final_loss.detach().cpu())
        adata.uns["STAGATE_loss"] = final_loss
    result = {
        "sample_id": args.sample_id,
        "method": "cluster_free_learned_refinement",
        "uses_ground_truth_for_refinement": False,
        "feature_columns": FEATURE_COLUMNS,
        "soft_target_weights": {
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
        },
        "n_spots": int(adata.n_obs),
        "n_evaluated_spots": int(evaluation.shape[0]),
        "n_clusters": args.clusters,
        "baseline_ari": baseline_ari,
        "baseline_metrics": baseline_metrics_path,
        "original_reencoding_ari": original_ari,
        "original_metrics": (
            str(original_metrics_path) if original_metrics_path.exists() else None
        ),
        "cluster_free_ari": float(ari),
        "delta_vs_baseline": (
            float(ari - baseline_ari) if baseline_ari is not None else None
        ),
        "delta_vs_original": (
            float(ari - original_ari) if original_ari is not None else None
        ),
        "requested_retain_ratio": args.retain_ratio,
        "actual_edge_retention_ratio": graph_report[
            "undirected_edge_retention_ratio"
        ],
        "minimum_degree": args.minimum_degree,
        "refined_isolated_node_count": graph_report[
            "refined_isolated_node_count"
        ],
        "scorer_train_size": scorer_result.train_size,
        "scorer_validation_size": scorer_result.validation_size,
        "scorer_epochs": args.scorer_epochs,
        "scorer_device": str(scorer_device),
        "soft_target_minimum": float(targets.min()),
        "soft_target_maximum": float(targets.max()),
        "soft_target_mean": float(targets.mean()),
        "soft_target_standard_deviation": float(targets.std()),
        "scorer_final_train_loss": scorer_result.history[-1]["train_loss"],
        "scorer_final_validation_loss": scorer_result.history[-1][
            "validation_loss"
        ],
        "final_reconstruction_loss": final_loss,
        "device": str(device),
        "seed": args.seed,
    }
    adata.write_h5ad(output_dir / f"{args.sample_id}_cluster_free_stagate.h5ad")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
