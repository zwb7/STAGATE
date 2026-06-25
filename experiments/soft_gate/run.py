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
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score

import STAGATE_pyG as STAGATE
from examples.rule_based_graph_refinement import (
    canonicalize_spatial_graph,
    clear_baseline_outputs,
    graph_connectivity_stats,
    load_baseline_ari,
    validate_baseline_adata,
)
from experiments.soft_gate.diagnostics import gate_diagnostics
from experiments.soft_gate.features import build_asg_edge_priors, validate_warmup_embedding
from experiments.soft_gate.training import canonical_variant, train_soft_gate_stagate

VARIANTS = [
    "baseline",
    "extra_training",
    "current_gate_only",
    "stabilized_unnormalized",
    "stabilized_renormalized",
    "uniform_gate",
    "shuffled_gate",
    "boundary_focused",
    # Backward-compatible aliases from the abandoned E3-v1 command set.
    "gate_only",
    "gate_distribution",
    "full",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E3-v2 Adaptive Soft Edge Gating for STAGATE."
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/soft_gate_v2"),
        help="Results are saved under <output-dir>/<variant>/<sample-id>/.",
    )
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument("--warmup-key", default="STAGATE")
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--original-metrics", type=Path, default=None)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Backward-compatible alias for --gate-epochs.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Train an internal no-gate warm-up model for this many epochs. "
        "If 0, use --warmup-key from the input h5ad.",
    )
    parser.add_argument(
        "--gate-epochs",
        type=int,
        default=1000,
        help="Number of post-warm-up ASG gate training epochs.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clipping", type=float, default=5.0)
    parser.add_argument("--gate-dim", type=int, default=16)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--lambda-budget", type=float, default=1.0)
    parser.add_argument("--g-min", type=float, default=0.80)
    parser.add_argument("--initial-mean-gate", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--logit-clip", type=float, default=5.0)
    parser.add_argument("--boundary-candidate-quantile", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--r-home", default=None)
    parser.add_argument("--r-user", default=None)
    parser.add_argument("--skip-plots", action="store_true")
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


def read_metric(path: Path | None, keys: tuple[str, ...]) -> float | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def default_original_metrics(sample_id: str) -> Path:
    return Path("results/rule_based/original") / sample_id / "metrics.json"


def validate_args(args: argparse.Namespace) -> None:
    if args.gate_dim <= 0:
        raise ValueError("--gate-dim must be positive")
    if args.epochs is not None and args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    if args.gate_epochs < 0:
        raise ValueError("--gate-epochs must be non-negative")
    if not 0.0 <= args.rho < 1.0:
        raise ValueError("--rho must be in [0, 1)")
    if not 0.0 <= args.g_min < 1.0:
        raise ValueError("--g-min must be in [0, 1)")
    if not args.g_min < args.initial_mean_gate < 1.0:
        raise ValueError("--initial-mean-gate must be in (g_min, 1)")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.logit_clip <= 0:
        raise ValueError("--logit-clip must be positive")
    if not 0.0 < args.boundary_candidate_quantile < 1.0:
        raise ValueError("--boundary-candidate-quantile must be in (0, 1)")


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    warnings.filterwarnings("ignore")
    validate_args(args)
    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        os.environ["R_USER"] = args.r_user
    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Baseline h5ad not found: {args.input_h5ad}")

    requested_variant = args.variant
    variant = canonical_variant(args.variant)
    output_dir = args.output_dir / variant / args.sample_id
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading baseline artifact from {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)
    validate_baseline_adata(adata, args.ground_truth_key, "embedding")
    existing_warmup_embedding = validate_warmup_embedding(adata, args.warmup_key)
    warmup_embedding = None if args.warmup_epochs > 0 else existing_warmup_embedding
    _, pairs = canonicalize_spatial_graph(adata.uns["Spatial_Net"], adata.obs_names)

    edge_prior_result = build_asg_edge_priors(
        adata,
        pairs,
        clusters=args.clusters,
        seed=args.seed,
        embedding_key=args.warmup_key,
        boundary_candidate_quantile=args.boundary_candidate_quantile,
    )
    edge_priors = edge_prior_result.table
    edge_priors.to_csv(output_dir / "edge_priors.csv", index=False)
    np.save(output_dir / "soft_assignments.npy", edge_prior_result.soft_assignments)

    original_graph_stats = graph_connectivity_stats(
        adata.uns["Spatial_Net"],
        adata.obs_names,
    )
    with (output_dir / "graph_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "method": "asg_stagate",
                "variant": variant,
                "requested_variant": requested_variant,
                "hard_pruning": False,
                "boundary_candidate_threshold": edge_prior_result.boundary_candidate_threshold,
                "n_boundary_candidate_edges": int(edge_priors["boundary_candidate"].sum()),
                "original_directed_edge_count": original_graph_stats["n_directed_edges"],
                "original_undirected_edge_count": original_graph_stats["n_undirected_edges"],
                "original_isolated_node_count": original_graph_stats["isolated_node_count"],
                "original_connected_component_count": original_graph_stats["connected_component_count"],
                "original_largest_component_ratio": original_graph_stats["largest_component_ratio"],
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    clear_baseline_outputs(adata)
    device = resolve_device(args.device)
    training_result = train_soft_gate_stagate(
        adata,
        edge_priors,
        warmup_embedding,
        variant=variant,
        hidden_dims=[args.hidden_dim, args.latent_dim],
        warmup_epochs=args.warmup_epochs,
        gate_epochs=args.epochs if args.epochs is not None else args.gate_epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clipping=args.gradient_clipping,
        gate_dim=args.gate_dim,
        rho=args.rho,
        lambda_budget=args.lambda_budget,
        key_added="STAGATE",
        random_seed=args.seed,
        save_loss=True,
        save_reconstruction=False,
        device=device,
        g_min=args.g_min,
        initial_mean_gate=args.initial_mean_gate,
        temperature=args.temperature,
        logit_clip=args.logit_clip,
    )
    adata = training_result.adata

    if args.warmup_epochs > 0:
        adata.obsm[args.warmup_key] = training_result.warmup_embedding


    gate_score_columns = [
        "pair_id",
        "node_a",
        "node_b",
        "node_a_index",
        "node_b_index",
        "distance",
        "soft_domain_consistency",
        "embedding_similarity",
        "node_a_entropy",
        "node_b_entropy",
        "boundary_candidate",
    ]
    gate_scores = edge_priors.loc[:, gate_score_columns].copy()
    gate_scores["gate"] = training_result.pair_gates
    gate_scores.to_csv(output_dir / "gate_scores.csv", index=False)

    diagnostics = gate_diagnostics(
        adata,
        edge_priors,
        training_result.pair_gates,
        training_result.effective_degree,
        ground_truth_key=args.ground_truth_key,
    )
    diagnostics["boundary_candidate_threshold"] = edge_prior_result.boundary_candidate_threshold
    diagnostics["n_boundary_candidate_edges"] = int(edge_priors["boundary_candidate"].sum())
    with (output_dir / "gate_diagnostics.json").open("w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2, ensure_ascii=False)

    with (output_dir / "training_history.json").open("w", encoding="utf-8") as file:
        json.dump(training_result.history, file, indent=2, ensure_ascii=False)

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
    original_ari = read_metric(
        original_metrics_path,
        ("rule_based_ari", "ari", "original_reencoding_ari"),
    )

    if not args.skip_plots:
        STAGATE.Stats_Spatial_Net(adata)
        save_current_figure(output_dir / "original_spatial_network_stats.png")
        sc.pl.umap(
            adata,
            color=["mclust", args.ground_truth_key],
            title=[f"ASG {variant} (ARI={ari:.2f})", "Ground Truth"],
            show=False,
        )
        save_current_figure(output_dir / "umap_clusters.png")
        sc.pl.spatial(
            adata,
            img_key="hires",
            color=["mclust", args.ground_truth_key],
            title=[f"ASG {variant} (ARI={ari:.2f})", "Ground Truth"],
            show=False,
        )
        save_current_figure(output_dir / "spatial_clusters.png")

    gate_summary = diagnostics["gate_summary"]
    result = {
        "sample_id": args.sample_id,
        "method": "asg_stagate",
        "variant": variant,
        "requested_variant": requested_variant,
        "uses_ground_truth_for_refinement": False,
        "hard_pruning": False,
        "renormalize_gate": training_result.renormalize_gate,
        "input_h5ad": str(args.input_h5ad),
        "ground_truth_key": args.ground_truth_key,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_evaluated_spots": int(evaluation.shape[0]),
        "n_clusters": args.clusters,
        "baseline_ari": baseline_ari,
        "baseline_metrics": baseline_metrics_path,
        "original_reencoding_ari": original_ari,
        "original_metrics": str(original_metrics_path) if original_metrics_path.exists() else None,
        "ari": float(ari),
        "soft_gate_ari": float(ari),
        "delta_vs_baseline": float(ari - baseline_ari) if baseline_ari is not None else None,
        "delta_vs_original": float(ari - original_ari) if original_ari is not None else None,
        "mean_gate": gate_summary["mean_gate"],
        "std_gate": gate_summary["std_gate"],
        "minimum_gate": gate_summary["minimum_gate"],
        "maximum_gate": gate_summary["maximum_gate"],
        "gate_p1": gate_summary["gate_p1"],
        "gate_p5": gate_summary["gate_p5"],
        "gate_p10": gate_summary["gate_p10"],
        "gate_p50": gate_summary["gate_p50"],
        "gate_p90": gate_summary["gate_p90"],
        "gate_p95": gate_summary["gate_p95"],
        "gate_p99": gate_summary["gate_p99"],
        "mean_effective_degree": gate_summary["mean_effective_degree"],
        "minimum_effective_degree": gate_summary["minimum_effective_degree"],
        "maximum_effective_degree": gate_summary["maximum_effective_degree"],
        "effective_degree_p1": gate_summary["effective_degree_p1"],
        "effective_degree_p5": gate_summary["effective_degree_p5"],
        "effective_degree_p50": gate_summary["effective_degree_p50"],
        "effective_degree_p95": gate_summary["effective_degree_p95"],
        "learned_bias": training_result.learned_bias,
        "rho": args.rho,
        "lambda_budget": args.lambda_budget,
        "g_min": args.g_min,
        "initial_mean_gate": args.initial_mean_gate,
        "temperature": args.temperature,
        "logit_clip": args.logit_clip,
        "boundary_candidate_quantile": args.boundary_candidate_quantile,
        "boundary_candidate_threshold": edge_prior_result.boundary_candidate_threshold,
        "n_boundary_candidate_edges": int(edge_priors["boundary_candidate"].sum()),
        "final_total_loss": training_result.final_losses.get("total_loss"),
        "final_reconstruction_loss": training_result.final_losses.get("reconstruction_loss"),
        "final_budget_loss": training_result.final_losses.get("budget_loss"),
        "device": str(device),
        "seed": args.seed,
        "hidden_dims": [args.hidden_dim, args.latent_dim],
        "warmup_epochs": args.warmup_epochs,
        "gate_epochs": args.epochs if args.epochs is not None else args.gate_epochs,
        "epochs": args.epochs if args.epochs is not None else args.gate_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gate_dim": args.gate_dim,
    }

    output_h5ad = output_dir / f"{args.sample_id}_asg_{variant}_stagate.h5ad"
    adata.write_h5ad(output_h5ad)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()