"""Phase 2 boundary-gated BA-STAGATE adapter.

This is the recommended replacement for the first smoke-test adapter. The
residual update is gated per spot:

    z'_i = z_i + gamma * g_i * MLP(LayerNorm(z_i))

For ``all_spot`` the gate is one for every spot. For ``boundary_only`` and
``boundary_adjacent`` the gate is one only on the selected boundary set and
zero on pseudo interior spots. STAGATE embeddings remain frozen.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn

from baseline_reporting import args_as_dict, get_runtime_metadata, write_json
from mclust_posterior import mclust_with_posterior
from phase1_boundary_diagnostics import (
    build_neighbor_lists,
    encode_ground_truth,
    matched_prediction_errors,
)
from phase2_adapter import (
    EXPERIMENT_MODES,
    adjacent_negative_indices,
    compute_prototypes,
    encode_pseudo_labels,
    evaluate_labels,
    load_boundary_scores,
    preservation_loss,
    prototype_loss,
    select_core_masks,
    select_train_mask,
    top_ratio_mask,
)
from phase2_adapter_aligned import perturbation_summary


class GatedResidualAdapter(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        embedding: torch.Tensor,
        gamma: float,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        return embedding + gamma * gate[:, None] * self.net(embedding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen boundary-gated BA-STAGATE adapter."
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--boundary-scores", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--experiment", choices=EXPERIMENT_MODES, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ba_stagate/phase2_adapter_gated"),
    )
    parser.add_argument("--embedding-key", default="STAGATE")
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--mclust-model", default="EEE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lambda-pres", type=float, default=1.0)
    parser.add_argument("--rho-boundary", type=float, default=1.0)
    parser.add_argument("--rho-interior", type=float, default=5.0)
    parser.add_argument("--boundary-ratio", type=float, default=0.2)
    parser.add_argument("--tau-train", type=float, default=0.6)
    parser.add_argument("--tau-core", type=float, default=0.8)
    parser.add_argument("--core-bottom-quantile", type=float, default=0.5)
    parser.add_argument("--fallback-core-ratio", type=float, default=0.2)
    parser.add_argument("--min-core-spots", type=int, default=5)
    parser.add_argument("--r-home", default=None)
    parser.add_argument("--r-user", default=None)
    parser.add_argument(
        "--save-h5ad",
        action="store_true",
        help="Save an h5ad with BA_STAGATE embedding. Disabled by default.",
    )
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def gate_for_experiment(experiment: str, boundary_mask: np.ndarray) -> np.ndarray:
    if experiment == "all_spot":
        return np.ones(boundary_mask.shape[0], dtype=np.float32)
    return boundary_mask.astype(np.float32)


def train_adapter(args: argparse.Namespace) -> dict[str, object]:
    warnings.filterwarnings("ignore")
    set_random_seed(args.seed)
    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        os.environ["R_USER"] = args.r_user
    if not args.input_h5ad.exists():
        raise FileNotFoundError(f"Input h5ad not found: {args.input_h5ad}")

    device = resolve_device(args.device)
    adata = sc.read_h5ad(args.input_h5ad)
    if args.embedding_key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm['{args.embedding_key}']")
    if "Spatial_Net" not in adata.uns:
        raise KeyError("Missing adata.uns['Spatial_Net']")
    if args.ground_truth_key not in adata.obs:
        raise KeyError(f"Missing adata.obs['{args.ground_truth_key}']")

    score_table = load_boundary_scores(args.boundary_scores, adata.obs_names)
    original_embedding = np.asarray(adata.obsm[args.embedding_key], dtype=np.float32)
    pseudo_labels_raw = score_table["pseudo_label"].to_numpy()
    pseudo_labels, original_label_values = encode_pseudo_labels(pseudo_labels_raw)
    confidence = score_table["pseudo_confidence"].to_numpy(dtype=np.float64)
    boundary_score = score_table["combined_boundary_score"].to_numpy(dtype=np.float64)
    boundary_mask = top_ratio_mask(boundary_score, args.boundary_ratio)
    residual_gate = gate_for_experiment(args.experiment, boundary_mask)
    gt_boundary = score_table["gt_boundary"].astype(bool).to_numpy()
    gt_labels, evaluated = encode_ground_truth(adata.obs[args.ground_truth_key])

    core_masks = select_core_masks(pseudo_labels, confidence, boundary_score, args)
    train_mask = select_train_mask(args.experiment, confidence, boundary_mask, args)
    if train_mask.sum() == 0:
        raise ValueError("No training spots selected; lower --tau-train or inspect scores.")

    missing_train_labels = sorted(set(np.unique(pseudo_labels[train_mask])).difference(core_masks))
    if missing_train_labels:
        raise ValueError(f"Missing prototypes for train labels: {missing_train_labels}")

    embedding_tensor = torch.as_tensor(original_embedding, dtype=torch.float32, device=device)
    gate_tensor = torch.as_tensor(residual_gate, dtype=torch.float32, device=device)
    adapter = GatedResidualAdapter(original_embedding.shape[1], args.dropout).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.adapter_lr)
    train_indices_np = np.flatnonzero(train_mask)
    train_indices = torch.as_tensor(train_indices_np, dtype=torch.long, device=device)
    preservation_weights = np.where(
        boundary_mask,
        args.rho_boundary,
        args.rho_interior,
    ).astype(np.float32)
    preservation_weights_tensor = torch.as_tensor(
        preservation_weights,
        dtype=torch.float32,
        device=device,
    )
    spatial_neighbors = build_neighbor_lists(adata.uns["Spatial_Net"], adata.obs_names)

    with torch.no_grad():
        prototypes, prototype_labels, label_to_proto = compute_prototypes(
            embedding_tensor,
            core_masks,
            device,
        )
    train_targets_np = np.asarray(
        [label_to_proto[int(label)] for label in pseudo_labels[train_indices_np]],
        dtype=np.int64,
    )
    train_targets = torch.as_tensor(train_targets_np, dtype=torch.long, device=device)
    adjacent_negatives = None
    if args.experiment == "boundary_adjacent":
        adjacent_negatives = adjacent_negative_indices(
            train_indices_np,
            pseudo_labels,
            spatial_neighbors,
            label_to_proto,
        )

    training_log: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        optimizer.zero_grad()
        shaped = adapter(embedding_tensor, args.gamma, gate_tensor)
        loss_proto = prototype_loss(
            shaped,
            prototypes,
            train_indices,
            train_targets,
            args.temperature,
            adjacent_negatives=adjacent_negatives,
        )
        loss_pres = preservation_loss(embedding_tensor, shaped, preservation_weights_tensor)
        loss = loss_proto + args.lambda_pres * loss_pres
        loss.backward()
        optimizer.step()

        row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "loss_proto": float(loss_proto.detach().cpu()),
            "loss_pres": float(loss_pres.detach().cpu()),
        }
        training_log.append(row)
        if row["loss"] < best_loss - 1e-7:
            best_loss = row["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in adapter.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.early_stop_patience:
            break

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.eval()
    with torch.no_grad():
        shaped_embedding = (
            adapter(embedding_tensor, args.gamma, gate_tensor)
            .detach()
            .cpu()
            .numpy()
        )

    ba_mclust = mclust_with_posterior(
        shaped_embedding,
        num_cluster=args.clusters,
        model_names=args.mclust_model,
        random_seed=args.seed,
    )
    ba_labels = ba_mclust["labels"]
    baseline_metrics = evaluate_labels(
        original_embedding,
        pseudo_labels,
        gt_labels,
        evaluated,
        gt_boundary,
        boundary_mask,
    )
    ba_metrics = evaluate_labels(
        shaped_embedding,
        ba_labels,
        gt_labels,
        evaluated,
        gt_boundary,
        boundary_mask,
    )
    ba_error = matched_prediction_errors(gt_labels, ba_labels, evaluated)
    baseline_error = matched_prediction_errors(gt_labels, pseudo_labels, evaluated)
    perturbation = perturbation_summary(
        original_embedding,
        shaped_embedding,
        gt_boundary,
        boundary_mask,
        pseudo_labels,
        ba_labels,
        evaluated,
    )

    output_dir = args.output_dir / args.sample_id / f"seed_{args.seed}" / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training_log).to_csv(output_dir / "training_log.csv", index=False)
    spot_perturbation = np.linalg.norm(shaped_embedding - original_embedding, axis=1)
    pd.DataFrame(
        {
            "spot_id": adata.obs_names,
            "pseudo_label": pseudo_labels_raw,
            "ba_mclust": ba_labels,
            "ba_confidence": ba_mclust["confidence"],
            "pseudo_boundary": boundary_mask,
            "residual_gate": residual_gate,
            "gt_boundary": gt_boundary,
            "baseline_error_after_hungarian": baseline_error,
            "ba_error_after_hungarian": ba_error,
            "embedding_l2_perturbation": spot_perturbation,
        }
    ).to_csv(output_dir / "spot_level_results.csv", index=False)
    np.save(output_dir / "BA_STAGATE.npy", shaped_embedding)
    np.save(output_dir / "ba_mclust_posterior.npy", ba_mclust["posterior"])
    torch.save(adapter.state_dict(), output_dir / "adapter_state.pt")

    if args.save_h5ad:
        adata.obsm["BA_STAGATE"] = shaped_embedding
        adata.obs["ba_stagate_mclust"] = pd.Categorical(ba_labels.astype(str))
        adata.write_h5ad(output_dir / f"{args.sample_id}_{args.experiment}_ba_stagate.h5ad")

    result = {
        "sample_id": args.sample_id,
        "experiment": args.experiment,
        "method": "BA_STAGATE_boundary_gated_posthoc_adapter",
        "input_h5ad": str(args.input_h5ad),
        "boundary_scores": str(args.boundary_scores),
        "n_spots": int(adata.n_obs),
        "n_train_spots": int(train_mask.sum()),
        "n_boundary_spots": int(boundary_mask.sum()),
        "n_gt_boundary_spots": int(gt_boundary.sum()),
        "residual_gate": {
            "mode": "all_spots" if args.experiment == "all_spot" else "boundary_only",
            "active_spots": int((residual_gate > 0).sum()),
        },
        "prototype_labels": [int(label) for label in prototype_labels],
        "core_spots_per_cluster": {
            str(original_label_values[label]): int(mask.sum())
            for label, mask in core_masks.items()
        },
        "config": args_as_dict(args),
        "baseline_metrics_from_phase1_pseudo_labels": baseline_metrics,
        "ba_metrics": ba_metrics,
        "metric_delta": {
            key: (
                ba_metrics[key] - baseline_metrics[key]
                if ba_metrics.get(key) is not None and baseline_metrics.get(key) is not None
                else None
            )
            for key in ba_metrics
        },
        "error_rates_after_hungarian": {
            "baseline_overall": (
                float(baseline_error[evaluated].mean()) if evaluated.any() else None
            ),
            "ba_overall": float(ba_error[evaluated].mean()) if evaluated.any() else None,
            "ba_gt_boundary": (
                float(ba_error[gt_boundary].mean()) if gt_boundary.any() else None
            ),
            "ba_gt_interior": (
                float(ba_error[(~gt_boundary) & evaluated].mean())
                if ((~gt_boundary) & evaluated).any()
                else None
            ),
        },
        "perturbation": perturbation,
        "epochs_ran": int(training_log[-1]["epoch"]) if training_log else 0,
        "best_loss": best_loss,
        "runtime": get_runtime_metadata(REPO_ROOT),
    }
    write_json(output_dir / "metrics_phase2.json", result)
    print(f"Gated Phase 2 adapter results saved to {output_dir.resolve()}")
    return result


def main() -> None:
    train_adapter(parse_args())


if __name__ == "__main__":
    main()
