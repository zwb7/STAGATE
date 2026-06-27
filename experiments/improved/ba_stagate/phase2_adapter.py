"""Phase 2 frozen post-hoc BA-STAGATE adapter.

This script trains a small residual MLP on fixed STAGATE embeddings. It does
not update STAGATE_pyG, does not reconstruct gene expression, and does not use
ground-truth labels for training. Ground truth is used only for evaluation.
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
import torch.nn.functional as F
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from baseline_reporting import args_as_dict, get_runtime_metadata, write_json
from mclust_posterior import mclust_with_posterior
from phase1_boundary_diagnostics import (
    build_neighbor_lists,
    encode_ground_truth,
    matched_prediction_errors,
    neighbor_disagreement,
)


EXPERIMENT_MODES = ("all_spot", "boundary_only", "boundary_adjacent")


class ResidualAdapter(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, embedding: torch.Tensor, gamma: float) -> torch.Tensor:
        return embedding + gamma * self.net(embedding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen post-hoc BA-STAGATE residual adapter."
    )
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--boundary-scores", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--experiment",
        choices=EXPERIMENT_MODES,
        required=True,
        help=(
            "all_spot=E1, boundary_only=E2, "
            "boundary_adjacent=E3 adjacent hard negatives"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ba_stagate/phase2_adapter"),
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


def load_boundary_scores(path: Path, obs_names: pd.Index) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Boundary score file not found: {path}")
    scores = pd.read_csv(path)
    required = {
        "spot_id",
        "pseudo_label",
        "pseudo_confidence",
        "combined_boundary_score",
        "pseudo_boundary",
        "gt_boundary",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"boundary_scores.csv is missing columns: {missing}")

    scores = scores.set_index("spot_id")
    missing_spots = obs_names.difference(scores.index)
    if len(missing_spots) > 0:
        raise ValueError(
            "boundary_scores.csv does not cover all AnnData spots; "
            f"first missing spot: {missing_spots[0]}"
        )
    return scores.loc[obs_names].reset_index()


def top_ratio_mask(scores: np.ndarray, ratio: float) -> np.ndarray:
    if not 0 < ratio < 1:
        raise ValueError("--boundary-ratio must be between 0 and 1")
    n_selected = max(1, int(np.ceil(scores.shape[0] * ratio)))
    selected = np.argsort(scores, kind="mergesort")[-n_selected:]
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[selected] = True
    return mask


def encode_pseudo_labels(raw_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_labels = np.asarray(sorted(pd.unique(raw_labels)))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    encoded = np.asarray([label_to_index[label] for label in raw_labels], dtype=np.int64)
    return encoded, unique_labels


def select_core_masks(
    labels: np.ndarray,
    confidence: np.ndarray,
    boundary_score: np.ndarray,
    args: argparse.Namespace,
) -> dict[int, np.ndarray]:
    core_masks: dict[int, np.ndarray] = {}
    for label in sorted(np.unique(labels)):
        cluster_mask = labels == label
        cluster_indices = np.flatnonzero(cluster_mask)
        if cluster_indices.size == 0:
            continue

        score_cutoff = np.quantile(
            boundary_score[cluster_indices],
            args.core_bottom_quantile,
        )
        core_mask = (
            cluster_mask
            & (confidence > args.tau_core)
            & (boundary_score <= score_cutoff)
        )
        if core_mask.sum() < args.min_core_spots:
            n_fallback = max(
                args.min_core_spots,
                int(np.ceil(cluster_indices.size * args.fallback_core_ratio)),
            )
            n_fallback = min(n_fallback, cluster_indices.size)
            order = np.argsort(confidence[cluster_indices], kind="mergesort")
            fallback_indices = cluster_indices[order[-n_fallback:]]
            core_mask = np.zeros(labels.shape[0], dtype=bool)
            core_mask[fallback_indices] = True
        core_masks[int(label)] = core_mask
    return core_masks


def compute_prototypes(
    embedding: torch.Tensor,
    core_masks: dict[int, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, list[int], dict[int, int]]:
    labels = sorted(core_masks)
    prototypes = []
    for label in labels:
        index = torch.as_tensor(np.flatnonzero(core_masks[label]), device=device)
        if index.numel() == 0:
            raise ValueError(f"No core spots available for pseudo cluster {label}")
        prototypes.append(embedding.index_select(0, index).mean(dim=0))
    prototype_tensor = torch.stack(prototypes, dim=0)
    label_to_proto = {label: index for index, label in enumerate(labels)}
    return prototype_tensor, labels, label_to_proto


def select_train_mask(
    experiment: str,
    confidence: np.ndarray,
    boundary_mask: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    high_confidence = confidence > args.tau_train
    if experiment == "all_spot":
        return high_confidence
    return boundary_mask & high_confidence


def adjacent_negative_indices(
    train_indices: np.ndarray,
    labels: np.ndarray,
    neighbors: list[np.ndarray],
    label_to_proto: dict[int, int],
) -> list[list[int]]:
    all_proto_indices = set(label_to_proto.values())
    result: list[list[int]] = []
    for spot_index in train_indices:
        own_label = int(labels[spot_index])
        adjacent_labels = {
            int(labels[neighbor])
            for neighbor in neighbors[spot_index]
            if int(labels[neighbor]) != own_label
        }
        negative_indices = [
            label_to_proto[label]
            for label in sorted(adjacent_labels)
            if label in label_to_proto
        ]
        if not negative_indices:
            own_proto = label_to_proto[own_label]
            negative_indices = sorted(all_proto_indices.difference({own_proto}))
        result.append(negative_indices)
    return result


def prototype_loss(
    z_shaped: torch.Tensor,
    prototypes: torch.Tensor,
    train_indices: torch.Tensor,
    train_proto_targets: torch.Tensor,
    temperature: float,
    adjacent_negatives: list[list[int]] | None = None,
) -> torch.Tensor:
    z_train = F.normalize(z_shaped.index_select(0, train_indices), dim=1)
    normalized_prototypes = F.normalize(prototypes, dim=1)

    if adjacent_negatives is None:
        logits = z_train @ normalized_prototypes.T / temperature
        return F.cross_entropy(logits, train_proto_targets)

    losses = []
    for row_index, negative_proto_indices in enumerate(adjacent_negatives):
        positive_proto = int(train_proto_targets[row_index].item())
        candidate_indices = [positive_proto] + [
            index for index in negative_proto_indices if index != positive_proto
        ]
        candidate = torch.as_tensor(
            candidate_indices,
            dtype=torch.long,
            device=z_shaped.device,
        )
        logits = (
            z_train[row_index : row_index + 1]
            @ normalized_prototypes.index_select(0, candidate).T
            / temperature
        )
        target = torch.zeros(1, dtype=torch.long, device=z_shaped.device)
        losses.append(F.cross_entropy(logits, target))
    return torch.stack(losses).mean()


def preservation_loss(
    original: torch.Tensor,
    shaped: torch.Tensor,
    preservation_weights: torch.Tensor,
) -> torch.Tensor:
    per_spot = torch.sum((shaped - original) ** 2, dim=1)
    return torch.mean(preservation_weights * per_spot)


def evaluate_labels(
    embedding: np.ndarray,
    labels: np.ndarray,
    gt_labels: np.ndarray,
    evaluated: np.ndarray,
    gt_boundary: np.ndarray,
    pseudo_boundary: np.ndarray,
) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "ari": None,
        "nmi": None,
        "ami": None,
        "silhouette": None,
        "gt_boundary_ari": None,
        "gt_boundary_nmi": None,
        "gt_interior_ari": None,
        "gt_interior_nmi": None,
        "pseudo_boundary_ari": None,
        "pseudo_interior_ari": None,
    }
    if not evaluated.any():
        return metrics

    metrics["ari"] = float(adjusted_rand_score(gt_labels[evaluated], labels[evaluated]))
    metrics["nmi"] = float(
        normalized_mutual_info_score(gt_labels[evaluated], labels[evaluated])
    )
    metrics["ami"] = float(adjusted_mutual_info_score(gt_labels[evaluated], labels[evaluated]))
    unique_labels = np.unique(labels[evaluated])
    if 1 < unique_labels.size < evaluated.sum():
        metrics["silhouette"] = float(silhouette_score(embedding[evaluated], labels[evaluated]))

    gt_interior = (~gt_boundary) & evaluated
    pseudo_interior = (~pseudo_boundary) & evaluated
    for prefix, mask in [
        ("gt_boundary", gt_boundary),
        ("gt_interior", gt_interior),
        ("pseudo_boundary", pseudo_boundary & evaluated),
        ("pseudo_interior", pseudo_interior),
    ]:
        if mask.sum() > 1:
            metrics[f"{prefix}_ari"] = float(adjusted_rand_score(gt_labels[mask], labels[mask]))
            if prefix.startswith("gt_"):
                metrics[f"{prefix}_nmi"] = float(
                    normalized_mutual_info_score(gt_labels[mask], labels[mask])
                )
    return metrics


def perturbation_summary(
    original: np.ndarray,
    shaped: np.ndarray,
    gt_boundary: np.ndarray,
    pseudo_boundary: np.ndarray,
    baseline_labels: np.ndarray,
    ba_labels: np.ndarray,
    evaluated: np.ndarray,
) -> dict[str, float | None]:
    perturbation = np.linalg.norm(shaped - original, axis=1)
    interior = (~gt_boundary) & evaluated
    pseudo_interior = (~pseudo_boundary) & evaluated
    return {
        "mean_l2": float(perturbation.mean()),
        "gt_boundary_mean_l2": (
            float(perturbation[gt_boundary].mean()) if gt_boundary.any() else None
        ),
        "gt_interior_mean_l2": (
            float(perturbation[interior].mean()) if interior.any() else None
        ),
        "pseudo_boundary_mean_l2": (
            float(perturbation[pseudo_boundary].mean()) if pseudo_boundary.any() else None
        ),
        "pseudo_interior_mean_l2": (
            float(perturbation[pseudo_interior].mean()) if pseudo_interior.any() else None
        ),
        "interior_changed_label_ratio": (
            float(np.mean(baseline_labels[interior] != ba_labels[interior]))
            if interior.any()
            else None
        ),
        "pseudo_interior_changed_label_ratio": (
            float(np.mean(baseline_labels[pseudo_interior] != ba_labels[pseudo_interior]))
            if pseudo_interior.any()
            else None
        ),
    }


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
    adapter = ResidualAdapter(original_embedding.shape[1], args.dropout).to(device)
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
        shaped = adapter(embedding_tensor, args.gamma)
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

        log_row = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "loss_proto": float(loss_proto.detach().cpu()),
            "loss_pres": float(loss_pres.detach().cpu()),
        }
        training_log.append(log_row)
        if log_row["loss"] < best_loss - 1e-7:
            best_loss = log_row["loss"]
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
        shaped_embedding = adapter(embedding_tensor, args.gamma).detach().cpu().numpy()

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
    pseudo_error = matched_prediction_errors(gt_labels, ba_labels, evaluated)
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
    pd.DataFrame(
        {
            "spot_id": adata.obs_names,
            "pseudo_label": pseudo_labels_raw,
            "ba_mclust": ba_labels,
            "ba_confidence": ba_mclust["confidence"],
            "pseudo_boundary": boundary_mask,
            "gt_boundary": gt_boundary,
            "baseline_error_after_hungarian": baseline_error,
            "ba_error_after_hungarian": pseudo_error,
            "embedding_l2_perturbation": np.linalg.norm(
                shaped_embedding - original_embedding,
                axis=1,
            ),
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
        "method": "BA_STAGATE_posthoc_adapter",
        "input_h5ad": str(args.input_h5ad),
        "boundary_scores": str(args.boundary_scores),
        "n_spots": int(adata.n_obs),
        "n_train_spots": int(train_mask.sum()),
        "n_boundary_spots": int(boundary_mask.sum()),
        "n_gt_boundary_spots": int(gt_boundary.sum()),
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
            "ba_overall": float(pseudo_error[evaluated].mean()) if evaluated.any() else None,
            "ba_gt_boundary": (
                float(pseudo_error[gt_boundary].mean()) if gt_boundary.any() else None
            ),
            "ba_gt_interior": (
                float(pseudo_error[(~gt_boundary) & evaluated].mean())
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
    print(f"Phase 2 adapter results saved to {output_dir.resolve()}")
    return result


def main() -> None:
    train_adapter(parse_args())


if __name__ == "__main__":
    main()
