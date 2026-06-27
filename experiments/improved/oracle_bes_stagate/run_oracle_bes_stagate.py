"""Run Oracle-BES-STAGATE feasibility experiments.

This script is intended for the remote server. It may train a model depending
on ``--experiment`` and ``--training-mode``. Do not run it locally for smoke
tests in this repository workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from baseline_reporting import args_as_dict, get_runtime_metadata
from STAGATE_pyG.utils import Transfer_pytorch_Data

from evaluate_oracle_bes import (
    compute_metrics,
    correction_stats,
    flatten_dict,
    label_change_metrics,
    matched_prediction_errors,
    mclust_with_posterior,
    perturbation_metrics,
    summarize_runs_to_markdown,
)
from losses import (
    adjacent_domain_prototype_loss,
    all_domain_prototype_loss,
    compute_prototypes,
    interior_preservation_loss,
    target_indices_for_spots,
)
from model import FrozenEmbeddingShaper, OracleBESSTAGATE
from oracle_boundary import (
    OracleBoundaryData,
    attach_boundary_obs,
    build_oracle_boundary_data,
    random_boundary_mask,
)


EXPERIMENTS = ("O0", "O1", "O2", "O3", "O4")
TRAINING_MODES = ("frozen_adapter", "warmup_last_layer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Oracle-BES-STAGATE feasibility experiments."
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/oracle_bes_stagate"),
    )
    parser.add_argument("--embedding-key", default="STAGATE")
    parser.add_argument("--ground-truth-key", default="Ground Truth")
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--mclust-model", default="EEE")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--r-home", default=None)
    parser.add_argument("--r-user", default=None)
    parser.add_argument(
        "--training-mode",
        choices=TRAINING_MODES,
        default="frozen_adapter",
        help=(
            "frozen_adapter uses the baseline STAGATE embedding from --input-h5ad. "
            "warmup_last_layer retrains an experiment-local STAGATE wrapper from "
            "the input expression matrix."
        ),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--warmup-ratio", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--lambda-bes", type=float, default=0.05)
    parser.add_argument("--lambda-pres", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient-clipping", type=float, default=5.0)
    parser.add_argument("--min-core-spots", type=int, default=5)
    parser.add_argument(
        "--no-save-h5ad",
        action="store_true",
        help="Skip embeddings.h5ad output. The default saves the h5ad artifact.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="After this run, regenerate oracle_bes_summary.md from metrics.csv files.",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, indent=2, ensure_ascii=False)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        with path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(json_ready(payload), file, sort_keys=False)
    except Exception:
        with path.open("w", encoding="utf-8") as file:
            json.dump(json_ready(payload), file, indent=2, ensure_ascii=False)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def validate_input_adata(
    adata: sc.AnnData,
    args: argparse.Namespace,
    require_embedding: bool,
) -> None:
    missing = []
    if "Spatial_Net" not in adata.uns:
        missing.append("adata.uns['Spatial_Net']")
    if args.ground_truth_key not in adata.obs:
        missing.append(f"adata.obs['{args.ground_truth_key}']")
    if require_embedding and args.embedding_key not in adata.obsm:
        missing.append(f"adata.obsm['{args.embedding_key}']")
    if missing:
        raise KeyError("Missing required fields: " + ", ".join(missing))


def all_label_core_masks(boundary_data: OracleBoundaryData) -> dict[int, np.ndarray]:
    return {
        int(label): (boundary_data.labels == label) & boundary_data.valid_mask
        for label in sorted(np.unique(boundary_data.labels[boundary_data.valid_mask]))
    }


def experiment_train_mask(
    args: argparse.Namespace,
    boundary_data: OracleBoundaryData,
) -> np.ndarray:
    if args.experiment == "O1":
        return boundary_data.valid_mask.copy()
    if args.experiment in {"O2", "O3"}:
        return boundary_data.gt_boundary_mask.copy()
    if args.experiment == "O4":
        return random_boundary_mask(
            boundary_data.valid_mask,
            target_count=int(boundary_data.gt_boundary_mask.sum()),
            seed=args.seed,
        )
    raise ValueError(f"No train mask for experiment {args.experiment}")


def experiment_core_masks(
    args: argparse.Namespace,
    boundary_data: OracleBoundaryData,
) -> dict[int, np.ndarray]:
    if args.experiment == "O1":
        return all_label_core_masks(boundary_data)
    return boundary_data.core_masks


def compute_oracle_losses(
    args: argparse.Namespace,
    refined_embedding: torch.Tensor,
    original_embedding: torch.Tensor,
    boundary_data: OracleBoundaryData,
    train_mask: np.ndarray,
    core_masks: dict[int, np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if int(train_mask.sum()) == 0:
        raise ValueError(f"No training spots selected for {args.experiment}")

    prototypes, _, label_to_proto = compute_prototypes(
        refined_embedding,
        boundary_data.labels,
        core_masks,
        device,
    )
    train_indices_np = np.flatnonzero(train_mask)
    train_indices = torch.as_tensor(train_indices_np, dtype=torch.long, device=device)
    targets_np = target_indices_for_spots(
        train_indices_np,
        boundary_data.labels,
        label_to_proto,
    )
    target_proto_indices = torch.as_tensor(targets_np, dtype=torch.long, device=device)

    if args.experiment == "O1":
        loss_bes = all_domain_prototype_loss(
            refined_embedding,
            prototypes,
            train_indices,
            target_proto_indices,
            args.temperature,
        )
    else:
        loss_bes = adjacent_domain_prototype_loss(
            refined_embedding,
            prototypes,
            train_indices_np,
            train_indices,
            target_proto_indices,
            boundary_data.adjacent_negative_labels,
            label_to_proto,
            args.temperature,
        )

    loss_pres = refined_embedding.sum() * 0.0
    if args.experiment == "O3":
        interior_indices = torch.as_tensor(
            np.flatnonzero(boundary_data.gt_interior_mask),
            dtype=torch.long,
            device=device,
        )
        loss_pres = interior_preservation_loss(
            original_embedding,
            refined_embedding,
            interior_indices,
        )
    loss = args.lambda_bes * loss_bes + args.lambda_pres * loss_pres
    return loss, {
        "loss_bes": float(loss_bes.detach().cpu()),
        "loss_pres": float(loss_pres.detach().cpu()),
    }


def train_frozen_adapter(
    args: argparse.Namespace,
    adata: sc.AnnData,
    boundary_data: OracleBoundaryData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | None]]]:
    original_embedding_np = np.asarray(adata.obsm[args.embedding_key], dtype=np.float32)
    original_embedding = torch.as_tensor(
        original_embedding_np,
        dtype=torch.float32,
        device=device,
    )
    shaper = FrozenEmbeddingShaper(
        original_embedding_np.shape[1],
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        shaper.parameters(),
        lr=args.adapter_learning_rate,
        weight_decay=0.0,
    )
    train_mask = experiment_train_mask(args, boundary_data)
    core_masks = experiment_core_masks(args, boundary_data)

    training_log: list[dict[str, float | int | None]] = []
    for epoch in range(1, args.epochs + 1):
        shaper.train()
        optimizer.zero_grad()
        refined = shaper(original_embedding, gamma=args.gamma)
        oracle_loss, loss_parts = compute_oracle_losses(
            args,
            refined,
            original_embedding,
            boundary_data,
            train_mask,
            core_masks,
            device,
        )
        oracle_loss.backward()
        torch.nn.utils.clip_grad_norm_(shaper.parameters(), args.gradient_clipping)
        optimizer.step()
        training_log.append(
            {
                "epoch": epoch,
                "loss": float(oracle_loss.detach().cpu()),
                "loss_rec": None,
                **loss_parts,
            }
        )

    shaper.eval()
    with torch.no_grad():
        refined_np = shaper(original_embedding, gamma=args.gamma).detach().cpu().numpy()
    return original_embedding_np, refined_np, training_log


def prepare_pyg_data(adata: sc.AnnData) -> Any:
    adata.X = sp.csr_matrix(adata.X)
    if "highly_variable" in adata.var.columns:
        used_adata = adata[:, adata.var["highly_variable"]].copy()
    else:
        used_adata = adata
    return Transfer_pytorch_Data(used_adata)


def train_warmup_last_layer(
    args: argparse.Namespace,
    adata: sc.AnnData,
    boundary_data: OracleBoundaryData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | None]]]:
    data = prepare_pyg_data(adata).to(device)
    model = OracleBESSTAGATE(
        hidden_dims=[data.x.shape[1], args.hidden_dim, args.latent_dim],
        gamma=args.gamma,
        dropout=args.dropout,
    ).to(device)
    warmup_epochs = int(round(args.epochs * args.warmup_ratio))
    warmup_epochs = max(0, min(args.epochs, warmup_epochs))
    joint_epochs = args.epochs - warmup_epochs
    training_log: list[dict[str, float | int | None]] = []

    warmup_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    for epoch in range(1, warmup_epochs + 1):
        model.train()
        warmup_optimizer.zero_grad()
        _, _, reconstruction = model(data.x, data.edge_index, apply_shaping=False)
        loss_rec = F.mse_loss(data.x, reconstruction)
        loss_rec.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
        warmup_optimizer.step()
        training_log.append(
            {
                "epoch": epoch,
                "stage": "warmup",
                "loss": float(loss_rec.detach().cpu()),
                "loss_rec": float(loss_rec.detach().cpu()),
                "loss_bes": None,
                "loss_pres": None,
            }
        )

    model.freeze_except_last_encoder_and_shaping()
    joint_optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_mask = experiment_train_mask(args, boundary_data)
    core_masks = experiment_core_masks(args, boundary_data)

    for step in range(1, joint_epochs + 1):
        epoch = warmup_epochs + step
        model.train()
        joint_optimizer.zero_grad()
        embedding, refined, reconstruction = model(
            data.x,
            data.edge_index,
            apply_shaping=True,
        )
        loss_rec = F.mse_loss(data.x, reconstruction)
        oracle_loss, loss_parts = compute_oracle_losses(
            args,
            refined,
            embedding,
            boundary_data,
            train_mask,
            core_masks,
            device,
        )
        loss = loss_rec + oracle_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            args.gradient_clipping,
        )
        joint_optimizer.step()
        training_log.append(
            {
                "epoch": epoch,
                "stage": "joint",
                "loss": float(loss.detach().cpu()),
                "loss_rec": float(loss_rec.detach().cpu()),
                **loss_parts,
            }
        )

    model.eval()
    with torch.no_grad():
        embedding, refined, _ = model(data.x, data.edge_index, apply_shaping=True)
    return (
        embedding.detach().cpu().numpy(),
        refined.detach().cpu().numpy(),
        training_log,
    )


def run_mclust_labels(
    embedding: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    return mclust_with_posterior(
        embedding,
        num_cluster=args.clusters,
        model_names=args.mclust_model,
        random_seed=args.seed,
    )


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "stage": "oracle_bes_stagate",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "scope": "oracle_feasibility_sanity_check",
        "args": args_as_dict(args),
        "oracle_usage": {
            "ground_truth_used_for": "oracle boundary/prototype loss only",
            "ground_truth_not_used_for": "mclust tuning or cluster count search",
        },
        "runtime": get_runtime_metadata(REPO_ROOT),
    }


def save_outputs(
    args: argparse.Namespace,
    adata: sc.AnnData,
    output_dir: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    training_log: list[dict[str, Any]],
    boundary_data: OracleBoundaryData,
    baseline_labels: np.ndarray,
    refined_labels: np.ndarray,
    baseline_errors: np.ndarray,
    refined_errors: np.ndarray,
    original_embedding: np.ndarray,
    refined_embedding: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(output_dir / "config.yaml", config)
    write_json(output_dir / "config.json", config)
    pd.DataFrame([flatten_dict(metrics)]).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(training_log).to_csv(output_dir / "training_log.csv", index=False)

    boundary_stats = {
        "n_spots": int(adata.n_obs),
        "n_valid_gt_spots": int(boundary_data.valid_mask.sum()),
        "n_gt_boundary_spots": int(boundary_data.gt_boundary_mask.sum()),
        "n_gt_interior_spots": int(boundary_data.gt_interior_mask.sum()),
        "gt_boundary_score_mean": float(boundary_data.gt_boundary_score.mean()),
        "gt_boundary_score_valid_mean": (
            float(boundary_data.gt_boundary_score[boundary_data.valid_mask].mean())
            if boundary_data.valid_mask.any()
            else None
        ),
    }
    pd.DataFrame([boundary_stats]).to_csv(output_dir / "boundary_stats.csv", index=False)

    correction = correction_stats(
        boundary_data.labels,
        baseline_labels,
        refined_labels,
        boundary_data.valid_mask,
    )
    pd.DataFrame([correction]).to_csv(output_dir / "correction_stats.csv", index=False)

    changed = baseline_labels != refined_labels
    labels = pd.DataFrame(
        {
            "spot_id": adata.obs_names,
            "gt_label": adata.obs["gt_label"].astype(str).to_numpy(),
            "gt_boundary_score": boundary_data.gt_boundary_score,
            "is_gt_boundary": boundary_data.gt_boundary_mask,
            "is_gt_interior": boundary_data.gt_interior_mask,
            "stagate_mclust": baseline_labels,
            "oracle_bes_mclust": refined_labels,
            "changed_label": changed,
            "correct_before": (~baseline_errors) & boundary_data.valid_mask,
            "correct_after": (~refined_errors) & boundary_data.valid_mask,
            "embedding_l2_perturbation": np.linalg.norm(
                refined_embedding - original_embedding,
                axis=1,
            ),
        }
    )
    labels.to_csv(output_dir / "labels.csv", index=False)

    adata.obsm["STAGATE"] = original_embedding
    adata.obsm["Oracle_BES_STAGATE"] = refined_embedding
    adata.obs["stagate_mclust"] = pd.Categorical(baseline_labels.astype(str))
    adata.obs["oracle_bes_mclust"] = pd.Categorical(refined_labels.astype(str))
    adata.obs["changed_label"] = changed
    adata.obs["correct_before"] = (~baseline_errors) & boundary_data.valid_mask
    adata.obs["correct_after"] = (~refined_errors) & boundary_data.valid_mask
    if not args.no_save_h5ad:
        adata.write_h5ad(output_dir / "embeddings.h5ad")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
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
    require_embedding = args.experiment == "O0" or args.training_mode == "frozen_adapter"
    validate_input_adata(adata, args, require_embedding=require_embedding)

    boundary_data = build_oracle_boundary_data(
        adata,
        ground_truth_key=args.ground_truth_key,
        min_core_spots=args.min_core_spots,
    )
    attach_boundary_obs(adata, boundary_data, args.ground_truth_key)

    if args.embedding_key in adata.obsm:
        baseline_embedding = np.asarray(adata.obsm[args.embedding_key], dtype=np.float32)
    else:
        baseline_embedding = None

    if baseline_embedding is None:
        baseline_labels = np.full(adata.n_obs, -1, dtype=np.int64)
    else:
        baseline_mclust = run_mclust_labels(baseline_embedding, args)
        baseline_labels = baseline_mclust["labels"]

    training_log: list[dict[str, Any]] = []
    if args.experiment == "O0":
        if baseline_embedding is None:
            raise KeyError(f"O0 requires adata.obsm['{args.embedding_key}']")
        original_embedding = baseline_embedding
        refined_embedding = baseline_embedding.copy()
        refined_labels = baseline_labels.copy()
    elif args.training_mode == "frozen_adapter":
        original_embedding, refined_embedding, training_log = train_frozen_adapter(
            args,
            adata,
            boundary_data,
            device,
        )
        refined_labels = run_mclust_labels(refined_embedding, args)["labels"]
    else:
        original_embedding, refined_embedding, training_log = train_warmup_last_layer(
            args,
            adata,
            boundary_data,
            device,
        )
        refined_labels = run_mclust_labels(refined_embedding, args)["labels"]
        if baseline_embedding is None:
            baseline_mclust = run_mclust_labels(original_embedding, args)
            baseline_labels = baseline_mclust["labels"]

    baseline_errors = matched_prediction_errors(
        boundary_data.labels,
        baseline_labels,
        boundary_data.valid_mask,
    )
    refined_errors = matched_prediction_errors(
        boundary_data.labels,
        refined_labels,
        boundary_data.valid_mask,
    )

    clustering = compute_metrics(
        refined_embedding,
        boundary_data.labels,
        refined_labels,
        boundary_data.valid_mask,
        boundary_data.gt_boundary_mask,
        boundary_data.gt_interior_mask,
    )
    perturbation = perturbation_metrics(
        original_embedding,
        refined_embedding,
        boundary_data.gt_boundary_mask,
        boundary_data.gt_interior_mask,
    )
    label_changes = label_change_metrics(
        baseline_labels,
        refined_labels,
        boundary_data.valid_mask,
        boundary_data.gt_boundary_mask,
        boundary_data.gt_interior_mask,
    )
    corrections = correction_stats(
        boundary_data.labels,
        baseline_labels,
        refined_labels,
        boundary_data.valid_mask,
    )
    metrics: dict[str, Any] = {
        "sample_id": args.sample_id,
        "experiment": args.experiment,
        "training_mode": args.training_mode,
        "input_h5ad": str(args.input_h5ad),
        "n_spots": int(adata.n_obs),
        "n_valid_gt_spots": int(boundary_data.valid_mask.sum()),
        "n_gt_boundary_spots": int(boundary_data.gt_boundary_mask.sum()),
        "n_gt_interior_spots": int(boundary_data.gt_interior_mask.sum()),
        "seed": args.seed,
        "clusters": args.clusters,
        "mclust_model": args.mclust_model,
        **clustering,
        **perturbation,
        **label_changes,
        **corrections,
        "final_loss": training_log[-1]["loss"] if training_log else None,
        "final_loss_rec": training_log[-1].get("loss_rec") if training_log else None,
        "final_loss_bes": training_log[-1].get("loss_bes") if training_log else None,
        "final_loss_pres": training_log[-1].get("loss_pres") if training_log else None,
    }

    run_dir = args.output_dir / args.sample_id / f"seed_{args.seed}" / args.experiment
    config = build_config(args)
    save_outputs(
        args,
        adata,
        run_dir,
        config,
        metrics,
        training_log,
        boundary_data,
        baseline_labels,
        refined_labels,
        baseline_errors,
        refined_errors,
        original_embedding,
        refined_embedding,
    )
    if args.summary:
        summarize_runs_to_markdown(
            args.output_dir,
            args.output_dir / "oracle_bes_summary.md",
        )
    print(f"Oracle-BES-STAGATE results saved to {run_dir.resolve()}")
    return metrics


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
