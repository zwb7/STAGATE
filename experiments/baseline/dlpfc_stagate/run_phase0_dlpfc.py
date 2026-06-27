"""Phase 0 single-seed STAGATE baseline runner for DLPFC sections.

This script keeps the official STAGATE_pyG model path unchanged and writes
the extra artifacts needed before boundary diagnostics:

- config.json
- metrics.json with ARI, NMI, AMI, and silhouette
- baseline h5ad and figures under results/stagate/<section_id>/

Run this script on the remote server only. It trains STAGATE.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import torch

import STAGATE_pyG as STAGATE
from baseline_reporting import (
    args_as_dict,
    compute_clustering_metrics,
    get_runtime_metadata,
    write_json,
)
from preprocessing import LOG_NORMALIZE, PREPROCESS_MODES, preprocess_expression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed Phase 0 STAGATE baseline for one DLPFC section."
    )
    parser.add_argument("--section-id", default="151673")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data1/zhangwenbo/Code/Dataset/LIBD"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/stagate"))
    parser.add_argument("--count-file", default=None)
    parser.add_argument("--truth-file", default=None)
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--radius", type=float, default=150.0)
    parser.add_argument(
        "--preprocess-mode",
        choices=PREPROCESS_MODES,
        default=LOG_NORMALIZE,
    )
    parser.add_argument("--n-top-genes", type=int, default=3000)
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
    parser.add_argument(
        "--mclust-model",
        default="EEE",
        help="mclust covariance model name; the official DLPFC baseline uses EEE.",
    )
    parser.add_argument("--r-home", default=None)
    parser.add_argument("--r-user", default=None)
    parser.add_argument(
        "--skip-paga",
        action="store_true",
        help="Skip spatial trajectory inference and its plot.",
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


def load_dlpfc(
    section_dir: Path,
    section_id: str,
    count_file: str,
    truth_file: Path,
) -> sc.AnnData:
    if not section_dir.exists():
        raise FileNotFoundError(f"DLPFC section directory not found: {section_dir}")
    if not truth_file.exists():
        raise FileNotFoundError(f"Ground-truth annotation not found: {truth_file}")

    adata = sc.read_visium(path=section_dir, count_file=count_file)
    adata.var_names_make_unique()

    annotations = pd.read_csv(
        truth_file,
        sep="\t",
        header=None,
        index_col=0,
        names=["Ground Truth"],
    )
    adata.obs["Ground Truth"] = annotations.reindex(adata.obs_names)["Ground Truth"]
    adata.obs["Ground Truth"] = adata.obs["Ground Truth"].astype("category")
    return adata


def build_config(
    args: argparse.Namespace,
    section_dir: Path,
    count_file: str,
    truth_file: Path,
    device: torch.device,
    preprocessing: dict[str, object] | None,
) -> dict[str, object]:
    training = {
        "hidden_dims": [args.hidden_dim, args.latent_dim],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
    }
    return {
        "stage": "phase0_baseline",
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "scope": "single_seed_only",
        "planned_phase0_sections": ["151673", "151674", "151676"],
        "args": args_as_dict(args),
        "data": {
            "data_root": str(args.data_root),
            "section_id": args.section_id,
            "section_dir": str(section_dir),
            "count_file": count_file,
            "truth_file": str(truth_file),
        },
        "preprocessing": preprocessing,
        "spatial_graph": {
            "model": "Radius",
            "radius": args.radius,
        },
        "training": training,
        "clustering": {
            "method": "mclust",
            "modelNames": args.mclust_model,
            "num_cluster": args.clusters,
            "random_seed": args.seed,
        },
        "runtime": get_runtime_metadata(REPO_ROOT),
    }


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    warnings.filterwarnings("ignore")

    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        os.environ["R_USER"] = args.r_user

    section_dir = args.data_root / args.section_id
    count_file = args.count_file if args.count_file else "filtered_feature_bc_matrix.h5"
    truth_file = (
        Path(args.truth_file)
        if args.truth_file
        else section_dir / f"{args.section_id}_truth.txt"
    )
    output_dir = args.output_dir / args.section_id
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Loading DLPFC section {args.section_id} from {section_dir}")
    adata = load_dlpfc(section_dir, args.section_id, count_file, truth_file)
    print(f"Loaded {adata.n_obs} spots and {adata.n_vars} genes")

    preprocess_expression(adata, args.n_top_genes, mode=args.preprocess_mode)

    sc.pl.spatial(
        adata,
        img_key="hires",
        color="Ground Truth",
        title="Ground Truth",
        show=False,
    )
    save_current_figure(output_dir / "ground_truth_spatial.png")

    STAGATE.Cal_Spatial_Net(adata, rad_cutoff=args.radius)
    STAGATE.Stats_Spatial_Net(adata)
    save_current_figure(output_dir / "spatial_network_stats.png")

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
        modelNames=args.mclust_model,
        random_seed=args.seed,
    )

    clustering_metrics, evaluation = compute_clustering_metrics(
        adata,
        label_key="mclust",
        truth_key="Ground Truth",
        embedding_key="STAGATE",
    )
    ari = clustering_metrics["ari"]
    print(f"Adjusted Rand Index: {ari:.4f}")

    sc.pl.umap(
        adata,
        color=["mclust", "Ground Truth"],
        title=[f"STAGATE (ARI={ari:.2f})", "Ground Truth"],
        show=False,
    )
    save_current_figure(output_dir / "umap_clusters.png")

    sc.pl.spatial(
        adata,
        img_key="hires",
        color=["mclust", "Ground Truth"],
        title=[f"STAGATE (ARI={ari:.2f})", "Ground Truth"],
        show=False,
    )
    save_current_figure(output_dir / "spatial_clusters.png")

    if not args.skip_paga:
        used_adata = adata[adata.obs["Ground Truth"].notna()].copy()
        sc.tl.paga(used_adata, groups="Ground Truth")
        sc.pl.paga_compare(
            used_adata,
            legend_fontsize=10,
            frameon=False,
            size=20,
            title=f"{args.section_id}_STAGATE",
            legend_fontoutline=2,
            show=False,
        )
        save_current_figure(output_dir / "paga_trajectory.png")

    final_loss = adata.uns.get("STAGATE_loss")
    if torch.is_tensor(final_loss):
        final_loss = float(final_loss.detach().cpu())
        adata.uns["STAGATE_loss"] = final_loss

    config = build_config(
        args,
        section_dir=section_dir,
        count_file=count_file,
        truth_file=truth_file,
        device=device,
        preprocessing=adata.uns.get("preprocessing"),
    )
    result = {
        "section_id": args.section_id,
        "method": "STAGATE_official_baseline",
        "phase0_scope": "single_seed_only",
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_evaluated_spots": int(evaluation.shape[0]),
        "n_clusters": args.clusters,
        "preprocess_mode": args.preprocess_mode,
        "n_top_genes": args.n_top_genes,
        **clustering_metrics,
        "final_reconstruction_loss": final_loss,
        "device": str(device),
        "seed": args.seed,
        "hidden_dims": [args.hidden_dim, args.latent_dim],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "spatial_graph": config["spatial_graph"],
        "mclust": config["clustering"],
        "data": config["data"],
        "runtime": config["runtime"],
    }

    adata.write_h5ad(output_dir / f"{args.section_id}_stagate.h5ad")
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "metrics.json", result)

    print(f"Results saved to {output_dir.resolve()}")
    return result


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
