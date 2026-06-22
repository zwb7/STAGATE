"""Train and evaluate STAGATE on a 10x Visium DLPFC section.

Expected data layout:

Data/
└── 151676/
    ├── 151676_filtered_feature_bc_matrix.h5
    ├── 151676_truth.txt
    └── spatial/

The evaluation is unsupervised: STAGATE is trained on all spots, and the
resulting clusters are compared with the available manual annotations by ARI.
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
import pandas as pd
import scanpy as sc
import torch
from sklearn.metrics import adjusted_rand_score

import STAGATE_pyG as STAGATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate STAGATE_pyG on one DLPFC section."
    )
    parser.add_argument("--section-id", default="151676")
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dlpfc"))
    parser.add_argument("--count-file", default=None)
    parser.add_argument("--truth-file", default=None)
    parser.add_argument("--clusters", type=int, default=7)
    parser.add_argument("--radius", type=float, default=150.0)
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        default="cuda:7",
        help="PyTorch device, for example cpu, cuda:0, or auto.",
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


def preprocess(adata: sc.AnnData, n_top_genes: int) -> None:
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    warnings.filterwarnings("ignore")

    if args.r_home:
        os.environ["R_HOME"] = args.r_home
    if args.r_user:
        os.environ["R_USER"] = args.r_user

    section_dir = args.data_root / args.section_id
    count_file = (
        args.count_file
        if args.count_file
        else f"filtered_feature_bc_matrix.h5"
    )
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

    preprocess(adata, args.n_top_genes)

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
        save_reconstrction=True,
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

    evaluation = adata.obs[["mclust", "Ground Truth"]].dropna()
    ari = adjusted_rand_score(
        evaluation["Ground Truth"].astype(str),
        evaluation["mclust"].astype(str),
    )
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

    # The current trainer stores the final loss as a tensor.
    final_loss = adata.uns.get("STAGATE_loss")
    if torch.is_tensor(final_loss):
        final_loss = float(final_loss.detach().cpu())
        adata.uns["STAGATE_loss"] = final_loss

    result = {
        "section_id": args.section_id,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_evaluated_spots": int(evaluation.shape[0]),
        "n_clusters": args.clusters,
        "ari": float(ari),
        "final_reconstruction_loss": final_loss,
        "device": str(device),
        "seed": args.seed,
    }

    adata.write_h5ad(output_dir / f"{args.section_id}_stagate.h5ad")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_dir.resolve()}")
    return result


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
