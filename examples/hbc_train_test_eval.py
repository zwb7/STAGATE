"""Train and evaluate STAGATE on the Human Breast Cancer Visium dataset.

Expected data layout:

Human_Breast_Cancer/
├── filtered_feature_bc_matrix.h5
├── hbrc_truth.csv
└── spatial/

The default evaluation uses the 20 fine-grained regions in the
``ground_truth`` column of ``hbrc_truth.csv``. Ground-truth labels are only
used after unsupervised STAGATE training and mclust clustering.
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
        description="Train and evaluate STAGATE_pyG on Human Breast Cancer."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data1/zhangwenbo/Code/Dataset/HBC"),
        help="Directory containing the Visium count file and spatial directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/stagate"),
    )
    parser.add_argument(
        "--dataset-id",
        default="HBC",
        help="Name of the subdirectory created under --output-dir.",
    )
    parser.add_argument(
        "--count-file",
        default="filtered_feature_bc_matrix.h5",
    )
    parser.add_argument(
        "--truth-file",
        type=Path,
        default=None,
        help="Defaults to <data-root>/hbrc_truth.csv.",
    )
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--annotation-column", default="ground_truth")
    parser.add_argument("--clusters", type=int, default=20)
    parser.add_argument(
        "--radius",
        type=float,
        default=400.0,
        help="Radius cutoff in full-resolution Visium pixel coordinates.",
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


def load_hbc(
    data_root: Path,
    count_file: str,
    truth_file: Path,
    id_column: str,
    annotation_column: str,
) -> sc.AnnData:
    if not data_root.exists():
        raise FileNotFoundError(f"HBC data directory not found: {data_root}")
    if not truth_file.exists():
        raise FileNotFoundError(f"HBC ground-truth file not found: {truth_file}")

    adata = sc.read_visium(path=data_root, count_file=count_file)
    adata.var_names_make_unique()

    annotations = pd.read_csv(truth_file)
    required_columns = {id_column, annotation_column}
    missing_columns = sorted(required_columns.difference(annotations.columns))
    if missing_columns:
        raise ValueError(
            f"Ground-truth file is missing columns: {missing_columns}. "
            f"Available columns: {list(annotations.columns)}"
        )
    if annotations[id_column].duplicated().any():
        raise ValueError(f"Ground-truth ID column contains duplicates: {id_column}")

    annotations = annotations.set_index(id_column)
    adata.obs["Ground Truth"] = annotations.reindex(adata.obs_names)[
        annotation_column
    ]
    n_matched = int(adata.obs["Ground Truth"].notna().sum())
    if n_matched == 0:
        raise ValueError(
            "No ground-truth barcodes matched the Visium count matrix. "
            "Check --id-column and the barcode suffixes."
        )
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

    truth_file = (
        args.truth_file
        if args.truth_file is not None
        else args.data_root / "hbc_truth.csv"
    )
    output_dir = args.output_dir / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Loading Human Breast Cancer data from {args.data_root}")
    adata = load_hbc(
        args.data_root,
        args.count_file,
        truth_file,
        args.id_column,
        args.annotation_column,
    )
    n_annotated = int(adata.obs["Ground Truth"].notna().sum())
    n_truth_classes = int(adata.obs["Ground Truth"].nunique())
    print(
        f"Loaded {adata.n_obs} spots and {adata.n_vars} genes; "
        f"matched {n_annotated} annotations across {n_truth_classes} classes"
    )
    if args.clusters != n_truth_classes:
        print(
            f"Warning: --clusters={args.clusters}, while the selected annotation "
            f"column contains {n_truth_classes} classes."
        )

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
    if adata.uns["Spatial_Net"].empty:
        raise ValueError(
            "The spatial graph has no edges. Increase --radius for this dataset."
        )
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
            legend_fontsize=8,
            frameon=False,
            size=20,
            title=f"{args.dataset_id}_STAGATE",
            legend_fontoutline=2,
            show=False,
        )
        save_current_figure(output_dir / "paga_trajectory.png")

    final_loss = adata.uns.get("STAGATE_loss")
    if torch.is_tensor(final_loss):
        final_loss = float(final_loss.detach().cpu())
        adata.uns["STAGATE_loss"] = final_loss

    result = {
        "dataset_id": args.dataset_id,
        "annotation_column": args.annotation_column,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_evaluated_spots": int(evaluation.shape[0]),
        "n_ground_truth_classes": n_truth_classes,
        "n_clusters": args.clusters,
        "radius": args.radius,
        "ari": float(ari),
        "final_reconstruction_loss": final_loss,
        "device": str(device),
        "seed": args.seed,
    }

    adata.write_h5ad(output_dir / f"{args.dataset_id}_stagate.h5ad")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_dir.resolve()}")
    return result


def main() -> None:
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()
