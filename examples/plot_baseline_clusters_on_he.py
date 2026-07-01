"""Plot baseline STAGATE clustering labels on the original H&E spatial image.

This script mirrors `plot_bagr_clusters_on_he.py`, but defaults to baseline
STAGATE naming. It attaches baseline labels to `adata.obs` and visualizes them
with `scanpy.pl.spatial`.

Example
-------
python examples/plot_baseline_clusters_on_he.py \
    --adata results/stagate/151674/adata.h5ad \
    --labels results/stagate_baseline/151674/pred_labels.csv \
    --barcode-col barcode \
    --cluster-col stagate_cluster \
    --output results/stagate_baseline/151674/stagate_spatial_he.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize baseline STAGATE clustering results on the original H&E slice."
    )
    parser.add_argument(
        "--adata",
        required=True,
        help="Path to an .h5ad file containing spatial coordinates and H&E image metadata.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help=(
            "CSV/TSV label file. It should contain one barcode column and one "
            "cluster-label column, or use the first column as barcode if "
            "--barcode-col is omitted."
        ),
    )
    parser.add_argument(
        "--cluster-col",
        default="stagate_cluster",
        help="Column in --labels containing baseline STAGATE cluster labels.",
    )
    parser.add_argument(
        "--barcode-col",
        default=None,
        help="Column in --labels containing spot barcodes. If omitted, the first column is used.",
    )
    parser.add_argument(
        "--obs-key",
        default="STAGATE",
        help="Name of the new adata.obs column used for plotting.",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Label-file delimiter. By default pandas infers CSV/TSV delimiter from suffix.",
    )
    parser.add_argument(
        "--library-id",
        default=None,
        help="Visium library_id passed to scanpy.pl.spatial when needed.",
    )
    parser.add_argument(
        "--img-key",
        default="hires",
        help="Image key passed to scanpy.pl.spatial, usually 'hires' or 'lowres'.",
    )
    parser.add_argument(
        "--spot-size",
        type=float,
        default=None,
        help="Spot size passed to scanpy.pl.spatial. Leave unset for Scanpy default.",
    )
    parser.add_argument(
        "--palette",
        default=None,
        help="Optional Matplotlib/Scanpy palette name, e.g. tab20.",
    )
    parser.add_argument(
        "--title",
        default="STAGATE",
        help="Plot title.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output image path, e.g. stagate_spatial_he.png or .pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--out-adata",
        default=None,
        help="Optional path to save the AnnData object with attached baseline labels.",
    )
    return parser.parse_args()


def read_labels(path: str, barcode_col: str | None, cluster_col: str, sep: str | None) -> pd.Series:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file does not exist: {labels_path}")

    read_sep = sep
    if read_sep is None:
        read_sep = "\t" if labels_path.suffix.lower() in {".tsv", ".txt"} else ","

    df = pd.read_csv(labels_path, sep=read_sep)
    if df.empty:
        raise ValueError(f"Label file is empty: {labels_path}")

    if barcode_col is None:
        barcode_col = df.columns[0]

    missing = [col for col in (barcode_col, cluster_col) if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s) in label file: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    labels = df[[barcode_col, cluster_col]].copy()
    labels[barcode_col] = labels[barcode_col].astype(str)
    labels = labels.drop_duplicates(subset=barcode_col, keep="first")
    return labels.set_index(barcode_col)[cluster_col]


def attach_labels(adata: sc.AnnData, labels: pd.Series, obs_key: str) -> sc.AnnData:
    adata = adata.copy()
    obs_names = adata.obs_names.astype(str)
    matched = obs_names.intersection(labels.index.astype(str))
    if len(matched) == 0:
        raise ValueError(
            "No overlapping barcodes between adata.obs_names and the label file. "
            "Check --barcode-col and whether barcode suffixes match."
        )

    adata.obs_names = obs_names
    adata.obs[obs_key] = labels.reindex(adata.obs_names)
    missing_count = int(adata.obs[obs_key].isna().sum())
    if missing_count:
        print(
            f"[WARN] {missing_count} / {adata.n_obs} spots have no baseline label; "
            "they will be shown as missing."
        )

    adata.obs[obs_key] = adata.obs[obs_key].astype("category")
    return adata


def plot_spatial(adata: sc.AnnData, args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plot_kwargs = {
        "color": args.obs_key,
        "img_key": args.img_key,
        "library_id": args.library_id,
        "show": False,
        "title": args.title,
    }
    if args.spot_size is not None:
        plot_kwargs["spot_size"] = args.spot_size
    if args.palette is not None:
        plot_kwargs["palette"] = args.palette

    sc.pl.spatial(adata, **plot_kwargs)
    fig = plt.gcf()
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved spatial plot to: {output}")


def main() -> None:
    args = parse_args()

    adata = sc.read_h5ad(args.adata)
    labels = read_labels(args.labels, args.barcode_col, args.cluster_col, args.sep)
    adata = attach_labels(adata, labels, args.obs_key)

    plot_spatial(adata, args)

    if args.out_adata:
        out_adata = Path(args.out_adata)
        out_adata.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out_adata)
        print(f"[OK] Saved annotated AnnData to: {out_adata}")


if __name__ == "__main__":
    main()
