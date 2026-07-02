import argparse
from pathlib import Path


def require_python_packages():
    missing = []
    for package, module in [
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("scanpy", "scanpy"),
    ]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        joined = " ".join(missing)
        raise SystemExit(
            "Missing Python package(s): " + ", ".join(missing) + "\n"
            "Install them in the visualization environment, for example:\n"
            f"  mamba install -c conda-forge {joined}\n"
            "or:\n"
            f"  conda install -c conda-forge {joined}\n"
            "or with pip if that is how the environment is managed:\n"
            f"  python -m pip install {joined}"
        )


require_python_packages()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render Seurat clustering results with scanpy.pl.spatial."
    )
    parser.add_argument("--data_dir", required=True, help="10x Visium sample directory.")
    parser.add_argument("--sample_id", required=True, help="Sample ID, e.g. 151507 or HBC.")
    parser.add_argument(
        "--metadata",
        required=True,
        help="metadata.csv written by run_seurat_spatial.R.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output root directory for PNG spatial plots.",
    )
    parser.add_argument(
        "--counts_file",
        default="filtered_feature_bc_matrix.h5",
        help="Count matrix file inside data_dir.",
    )
    parser.add_argument(
        "--library_id",
        default=None,
        help="Scanpy spatial library_id. Defaults to sample_id.",
    )
    parser.add_argument(
        "--img_key",
        default="hires",
        choices=["hires", "lowres"],
        help="Spatial image key used by sc.pl.spatial.",
    )
    parser.add_argument("--spot_size", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_metadata(path):
    metadata = pd.read_csv(path, index_col=0)
    metadata.index = metadata.index.astype(str)
    return metadata


def resolve_library_id(adata, sample_id, requested):
    spatial = adata.uns.get("spatial", {})
    if requested is not None:
        return requested
    if sample_id in spatial:
        return sample_id
    if len(spatial) == 1:
        return next(iter(spatial))
    return sample_id


def attach_metadata(adata, metadata):
    common = adata.obs_names.intersection(metadata.index)
    if len(common) == 0:
        raise ValueError("No shared barcodes between Visium data and Seurat metadata.")

    for column in metadata.columns:
        adata.obs[column] = pd.NA
        adata.obs.loc[common, column] = metadata.loc[common, column].astype(str)

    if "seurat_clusters" in adata.obs:
        adata.obs["seurat_clusters"] = adata.obs["seurat_clusters"].astype("category")
    else:
        raise ValueError("metadata.csv does not contain seurat_clusters.")

    if "ground_truth" in adata.obs:
        adata.obs["ground_truth"] = adata.obs["ground_truth"].astype("category")


def spatial_png(adata, color, library_id, args, out_path):
    kwargs = {
        "color": color,
        "library_id": library_id,
        "img_key": args.img_key,
        "show": False,
        "title": f"{args.sample_id} {color}",
    }
    if args.spot_size is not None:
        kwargs["spot_size"] = args.spot_size

    sc.pl.spatial(adata, **kwargs)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close("all")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / args.sample_id
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_visium(
        path=args.data_dir,
        count_file=args.counts_file,
        library_id=args.library_id or args.sample_id,
    )
    metadata = read_metadata(args.metadata)
    attach_metadata(adata, metadata)
    library_id = resolve_library_id(adata, args.sample_id, args.library_id)

    spatial_png(
        adata,
        "seurat_clusters",
        library_id,
        args,
        out_dir / "spatial_clusters.png",
    )

    if "ground_truth" in adata.obs and adata.obs["ground_truth"].notna().any():
        spatial_png(
            adata,
            "ground_truth",
            library_id,
            args,
            out_dir / "spatial_ground_truth.png",
        )


if __name__ == "__main__":
    main()
