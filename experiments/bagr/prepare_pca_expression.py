"""Prepare PCA expression features for BAGR edge-risk scoring.

This backfills ``pca_expression.npy`` for Step 1 baseline directories. It uses
raw DLPFC Visium data, applies the same lightweight expression preprocessing as
``run_stagate_baseline.py`` by default, aligns rows to ``pred_labels.csv`` spot
order, and writes only a compact PCA matrix. It does not train STAGATE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pca_expression.npy for BAGR from raw DLPFC data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Step 1 baseline directory containing pred_labels.csv and metrics_global.json.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("dataset/DLPFC"),
        help="DLPFC dataset root containing <sample-id>/filtered_feature_bc_matrix.h5.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="DLPFC sample id. Defaults to metrics_global.json sample_id or parent directory.",
    )
    parser.add_argument("--count-file", default="filtered_feature_bc_matrix.h5")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npy path. Defaults to <input-dir>/pca_expression.npy.",
    )
    parser.add_argument("--n-top-genes", type=int, default=3000)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument(
        "--preprocess-mode",
        choices=["log-normalize", "none"],
        default="log-normalize",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing pca_expression.npy.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def infer_sample_id(input_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    metrics = read_json(input_dir / "metrics_global.json")
    sample_id = metrics.get("sample_id")
    if sample_id:
        return str(sample_id)
    if input_dir.name.startswith("seed_") and input_dir.parent.name:
        return input_dir.parent.name
    return input_dir.name


def load_spot_order(input_dir: Path) -> list[str]:
    path = input_dir / "pred_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pred_labels.csv: {path}")
    labels = pd.read_csv(path, dtype={"spot_id": str})
    if "spot_id" not in labels.columns:
        raise ValueError(f"{path} is missing spot_id column")
    if labels["spot_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicated spot_id values")
    return labels["spot_id"].astype(str).tolist()


def preprocess_expression(adata, mode: str, n_top_genes: int) -> None:
    if mode == "none":
        adata.uns["preprocessing"] = {"mode": "none"}
        return
    if n_top_genes <= 0:
        raise ValueError("--n-top-genes must be positive")
    import scanpy as sc

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat_v3",
        n_top_genes=n_top_genes,
    )
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.uns["preprocessing"] = {
        "mode": "log-normalize",
        "n_top_genes": int(n_top_genes),
        "normalization": "scanpy.normalize_total(target_sum=1e4)+log1p",
        "hvg_method": "scanpy.highly_variable_genes(flavor='seurat_v3')",
    }


def matrix_to_numpy(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64)


def compute_pca(matrix: np.ndarray, n_components: int) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"Expression matrix must be 2D, got {matrix.shape}")
    max_components = min(matrix.shape[0], matrix.shape[1])
    if n_components <= 0:
        raise ValueError("--n-components must be positive")
    n_components = min(n_components, max_components)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    return np.asarray(centered @ vt[:n_components].T, dtype=np.float32)


def prepare_pca_expression(args: argparse.Namespace) -> Path:
    import scanpy as sc

    sample_id = infer_sample_id(args.input_dir, args.sample_id)
    output = args.output or args.input_dir / "pca_expression.npy"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite.")

    spot_order = load_spot_order(args.input_dir)
    sample_dir = args.data_root / sample_id
    count_path = sample_dir / args.count_file
    if not sample_dir.exists():
        raise FileNotFoundError(f"DLPFC sample directory not found: {sample_dir}")
    if not count_path.exists():
        raise FileNotFoundError(f"DLPFC count file not found: {count_path}")

    print(f"Loading raw DLPFC slice {sample_id} from {sample_dir}")
    adata = sc.read_visium(path=sample_dir, count_file=args.count_file)
    adata.var_names_make_unique()
    preprocess_expression(adata, args.preprocess_mode, args.n_top_genes)

    missing_spots = [spot for spot in spot_order if spot not in set(adata.obs_names.astype(str))]
    if missing_spots:
        raise ValueError(
            "pred_labels.csv contains spots absent from raw AnnData, for example: "
            + ", ".join(missing_spots[:5])
        )
    adata = adata[spot_order, :].copy()
    if "highly_variable" in adata.var:
        used = adata[:, adata.var["highly_variable"].to_numpy()].copy()
    else:
        used = adata
    expression = matrix_to_numpy(used.X)
    pca = compute_pca(expression, args.n_components)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, pca)
    meta = {
        "sample_id": sample_id,
        "input_dir": str(args.input_dir),
        "data_root": str(args.data_root),
        "count_file": args.count_file,
        "output": str(output),
        "n_spots": int(pca.shape[0]),
        "n_components": int(pca.shape[1]),
        "n_expression_features": int(expression.shape[1]),
        "preprocess_mode": args.preprocess_mode,
        "n_top_genes": int(args.n_top_genes),
        "row_order_source": str(args.input_dir / "pred_labels.csv"),
    }
    write_json(output.with_suffix(".json"), meta)
    print(f"Saved PCA expression features to {output.resolve()} with shape {pca.shape}")
    return output


def main() -> None:
    prepare_pca_expression(parse_args())


if __name__ == "__main__":
    main()
