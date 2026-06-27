"""Run mclust and save posterior probabilities for BA-STAGATE-Calib.

This script is intentionally kept outside ``STAGATE_pyG/`` so the official
baseline helper ``mclust_R`` remains unchanged. It reads an AnnData file with a
precomputed embedding, runs R mclust, and writes:

  - ``adata.obs[label_key]``
  - ``adata.obsm[posterior_key]``
  - ``adata.obs[confidence_key]``
  - ``adata.obs[uncertainty_key]``
  - ``adata.uns[posterior_key + "_labels"]``

The output .h5ad can then be used by ``04_run_ba_stagate_calib.py`` with
``--confidence-key`` and ``--posterior-key``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def read_adata(path: Path):
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError(
            "anndata is required to read .h5ad files. Install it in the server "
            "environment before running this script."
        ) from exc
    return ad.read_h5ad(path)


def validate_embedding(adata, used_obsm: str) -> np.ndarray:
    if used_obsm not in adata.obsm:
        raise KeyError(f"adata.obsm[{used_obsm!r}] not found.")

    embedding = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if embedding.ndim != 2:
        raise ValueError(f"{used_obsm} must be 2D, got shape {embedding.shape}.")
    if not np.isfinite(embedding).all():
        raise ValueError(f"{used_obsm} contains NaN or infinite values.")
    return embedding


def run_mclust_with_posterior(
    embedding: np.ndarray,
    num_cluster: int,
    model_names: str,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects.vectors import FloatVector
        from rpy2.robjects.vectors import IntVector
        from rpy2.robjects.vectors import StrVector
    except ImportError as exc:
        raise RuntimeError(
            "rpy2 is required to run mclust from Python. Use the same server "
            "environment used for the STAGATE baseline mclust step."
        ) from exc

    np.random.seed(random_seed)
    robjects.r.library("mclust")
    robjects.r["set.seed"](random_seed)

    r_embedding = robjects.r["matrix"](
        FloatVector(embedding.ravel(order="C")),
        nrow=embedding.shape[0],
        ncol=embedding.shape[1],
        byrow=True,
    )
    r_embedding = robjects.r["colnames<-"](
        r_embedding,
        StrVector([f"STAGATE_{idx + 1}" for idx in range(embedding.shape[1])]),
    )

    res = robjects.r["Mclust"](
        r_embedding,
        G=IntVector([num_cluster]),
        modelNames=StrVector([model_names]),
    )

    labels = np.asarray(list(res.rx2("classification")), dtype=int)
    posterior = np.asarray(res.rx2("z"), dtype=np.float64)
    if posterior.ndim != 2:
        raise ValueError(f"mclust posterior z must be 2D, got shape {posterior.shape}.")
    if posterior.shape[0] != embedding.shape[0]:
        raise ValueError(
            "mclust posterior row count does not match embedding row count: "
            f"{posterior.shape[0]} vs {embedding.shape[0]}."
        )

    posterior_labels = [str(idx) for idx in range(1, posterior.shape[1] + 1)]
    observed_labels = sorted(str(item) for item in np.unique(labels))
    if observed_labels != posterior_labels:
        raise ValueError(
            "mclust classification labels do not match posterior column labels. "
            f"classification={observed_labels}, posterior={posterior_labels}"
        )

    return labels, posterior, posterior_labels


def write_outputs(
    adata,
    labels: np.ndarray,
    posterior: np.ndarray,
    posterior_labels: List[str],
    label_key: str,
    posterior_key: str,
    confidence_key: str,
    uncertainty_key: str,
) -> None:
    confidence = posterior.max(axis=1)
    uncertainty = 1.0 - confidence

    adata.obs[label_key] = pd.Categorical(labels.astype(str))
    adata.obsm[posterior_key] = posterior
    adata.obs[confidence_key] = confidence
    adata.obs[uncertainty_key] = uncertainty
    adata.uns[posterior_key + "_labels"] = posterior_labels


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run R mclust and save posterior probabilities into AnnData."
    )
    parser.add_argument("--adata", required=True, type=Path, help="Input .h5ad path.")
    parser.add_argument(
        "--output-adata",
        required=True,
        type=Path,
        help="Output .h5ad path with mclust posterior fields added.",
    )
    parser.add_argument("--used-obsm", default="STAGATE")
    parser.add_argument("--num-cluster", required=True, type=int)
    parser.add_argument("--modelNames", default="EEE")
    parser.add_argument("--random-seed", default=2020, type=int)
    parser.add_argument("--label-key", default="mclust")
    parser.add_argument("--posterior-key", default="mclust_posterior")
    parser.add_argument("--confidence-key", default="mclust_confidence")
    parser.add_argument("--uncertainty-key", default="mclust_uncertainty")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    adata = read_adata(args.adata)
    embedding = validate_embedding(adata, args.used_obsm)
    labels, posterior, posterior_labels = run_mclust_with_posterior(
        embedding=embedding,
        num_cluster=args.num_cluster,
        model_names=args.modelNames,
        random_seed=args.random_seed,
    )
    write_outputs(
        adata=adata,
        labels=labels,
        posterior=posterior,
        posterior_labels=posterior_labels,
        label_key=args.label_key,
        posterior_key=args.posterior_key,
        confidence_key=args.confidence_key,
        uncertainty_key=args.uncertainty_key,
    )

    args.output_adata.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output_adata)

    config = {
        "input_adata": str(args.adata),
        "output_adata": str(args.output_adata),
        "used_obsm": args.used_obsm,
        "num_cluster": args.num_cluster,
        "modelNames": args.modelNames,
        "random_seed": args.random_seed,
        "label_key": args.label_key,
        "posterior_key": args.posterior_key,
        "confidence_key": args.confidence_key,
        "uncertainty_key": args.uncertainty_key,
        "posterior_labels": posterior_labels,
    }
    config_path = args.output_adata.with_suffix(".mclust_posterior_config.json")
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print("mclust_with_posterior completed.")
    print(f"Saved: {args.output_adata}")
    print(f"Labels: adata.obs[{args.label_key!r}]")
    print(f"Posterior: adata.obsm[{args.posterior_key!r}]")
    print(f"Confidence: adata.obs[{args.confidence_key!r}]")


if __name__ == "__main__":
    main()
