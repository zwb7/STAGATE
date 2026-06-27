"""Reporting helpers for reproducible baseline experiment artifacts."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def json_ready(value: Any) -> Any:
    """Convert common experiment values into JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, indent=2, ensure_ascii=False)


def get_git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def get_package_versions(package_names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def get_runtime_metadata(repo_root: Path) -> dict[str, Any]:
    return {
        "git_commit": get_git_commit(repo_root),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "package_versions": get_package_versions(
            [
                "anndata",
                "matplotlib",
                "numpy",
                "pandas",
                "scanpy",
                "scikit-learn",
                "scipy",
                "torch",
                "torch-geometric",
                "rpy2",
            ]
        ),
    }


def args_as_dict(args: Any) -> dict[str, Any]:
    return {key: json_ready(value) for key, value in vars(args).items()}


def compute_clustering_metrics(
    adata: Any,
    label_key: str,
    truth_key: str,
    embedding_key: str,
) -> tuple[dict[str, float | None], Any]:
    evaluation = adata.obs[[label_key, truth_key]].dropna()
    truth = evaluation[truth_key].astype(str)
    predicted = evaluation[label_key].astype(str)
    metrics: dict[str, float | None] = {
        "ari": float(adjusted_rand_score(truth, predicted)),
        "nmi": float(normalized_mutual_info_score(truth, predicted)),
        "ami": float(adjusted_mutual_info_score(truth, predicted)),
        "silhouette": None,
    }

    n_labels = int(predicted.nunique())
    if 1 < n_labels < evaluation.shape[0]:
        positions = adata.obs_names.get_indexer(evaluation.index)
        if np.all(positions >= 0):
            embedding = np.asarray(adata.obsm[embedding_key])[positions]
            metrics["silhouette"] = float(silhouette_score(embedding, predicted))

    return metrics, evaluation
