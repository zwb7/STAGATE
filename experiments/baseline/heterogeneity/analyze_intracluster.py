#!/usr/bin/env python
"""Detect within-cluster substructure in spatial transcriptomics outputs.

This script is intended for post-hoc baseline analysis. It reads an existing
AnnData file, subsets each sample and parent cluster, then checks whether the
cluster contains stable discrete subclusters or stronger continuous spatial
heterogeneity. It does not train or evaluate a model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


DEFAULT_RESOLUTIONS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
QC_COLUMNS = ("total_counts", "n_genes_by_counts", "pct_counts_mt")


@dataclass
class ClusterResult:
    dataset: str
    sample: str
    parent_cluster: str
    n_spots: int
    status: str
    conclusion: str
    best_resolution: float | None = None
    best_n_subclusters: int | None = None
    best_stability_ari: float | None = None
    best_silhouette: float | None = None
    same_label_neighbor_fraction: float | None = None
    mean_subcluster_moran_i: float | None = None
    max_embedding_moran_i: float | None = None
    max_qc_eta_squared: float | None = None
    label_ari: float | None = None
    marker_genes_padj05: int | None = None
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc within-cluster heterogeneity analysis for DLPFC/HBC "
            "baseline STAGATE outputs."
        )
    )
    parser.add_argument("--input", required=True, help="Input .h5ad file.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--dataset", default="unknown", help="Dataset name.")
    parser.add_argument(
        "--sample-key",
        default="sample",
        help="obs column for sample IDs. If absent, all spots are treated as one sample.",
    )
    parser.add_argument(
        "--cluster-key",
        default="mclust",
        help="obs column containing parent baseline clusters.",
    )
    parser.add_argument(
        "--label-key",
        default=None,
        help="optional obs column for reference labels, e.g. DLPFC ground_truth.",
    )
    parser.add_argument(
        "--embedding-key",
        default="X_STAGATE",
        help="obsm key used for neighbor graph and silhouette. Falls back to PCA.",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help="obsm key containing spatial coordinates.",
    )
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=list(DEFAULT_RESOLUTIONS),
        help="Leiden resolutions to scan inside each parent cluster.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds for stability analysis.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=10,
        help="Target neighbor count for within-cluster graphs.",
    )
    parser.add_argument(
        "--spatial-neighbors",
        type=int,
        default=6,
        help="Neighbor count used for spatial contiguity and Moran's I.",
    )
    parser.add_argument(
        "--min-spots",
        type=int,
        default=40,
        help="Skip parent clusters with fewer spots.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.75,
        help="Mean pairwise ARI threshold for stable subclusters.",
    )
    parser.add_argument(
        "--silhouette-threshold",
        type=float,
        default=0.10,
        help="Minimum silhouette score for discrete subcluster evidence.",
    )
    parser.add_argument(
        "--continuous-moran-threshold",
        type=float,
        default=0.25,
        help="Embedding Moran's I threshold for continuous spatial heterogeneity.",
    )
    parser.add_argument(
        "--max-marker-genes",
        type=int,
        default=20,
        help="Number of ranked marker genes to export per candidate subcluster.",
    )
    return parser.parse_args()


def safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text or "NA"


def as_dense_1d(values) -> np.ndarray:
    if sparse.issparse(values):
        return np.asarray(values.toarray()).ravel()
    return np.asarray(values).ravel()


def ensure_qc_columns(adata: ad.AnnData) -> None:
    if "total_counts" not in adata.obs:
        adata.obs["total_counts"] = as_dense_1d(adata.X.sum(axis=1))
    if "n_genes_by_counts" not in adata.obs:
        if sparse.issparse(adata.X):
            adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()
        else:
            adata.obs["n_genes_by_counts"] = (np.asarray(adata.X) > 0).sum(axis=1)
    if "pct_counts_mt" not in adata.obs:
        mt_mask = adata.var_names.astype(str).str.upper().str.startswith("MT-")
        if mt_mask.any():
            mt_counts = as_dense_1d(adata[:, mt_mask].X.sum(axis=1))
            totals = np.asarray(adata.obs["total_counts"], dtype=float)
            adata.obs["pct_counts_mt"] = np.divide(
                mt_counts * 100.0,
                totals,
                out=np.zeros_like(mt_counts, dtype=float),
                where=totals > 0,
            )


def get_representation(
    adata_sub: ad.AnnData, embedding_key: str, n_comps: int = 30
) -> tuple[str, np.ndarray]:
    if embedding_key in adata_sub.obsm:
        matrix = np.asarray(adata_sub.obsm[embedding_key])
        return embedding_key, matrix

    n_obs, n_vars = adata_sub.n_obs, adata_sub.n_vars
    n_pcs = max(2, min(n_comps, n_obs - 1, n_vars - 1))
    if "X_pca" not in adata_sub.obsm or adata_sub.obsm["X_pca"].shape[1] < n_pcs:
        sc.pp.pca(adata_sub, n_comps=n_pcs)
    return "X_pca", np.asarray(adata_sub.obsm["X_pca"])


def pairwise_mean_ari(label_sets: list[np.ndarray]) -> float:
    if len(label_sets) < 2:
        return float("nan")
    scores = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            scores.append(adjusted_rand_score(label_sets[i], label_sets[j]))
    return float(np.mean(scores)) if scores else float("nan")


def safe_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return float("nan")
    try:
        return float(silhouette_score(matrix, labels))
    except ValueError:
        return float("nan")


def eta_squared_by_group(values: np.ndarray, labels: np.ndarray) -> float:
    valid = np.isfinite(values)
    values = values[valid]
    labels = labels[valid]
    if len(values) == 0 or len(np.unique(labels)) < 2:
        return float("nan")

    grand_mean = float(np.mean(values))
    total_ss = float(np.sum((values - grand_mean) ** 2))
    if total_ss == 0:
        return 0.0

    between_ss = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        between_ss += len(group) * float((np.mean(group) - grand_mean) ** 2)
    return float(between_ss / total_ss)


def row_normalized_knn(coords: np.ndarray, n_neighbors: int) -> sparse.csr_matrix | None:
    n_obs = coords.shape[0]
    if n_obs <= 2 or coords.shape[1] < 2:
        return None
    k = max(1, min(n_neighbors, n_obs - 1))
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)[:, 1:]
    rows = np.repeat(np.arange(n_obs), k)
    cols = indices.ravel()
    data = np.ones(len(rows), dtype=float)
    graph = sparse.coo_matrix((data, (rows, cols)), shape=(n_obs, n_obs)).tocsr()
    graph = graph.maximum(graph.T)
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    return sparse.diags(inv).dot(graph).tocsr()


def moran_i(values: np.ndarray, weights: sparse.csr_matrix | None) -> float:
    if weights is None:
        return float("nan")
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() < 3:
        return float("nan")
    if not valid.all():
        x = x[valid]
        weights = weights[valid][:, valid]

    centered = x - float(np.mean(x))
    denominator = float(np.sum(centered**2))
    weight_sum = float(weights.sum())
    if denominator == 0 or weight_sum == 0:
        return float("nan")
    numerator = float(centered @ weights.dot(centered))
    return float((len(x) / weight_sum) * (numerator / denominator))


def spatial_label_metrics(
    coords: np.ndarray | None, labels: np.ndarray, n_neighbors: int
) -> tuple[float, float]:
    if coords is None:
        return float("nan"), float("nan")
    weights = row_normalized_knn(coords, n_neighbors)
    if weights is None:
        return float("nan"), float("nan")

    neighbor_rows, neighbor_cols = weights.nonzero()
    same = labels[neighbor_rows] == labels[neighbor_cols]
    same_fraction = float(np.mean(same)) if len(same) else float("nan")

    label_morans = []
    for label in np.unique(labels):
        one_hot = (labels == label).astype(float)
        label_morans.append(moran_i(one_hot, weights))
    return same_fraction, float(np.nanmean(label_morans))


def embedding_spatial_moran(
    coords: np.ndarray | None, matrix: np.ndarray, n_neighbors: int, max_dims: int = 3
) -> float:
    if coords is None:
        return float("nan")
    weights = row_normalized_knn(coords, n_neighbors)
    if weights is None:
        return float("nan")

    scores = []
    for dim in range(min(max_dims, matrix.shape[1])):
        scores.append(moran_i(matrix[:, dim], weights))
    return float(np.nanmax(scores)) if scores else float("nan")


def run_leiden_grid(
    adata_sub: ad.AnnData,
    use_rep: str,
    matrix: np.ndarray,
    resolutions: Iterable[float],
    seeds: Iterable[int],
    n_neighbors: int,
) -> tuple[pd.DataFrame, dict[tuple[float, int], np.ndarray]]:
    labels_by_run: dict[tuple[float, int], np.ndarray] = {}
    rows = []

    local_neighbors = max(2, min(n_neighbors, adata_sub.n_obs - 1))
    sc.pp.neighbors(adata_sub, n_neighbors=local_neighbors, use_rep=use_rep)

    for resolution in resolutions:
        run_labels = []
        run_silhouettes = []
        run_cluster_counts = []
        for seed in seeds:
            key = f"leiden_r{resolution:g}_s{seed}"
            sc.tl.leiden(
                adata_sub,
                resolution=resolution,
                random_state=seed,
                key_added=key,
            )
            labels = adata_sub.obs[key].astype(str).to_numpy()
            labels_by_run[(float(resolution), int(seed))] = labels
            run_labels.append(labels)
            run_cluster_counts.append(len(np.unique(labels)))
            run_silhouettes.append(safe_silhouette(matrix, labels))

        rows.append(
            {
                "resolution": float(resolution),
                "mean_pairwise_ari": pairwise_mean_ari(run_labels),
                "mean_n_subclusters": float(np.mean(run_cluster_counts)),
                "median_n_subclusters": float(np.median(run_cluster_counts)),
                "mean_silhouette": float(np.nanmean(run_silhouettes)),
                "min_n_subclusters": int(np.min(run_cluster_counts)),
                "max_n_subclusters": int(np.max(run_cluster_counts)),
            }
        )

    return pd.DataFrame(rows), labels_by_run


def choose_resolution(
    stability: pd.DataFrame, stability_threshold: float
) -> pd.Series | None:
    candidates = stability[
        (stability["mean_n_subclusters"] >= 2.0)
        & (stability["mean_pairwise_ari"] >= stability_threshold)
    ].copy()
    if candidates.empty:
        candidates = stability[stability["mean_n_subclusters"] >= 2.0].copy()
    if candidates.empty:
        return None

    candidates["silhouette_rank"] = candidates["mean_silhouette"].fillna(-999.0)
    candidates = candidates.sort_values(
        ["mean_pairwise_ari", "silhouette_rank", "mean_n_subclusters"],
        ascending=[False, False, True],
    )
    return candidates.iloc[0]


def write_markers(
    adata_sub: ad.AnnData,
    labels: np.ndarray,
    out_dir: Path,
    max_marker_genes: int,
) -> int:
    marker_key = "candidate_subcluster"
    ranked_key = "rank_genes_candidate_subcluster"
    adata_sub.obs[marker_key] = pd.Categorical(labels)

    try:
        sc.tl.rank_genes_groups(
            adata_sub,
            groupby=marker_key,
            method="wilcoxon",
            key_added=ranked_key,
        )
    except Exception as exc:  # pragma: no cover - depends on input matrix state
        (out_dir / "marker_error.txt").write_text(str(exc), encoding="utf-8")
        return 0

    groups = list(adata_sub.uns[ranked_key]["names"].dtype.names)
    rows = []
    significant_count = 0
    for group in groups:
        names = adata_sub.uns[ranked_key]["names"][group][:max_marker_genes]
        scores = adata_sub.uns[ranked_key]["scores"][group][:max_marker_genes]
        pvals_adj = adata_sub.uns[ranked_key]["pvals_adj"][group][:max_marker_genes]
        logfoldchanges = adata_sub.uns[ranked_key].get("logfoldchanges", None)
        if logfoldchanges is not None:
            logfoldchanges = logfoldchanges[group][:max_marker_genes]
        else:
            logfoldchanges = [np.nan] * len(names)

        for rank, (gene, score, padj, logfc) in enumerate(
            zip(names, scores, pvals_adj, logfoldchanges), start=1
        ):
            if np.isfinite(padj) and padj < 0.05:
                significant_count += 1
            rows.append(
                {
                    "subcluster": group,
                    "rank": rank,
                    "gene": gene,
                    "score": score,
                    "pvals_adj": padj,
                    "logfoldchanges": logfc,
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "candidate_markers.csv", index=False)
    return significant_count


def contingency_table(
    labels: np.ndarray,
    reference: pd.Series,
    out_path: Path,
) -> float:
    ref_values = reference.astype(str).to_numpy()
    table = pd.crosstab(
        pd.Series(labels, name="candidate_subcluster"),
        pd.Series(ref_values, name="reference_label"),
    )
    table.to_csv(out_path)
    if len(np.unique(ref_values)) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(adjusted_rand_score(ref_values, labels))


def classify_result(
    n_subclusters: int | None,
    stability: float | None,
    silhouette: float | None,
    marker_count: int | None,
    max_embedding_moran: float | None,
    stability_threshold: float,
    silhouette_threshold: float,
    continuous_moran_threshold: float,
) -> str:
    has_discrete_evidence = (
        n_subclusters is not None
        and n_subclusters >= 2
        and stability is not None
        and np.isfinite(stability)
        and stability >= stability_threshold
        and silhouette is not None
        and np.isfinite(silhouette)
        and silhouette >= silhouette_threshold
        and marker_count is not None
        and marker_count > 0
    )
    if has_discrete_evidence:
        return "discrete_subclusters"

    has_continuous_evidence = (
        max_embedding_moran is not None
        and np.isfinite(max_embedding_moran)
        and max_embedding_moran >= continuous_moran_threshold
    )
    if has_continuous_evidence:
        return "continuous_heterogeneity"
    return "no_strong_substructure"


def analyze_parent_cluster(
    adata: ad.AnnData,
    dataset: str,
    sample_value: object,
    parent_cluster: object,
    args: argparse.Namespace,
    out_root: Path,
) -> ClusterResult:
    sample_text = str(sample_value)
    cluster_text = str(parent_cluster)
    out_dir = out_root / safe_name(sample_text) / f"cluster_{safe_name(cluster_text)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    subset_mask = (adata.obs[args.sample_key].astype(str) == sample_text) & (
        adata.obs[args.cluster_key].astype(str) == cluster_text
    )
    adata_sub = adata[subset_mask].copy()

    result = ClusterResult(
        dataset=dataset,
        sample=sample_text,
        parent_cluster=cluster_text,
        n_spots=adata_sub.n_obs,
        status="ok",
        conclusion="not_evaluated",
    )

    if adata_sub.n_obs < args.min_spots:
        result.status = "skipped"
        result.conclusion = "too_few_spots"
        result.notes = f"n_spots < min_spots ({args.min_spots})"
        return result

    try:
        use_rep, matrix = get_representation(adata_sub, args.embedding_key)
    except Exception as exc:
        result.status = "failed"
        result.conclusion = "representation_error"
        result.notes = str(exc)
        return result

    coords = None
    if args.spatial_key in adata_sub.obsm:
        coords = np.asarray(adata_sub.obsm[args.spatial_key])

    result.max_embedding_moran_i = embedding_spatial_moran(
        coords, matrix, args.spatial_neighbors
    )

    try:
        stability_df, labels_by_run = run_leiden_grid(
            adata_sub,
            use_rep,
            matrix,
            args.resolutions,
            args.seeds,
            args.n_neighbors,
        )
    except Exception as exc:
        result.status = "failed"
        result.conclusion = "leiden_error"
        result.notes = str(exc)
        return result

    stability_df.to_csv(out_dir / "stability_by_resolution.csv", index=False)
    best = choose_resolution(stability_df, args.stability_threshold)
    if best is None:
        result.conclusion = classify_result(
            None,
            None,
            None,
            None,
            result.max_embedding_moran_i,
            args.stability_threshold,
            args.silhouette_threshold,
            args.continuous_moran_threshold,
        )
        return result

    best_resolution = float(best["resolution"])
    chosen_seed = int(args.seeds[0])
    labels = labels_by_run[(best_resolution, chosen_seed)]
    result.best_resolution = best_resolution
    result.best_n_subclusters = len(np.unique(labels))
    result.best_stability_ari = float(best["mean_pairwise_ari"])
    result.best_silhouette = safe_silhouette(matrix, labels)

    label_obs = pd.DataFrame(
        {
            "spot_id": adata_sub.obs_names,
            "sample": sample_text,
            "parent_cluster": cluster_text,
            "candidate_subcluster": labels,
        }
    )
    label_obs.to_csv(out_dir / "candidate_subcluster_labels.csv", index=False)

    same_fraction, mean_moran = spatial_label_metrics(
        coords, labels, args.spatial_neighbors
    )
    result.same_label_neighbor_fraction = same_fraction
    result.mean_subcluster_moran_i = mean_moran

    qc_eta = []
    for column in QC_COLUMNS:
        if column in adata_sub.obs:
            values = pd.to_numeric(adata_sub.obs[column], errors="coerce").to_numpy()
            qc_eta.append(eta_squared_by_group(values, labels))
    result.max_qc_eta_squared = float(np.nanmax(qc_eta)) if qc_eta else float("nan")

    if args.label_key and args.label_key in adata_sub.obs:
        result.label_ari = contingency_table(
            labels,
            adata_sub.obs[args.label_key],
            out_dir / "reference_label_contingency.csv",
        )

    result.marker_genes_padj05 = write_markers(
        adata_sub, labels, out_dir, args.max_marker_genes
    )
    result.conclusion = classify_result(
        result.best_n_subclusters,
        result.best_stability_ari,
        result.best_silhouette,
        result.marker_genes_padj05,
        result.max_embedding_moran_i,
        args.stability_threshold,
        args.silhouette_threshold,
        args.continuous_moran_threshold,
    )
    return result


def prepare_obs_keys(adata: ad.AnnData, args: argparse.Namespace) -> None:
    if args.cluster_key not in adata.obs:
        raise KeyError(f"cluster key '{args.cluster_key}' not found in adata.obs")

    if args.sample_key not in adata.obs:
        adata.obs[args.sample_key] = "all"


def write_run_config(args: argparse.Namespace, out_root: Path, adata: ad.AnnData) -> None:
    config = vars(args).copy()
    config["n_obs"] = int(adata.n_obs)
    config["n_vars"] = int(adata.n_vars)
    config["obs_columns"] = list(map(str, adata.obs.columns))
    config["obsm_keys"] = list(map(str, adata.obsm.keys()))
    (out_root / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    prepare_obs_keys(adata, args)
    ensure_qc_columns(adata)
    write_run_config(args, out_root, adata)

    results: list[ClusterResult] = []
    sample_values = sorted(adata.obs[args.sample_key].astype(str).unique())
    for sample_value in sample_values:
        sample_mask = adata.obs[args.sample_key].astype(str) == sample_value
        parent_clusters = sorted(adata.obs.loc[sample_mask, args.cluster_key].astype(str).unique())
        for parent_cluster in parent_clusters:
            result = analyze_parent_cluster(
                adata,
                args.dataset,
                sample_value,
                parent_cluster,
                args,
                out_root,
            )
            results.append(result)

    summary = pd.DataFrame([asdict(result) for result in results])
    summary.to_csv(out_root / "summary.csv", index=False)
    print(f"Wrote {len(summary)} parent-cluster summaries to {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
