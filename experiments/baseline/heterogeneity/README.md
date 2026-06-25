# Baseline Intracluster Heterogeneity Analysis

This folder contains post-hoc analysis code for asking whether a baseline
cluster contains discrete subclusters or stronger continuous heterogeneity.
It reads existing `.h5ad` outputs and does not train, infer, or evaluate a
model.

## What the script checks

For each sample and parent cluster, `analyze_intracluster.py` scans Leiden
resolutions inside the parent cluster and repeats clustering across random
seeds. It exports:

- `summary.csv`: one row per sample and parent cluster.
- `run_config.json`: input keys, parameters, and AnnData dimensions.
- `stability_by_resolution.csv`: mean pairwise ARI across seeds for each
  resolution.
- `candidate_subcluster_labels.csv`: candidate subcluster assignment for the
  selected resolution and first seed.
- `candidate_markers.csv`: top marker genes for candidate subclusters.
- `reference_label_contingency.csv`: candidate subcluster versus reference
  label table when `--label-key` is provided.

The `conclusion` column is a screening label, not proof:

- `discrete_subclusters`: stable repeated subclustering, non-trivial
  silhouette, and at least one adjusted-p marker gene.
- `continuous_heterogeneity`: no strong discrete evidence, but spatially
  autocorrelated embedding dimensions suggest a gradient.
- `no_strong_substructure`: neither criterion is strong under current
  thresholds.
- `too_few_spots`: skipped by `--min-spots`.

## DLPFC example

Run this on the remote server after baseline outputs exist:

```bash
python experiments/baseline/heterogeneity/analyze_intracluster.py \
  --dataset DLPFC \
  --input results/baseline/dlpfc/151673/stagate_output.h5ad \
  --out results/heterogeneity/DLPFC/151673 \
  --sample-key sample \
  --cluster-key mclust \
  --label-key ground_truth \
  --embedding-key X_STAGATE \
  --spatial-key spatial
```

If each `.h5ad` contains only one slice and does not have a `sample` column,
the script automatically treats all spots as one sample.

## HBC example

```bash
python experiments/baseline/heterogeneity/analyze_intracluster.py \
  --dataset HBC \
  --input results/baseline/hbc/sample1/stagate_output.h5ad \
  --out results/heterogeneity/HBC/sample1 \
  --sample-key sample \
  --cluster-key mclust \
  --embedding-key X_STAGATE \
  --spatial-key spatial
```

For HBC, interpret subclusters alongside tumor, immune, stromal,
proliferation, hypoxia, and QC signatures. Stable expression subclusters can
still reflect mixed cell composition or tumor-stroma interfaces rather than a
new biological state.

## Suggested interpretation checks

Use a conservative rule before reporting a real subcluster:

- It is stable across seeds and nearby resolutions.
- It has marker genes after multiple testing correction.
- It is spatially coherent or has a biologically interpretable spatial pattern.
- It is not mainly explained by total counts, detected genes, mitochondrial
  percentage, edge effects, or sample-specific artifacts.
- Similar structure appears in more than one biologically comparable sample.

Do not modify the official `STAGATE_pyG/` baseline to run this analysis.
