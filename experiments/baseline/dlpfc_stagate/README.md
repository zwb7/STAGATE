# Phase 0 STAGATE Baseline

This directory records the fixed single-seed DLPFC baseline protocol used
before boundary diagnostics or BA-STAGATE experiments. It intentionally
excludes the later multi-seed requirement.

## Scope

- Keep `STAGATE_pyG/` as the official baseline implementation.
- Run the Phase 0 DLPFC baseline entrypoint with fixed seed `0`.
- Save `config.json`, `metrics.json`, figures, and the baseline `.h5ad` under
  `results/stagate/<section_id>/`.
- Report ARI, NMI, AMI, and silhouette for the mclust labels on STAGATE
  embeddings.

## DLPFC Seed-0 Commands

Run these on the remote server only:

```bash
python experiments/baseline/dlpfc_stagate/run_phase0_dlpfc.py --section-id 151673 --seed 0 --device cuda:7
python experiments/baseline/dlpfc_stagate/run_phase0_dlpfc.py --section-id 151674 --seed 0 --device cuda:7
python experiments/baseline/dlpfc_stagate/run_phase0_dlpfc.py --section-id 151676 --seed 0 --device cuda:7
```

Use `--data-root`, `--r-home`, `--r-user`, or `--device` to match the server
environment when needed.

## Required Artifacts

Each section output should contain:

- `config.json`
- `metrics.json`
- `<section_id>_stagate.h5ad`
- `ground_truth_spatial.png`
- `spatial_network_stats.png`
- `umap_clusters.png`
- `spatial_clusters.png`
- `paga_trajectory.png`, unless `--skip-paga` is used

The `metrics.json` and `config.json` files include preprocessing settings,
spatial graph parameters, STAGATE training parameters, mclust parameters,
runtime dependency versions, and the Git commit when available.
