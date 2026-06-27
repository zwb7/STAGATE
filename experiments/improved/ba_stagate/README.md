# Phase 1 Boundary Diagnostics

Phase 1 tests whether BA-STAGATE's premise is plausible before training any
adapter:

- Do STAGATE pseudo-label errors concentrate near ground-truth boundaries?
- Does a pseudo-label boundary score recover those regions?
- Are mclust posterior confidence values informative enough for later
  prototype selection?

The scripts in this directory read Phase 0 `.h5ad` files only. They do not
train STAGATE, train BA-STAGATE, use a GPU, or modify `STAGATE_pyG/`.

## Server Commands

Run after Phase 0 artifacts exist:

```bash
python experiments/improved/ba_stagate/boundary_diagnostics.py --sample-id 151673 --input-h5ad results/stagate/151673/151673_stagate.h5ad --clusters 7 --seed 0
python experiments/improved/ba_stagate/boundary_diagnostics.py --sample-id 151674 --input-h5ad results/stagate/151674/151674_stagate.h5ad --clusters 7 --seed 0
python experiments/improved/ba_stagate/boundary_diagnostics.py --sample-id 151676 --input-h5ad results/stagate/151676/151676_stagate.h5ad --clusters 7 --seed 0
```

Use `--r-home` and `--r-user` if the server R environment needs them.

## Outputs

Each sample writes to:

```text
results/ba_stagate/phase1_boundary_diagnostics/<sample_id>/
```

Required files:

- `boundary_scores.csv`
- `diagnostics_metrics.json`
- `mclust_posterior.npy`
- `<sample_id>_phase1_diagnostics.h5ad`
- `figures/spatial_boundary_score.png`
- `figures/embedding_boundary_score.png`
- `figures/combined_boundary_score.png`
- `figures/confidence_spatial.png`
- `figures/error_spatial.png`
- `figures/gt_boundary_spatial.png`
- `figures/pseudo_boundary_spatial.png`

## Decision Rule

Proceed to Phase 2 only if high combined-boundary-score spots have a clearly
higher error rate than pseudo interior spots, pseudo boundaries overlap with
GT boundaries, and boundary ARI is meaningfully worse than interior ARI.
