# Recommended Phase 2 Commands

Use `phase2_adapter_aligned.py` for reported runs. It trains the same adapter
as `phase2_adapter.py`, but computes `interior_changed_label_ratio` after
Hungarian alignment between baseline pseudo labels and BA-STAGATE mclust labels.

## Smoke Test

```bash
python experiments/improved/ba_stagate/phase2_adapter_aligned.py --sample-id 151673 --input-h5ad results/stagate/151673/151673_stagate.h5ad --boundary-scores results/ba_stagate/phase1_boundary_diagnostics/151673/boundary_scores.csv --experiment all_spot --clusters 7 --seed 0 --device cuda:7
python experiments/improved/ba_stagate/phase2_adapter_aligned.py --sample-id 151673 --input-h5ad results/stagate/151673/151673_stagate.h5ad --boundary-scores results/ba_stagate/phase1_boundary_diagnostics/151673/boundary_scores.csv --experiment boundary_only --clusters 7 --seed 0 --device cuda:7
```

## First Main Round

```bash
python experiments/improved/ba_stagate/phase2_adapter_aligned.py --sample-id 151676 --input-h5ad results/stagate/151676/151676_stagate.h5ad --boundary-scores results/ba_stagate/phase1_boundary_diagnostics/151676/boundary_scores.csv --experiment all_spot --clusters 7 --seed 0 --device cuda:7
python experiments/improved/ba_stagate/phase2_adapter_aligned.py --sample-id 151676 --input-h5ad results/stagate/151676/151676_stagate.h5ad --boundary-scores results/ba_stagate/phase1_boundary_diagnostics/151676/boundary_scores.csv --experiment boundary_only --clusters 7 --seed 0 --device cuda:7
```

## E3 After E2 Passes

```bash
python experiments/improved/ba_stagate/phase2_adapter_aligned.py --sample-id 151673 --input-h5ad results/stagate/151673/151673_stagate.h5ad --boundary-scores results/ba_stagate/phase1_boundary_diagnostics/151673/boundary_scores.csv --experiment boundary_adjacent --clusters 7 --seed 0 --device cuda:7
```
