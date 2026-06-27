# Phase 2 Adapter Commands

Phase 2 trains only a frozen post-hoc adapter on STAGATE embeddings. It does
not update `STAGATE_pyG/` and does not use ground truth during training.

## Smoke Test

Run `151673`, seed `0`, first:

```bash
python experiments/improved/ba_stagate/phase2_adapter.py \
  --sample-id 151673 \
  --input-h5ad results/stagate/151673/151673_stagate.h5ad \
  --boundary-scores results/ba_stagate/phase1/151673/boundary_scores.csv \
  --experiment all_spot \
  --clusters 7 \
  --seed 0 \
  --device cuda:7

python experiments/improved/ba_stagate/phase2_adapter.py \
  --sample-id 151673 \
  --input-h5ad results/stagate/151673/151673_stagate.h5ad \
  --boundary-scores results/ba_stagate/phase1/151673/boundary_scores.csv \
  --experiment boundary_only \
  --clusters 7 \
  --seed 0 \
  --device cuda:7
```

Check `metrics_phase2.json` before running more samples:

- `perturbation.gt_boundary_mean_l2 > perturbation.gt_interior_mean_l2`
- `perturbation.interior_changed_label_ratio <= 0.10`
- `ba_metrics.gt_boundary_ari` improves or does not degrade materially
- `metric_delta.gt_interior_ari >= -0.01`

## E1/E2 Main Round

Run `all_spot` and `boundary_only` for `151673` and `151676` first. Add
`151674` later as a stress case.

```bash
python experiments/improved/ba_stagate/phase2_adapter.py --sample-id 151676 --input-h5ad results/stagate/151676/151676_stagate.h5ad --boundary-scores results/ba_stagate/phase1/151676/boundary_scores.csv --experiment all_spot --clusters 7 --seed 0 --device cuda:7
python experiments/improved/ba_stagate/phase2_adapter.py --sample-id 151676 --input-h5ad results/stagate/151676/151676_stagate.h5ad --boundary-scores results/ba_stagate/phase1/151676/boundary_scores.csv --experiment boundary_only --clusters 7 --seed 0 --device cuda:7
```

## E3

Only run E3 after E2 is better than E0/E1:

```bash
python experiments/improved/ba_stagate/phase2_adapter.py --sample-id 151673 --input-h5ad results/stagate/151673/151673_stagate.h5ad --boundary-scores results/ba_stagate/phase1/151673/boundary_scores.csv --experiment boundary_adjacent --clusters 7 --seed 0 --device cuda:7
```
