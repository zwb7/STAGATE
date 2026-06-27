# Phase 2 Gated Adapter Commands

Use this entrypoint after the first smoke test showed that the ungated adapter
perturbed interior spots more than boundary spots.

## Smoke Test

```bash
python experiments/improved/ba_stagate/phase2_adapter_gated.py --sample-id 151673 --input-h5ad results/stagate/151673/151673_stagate.h5ad --boundary-scores results/ba_stagate/phase1/151673/boundary_scores.csv --experiment all_spot --clusters 7 --seed 0 --device cuda:7
python experiments/improved/ba_stagate/phase2_adapter_gated.py --sample-id 151673 --input-h5ad results/stagate/151673/151673_stagate.h5ad --boundary-scores results/ba_stagate/phase1/151673/boundary_scores.csv --experiment boundary_only --clusters 7 --seed 0 --device cuda:7
```

The output directory is:

```text
results/ba_stagate/phase2_adapter_gated/<sample_id>/seed_<seed>/<experiment>/
```

Pass criteria:

- `boundary_only` has `residual_gate.active_spots == n_boundary_spots`
- `boundary_only` has `perturbation.gt_boundary_mean_l2 > perturbation.gt_interior_mean_l2`
- `boundary_only` has `perturbation.interior_changed_label_ratio <= 0.10`
- `metric_delta.gt_interior_ari >= -0.01`
- `ba_metrics.gt_boundary_ari` should improve or not materially degrade
