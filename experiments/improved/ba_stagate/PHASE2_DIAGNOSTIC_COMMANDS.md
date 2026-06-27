# Phase 2 Diagnostic Commands

These diagnostics should run before E3, multi-seed, or stress-case expansion.

## Mclust Jitter Stability

```bash
python experiments/improved/ba_stagate/phase2_mclust_stability.py --sample-id 151673 --input-h5ad results/stagate/151673/151673_stagate.h5ad --clusters 7 --seed 0
python experiments/improved/ba_stagate/phase2_mclust_stability.py --sample-id 151676 --input-h5ad results/stagate/151676/151676_stagate.h5ad --clusters 7 --seed 0
```

Outputs:

```text
results/ba_stagate/phase2_diagnostics/mclust_stability/<sample_id>/seed_0/
```

## Fixed-Prototype Evaluation

Run this against existing gated E2 embeddings:

```bash
python experiments/improved/ba_stagate/phase2_fixed_prototype_eval.py --sample-id 151673 --experiment boundary_only --input-h5ad results/stagate/151673/151673_stagate.h5ad --boundary-scores results/ba_stagate/phase1_boundary_diagnostics/151673/boundary_scores.csv --ba-embedding results/ba_stagate/phase2_adapter_gated/151673/seed_0/boundary_only/BA_STAGATE.npy
python experiments/improved/ba_stagate/phase2_fixed_prototype_eval.py --sample-id 151676 --experiment boundary_only --input-h5ad results/stagate/151676/151676_stagate.h5ad --boundary-scores results/ba_stagate/phase1_boundary_diagnostics/151676/boundary_scores.csv --ba-embedding results/ba_stagate/phase2_adapter_gated/151676/seed_0/boundary_only/BA_STAGATE.npy
```

Outputs:

```text
results/ba_stagate/phase2_diagnostics/fixed_prototype/<sample_id>/boundary_only/
```
