# Recommended Phase 1 Commands

Use `phase1_boundary_diagnostics.py` as the Phase 1 entrypoint. It includes
Hungarian label matching before computing per-spot pseudo-label errors.

```bash
python experiments/improved/ba_stagate/phase1_boundary_diagnostics.py --sample-id 151673 --input-h5ad results/stagate/151673/151673_stagate.h5ad --clusters 7 --seed 0
python experiments/improved/ba_stagate/phase1_boundary_diagnostics.py --sample-id 151674 --input-h5ad results/stagate/151674/151674_stagate.h5ad --clusters 7 --seed 0
python experiments/improved/ba_stagate/phase1_boundary_diagnostics.py --sample-id 151676 --input-h5ad results/stagate/151676/151676_stagate.h5ad --clusters 7 --seed 0
```

Outputs are written under:

```text
results/ba_stagate/phase1_boundary_diagnostics/<sample_id>/
```
