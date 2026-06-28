# Oracle-BES-STAGATE

This directory contains an isolated oracle feasibility / sanity-check
experiment. It does not modify the official `STAGATE_pyG/` baseline.

The experiment asks whether ground-truth-label-guided BES-style boundary
shaping can improve STAGATE embeddings on DLPFC. Ground truth labels are used
only to define oracle boundary loss and prototypes. Final labels are still
obtained by running mclust on the embedding.

## Experiments

- `O0`: baseline STAGATE embedding from `adata.obsm["STAGATE"]`, then mclust.
- `O1`: all labeled spots supervised prototype loss.
- `O2`: GT-boundary BES loss with adjacent GT-domain negatives.
- `O3`: O2 plus GT-interior preservation loss.
- `O4`: random-boundary negative control with the same boundary count as O2.

## Training modes

- `frozen_adapter`: trains only a residual shaping head on the existing
  baseline embedding in `--input-h5ad`. This is the least invasive first pass.
- `warmup_last_layer`: trains an experiment-local STAGATE wrapper from the
  input expression matrix, then freezes most parameters and trains the last
  encoder layer plus the shaping head. This mode is server-only and may be slow.

## Example server command

```bash
python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py \
  --sample-id 151676 \
  --input-h5ad results/stagate/151676/151676_stagate.h5ad \
  --experiment O3 \
  --clusters 7 \
  --seed 0 \
  --device cuda:7 \
  --gamma 0.05 \
  --lambda-bes 0.05 \
  --lambda-pres 1.0 \
  --temperature 0.5 \
  --training-mode frozen_adapter
```

Use `--training-mode warmup_last_layer` only on the remote server after the
frozen adapter pass is inspected.

## Outputs

Each run writes:

```text
results/oracle_bes_stagate/{sample_id}/seed_{seed}/{experiment}/
  config.yaml
  config.json
  metrics.csv
  training_log.csv
  boundary_stats.csv
  correction_stats.csv
  labels.csv
  embeddings.h5ad
```

`embeddings.h5ad` stores:

- `adata.obsm["STAGATE"]`
- `adata.obsm["Oracle_BES_STAGATE"]`
- `adata.obs["gt_label"]`
- `adata.obs["gt_boundary_score"]`
- `adata.obs["is_gt_boundary"]`
- `adata.obs["is_gt_interior"]`
- `adata.obs["stagate_mclust"]`
- `adata.obs["oracle_bes_mclust"]`
- `adata.obs["changed_label"]`
- `adata.obs["correct_before"]`
- `adata.obs["correct_after"]`

## First-round order

Run on the server only:

```bash
bash experiments/improved/oracle_bes_stagate/scripts/run_151676.sh
```

If `O2` / `O3` do not improve over `O0`, pause. If they improve, continue with
`151673`, then `151674` and `151507`.

## O6 hard boundary-gated mechanism check

After the frozen-adapter No-Go result, O6 is the final narrow mechanism check:
interior spots are preserved by construction, not by a preservation loss.

```text
z'_i = z_i                         for GT-interior spots
z'_i = z_i + gamma * h(z_i)         for GT-boundary spots
```

Run only on `151676` first:

```bash
bash experiments/improved/oracle_bes_stagate/scripts/run_151676_o6.sh
```

The O6 output includes the normal full-mclust metrics on `Oracle_BES_STAGATE`
and additional `boundary_relabel_*` metrics for boundary-only fixed-prototype
assignment, where interior labels are copied from O0 and only GT-boundary spots
are reassigned to fixed O0 interior cluster prototypes.