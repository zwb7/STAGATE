# Experiments

This directory stores reproducible experiment entrypoints and comparison-method
scripts. Keep official baseline code in `STAGATE_pyG/`; put runnable experiment
wrappers, method-specific scripts, configs, and documentation here.

Suggested layout:

```text
experiments/
  baseline/
    seurat/
    stagate/
  improved/
  comparisons/
```

Guidelines:

- Do not commit datasets, `.h5ad` files, model weights, generated figures, or
  large result artifacts.
- Save run outputs under `results/`, not under `STAGATE_pyG/`.
- Record sample IDs, random seeds, software versions, parameters, logs, metrics,
  and the Git commit used for each experiment.
- Keep baseline and improved-method code/configuration isolated.
- Run full data loading, model training, inference, and evaluation only on the
  remote server.
