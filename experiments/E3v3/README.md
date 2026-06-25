# E3v3: Boundary-aware Prototype-Guided Soft Graph Refinement

This directory contains an isolated E3v3 implementation. It does not modify
`STAGATE_pyG/`, so the official STAGATE baseline remains available for direct
reproduction.

## Implementation scope

The current MVP implements:

1. STAGATE-compatible warm-up training.
2. Soft domain assignment from warm-up embeddings.
3. Pseudo-boundary score from assignment uncertainty and local inconsistency.
4. Boundary-only prototype margin loss.
5. Boundary-aware soft edge gate.
6. Gate budget loss and preserve loss.
7. Optional clustering and metric export in the server-side runner.

The current MVP intentionally does not implement expression pruning, random
pruning, hard-pruning training, MMD/OT priors, long-range edges, or multi-scale
graphs.

## Files

- `boundary.py`: pseudo-boundary mining, prototype loss, gate diagnostics.
- `gated_gat_conv.py`: STAGATE-compatible GAT layer with optional soft edge
  gate support.
- `model.py`: E3v3 STAGATE autoencoder and boundary-aware edge gate.
- `train_e3v3.py`: two-stage warm-up and E3v3 training function.
- `run_dlpfc_e3v3.py`: server-side command-line entry point.

## Design notes

The warm-up uses the same architecture shape as official STAGATE. During stage
2, the gate is computed from detached warm-up or previous-epoch embeddings to
avoid circular dependence between the gate and the encoder output. The gate is
then applied only to the encoder attention propagation layer. Decoder behavior
is left STAGATE-compatible.

Soft assignments are estimated from warm-up embeddings without using ground
truth labels. The default uses `sklearn.mixture.GaussianMixture`; if that fails,
the code falls back to a KMeans-distance softmax assignment.

## Example server command

Run this only on the remote server according to the project protocol:

```bash
python experiments.E3v3.run_dlpfc_e3v3.py \
  --input-h5ad results/stagate/151674/151674_stagate.h5ad \
  --output-dir results/E3v3/151674 \
  --n-clusters 7 \
  --seed 0 \
  --warmup-epochs 500 \
  --stage2-epochs 500 \
  --rad-cutoff 150 \
  --truth-key GroundTruth \
  --run-mclust
```

The script writes an `.h5ad` result and a `metrics.json` file under the output
directory. Do not commit generated `.h5ad`, model weights, figures, or other
large run artifacts.
