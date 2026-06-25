# E3 Soft Gate Summary

## ARI by method

| Method | 151674 | 151676 | HBC | Mean ARI | Mean Δ vs Original |
|---|---:|---:|---:|---:|---:|
| baseline | 0.6147 | 0.6194 | 0.4556 | 0.5632 |  |
| original | 0.4538 | 0.5192 | 0.4563 | 0.4764 |  |
| oracle | 0.6536 | 0.6420 | 0.4671 | 0.5876 |  |
| expression | 0.3681 | 0.4377 | 0.4422 | 0.4160 |  |
| embedding | 0.4463 | 0.4321 | 0.4357 | 0.4380 |  |
| cluster_free | 0.4750 | 0.6177 | 0.4451 | 0.5126 | 0.0362 |
| extra_training | 0.6265 | 0.4256 | 0.4831 | 0.5117 | 0.0353 |
| gate_only | 0.6833 | 0.5813 | 0.4415 | 0.5687 | 0.0922 |
| gate_distribution | 0.3567 | 0.4306 | 0.4622 | 0.4165 | -0.0599 |
| full | 0.3882 | 0.4330 | 0.4988 | 0.4400 | -0.0365 |

## Gate diagnosis: full variant

| Sample | Mean gate | Min eff. degree | Gate AUC | Low-5 enrich | Low-5 CDR | Low-5 SDR |
|---|---:|---:|---:|---:|---:|---:|
| 151674 | 1.0000 | 0.0000 | 0.5063 | 1.0000 | 1.0000 | 1.0000 |
| 151676 | 0.9998 | 0.0000 | 0.4708 | 0.7806 | 0.0391 | 0.0513 |
| HBC | 0.9999 | 2.0000 | 0.5205 | 1.4628 | 0.0735 | 0.0467 |

## Quick interpretation

- Full mean Δ vs original: `-0.0365`.
- Full mean ARI advantage over extra_training: `-0.0717`.
- Full mean bottom-5 low-gate enrichment: `1.0811`.
- Full mean bottom-5 CDR minus SDR: `0.0049`.

Decision rules:

- If `full` does not exceed `extra_training`, the gate mechanism is not yet supported.
- If bottom-5 enrichment is not greater than 1, low-gate edges are not enriched for cross-domain edges.
- If bottom-5 CDR is not greater than SDR, do not proceed to hard pruning.
- If gate AUC is <= 0.5, treat the gate as mechanistically invalid even if ARI improves.
