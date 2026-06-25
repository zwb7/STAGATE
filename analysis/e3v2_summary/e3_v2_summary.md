# E3-v2 Result Summary

## Variant-level summary

| variant | n_runs | ari_mean | delta_vs_original_mean | gate_auc_for_cross_domain_mean | low5_enrichment_mean | low5_cdr_over_sdr_mean |
| --- | --- | --- | --- | --- | --- | --- |
| extra_training | 1 | 0.399530 | -0.054309 | 0.500000 | 1.000000 | 1.000000 |
| current_gate_only | 1 | 0.476956 | 0.023118 | 0.412291 | 0.000000 | 0.000000 |
| stabilized_unnormalized | 1 | 0.460111 | 0.006272 | 0.541463 | 1.866491 | 2.053733 |
| stabilized_renormalized | 1 | 0.514725 | 0.060887 | 0.528626 | 1.072240 | 1.080452 |
| uniform_gate | 1 | 0.417578 | -0.036261 | 0.500000 | 1.000000 | 1.000000 |
| shuffled_gate | 1 | 0.402680 | -0.051158 | 0.509879 | 0.873677 | 0.862216 |

## Run-level summary

| sample_id | variant | seed | ari | delta_vs_original | mean_gate | gate_auc_for_cross_domain | low5_enrichment | low5_cdr_over_sdr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 151674 | extra_training | 0 | 0.399530 | -0.054309 | 1.000000 | 0.500000 | 1.000000 | 1.000000 |
| 151674 | current_gate_only | 0 | 0.476956 | 0.023118 | 0.978566 | 0.412291 | 0.000000 | 0.000000 |
| 151674 | stabilized_unnormalized | 0 | 0.460111 | 0.006272 | 0.945976 | 0.541463 | 1.866491 | 2.053733 |
| 151674 | stabilized_renormalized | 0 | 0.514725 | 0.060887 | 0.944218 | 0.528626 | 1.072240 | 1.080452 |
| 151674 | uniform_gate | 0 | 0.417578 | -0.036261 | 0.950000 | 0.500000 | 1.000000 | 1.000000 |
| 151674 | shuffled_gate | 0 | 0.402680 | -0.051158 | 0.944609 | 0.509879 | 0.873677 | 0.862216 |

## Minimal decision checks

- Main method should be `stabilized_renormalized`.
- It should outperform `extra_training` on mean ARI.
- It should not be matched by `uniform_gate` or `shuffled_gate`.
- Mechanism is weak if low-gate enrichment <= 1 or Gate AUC <= 0.5.
