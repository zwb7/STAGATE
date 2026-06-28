#!/usr/bin/env bash
set -euo pipefail

SAMPLE_ID=151676
INPUT_H5AD="results/stagate/${SAMPLE_ID}/${SAMPLE_ID}_stagate.h5ad"

python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py \
  --sample-id "${SAMPLE_ID}" \
  --input-h5ad "${INPUT_H5AD}" \
  --experiment O6 \
  --run-tag hardmask_default \
  --clusters 7 \
  --seed 0 \
  --device cuda:7 \
  --training-mode frozen_adapter \
  --gamma 0.05 \
  --lambda-bes 0.05 \
  --lambda-mag 0.0 \
  --temperature 0.5 \
  --summary