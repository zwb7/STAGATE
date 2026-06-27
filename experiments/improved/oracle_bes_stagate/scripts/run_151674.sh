#!/usr/bin/env bash
set -euo pipefail

SAMPLE_ID=151674
INPUT_H5AD="results/stagate/${SAMPLE_ID}/${SAMPLE_ID}_stagate.h5ad"
COMMON_ARGS=(
  --sample-id "${SAMPLE_ID}"
  --input-h5ad "${INPUT_H5AD}"
  --clusters 7
  --seed 0
  --device cuda:7
  --training-mode frozen_adapter
  --gamma 0.05
  --lambda-bes 0.05
  --lambda-pres 1.0
  --temperature 0.5
)

python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" --experiment O0
python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" --experiment O1
python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" --experiment O2
python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" --experiment O3
python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" --experiment O4 --summary
