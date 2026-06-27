#!/usr/bin/env bash
set -euo pipefail

SAMPLE_ID=151676
INPUT_H5AD="results/stagate/${SAMPLE_ID}/${SAMPLE_ID}_stagate.h5ad"
COMMON_ARGS=(
  --sample-id "${SAMPLE_ID}"
  --input-h5ad "${INPUT_H5AD}"
  --clusters 7
  --seed 0
  --device cuda:7
  --training-mode frozen_adapter
  --temperature 0.5
)

# Boundary-loss strength diagnostics.
python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" \
  --experiment O2 --run-tag gamma002_bes001 \
  --gamma 0.02 --lambda-bes 0.01 --lambda-pres 1.0

python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" \
  --experiment O2 --run-tag gamma002_bes005 \
  --gamma 0.02 --lambda-bes 0.05 --lambda-pres 1.0

python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" \
  --experiment O2 --run-tag gamma005_bes001 \
  --gamma 0.05 --lambda-bes 0.01 --lambda-pres 1.0

# O3 preservation diagnostics: test whether the default lambda_pres=1.0
# suppresses useful boundary movement too strongly.
python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" \
  --experiment O3 --run-tag gamma002_bes005_pres01 \
  --gamma 0.02 --lambda-bes 0.05 --lambda-pres 0.1

python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" \
  --experiment O3 --run-tag gamma002_bes005_pres02 \
  --gamma 0.02 --lambda-bes 0.05 --lambda-pres 0.2

python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" \
  --experiment O3 --run-tag gamma005_bes005_pres02 \
  --gamma 0.05 --lambda-bes 0.05 --lambda-pres 0.2

python experiments/improved/oracle_bes_stagate/run_oracle_bes_stagate.py "${COMMON_ARGS[@]}" \
  --experiment O3 --run-tag gamma005_bes005_pres05 \
  --gamma 0.05 --lambda-bes 0.05 --lambda-pres 0.5 --summary