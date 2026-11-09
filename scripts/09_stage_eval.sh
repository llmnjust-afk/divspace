#!/usr/bin/env bash
# Re-runs only the evaluation stages for models whose direction artifact exists.
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${1:-${MODELS:-}}
[ -n "$MODELS_LIST" ] || { echo "no models given" >&2; exit 2; }
export MODELS="$MODELS_LIST"

for stage in 03_steer 04_controls 04b_ablate 05_metrics; do
  echo "########## $stage ##########"
  bash "scripts/${stage}.sh"; echo "${stage} rc=$?"
done
