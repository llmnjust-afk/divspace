#!/usr/bin/env bash
# Confirmatory block: both waves, then every comparison and a publication.
set -uo pipefail
cd "$(dirname "$0")/.."

for wave in 1 2; do
  echo "########## confirm wave ${wave} ##########"
  bash scripts/10_confirmatory.sh "$wave" 2>&1 | tee "runs/confirm_wave${wave}.log" | tail -3
done

echo "########## analysis ##########"
PYTHONPATH=. "${PYTHON:-python3}" analysis/compare_controls.py --results results --out results/CONTROLS.md || true
PYTHONPATH=. "${PYTHON:-python3}" analysis/summarise.py --results results --out results/SUMMARY.md || true
shopt -s nullglob
attr_files=(results/*/attribution.json)
metric_files=(results/*/metrics_sweep.json)
shopt -u nullglob
if [ ${#attr_files[@]} -gt 0 ] && [ ${#metric_files[@]} -gt 0 ]; then
  PYTHONPATH=. "${PYTHON:-python3}" analysis/plot_pareto.py \
    --attribution "${attr_files[@]}" --metrics "${metric_files[@]}" --out-dir figures || true
fi

echo "########## publish ##########"
bash scripts/06_publish.sh || echo "publish failed"
echo "########## finished ##########"
