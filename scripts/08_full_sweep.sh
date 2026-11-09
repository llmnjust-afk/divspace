#!/usr/bin/env bash
# Two waves of models on four cards, then one publication at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs results figures

WAVE1=${WAVE1:-"Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct EleutherAI/pythia-2.8b HuggingFaceTB/SmolLM2-1.7B-Instruct"}
WAVE1_STAGES=${WAVE1_STAGES:-"03_steer 04_controls 04b_ablate 05_metrics"}
WAVE2=${WAVE2:-"EleutherAI/pythia-1b Qwen/Qwen2.5-1.5B-Instruct EleutherAI/pythia-410m EleutherAI/pythia-160m"}

missing=0
for model in $WAVE1; do
  slug=$(echo "$model" | tr '/' '_')
  [ -f "results/${slug}/directions.pt" ] || missing=$((missing + 1))
done

echo "########## wave 1 ##########"
if [ "$missing" -eq 0 ]; then
  echo "reusing existing direction artifacts"
  STAGES="$WAVE1_STAGES" bash scripts/09_stage_eval.sh "$WAVE1" 2>&1 | tee runs/wave1.log | tail -4
else
  MODELS="$WAVE1" PUBLISH=0 bash scripts/07_run_all.sh 2>&1 | tee runs/wave1.log | tail -4
fi

echo "########## wave 2 ##########"
MODELS="$WAVE2" PUBLISH=0 bash scripts/07_run_all.sh 2>&1 | tee runs/wave2.log | tail -4

echo "########## summary ##########"
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
if [ "${PUBLISH:-1}" = "1" ]; then
  bash scripts/06_publish.sh || echo "publish failed"
fi
echo "########## finished ##########"
