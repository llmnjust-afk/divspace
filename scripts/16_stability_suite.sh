#!/usr/bin/env bash
# Confirmatory block 5: budget-matched and between-prompt stability.
# One generation pass per model answers three questions at once: is a within-prompt
# direction reproducible, does the pooled statistic survive the same per-arm budget,
# and is the pooled vector the between-prompt identity direction.
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"Qwen/Qwen2.5-7B-Instruct EleutherAI/pythia-1b Qwen/Qwen2.5-1.5B-Instruct EleutherAI/pythia-410m"}
GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
N_SAMPLES=${N_SAMPLES:-32}
SIZES=${SIZES:-"1,2,4,8,20,40"}
REPS=${REPS:-20}
PY=${PYTHON:-python3}
mkdir -p runs results

run_model() {
  local model=$1 gpu=$2 slug=$3 log layers
  log="runs/suite_${slug}.log"
  export CUDA_VISIBLE_DEVICES="$gpu"
  : > "$log"
  if [ ! -f "results/${slug}/directions.json" ]; then
    echo "skip ${model}: no directions.json" | tee -a "$log"
    return 1
  fi
  layers=$("$PY" -c "import json,sys; print(','.join(str(x) for x in json.load(open(sys.argv[1]))['layers']))" \
    "results/${slug}/directions.json") || return 1
  echo "[gpu${gpu}] ${model}: stability suite over layers ${layers}" | tee -a "$log"
  "$PY" -m divspace.cli stability-suite --model "$model" --device cuda:0 \
    --artifact "results/${slug}/directions.pt" --layers "$layers" \
    --n-samples "$N_SAMPLES" --sizes "$SIZES" --reps "$REPS" \
    --batch-size "$BATCH_SIZE" --max-new-tokens 160 \
    --out "results/${slug}/stability_suite.json" >> "$log" 2>&1
}

pids=()
i=0
for MODEL in $MODELS_LIST; do
  SLUG=$(echo "$MODEL" | tr '/' '_')
  run_model "$MODEL" "$(( i % GPUS ))" "$SLUG" &
  pids+=($!)
  i=$(( i + 1 ))
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done

"$PY" analysis/stability_suite.py --results results --out results/STABILITY_SUITE.md || fail=1
"$PY" analysis/pairwise_consistency.py --results results --out results/PAIRWISE.md || fail=1
echo "stability suite finished (failures: $fail)"
exit "$fail"
