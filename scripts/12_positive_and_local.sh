#!/usr/bin/env bash
# Confirmatory block 2, one model per card:
#   (a) positive control - a labelled yes/no direction through the same pipeline
#   (b) prompt-local steering against a random and a mismatched direction
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"Qwen/Qwen2.5-7B-Instruct EleutherAI/pythia-1b Qwen/Qwen2.5-1.5B-Instruct EleutherAI/pythia-410m"}
GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
COEFS=${COEFS:--2 2}
LOCAL_COEFS=${LOCAL_COEFS:--2 2}
PY=${PYTHON:-python3}
mkdir -p runs results

run_model() {
  local model=$1 gpu=$2 slug=$3 log layer artifact
  log="runs/c2_${slug}.log"
  artifact="results/${slug}/directions.pt"
  export CUDA_VISIBLE_DEVICES="$gpu"

  layer=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); L=d['layers']; print(L[len(L)//2])" \
    "results/${slug}/directions.json" 2>/dev/null)
  [ -n "$layer" ] || layer=12

  echo "[gpu${gpu}] ${model}: positive control (yes/no)"
  "$PY" -m divspace.cli contrast --model "$model" --device cuda:0 \
    --pairs data/yesno_fit.json --layers "$layer" --batch-size "$BATCH_SIZE" \
    --out "results/${slug}/contrast.pt" >> "$log" 2>&1 || return 1

  for control in none random shuffled orthogonal; do
    local extra=()
    if [ "$control" != "none" ]; then extra=(--control "$control" --condition "$control"); fi
    "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
      --artifact "results/${slug}/contrast.pt" --prompt-file data/yesno_test.txt \
      --coefs -4 -2 2 4 ${extra[@]+"${extra[@]}"} --include-baseline \
      --n-samples 8 --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
      --out "results/${slug}/yesno_${control}.jsonl" >> "$log" 2>&1 \
      || echo "warn: ${control} control failed for ${model}"
  done

  echo "[gpu${gpu}] ${model}: prompt-local steering"
  "$PY" -m divspace.cli local --model "$model" --device cuda:0 \
    --artifact "$artifact" --layer "$layer" --prompt-set all \
    --coefs $LOCAL_COEFS --n-fit 16 --n-samples 8 --batch-size "$BATCH_SIZE" \
    --max-new-tokens 160 --out "results/${slug}/local.jsonl" >> "$log" 2>&1 || return 1

  echo "[gpu${gpu}] ${model}: metrics"
  "$PY" -m divspace.cli metrics --model "$model" --device cuda:0 \
    --results "results/${slug}/local.jsonl" --artifact "$artifact" \
    --batch-size "$BATCH_SIZE" --out "results/${slug}/metrics_local.json" >> "$log" 2>&1 || return 1

  "$PY" analysis/yesno.py --results "results/${slug}/yesno_none.jsonl" \
    --out "results/${slug}/yesno_real.json" > "runs/yesno_${slug}.txt" 2>&1 || true
  for control in random shuffled orthogonal; do
    "$PY" analysis/yesno.py --results "results/${slug}/yesno_${control}.jsonl" \
      --out "results/${slug}/yesno_${control}.json" >> "runs/yesno_${slug}.txt" 2>&1 || true
  done
}

read -r -a ARR <<< "$MODELS_LIST"
i=0
pids=()
for MODEL in "${ARR[@]}"; do
  GPU=$(( i % GPUS ))
  SLUG=$(echo "$MODEL" | tr '/' '_')
  if [ ! -f "results/${SLUG}/directions.pt" ]; then echo "skip ${MODEL}: no artifact"; i=$((i+1)); continue; fi
  run_model "$MODEL" "$GPU" "$SLUG" &
  pids+=($!)
  i=$(( i + 1 ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "confirm block 2 finished (failures: $fail)"
exit "$fail"
