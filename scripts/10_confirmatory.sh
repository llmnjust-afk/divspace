#!/usr/bin/env bash
# Three confirmatory experiments, one model per card:
#   1. re-extract, now recording per-prompt direction agreement
#   2. ablation placebos (random / shuffled / orthogonal direction removed)
#   3. low-temperature sweep plus matched low-temperature controls
set -uo pipefail
cd "$(dirname "$0")/.."

WAVE=${1:-1}
WAVE1=${WAVE1:-"Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct EleutherAI/pythia-2.8b HuggingFaceTB/SmolLM2-1.7B-Instruct"}
WAVE2=${WAVE2:-"EleutherAI/pythia-1b Qwen/Qwen2.5-1.5B-Instruct EleutherAI/pythia-410m EleutherAI/pythia-160m"}
GPUS=${GPUS:-4}
N_SAMPLES=${N_SAMPLES:-16}
EVAL_SAMPLES=${EVAL_SAMPLES:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
LOW_T=${LOW_T:-0.4}
COEFS=${COEFS:--3 -2 -1 1 2 3}
CONTROL_COEFS=${CONTROL_COEFS:--2 2}
PY=${PYTHON:-python3}
mkdir -p runs results

if [ "$WAVE" = "1" ]; then MODELS_LIST=$WAVE1; else MODELS_LIST=$WAVE2; fi

run_model() {
  local model=$1 gpu=$2 slug=$3 log
  log="runs/confirm_${slug}.log"
  export CUDA_VISIBLE_DEVICES="$gpu"

  echo "[gpu${gpu}] ${model}: extraction"
  "$PY" -m divspace.cli extract --model "$model" --device cuda:0 \
    --prompt-set all --n-samples "$N_SAMPLES" --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" --min-samples 8 \
    --out "results/${slug}/directions.pt" > "runs/extract_${slug}.log" 2>&1 || return 1

  echo "[gpu${gpu}] ${model}: ablation placebos"
  for control in random shuffled orthogonal; do
    "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
      --artifact "results/${slug}/directions.pt" --mode ablate --control "$control" \
      --condition "ablate_${control}" --prompt-set all --n-samples "$EVAL_SAMPLES" \
      --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
      --out "results/${slug}/ablate_${control}.jsonl" >> "$log" 2>&1 || return 1
  done

  echo "[gpu${gpu}] ${model}: low-temperature sweep at T=${LOW_T}"
  "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
    --artifact "results/${slug}/directions.pt" --prompt-set all \
    --coefs $COEFS --n-samples "$EVAL_SAMPLES" --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" --temperature "$LOW_T" --include-baseline \
    --out "results/${slug}/sweep_lt.jsonl" >> "$log" 2>&1 || return 1

  for control in random shuffled orthogonal; do
    "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
      --artifact "results/${slug}/directions.pt" --control "$control" --condition "$control" \
      --coefs $CONTROL_COEFS --prompt-set all --n-samples "$EVAL_SAMPLES" \
      --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$LOW_T" \
      --out "results/${slug}/lt_control_${control}.jsonl" >> "$log" 2>&1 || return 1
  done

  echo "[gpu${gpu}] ${model}: metrics"
  for name in ablate_random ablate_shuffled ablate_orthogonal sweep_lt lt_control_random lt_control_shuffled lt_control_orthogonal; do
    "$PY" -m divspace.cli metrics --model "$model" --device cuda:0 \
      --results "results/${slug}/${name}.jsonl" --artifact "results/${slug}/directions.pt" \
      --batch-size "$BATCH_SIZE" --out "results/${slug}/metrics_${name}.json" \
      >> "$log" 2>&1 || return 1
  done
}

read -r -a ARR <<< "$MODELS_LIST"
i=0
pids=()
for MODEL in "${ARR[@]}"; do
  GPU=$(( i % GPUS ))
  SLUG=$(echo "$MODEL" | tr '/' '_')
  if [ ! -f "results/${SLUG}/directions.pt" ]; then echo "skip ${MODEL}: no direction artifact"; i=$((i+1)); continue; fi
  run_model "$MODEL" "$GPU" "$SLUG" &
  pids+=($!)
  i=$(( i + 1 ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "wave ${WAVE} finished (failures: $fail)"
exit "$fail"
