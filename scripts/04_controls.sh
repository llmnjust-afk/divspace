#!/usr/bin/env bash
# Stage 4: placebo controls. Run AFTER the main sweep, one process per card.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct"}
LAYERS=${LAYERS:-}
CONTROL_COEFS=${CONTROL_COEFS:--2 2}
N_SAMPLES=${N_SAMPLES:-8}
EVAL_SAMPLES=${EVAL_SAMPLES:-8}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-160}
PROMPT_SET=${PROMPT_SET:-all}
GPUS=${GPUS:-4}
mkdir -p runs

run_controls() {
  local model=$1 gpu=$2 slug=$3 control
  local layer_arg=()
  if [ -n "$LAYERS" ]; then layer_arg=(--layers "$LAYERS"); fi
  for control in random shuffled orthogonal; do
    echo "[gpu${gpu}] ${control} control for ${model}"
    CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON:-python3}" -m divspace.cli steer \
      --model "$model" --device cuda:0 \
      --artifact "results/${slug}/directions.pt" \
      ${layer_arg[@]+"${layer_arg[@]}"} --coefs $CONTROL_COEFS --control "$control" \
      --condition "$control" --n-samples "$EVAL_SAMPLES" --prompt-set "$PROMPT_SET" --batch-size "$BATCH_SIZE" --max-new-tokens "$MAX_NEW_TOKENS" \
      --out "results/${slug}/control_${control}.jsonl" \
      > "runs/control_${control}_${slug}.log" 2>&1 || return 1
  done
}

read -r -a ARR <<< "$MODELS_LIST"
i=0
pids=()
for MODEL in "${ARR[@]}"; do
  GPU=$(( i % GPUS ))
  SLUG=$(echo "$MODEL" | tr '/' '_')
  if [ ! -f "results/${SLUG}/directions.pt" ]; then echo "skip ${MODEL}: no direction artifact"; continue; fi
  run_controls "$MODEL" "$GPU" "$SLUG" &
  pids+=($!)
  i=$(( i + 1 ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "controls finished (failures: $fail)"
exit "$fail"
