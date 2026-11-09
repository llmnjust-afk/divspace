#!/usr/bin/env bash
# Stage 1: fit diversity directions. One model per GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct"}
N_SAMPLES=${N_SAMPLES:-16}
PROMPT_SET=${PROMPT_SET:-all}
BATCH_SIZE=${BATCH_SIZE:-4}
GPUS=${GPUS:-4}

read -r -a ARR <<< "$MODELS_LIST"
i=0
pids=()
for MODEL in "${ARR[@]}"; do
  GPU=$(( i % GPUS ))
  SLUG=$(echo "$MODEL" | tr '/' '_')
  mkdir -p "results/${SLUG}" "runs/extract"
  echo "[gpu${GPU}] extract ${MODEL}"
  CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON:-python3}" -m divspace.cli extract \
    --model "$MODEL" --device cuda:0 \
    --prompt-set "$PROMPT_SET" --n-samples "$N_SAMPLES" --batch-size "$BATCH_SIZE" \
    --out "results/${SLUG}/directions.pt" \
    > "runs/extract/${SLUG}.log" 2>&1 &
  pids+=($!)
  i=$(( i + 1 ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "extract finished (failures: $fail)"
exit "$fail"
