#!/usr/bin/env bash
# Stage 4: placebo controls. Run AFTER the main sweep, on the winning layer only.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct"}
LAYERS=${LAYERS:-16}
COEFS=${COEFS:--2 2}
GPUS=${GPUS:-4}

read -r -a ARR <<< "$MODELS_LIST"
i=0
pids=()
for MODEL in "${ARR[@]}"; do
  SLUG=$(echo "$MODEL" | tr '/' '_')
  k=0
  for CONTROL in random shuffled orthogonal; do
    GPU=$(( (i + k) % GPUS ))
    echo "[gpu${GPU}] ${CONTROL} control for ${MODEL}"
    CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON:-python3}" -m divspace.cli steer \
      --model "$MODEL" --device cuda:0 \
      --artifact "results/${SLUG}/directions.pt" \
      --layers "$LAYERS" --coefs $COEFS --control "$CONTROL" \
      --condition "$CONTROL" --n-samples 8 \
      --out "results/${SLUG}/control_${CONTROL}.jsonl" \
      > "runs/control_${CONTROL}_${SLUG}.log" 2>&1 &
    pids+=($!)
    k=$(( k + 1 ))
  done
  i=$(( i + 1 ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit "$fail"
