#!/usr/bin/env bash
# Stage 3: layer x coefficient steering sweep on the layers that attribution ranked highest.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct"}
LAYERS=${LAYERS:-""}
COEFS=${COEFS:--3 -2 -1 1 2 3}
N_SAMPLES=${N_SAMPLES:-8}
GPUS=${GPUS:-4}

read -r -a ARR <<< "$MODELS_LIST"
i=0
pids=()
for MODEL in "${ARR[@]}"; do
  GPU=$(( i % GPUS ))
  SLUG=$(echo "$MODEL" | tr '/' '_')
  LAYER_ARG=()
  if [ -n "$LAYERS" ]; then LAYER_ARG=(--layers "$LAYERS"); fi
  echo "[gpu${GPU}] steer ${MODEL}"
  CUDA_VISIBLE_DEVICES="$GPU" python -m divspace.cli steer \
    --model "$MODEL" --device cuda:0 \
    --artifact "results/${SLUG}/directions.pt" \
    "${LAYER_ARG[@]}" --coefs $COEFS --n-samples "$N_SAMPLES" --include-baseline \
    --out "results/${SLUG}/sweep.jsonl" \
    > "runs/steer_${SLUG}.log" 2>&1 &
  pids+=($!)
  i=$(( i + 1 ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit "$fail"
