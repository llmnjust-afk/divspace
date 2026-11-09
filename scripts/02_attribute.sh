#!/usr/bin/env bash
# Stage 2: attribution patching over layers.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct"}
N_PAIRS=${N_PAIRS:-48}
GPUS=${GPUS:-4}
mkdir -p runs

read -r -a ARR <<< "$MODELS_LIST"
i=0
pids=()
for MODEL in "${ARR[@]}"; do
  GPU=$(( i % GPUS ))
  SLUG=$(echo "$MODEL" | tr '/' '_')
  if [ ! -f "results/${SLUG}/directions.pt" ]; then echo "skip ${MODEL}: no direction artifact"; continue; fi
  echo "[gpu${GPU}] attribute ${MODEL}"
  CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON:-python3}" -m divspace.cli attribute \
    --model "$MODEL" --device cuda:0 \
    --artifact "results/${SLUG}/directions.pt" \
    --n-pairs "$N_PAIRS" \
    --out "results/${SLUG}/attribution.json" \
    > "runs/attribute_${SLUG}.log" 2>&1 &
  pids+=($!)
  i=$(( i + 1 ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit "$fail"
