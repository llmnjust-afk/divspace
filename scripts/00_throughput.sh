#!/usr/bin/env bash
# Week 1: what fits on one 4090, and how fast does it run.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-0}
MAX_NEW=${MAX_NEW:-128}

for MODEL in "${MODELS:-EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct}"; do
  SLUG=$(echo "$MODEL" | tr '/' '_')
  echo "== $MODEL =="
  CUDA_VISIBLE_DEVICES="$GPU" "${PYTHON:-python3}" -m divspace.cli probe \
    --model "$MODEL" --device cuda:0 --max-new-tokens "$MAX_NEW" \
    --out "results/probe/${SLUG}.json"
done
