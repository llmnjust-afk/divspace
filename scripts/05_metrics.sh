#!/usr/bin/env bash
# Stage 5: score every results file, one model at a time per card.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct"}
BATCH_SIZE=${BATCH_SIZE:-4}
GPUS=${GPUS:-4}
mkdir -p runs

score_model() {
  local model=$1 gpu=$2 slug=$3 name path
  for name in sweep control_random control_shuffled control_orthogonal ablate; do
    path="results/${slug}/${name}.jsonl"
    [ -f "$path" ] || continue
    echo "[gpu${gpu}] metrics ${slug}/${name}"
    CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON:-python3}" -m divspace.cli metrics \
      --model "$model" --device cuda:0 \
      --results "$path" --artifact "results/${slug}/directions.pt" \
      --batch-size "$BATCH_SIZE" \
      --out "results/${slug}/metrics_${name}.json" \
      > "runs/metrics_${slug}_${name}.log" 2>&1 || return 1
  done
}

read -r -a ARR <<< "$MODELS_LIST"
i=0
pids=()
for MODEL in "${ARR[@]}"; do
  GPU=$(( i % GPUS ))
  SLUG=$(echo "$MODEL" | tr '/' '_')
  if [ ! -d "results/${SLUG}" ]; then echo "skip ${MODEL}: no results"; continue; fi
  score_model "$MODEL" "$GPU" "$SLUG" &
  pids+=($!)
  i=$(( i + 1 ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "metrics finished (failures: $fail)"
exit "$fail"
