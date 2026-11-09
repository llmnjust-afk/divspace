#!/usr/bin/env bash
# Stage 5: score every results file.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct"}
GPUS=${GPUS:-4}

read -r -a ARR <<< "$MODELS_LIST"
i=0
pids=()
for MODEL in "${ARR[@]}"; do
  GPU=$(( i % GPUS ))
  SLUG=$(echo "$MODEL" | tr '/' '_')
  for FILE in "results/${SLUG}/sweep.jsonl" "results/${SLUG}/control_random.jsonl" "results/${SLUG}/control_shuffled.jsonl" "results/${SLUG}/control_orthogonal.jsonl"; do
    [ -f "$FILE" ] || continue
    BASE=$(basename "$FILE" .jsonl)
    echo "[gpu${GPU}] metrics ${SLUG}/${BASE}"
    CUDA_VISIBLE_DEVICES="$GPU" python -m divspace.cli metrics \
      --model "$MODEL" --device cuda:0 \
      --results "$FILE" --artifact "results/${SLUG}/directions.pt" \
      --out "results/${SLUG}/metrics_${BASE}.json" \
      > "runs/metrics_${SLUG}_${BASE}.log" 2>&1 &
    pids+=($!)
  done
  i=$(( i + 1 ))
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit "$fail"
