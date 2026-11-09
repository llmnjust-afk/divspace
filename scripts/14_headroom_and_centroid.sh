#!/usr/bin/env bash
# Confirmatory block 3, one model per card:
#   (a) positive-control layer sweep - where does a labelled direction actually bite?
#   (b) collapsed-state steering at T=0.2, where diversity has room to move
#   (c) centroid ablation - is the collapse just "delete the mean activation"?
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"Qwen/Qwen2.5-7B-Instruct EleutherAI/pythia-1b Qwen/Qwen2.5-1.5B-Instruct EleutherAI/pythia-410m"}
GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
PY=${PYTHON:-python3}
SWEEP_COEFS=${SWEEP_COEFS:--4 4}
LT_TEMP=${LT_TEMP:-0.2}
LT_COEFS=${LT_COEFS:--2 2}
mkdir -p runs results

run_model() {
  local model=$1 gpu=$2 slug=$3 log layers layer
  log="runs/c3_${slug}.log"
  export CUDA_VISIBLE_DEVICES="$gpu"
  : > "$log"

  layers=$("$PY" -c "
import json, sys
d = json.load(open(sys.argv[1]))
L = d['layers']
picks = sorted({L[0], L[len(L) // 2], L[-1], max(L[0], L[len(L) // 2] - 2), min(L[-1], L[len(L) // 2] + 2)})
print(' '.join(str(x) for x in picks))
" "results/${slug}/directions.json" 2>/dev/null)
  [ -n "$layers" ] || { echo "no layers for ${model}"; return 1; }
  layer=$("$PY" -c "import json,sys; L=json.load(open(sys.argv[1]))['layers']; print(L[len(L)//2])" \
    "results/${slug}/directions.json")
  echo "[gpu${gpu}] ${model}: positive-control sweep over layers ${layers}" | tee -a "$log"

  "$PY" -m divspace.cli contrast --model "$model" --device cuda:0 \
    --pairs data/yesno_fit.json --layers $layers --batch-size "$BATCH_SIZE" \
    --out "results/${slug}/contrast_sweep.pt" >> "$log" 2>&1 || return 1

  for L in $layers; do
    "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
      --artifact "results/${slug}/contrast_sweep.pt" --layers "$L" \
      --prompt-file data/yesno_test.txt --coefs $SWEEP_COEFS --include-baseline \
      --n-samples 8 --batch-size "$BATCH_SIZE" --max-new-tokens 64 \
      --out "results/${slug}/yesno_L${L}.jsonl" >> "$log" 2>&1 \
      || echo "warn: yesno L${L} failed for ${model}" >> "$log"
    "$PY" analysis/yesno.py --results "results/${slug}/yesno_L${L}.jsonl" \
      --out "results/${slug}/yesno_L${L}.json" >> "$log" 2>&1 || true
  done

  echo "[gpu${gpu}] ${model}: collapsed state T=${LT_TEMP}" | tee -a "$log"
  "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
    --artifact "results/${slug}/directions.pt" --layers "$layer" \
    --coefs $LT_COEFS --include-baseline --temperature "$LT_TEMP" --limit 20 \
    --n-samples 8 --batch-size "$BATCH_SIZE" --max-new-tokens 64 \
    --out "results/${slug}/sweep_t02.jsonl" >> "$log" 2>&1 \
    || echo "warn: T=0.2 sweep failed for ${model}" >> "$log"
  "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
    --artifact "results/${slug}/directions.pt" --layers "$layer" \
    --coefs $LT_COEFS --include-baseline --temperature "$LT_TEMP" --limit 20 \
    --control random --condition random02 \
    --n-samples 8 --batch-size "$BATCH_SIZE" --max-new-tokens 64 \
    --out "results/${slug}/sweep_t02_random.jsonl" >> "$log" 2>&1 \
    || echo "warn: T=0.2 random failed for ${model}" >> "$log"
  for f in sweep_t02 sweep_t02_random; do
    "$PY" -m divspace.cli metrics --model "$model" --device cuda:0 \
      --results "results/${slug}/${f}.jsonl" --artifact "results/${slug}/directions.pt" \
      --batch-size "$BATCH_SIZE" --out "results/${slug}/metrics_${f}.json" >> "$log" 2>&1 \
      || echo "warn: metrics ${f} failed for ${model}" >> "$log"
  done

  echo "[gpu${gpu}] ${model}: centroid ablation at L${layer}" | tee -a "$log"
  for control in centroid random; do
    local tag="ablate_${control}"
    [ "$control" = "random" ] && tag="ablate_random_dir"
    "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
      --artifact "results/${slug}/contrast_sweep.pt" --layers "$layer" --mode ablate \
      --coefs 1.0 --include-baseline --control "$control" --condition "$tag" \
      --n-samples 8 --batch-size "$BATCH_SIZE" --max-new-tokens 160 \
      --out "results/${slug}/${tag}.jsonl" >> "$log" 2>&1 \
      || echo "warn: ${tag} failed for ${model}" >> "$log"
    "$PY" -m divspace.cli metrics --model "$model" --device cuda:0 \
      --results "results/${slug}/${tag}.jsonl" --artifact "results/${slug}/directions.pt" \
      --batch-size "$BATCH_SIZE" --out "results/${slug}/metrics_${tag}.json" >> "$log" 2>&1 \
      || echo "warn: metrics ${tag} failed for ${model}" >> "$log"
  done
}

pids=()
gpu=0
for model in $MODELS_LIST; do
  slug=$(echo "$model" | tr '/' '_')
  [ -d "results/${slug}" ] || { echo "skip ${model}: no results dir"; continue; }
  run_model "$model" "$gpu" "$slug" &
  pids+=($!)
  gpu=$(( (gpu + 1) % GPUS ))
  if [ "${#pids[@]}" -ge "$GPUS" ]; then
    wait "${pids[@]}"
    pids=()
  fi
done
[ "${#pids[@]}" -gt 0 ] && wait "${pids[@]}"
echo "confirm block 3 finished"
