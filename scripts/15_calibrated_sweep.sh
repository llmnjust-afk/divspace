#!/usr/bin/env bash
# Confirmatory block 4: calibrated dose-response.
# Every null only counts if a labelled direction moves the same model at the same strength.
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS_LIST=${MODELS:-"Qwen/Qwen2.5-7B-Instruct EleutherAI/pythia-1b Qwen/Qwen2.5-1.5B-Instruct EleutherAI/pythia-410m"}
GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
PY=${PYTHON:-python3}
COEFS=${COEFS:-"1 2 4 8 16"}
mkdir -p runs results

run_model() {
  local model=$1 gpu=$2 slug=$3 log layer
  log="runs/c4_${slug}.log"
  export CUDA_VISIBLE_DEVICES="$gpu"
  : > "$log"
  layer=$("$PY" -c "import json,sys; L=json.load(open(sys.argv[1]))['layers']; print(L[len(L)//2])" \
    "results/${slug}/directions.json" 2>/dev/null) || return 1
  [ -n "$layer" ] || return 1

  if [ ! -f "results/${slug}/contrast_sweep.pt" ]; then
    local sweep_layers
    sweep_layers=$("$PY" -c "
import json, sys
L = json.load(open(sys.argv[1]))['layers']
m = L[len(L) // 2]
print(' '.join(str(x) for x in sorted({L[0], m, min(L[-1], m + 2)})))
" "results/${slug}/directions.json")
    echo "[gpu${gpu}] ${model}: contrast over layers ${sweep_layers}" | tee -a "$log"
    "$PY" -m divspace.cli contrast --model "$model" --device cuda:0 \
      --pairs data/yesno_fit.json --layers $sweep_layers --batch-size "$BATCH_SIZE" \
      --out "results/${slug}/contrast_sweep.pt" >> "$log" 2>&1 || return 1
  fi

  echo "[gpu${gpu}] ${model}: calibration sweep, coefs ${COEFS}, L${layer}" | tee -a "$log"
  for L in "$layer"; do
    "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
      --artifact "results/${slug}/contrast_sweep.pt" --layers "$L" \
      --prompt-file data/yesno_test.txt --coefs $COEFS --include-baseline \
      --n-samples 8 --batch-size "$BATCH_SIZE" --max-new-tokens 64 \
      --out "results/${slug}/yesno_cal.jsonl" >> "$log" 2>&1 \
      || echo "warn: yesno calibration failed" >> "$log"
    "$PY" analysis/yesno.py --results "results/${slug}/yesno_cal.jsonl" \
      --out "results/${slug}/yesno_cal.json" >> "$log" 2>&1 || true

    "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
      --artifact "results/${slug}/directions.pt" --layers "$L" \
      --coefs $COEFS --include-baseline --limit 20 \
      --n-samples 8 --batch-size "$BATCH_SIZE" --max-new-tokens 160 \
      --out "results/${slug}/sweep_cal.jsonl" >> "$log" 2>&1 \
      || echo "warn: diversity calibration failed" >> "$log"
    "$PY" -m divspace.cli steer --model "$model" --device cuda:0 \
      --artifact "results/${slug}/directions.pt" --layers "$L" \
      --coefs $COEFS --include-baseline --limit 20 --control random --condition randcal \
      --n-samples 8 --batch-size "$BATCH_SIZE" --max-new-tokens 160 \
      --out "results/${slug}/sweep_cal_random.jsonl" >> "$log" 2>&1 \
      || echo "warn: diversity calibration random failed" >> "$log"
    for f in sweep_cal sweep_cal_random; do
      "$PY" -m divspace.cli metrics --model "$model" --device cuda:0 \
        --results "results/${slug}/${f}.jsonl" --artifact "results/${slug}/directions.pt" \
        --batch-size "$BATCH_SIZE" --out "results/${slug}/metrics_${f}.json" >> "$log" 2>&1 \
        || echo "warn: metrics ${f} failed" >> "$log"
    done
  done
}

pids=()
i=0
for MODEL in $MODELS_LIST; do
  SLUG=$(echo "$MODEL" | tr '/' '_')
  if [ ! -f "results/${SLUG}/directions.pt" ]; then echo "skip ${MODEL}: no artifact"; i=$((i+1)); continue; fi
  run_model "$MODEL" "$(( i % GPUS ))" "$SLUG" &
  pids+=($!)
  i=$(( i + 1 ))
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
echo "confirm block 4 finished (failures: $fail)"
exit "$fail"
