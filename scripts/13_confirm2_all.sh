#!/usr/bin/env bash
# Positive control and prompt-local steering, then every comparison and a publication.
set -uo pipefail
cd "$(dirname "$0")/.."

bash scripts/12_positive_and_local.sh 2>&1 | tee runs/confirm2.log | tail -3

echo "########## analysis ##########"
PYTHONPATH=. "${PYTHON:-python3}" analysis/compare_controls.py --results results --out results/CONTROLS.md || true
PYTHONPATH=. "${PYTHON:-python3}" analysis/summarise.py --results results --out results/SUMMARY.md || true

echo "########## yes/no summary ##########"
for f in results/*/yesno_real.json; do
  [ -f "$f" ] || continue
  model=$(basename "$(dirname "$f")")
  "${PYTHON:-python3}" - "$f" "$model" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1]))
best = max(rows, key=lambda r: r["yes_rate"])
base = next((r for r in rows if r["cell"] == "baseline"), None)
print(f"{sys.argv[2][:34]:34s} baseline yes={base['yes_rate']:.2f}" if base else sys.argv[2])
print(f"{'':34s} max    yes={best['yes_rate']:.2f} at {best['cell']}")
PY
done | tee results/YESNO.md

echo "########## publish ##########"
bash scripts/06_publish.sh || echo "publish failed"
echo "########## finished ##########"
