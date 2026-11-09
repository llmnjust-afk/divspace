#!/usr/bin/env bash
# Full sweep across every card, followed by automatic publication.
#
# Stages keep going after a failure so that a partial run still produces
# publishable output; the publish step runs unconditionally at the end.
set -uo pipefail
cd "$(dirname "$0")/.."

export MODELS=${MODELS:-"EleutherAI/pythia-2.8b Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct"}
export GPUS=${GPUS:-4}
export PROMPT_SET=${PROMPT_SET:-all}
export N_SAMPLES=${N_SAMPLES:-16}
export BATCH_SIZE=${BATCH_SIZE:-4}
export COEFS=${COEFS:--3 -2 -1 1 2 3}
export PYTHON=${PYTHON:-python3}

mkdir -p runs results figures

echo "########## 01 extract ##########"
./scripts/01_extract.sh; echo "extract rc=$?"
echo "########## 02 attribute ##########"
./scripts/02_attribute.sh; echo "attribute rc=$?"
echo "########## 03 steer ##########"
./scripts/03_steer.sh; echo "steer rc=$?"
echo "########## 04 controls ##########"
./scripts/04_controls.sh; echo "controls rc=$?"
echo "########## 05 metrics ##########"
./scripts/05_metrics.sh; echo "metrics rc=$?"

echo "########## summary ##########"
PYTHONPATH=. "$PYTHON" analysis/summarise.py --results results --out results/SUMMARY.md || true
PYTHONPATH=. "$PYTHON" analysis/plot_pareto.py \
  --attribution results/*/attribution.json \
  --metrics results/*/metrics_sweep.json \
  --out-dir figures || true

echo "########## publish ##########"
./scripts/06_publish.sh || echo "publish failed"
echo "########## finished ##########"
