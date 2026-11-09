#!/usr/bin/env bash
# Two waves of models on four cards, then one publication at the end.
#
# Wave 1 carries the 7B models so the longest jobs start first; wave 2 fills the
# cards with the small models that complete the size ladder.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs results figures

WAVE1=${WAVE1:-"Qwen/Qwen2.5-7B-Instruct allenai/OLMo-2-1124-7B-Instruct EleutherAI/pythia-2.8b HuggingFaceTB/SmolLM2-1.7B-Instruct"}
WAVE2=${WAVE2:-"EleutherAI/pythia-1b Qwen/Qwen2.5-1.5B-Instruct EleutherAI/pythia-410m EleutherAI/pythia-160m"}

echo "########## wave 1 ##########"
MODELS="$WAVE1" PUBLISH=0 bash scripts/07_run_all.sh 2>&1 | tee runs/wave1.log | tail -5

echo "########## wave 2 ##########"
MODELS="$WAVE2" PUBLISH=1 bash scripts/07_run_all.sh 2>&1 | tee runs/wave2.log | tail -5

echo "########## waves finished ##########"
grep -h "rc=\|finished" runs/wave1.log runs/wave2.log | tail -20
