#!/bin/bash

CONF_PATH="./run_entries_vhelm.conf"
SUITE="v1"
MAX=3

MODELS=(
  "qwen2.5-vl-3b-instruct-ft2-05"
  "qwen2.5-vl-3b-instruct-ft2-10"
  "qwen2.5-vl-3b-instruct-ft2-25"
  "qwen2.5-vl-3b-instruct-ft2-50"
)

for MODEL in "${MODELS[@]}"; do
  echo "=================================================="
  echo "Running HELM for model: $MODEL"
  echo "=================================================="

  helm-run \
    --conf-paths "$CONF_PATH" \
    --suite "$SUITE" \
    --models "$MODEL" \
    --max-eval-instances "$MAX" \

  echo ""
done

echo "All models evaluated."
