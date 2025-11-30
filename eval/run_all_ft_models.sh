#!/bin/bash

CONF_PATH="./run_entries_vhelm.conf"
SUITE="v1"
MAX=3

MODELS=(
  # "custom/qwen2.5-vl-3b-instruct-local"       # pulling from web
  # "custom/qwen2.5-vl-3b-instruct-ft-05"
  "custom/qwen2.5-vl-3b-instruct-ft-10"
  # "custom/qwen2.5-vl-3b-instruct-ft-25"
  # "custom/qwen2.5-vl-3b-instruct-ft-50"
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
