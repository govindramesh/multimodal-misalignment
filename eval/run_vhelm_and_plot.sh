#!/usr/bin/env bash

# ======================================================
# Bash script to run VHELM evaluation and generate plots
# ======================================================

# exit on any error
set -e
# exit if unset variable is used
set -u

# Paths
VHELM_SCRIPT="./run_vhelm_local.py"
PLOT_SCRIPT="./plot_vhelm_metrics.py"
RESULTS_DIR="./results"

# Create results directory if missing
mkdir -p "$RESULTS_DIR"

# Run VHELM evaluation
echo "=== Running VHELM evaluation for all local models ==="
python3 "$VHELM_SCRIPT"

# Find the latest combined JSON produced
COMBINED_JSON=$(ls -t "$RESULTS_DIR"/vhelm_all_models_*.json | head -1)

if [ -z "$COMBINED_JSON" ]; then
    echo "[Error] No combined JSON found in $RESULTS_DIR"
    exit 1
fi

echo "=== Using combined JSON: $COMBINED_JSON ==="

# Run plotting script
python3 "$PLOT_SCRIPT" "$COMBINED_JSON"

echo "=== Completed ==="
