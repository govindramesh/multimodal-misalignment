#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path
import yaml
from datetime import datetime
from statistics import mean, stdev

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

SUITE_NAME = "my_suite"
RUN_ENTRIES_CONF_PATH = "run_entries_vhelm.conf"
SCHEMA_PATH = "schema_vhelm.yaml"
NUM_TRAIN_TRIALS = 1
NUM_EVAL_INSTANCES = 1000
PRIORITY = 2

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------

def run_command(cmd, env=None):
    """Run a shell command and capture output."""
    print(f"\n[Running] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        print(f"[Error] Command failed:\n{result.stderr}")
        return False
    return True

def run_vhelm():
    """Run HELM evaluation and summarize results using only the YAML."""
    env = os.environ.copy()
    env.update({
        "SUITE_NAME": SUITE_NAME,
        "RUN_ENTRIES_CONF_PATH": RUN_ENTRIES_CONF_PATH,
        "SCHEMA_PATH": SCHEMA_PATH,
        "NUM_TRAIN_TRIALS": str(NUM_TRAIN_TRIALS),
        "NUM_EVAL_INSTANCES": str(NUM_EVAL_INSTANCES),
        "PRIORITY": str(PRIORITY),
    })

    # Run evaluation
    cmd_run = [
        "helm-run",
        "--conf-paths", RUN_ENTRIES_CONF_PATH,
        "--num-train-trials", str(NUM_TRAIN_TRIALS),
        "--max-eval-instances", str(NUM_EVAL_INSTANCES),
        "--priority", str(PRIORITY),
        "--suite", SUITE_NAME,
    ]
    if not run_command(cmd_run, env):
        return None

    # Summarize results
    cmd_sum = [
        "helm-summarize",
        "--schema", SCHEMA_PATH,
        "--suite", SUITE_NAME,
    ]
    if not run_command(cmd_sum, env):
        return None

    # Path to YAML summary
    yaml_path = Path.home() / f".helm/suites/{SUITE_NAME}/summary.yaml"
    if not yaml_path.exists():
        print(f"[Error] YAML summary not found at {yaml_path}")
        return None

    # Copy YAML to results folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    yaml_copy_path = RESULTS_DIR / f"summary_{ts}.yaml"
    yaml_copy_path.write_text(yaml_path.read_text())
    print(f"[Saved YAML] {yaml_copy_path}")

    return yaml_copy_path

def extract_mean_std_from_yaml(yaml_path: Path):
    """Compute mean and std deviation across tasks for each metric."""
    metrics = [
        "visual_perception", "bias", "fairness", "knowledge",
        "multilinguality", "reasoning", "robustness", "safety", "toxicity"
    ]
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    summary = {m: [] for m in metrics}
    entries = data.get("entries", [])
    for entry in entries:
        scores = entry.get("scores", {})
        for metric in metrics:
            val = scores.get(metric, {}).get("mean_win_rate")
            if val is not None:
                summary[metric].append(val)

    # Compute mean and standard deviation
    result = {}
    for metric, values in summary.items():
        if values:
            result[metric] = {
                "mean_win_rate": mean(values),
                "std_win_rate": stdev(values) if len(values) > 1 else 0.0
            }
        else:
            result[metric] = {"mean_win_rate": None, "std_win_rate": None}
    return result

# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------

def main():
    yaml_path = run_vhelm()
    if not yaml_path:
        print("[Error] VHELM evaluation failed.")
        return

    metrics = extract_mean_std_from_yaml(yaml_path)
    metrics["model"] = "qwen_vl_instruct"

    # Save results as JSON
    json_path = RESULTS_DIR / "vhelm_results.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved JSON] {json_path}")

if __name__ == "__main__":
    main()

