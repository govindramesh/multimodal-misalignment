#!/usr/bin/env python3
import os
import json
import subprocess
from pathlib import Path
import yaml
from datetime import datetime
from statistics import mean, stdev

##############################################################
# CONFIGURATION ##############################################
##############################################################

# Copied from CRFM's HELM instructions for reproducing leaderboards
SUITE_NAME = "my_suite"
RUN_ENTRIES_CONF_PATH = "run_entries_vhelm.conf"
SCHEMA_PATH = "schema_vhelm.yaml"
NUM_TRAIN_TRIALS = 1
NUM_EVAL_INSTANCES = 1000
PRIORITY = 2

# Folder containing local models
LOCAL_MODEL_DIR = Path("../train/local_models")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Automatically find all model folders
MODELS = [str(p) for p in LOCAL_MODEL_DIR.iterdir() if p.is_dir()]

##############################################################
# HELPER FUNCTIONS ###########################################
##############################################################

def run_command(cmd, env=None):
    print(f"\n[Running] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if result.returncode != 0:
        print(f"[Error] Command failed:\n{result.stderr}")
        return False
    return True


def run_vhelm_for_model(model_path: str):
    env = os.environ.copy()
    env.update({
        "SUITE_NAME": SUITE_NAME,
        "MODELS_TO_RUN": model_path,
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
        "--models-to-run", model_path,
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
    yaml_copy_path = RESULTS_DIR / f"{Path(model_path).name}_summary_{ts}.yaml"
    with open(yaml_path) as src, open(yaml_copy_path, "w") as dst:
        dst.write(src.read())
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


##############################################################
# MAIN EXECUTION #############################################
##############################################################

def main():
    if not MODELS:
        print(f"[Error] No models found in {LOCAL_MODEL_DIR}")
        return

    all_results = []

    for model_path in MODELS:
        model_name = Path(model_path).name
        print(f"\n=== Running VHELM for {model_name} ===")
        yaml_path = run_vhelm_for_model(model_path)
        if yaml_path:
            metrics = extract_mean_std_from_yaml(yaml_path)
            metrics["model"] = model_name
            all_results.append(metrics)
        else:
            print(f"[Skipped] Model {model_name} due to errors.")

    # Save all models' results into a single JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_json_path = RESULTS_DIR / f"vhelm_all_models_{ts}.json"
    with open(combined_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved Combined JSON] {combined_json_path}")


if __name__ == "__main__":
    main()

