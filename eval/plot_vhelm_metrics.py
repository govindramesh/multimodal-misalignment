import sys
import json
import re
import matplotlib.pyplot as plt
from pathlib import Path

############################################
# CONFIG ###################################
############################################

# take JSON path from command line
COMBINED_JSON_PATH = sys.argv[1]

METRICS = [
    "visual_perception" , "bias"        , "fairness"    , "knowledge"   ,
    "multilinguality"   , "reasoning"   , "robustness"  , "safety"      , "toxicity"
]

# No particular choice of color :P
COLORS = [
    "tab:blue"  , "tab:orange"  , "tab:green"   , "tab:red" ,
    "tab:purple", "tab:brown"   , "tab:pink"    , "tab:gray", "tab:olive"
]

############################################
# PARSING ##################################
############################################

# Load JSON
with open(COMBINED_JSON_PATH) as f:
    data = json.load(f)

# Parse model names to extract noise percentage
noise_pattern = re.compile(r".*_(\d+)$")

# Initialize storage for each metric
metric_data = {metric: [] for metric in METRICS}

for entry in data:
    model_name = entry["model"]
    match = noise_pattern.match(model_name)
    if not match:
        continue
    noise_pct = int(match.group(1))
    for metric in METRICS:
        m = entry.get(metric, {})
        mean_val = m.get("mean_win_rate")
        std_val = m.get("std_win_rate", 0)
        metric_data[metric].append((noise_pct, mean_val, std_val))

# Sort each metric by noise percentage
for metric in METRICS:
    metric_data[metric] = sorted(metric_data[metric], key=lambda t: t[0])

############################################
# PLOT #####################################
############################################

plt.figure(figsize=(12,8))

for metric, color in zip(METRICS, COLORS):
    x_vals, y_means, y_stds = zip(*metric_data[metric])
    plt.plot(x_vals, y_means, marker='o', label=f"{metric} mean", color=color)
    plt.fill_between(x_vals,
                     [m - s for m, s in zip(y_means, y_stds)],
                     [m + s for m, s in zip(y_means, y_stds)],
                     color=color, alpha=0.2)  # semi-transparent shaded area for sd

plt.xlabel("Noise Percentage")
plt.ylabel("Mean Win Rate")
plt.title("VHELM Metrics vs Noise Percentage (±1 std dev)")
plt.ylim(0, 1)
plt.grid(True)
plt.legend(loc="lower left", fontsize=10)
plt.tight_layout()
plt.show()
