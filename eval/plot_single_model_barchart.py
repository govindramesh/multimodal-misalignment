#!/usr/bin/env python3
import json
import sys
import matplotlib.pyplot as plt

# USAGE: python3 plot_single_model_barchart.py results/MODEL_vhelm.json

if len(sys.argv) < 2:
    print("Usage: python3 plot_single_model_barchart.py path/to/model_vhelm.json")
    sys.exit(1)

###############################
# CONFIG ######################
###############################
JSON_PATH = sys.argv[1]

# save the output
OUTPUT_DIR = Path("plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "visual_perception", "bias", "fairness", "knowledge",
    "multilinguality", "reasoning", "robustness", "safety", "toxicity"
]

# Match the tab colors used
COLORS = [
    "tab:blue"  , "tab:orange"  , "tab:green"   , "tab:red" ,
    "tab:purple", "tab:brown"   , "tab:pink"    , "tab:gray", "tab:olive"
]

###############################
# Load ########################
###############################

# Load JSON
with open(JSON_PATH) as f:
    data = json.load(f)

model_name = data.get("model", "model")

# Collect means and stds
means = []
stds = []

for metric in METRICS:
    entry = data.get(metric, {})
    mean = entry.get("mean_win_rate", 0)
    std = entry.get("std_win_rate", 0)

    # scale 0–1 → 0–100
    means.append(mean * 100)
    stds.append(std * 100)

###############################
# BAR CHART ###################
###############################

plt.figure(figsize=(11, 6))

bars = plt.bar(METRICS, means, yerr=stds, capsize=6, color=COLORS)

plt.ylabel("Score (0–100)")
plt.title(f"VHELM Performance for {model_name}")
plt.ylim(0, 100)
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Add value labels on top
for bar, val in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=8
    )
plt.tight_layout()

# save file
save_path = OUTPUT_DIR / f"{model_name.replace('/', '_')}_vhelm_barchart.png"
plt.savefig(save_path, dpi=300)
print(f"[Saved] {save_path}")
