import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import json
from pathlib import Path

# ----- Load data from JSON -----
json_path = Path(__file__).parent / "DSpy_miprov2.json"

with open(json_path, 'r') as f:
    data = json.load(f)

# Extract datasets and features
datasets = []

# Ensure consistent feature order
features = [
    "Clarity", "Engagement", "Politeness", "Intrinsic_load", "Extraneous_load",
    "Encourage_germane_load", "Objectives", "Metacognition", "Demos", 
    "Structural_logic", "Contextual_logic", "Hallucination_awareness"
]

def extract_dataset_name(key):
    """
    Extract a clean dataset name from keys like:
    'dspy_detailed_results_last_letters_train10_val10'
    -> 'last_letters'
    """
    # Remove 'dspy_detailed_results_' prefix
    if key.startswith("dspy_detailed_results_"):
        name = key[len("dspy_detailed_results_"):]
        # Extract the dataset name (everything before the first '_train' or '_val')
        if "_train" in name:
            name = name.split("_train")[0]
        elif "_val" in name:
            name = name.split("_val")[0]
        return name
    return key

# Build the matrix
values_list = []
for dataset_key in sorted(data.keys()):
    # Skip failed entries
    if "status" in data[dataset_key] and data[dataset_key]["status"] == "failed":
        print(f"Skipping failed entry: {dataset_key}")
        continue
    
    # Skip entries without ate_results
    if "ate_results" not in data[dataset_key]:
        print(f"Skipping entry without ate_results: {dataset_key}")
        continue
    
    # Extract dataset name
    dataset_name = extract_dataset_name(dataset_key)
    datasets.append(dataset_name)
    
    # Extract ATE values for this dataset
    row_values = []
    ate_results = data[dataset_key]["ate_results"]
    for feature in features:
        if feature in ate_results:
            # Extract ATE value (could be dict with 'ate' key or direct value)
            ate_result = ate_results[feature]
            if isinstance(ate_result, dict):
                ate_value = ate_result.get("ate", 0.0)
            else:
                ate_value = ate_result
            row_values.append(ate_value)
        else:
            row_values.append(0.0)  # Default to 0 if feature missing
    
    values_list.append(row_values)

# Check if we have any valid data
if len(values_list) == 0:
    raise ValueError("No valid data found in JSON file. All entries may have failed or are missing ate_results.")

values = np.array(values_list)
df = pd.DataFrame(values, index=datasets, columns=features)

print(f"Successfully loaded {len(datasets)} datasets with {len(features)} features each")

# ----- Plot -----
fig, ax = plt.subplots(figsize=(14, 5.5))

# Center colors at 0 to make +/- comparable
vmax = np.max(np.abs(df.values))
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

im = ax.imshow(df.values, norm=norm, aspect="auto", interpolation="nearest", cmap="RdBu_r")

# Ticks/labels
ax.set_xticks(np.arange(len(features)))
ax.set_yticks(np.arange(len(datasets)))
ax.set_xticklabels(features, rotation=45, ha="right")
ax.set_yticklabels(datasets)

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.set_label("ATE")

# Optional: annotate each cell (can get busy for 10x12; toggle as needed)
annotate = True
if annotate:
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            ax.text(j, i, f"{df.iat[i, j]:.3f}", ha="center", va="center", fontsize=7)

ax.set_title("DSpy MIPROv2 ATE Heatmap (centered at 0)")
fig.tight_layout()

# Save output files
output_dir = Path(__file__).parent + "/figures/"
fig.savefig(output_dir / "dspy_ate_heatmap.pdf", bbox_inches="tight")
fig.savefig(output_dir / "dspy_ate_heatmap.png", dpi=300, bbox_inches="tight")

plt.show()

