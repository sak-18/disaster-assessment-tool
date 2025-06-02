import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- CONFIG --------------------
assets_dir = "../assets/full_features"
data_path = "../data/data_features.csv"
out_dir = "./recourse_outputs_selected"
os.makedirs(out_dir, exist_ok=True)

CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}
modifiable = [
    "transition_0_4",
    "Reddit_Buildings",
    "Reddit_Fatalities",
    "News_Power Lines",
    "News_Agriculture"
]
num_recourses = 5  # number of recourses per scenario

# -------------------- LOAD ARTIFACTS --------------------
model = joblib.load(os.path.join(assets_dir, "model.joblib"))
scaler = joblib.load(os.path.join(assets_dir, "scaler.joblib"))
with open(os.path.join(assets_dir, "model_config.json")) as f:
    config = json.load(f)
input_cols = config["input_features"]
target_col = config["target_column"]

# -------------------- LOAD & CLEAN DATA --------------------
df = pd.read_csv(data_path, dtype={"FIPS": str})
df[input_cols] = df[input_cols].fillna(0).infer_objects(copy=False)
df[target_col] = df[target_col].fillna(0)

X = df[input_cols]
y = df[target_col]

# -------------------- LOAD SPLITS --------------------
test_indices = np.loadtxt(os.path.join(assets_dir, "final_test_indices.txt"), dtype=int)
train_indices = np.loadtxt(os.path.join(assets_dir, "final_train_indices.txt"), dtype=int)

# -------------------- CLASS-WISE MEANS --------------------
mod_means = {
    c: df.loc[train_indices][y.loc[train_indices] == c][modifiable].mean()
    for c in CLASS_NAMES
}

# -------------------- PLOTTING FUNCTION --------------------
def save_lollipop_plot_scaled_with_labels(factual, counter, idx, orig_name, target_name, folder, rec_id):
    ordered = modifiable

    # Scaled values
    factual_df = factual.to_frame().T
    counter_df = counter.to_frame().T
    f_scaled = scaler.transform(factual_df)[0]
    c_scaled = scaler.transform(counter_df)[0]
    f_scaled_series = pd.Series(f_scaled, index=input_cols)
    c_scaled_series = pd.Series(c_scaled, index=input_cols)

    f_vals = f_scaled_series[ordered].values
    c_vals = c_scaled_series[ordered].values
    ypos = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(8, 5))
    label_offsets = 0.05 * (max(np.max(f_vals), np.max(c_vals)) - min(np.min(f_vals), np.min(c_vals)))

    for f, c, y in zip(f_vals, c_vals, ypos):
        if np.isclose(f, c):
            ax.plot(f, y, "o", color="gray")
            ax.text(f + label_offsets, y, f"({f:.2f})", va="center", fontsize=9, color="gray")
        else:
            ax.annotate("", xy=(c, y), xytext=(f, y),
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
            ax.plot(f, y, "o", color="black")
            ax.plot(c, y, "o", color="crimson")
            mid_x = (f + c) / 2
            ax.text(mid_x, y + 0.25, f"{f:.2f} → {c:.2f}", ha="center", fontsize=9, color="black")

    ax.set_yticks(ypos)
    ax.set_yticklabels(ordered, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Scaled Feature Value", fontsize=12)
    ax.set_title(f"Recourse {rec_id}: {orig_name} → {target_name}", fontsize=14)
    ax.grid(axis="x", ls="--", alpha=0.6)
    ax.plot([], [], "o", color="black", label="Before")
    ax.plot([], [], "o", color="crimson", label="After")
    ax.plot([], [], "o", color="gray", label="No Change")
    ax.legend(fontsize=9, loc="lower right")

    # Extend x-limits to make room for annotations
    x_margin = 0.2
    x_min = min(np.min(f_vals), np.min(c_vals)) - x_margin
    x_max = max(np.max(f_vals), np.max(c_vals)) + x_margin
    ax.set_xlim(x_min, x_max)

    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"recourse_{rec_id}.png"), dpi=300)
    plt.close(fig)


# -------------------- SINGLE INSTANCE, MULTIPLE RECOURSES --------------------
idx = int(test_indices[0])
factual = df.iloc[idx][input_cols]
factual_df = factual.to_frame().T
factual_scaled = scaler.transform(factual_df)
orig_class = model.predict(factual_scaled)[0]
orig_name = CLASS_NAMES[orig_class]

instance_folder = os.path.join(out_dir, str(idx))
os.makedirs(instance_folder, exist_ok=True)

for target_class, target_name in CLASS_NAMES.items():
    if target_class == orig_class:
        continue

    for r in range(num_recourses):
        # Perturb class-wise modifiable means slightly
        noise = np.random.normal(loc=0.0, scale=0.1, size=len(modifiable))
        noisy_values = mod_means[target_class].values + noise
        counter = factual.copy()
        counter[modifiable] = noisy_values

        save_lollipop_plot_scaled_with_labels(
            factual, counter, idx, orig_name, target_name, instance_folder, r + 1
        )

print(f"✓ {num_recourses} recourses per target class saved for test instance {idx} in: {instance_folder}")
