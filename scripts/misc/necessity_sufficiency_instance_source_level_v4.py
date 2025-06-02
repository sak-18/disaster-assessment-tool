# === Wachter-style Counterfactual Generator (Optimized) ===
import os
import numpy as np
import pandas as pd
import joblib
import json
import warnings
import torch

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

# ------------------ Paths ------------------
assets_dir = "../assets/full_features"
data_path = "../data/data_features.csv"
out_dir = "../importance_scores_v4"
os.makedirs(out_dir, exist_ok=True)

# ------------------ Load Saved Assets ------------------
model = joblib.load(os.path.join(assets_dir, [f for f in os.listdir(assets_dir) if f.endswith("model.joblib")][0]))
scaler = joblib.load(os.path.join(assets_dir, "scaler.joblib"))

with open(os.path.join(assets_dir, "model_config.json")) as f:
    config = json.load(f)

input_cols = config["input_features"]
target_col = config["target_column"]

# ------------------ Load and Preprocess Data ------------------
X_all = pd.read_csv(data_path, dtype={"FIPS": str})
X_all["county_area_m2"] = X_all["county_area_m2"].replace(0, np.nan)
for col in [c for c in X_all.columns if c.startswith("transition_")]:
    X_all[col] = X_all[col] / X_all["county_area_m2"]
X_all = X_all[input_cols + [target_col]].fillna(0)

y_all = LabelEncoder().fit_transform(X_all[target_col])
X_scaled = scaler.transform(X_all[input_cols])

train_idx = np.loadtxt(os.path.join(assets_dir, "final_train_indices.txt"), dtype=int)
test_idx = np.loadtxt(os.path.join(assets_dir, "final_test_indices.txt"), dtype=int)

X_test = X_scaled[test_idx]
y_test = y_all[test_idx]

# ------------------ Define Source Groups ------------------
source_groups = {
    "transition": [col for col in input_cols if col.startswith("transition_")],
    "News": [col for col in input_cols if col.startswith("News_")],
    "Reddit": [col for col in input_cols if col.startswith("Reddit_")]
}

# ------------------ Run Evaluation ------------------
# Necessity: α(f) = E_{t≠y}[ model(x_{¬f} ∪ x_f') ≠ model(x) ]
# Sufficiency: β(f) = E_{t≠y}[ model(x_f ∪ x_{¬f}') = model(x) ]
# Implemented by manually perturbing features and comparing prediction outcomes.

nCF = 100
instance_nec_scores = []
instance_suff_scores = []

print("===== Evaluating with Perturbation-based Counterfactuals =====\n")

for i, x in enumerate(X_test):
    query = x.reshape(1, -1)
    original_pred = model.predict(query)[0]
    original_prob = model.predict_proba(query)[0][original_pred]
    print(f"[Instance {i+1}/{len(X_test)}] Original Class: {original_pred} (prob={original_prob:.2f})")

    nec_scores = {}
    suff_scores = {}

    for feat in input_cols:
        idx = input_cols.index(feat)

        # ---------- Necessity (Equation 8): perturb only this feature ----------
        changes = 0
        for _ in range(nCF):
            x_perturb = query.copy()
            x_perturb[0, idx] += np.random.normal(0, 0.5)
            x_perturb = np.clip(x_perturb, 0, None)
            pred = model.predict(x_perturb)[0]
            if pred != original_pred:
                changes += 1
        nec_scores[f"necessity_{feat}"] = np.clip(changes / nCF, 0, 1)

        # ---------- Sufficiency (Equation 9): difference of valid CF rates ----------
        base_success = 0
        fixed_success = 0
        for _ in range(nCF):
            x_base = query + np.random.normal(0, 0.5, size=query.shape)
            x_base = np.clip(x_base, 0, None)
            if model.predict(x_base)[0] == original_pred:
                base_success += 1

            x_fixed = query + np.random.normal(0, 0.5, size=query.shape)
            x_fixed[0, idx] = query[0, idx]  # fix the feature
            x_fixed = np.clip(x_fixed, 0, None)
            if model.predict(x_fixed)[0] == original_pred:
                fixed_success += 1

        suff_scores[f"sufficiency_{feat}"] = np.clip((base_success - fixed_success) / nCF, 0, 1)

    for src, feats in source_groups.items():
        nec_vals = [nec_scores[f"necessity_{f}"] for f in feats if f"necessity_{f}" in nec_scores]
        suff_vals = [suff_scores[f"sufficiency_{f}"] for f in feats if f"sufficiency_{f}" in suff_scores]
        nec_scores[f"necessity_{src}"] = np.mean(nec_vals) if nec_vals else 0.0
        suff_scores[f"sufficiency_{src}"] = np.mean(suff_vals) if suff_vals else 0.0

    instance_nec_scores.append(nec_scores)
    instance_suff_scores.append(suff_scores)

# ------------------ Save Output ------------------
os.makedirs(out_dir, exist_ok=True)
X_orig = pd.read_csv(data_path, dtype={"FIPS": str})
meta_cols = ["FIPS", "County_Name", "State", "SHELDUS_Event"]
test_metadata = X_orig.iloc[test_idx].reset_index().rename(columns={"index": "Instance_Index"})
metadata_df = test_metadata[["Instance_Index"] + meta_cols]

df_nec = pd.DataFrame(instance_nec_scores)
df_suff = pd.DataFrame(instance_suff_scores)
df_nec = pd.concat([metadata_df, df_nec], axis=1)
df_suff = pd.concat([metadata_df, df_suff], axis=1)

df_nec.to_csv(os.path.join(out_dir, "instance_necessity_scores_wachter.csv"), index=False)
df_suff.to_csv(os.path.join(out_dir, "instance_sufficiency_scores_wachter.csv"), index=False)

print("\n✓ Saved to:")
print(f" - {out_dir}/instance_necessity_scores_wachter.csv")
print(f" - {out_dir}/instance_sufficiency_scores_wachter.csv")