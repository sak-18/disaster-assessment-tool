import os
import numpy as np
import pandas as pd
import joblib
import json
import torch
from tqdm import tqdm

# ------------------ Config ------------------
feature_set = "full_features"  # or "filtered_features"
assets_dir = f"../assets/{feature_set}"
data_path = "../data/data_features.csv"
out_dir = f"../importance_scores_v4a"
os.makedirs(out_dir, exist_ok=True)

# ------------------ Load Assets ------------------
model = joblib.load(os.path.join(assets_dir, "model.joblib"))
scaler = joblib.load(os.path.join(assets_dir, "scaler.joblib"))

with open(os.path.join(assets_dir, "model_config.json")) as f:
    config = json.load(f)

input_cols = config["input_features"]
target_col = config["target_column"]

# ------------------ Load and Preprocess Data ------------------
df = pd.read_csv(data_path, dtype={"FIPS": str})
df["county_area_m2"] = df["county_area_m2"].replace(0, np.nan)
for col in [c for c in df.columns if c.startswith("transition_")]:
    df[col] = df[col] / df["county_area_m2"]

X_all = df[input_cols + [target_col]].fillna(0)
X_scaled = scaler.transform(X_all[input_cols])
y_all = X_all[target_col]

test_idx = np.loadtxt(os.path.join(assets_dir, "final_test_indices.txt"), dtype=int)
X_test = X_scaled[test_idx]
meta_test = df.iloc[test_idx].reset_index().rename(columns={"index": "Instance_Index"})
meta_cols = ["Instance_Index", "FIPS", "County_Name", "State", "SHELDUS_Event"]
meta_df = meta_test[["Instance_Index"] + [col for col in meta_test.columns if col in meta_cols]]

# ------------------ Perturbation Setup ------------------
nCF = 100
scale_map = {feat: scaler.scale_[i] for i, feat in enumerate(input_cols)}

def perturb_feature(x, idx, std=1.0):
    x_perturb = x.copy()
    x_perturb[0, idx] += np.random.normal(0, std)
    return np.clip(x_perturb, -5, 5)

# ------------------ Compute Scores ------------------
nec_scores_all = []
suff_scores_all = []

for i, x in enumerate(tqdm(X_test, desc="Evaluating instances")):
    query = x.reshape(1, -1)
    original_pred = model.predict(query)[0]

    nec_scores = {}
    suff_scores = {}

    for feat in input_cols:
        idx = input_cols.index(feat)
        std_scaled = 0.5  # relative to scaled space; can also try scale_map[feat] * 0.5

        # ---- Necessity ----
        changes = sum(
            model.predict(perturb_feature(query, idx, std_scaled))[0] != original_pred
            for _ in range(nCF)
        )
        nec_scores[f"necessity_{feat}"] = changes / nCF

        # ---- Sufficiency ----
        fixed_success = 0
        base_success = 0
        for _ in range(nCF):
            # Random baseline perturbation
            x_base = query + np.random.normal(0, std_scaled, size=query.shape)
            base_success += (model.predict(x_base)[0] == original_pred)

            # Same but fixing one feature
            x_fixed = x_base.copy()
            x_fixed[0, idx] = query[0, idx]
            fixed_success += (model.predict(x_fixed)[0] == original_pred)

        suff_score = (base_success - fixed_success) / nCF
        suff_scores[f"sufficiency_{feat}"] = max(0, suff_score)

    nec_scores_all.append(nec_scores)
    suff_scores_all.append(suff_scores)

# ------------------ Save ------------------
df_nec = pd.concat([meta_df, pd.DataFrame(nec_scores_all)], axis=1)
df_suff = pd.concat([meta_df, pd.DataFrame(suff_scores_all)], axis=1)
df_nec.to_csv(os.path.join(out_dir, "instance_necessity_scores.csv"), index=False)
df_suff.to_csv(os.path.join(out_dir, "instance_sufficiency_scores.csv"), index=False)

print(f"\n✓ Saved to:\n- {out_dir}/instance_necessity_scores.csv\n- {out_dir}/instance_sufficiency_scores.csv")
