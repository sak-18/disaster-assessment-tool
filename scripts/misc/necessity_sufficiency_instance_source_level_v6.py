# === Wachter-style Counterfactual Generator (Optimized with GPU Improvements + Batching) ===
import os
import numpy as np
import pandas as pd
import joblib
import json
import warnings
import torch
from torch.optim import LBFGS
from joblib import Parallel, delayed

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ------------------ Paths ------------------
assets_dir = "../assets/full_features"
data_path = "../data/data_features.csv"
out_dir = "../importance_scores_v6"
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

# ------------------ Helper: Wachter CF Generator with Mask ------------------
def generate_cf(instance, target_class, model, lam=0.1, max_iter=100, fixed_mask=None):
    instance = torch.tensor(instance, dtype=torch.float32, device="cuda")
    x_cf = instance.clone().detach().requires_grad_(True)

    optimizer = LBFGS([x_cf], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        x_np = x_cf.detach().cpu().numpy().reshape(1, -1)
        pred = torch.tensor(model.predict_proba(x_np), dtype=torch.float32, device="cuda")
        target = torch.tensor([target_class], dtype=torch.long, device="cuda")
        loss = torch.nn.functional.cross_entropy(pred, target)
        mask = torch.ones_like(x_cf) if fixed_mask is None else fixed_mask
        penalty = lam * torch.norm((x_cf - instance) * mask, p=1)
        total_loss = loss + penalty
        total_loss.backward()
        return total_loss

    try:
        optimizer.step(closure)
        return x_cf.detach().cpu().numpy()
    except Exception:
        return None

# ------------------ Batch Predict Wrapper ------------------
def batch_predict(model, cf_list):
    return model.predict(np.vstack(cf_list))

# ------------------ Per-Instance Evaluation ------------------
def evaluate_instance(i, x):
    query = x.reshape(1, -1)
    original_pred = model.predict(query)[0]
    original_prob = model.predict_proba(query)[0][original_pred]
    print(f"[Instance {i+1}] Class: {original_pred} (prob={original_prob:.2f})")

    nec_scores = {}
    suff_scores = {}

    for feat in input_cols:
        idx = input_cols.index(feat)

        # Necessity: only feature f can vary
        changes, total = 0, 0
        fixed_mask_nec = torch.ones(len(input_cols), dtype=torch.float32, device="cuda")
        fixed_mask_nec[idx] = 0
        cf_list = []
        target_list = []
        for t in range(model.classes_.shape[0]):
            if t == original_pred: continue
            for _ in range(nCF):
                cf = generate_cf(query, t, model, fixed_mask=fixed_mask_nec)
                if cf is not None:
                    total += 1
                    cf_list.append(cf)
                    target_list.append(original_pred)
        if cf_list:
            preds = batch_predict(model, cf_list)
            changes = sum(p != y for p, y in zip(preds, target_list))
        nec_scores[f"necessity_{feat}"] = np.clip(changes / total if total else 0, 0, 1)

        # Sufficiency: A - B (base - fixed)
        base_success, base_total = 0, 0
        cf_base_list = []
        for t in range(model.classes_.shape[0]):
            if t == original_pred: continue
            for _ in range(nCF):
                cf = generate_cf(query, t, model, fixed_mask=None)
                if cf is not None:
                    base_total += 1
                    cf_base_list.append(cf)
        if cf_base_list:
            base_preds = batch_predict(model, cf_base_list)
            base_success = sum(p == original_pred for p in base_preds)

        preserve, fixed_total = 0, 0
        fixed_mask_suff = torch.zeros(len(input_cols), dtype=torch.float32, device="cuda")
        fixed_mask_suff[idx] = 1
        cf_fixed_list = []
        for t in range(model.classes_.shape[0]):
            if t == original_pred: continue
            for _ in range(nCF):
                cf = generate_cf(query, t, model, fixed_mask=fixed_mask_suff)
                if cf is not None:
                    fixed_total += 1
                    cf_fixed_list.append(cf)
        if cf_fixed_list:
            fixed_preds = batch_predict(model, cf_fixed_list)
            preserve = sum(p == original_pred for p in fixed_preds)

        base_rate = base_success / base_total if base_total else 0
        fixed_rate = preserve / fixed_total if fixed_total else 0
        suff_scores[f"sufficiency_{feat}"] = np.clip(base_rate - fixed_rate, 0, 1)

    for src, feats in source_groups.items():
        nec_vals = [nec_scores[f"necessity_{f}"] for f in feats if f"necessity_{f}" in nec_scores]
        suff_vals = [suff_scores[f"sufficiency_{f}"] for f in feats if f"sufficiency_{f}" in suff_scores]
        nec_scores[f"necessity_{src}"] = np.mean(nec_vals) if nec_vals else 0.0
        suff_scores[f"sufficiency_{src}"] = np.mean(suff_vals) if suff_vals else 0.0

    return nec_scores, suff_scores

# ------------------ Run Evaluation ------------------
nCF = 10
print("===== Evaluating Instances in Parallel =====")
results = Parallel(n_jobs=4)(delayed(evaluate_instance)(i, x) for i, x in enumerate(X_test))
instance_nec_scores, instance_suff_scores = zip(*results)

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
