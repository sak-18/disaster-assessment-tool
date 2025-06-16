#!/usr/bin/env python
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ─── CONFIG ───────────────────────────────────────────────────────────────────

ASSETS_DIR       = "../assets/full_features_v6"
DATA_PATH        = "../data/data_features.csv"
GROUPINGS_PATH   = "../assets/groupings/feature_groupings.csv"
TARGET_COL       = "Property_Damage_GT"

# ─── LOAD SAVED ASSETS ──────────────────────────────────────────────────────────

# 1) Load predictive model and scaler
model_path  = os.path.join(ASSETS_DIR, "model.joblib")
scaler_path = os.path.join(ASSETS_DIR, "scaler.joblib")
config_path = os.path.join(ASSETS_DIR, "model_config.json")
idx_path    = os.path.join(ASSETS_DIR, "final_test_indices.txt")

model  = joblib.load(model_path)
scaler = joblib.load(scaler_path)

with open(config_path) as f:
    cfg = json.load(f)

# 2) Determine feature list
if "features" in cfg:
    input_cols = cfg["features"]
else:
    print("⚠️  'features' not found in model_config.json, falling back to groupings.")
    grp = pd.read_csv(GROUPINGS_PATH)
    valid = set(grp["Feature"])
    input_cols = [c for c in pd.read_csv(DATA_PATH).columns
                  if c in valid and c != TARGET_COL]

# 3) Load data + test indices
df       = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
test_idx = np.loadtxt(idx_path, dtype=int)

# ─── PREPARE TEST SPLIT ─────────────────────────────────────────────────────────

# Extract X_test and y_test
X_test = df.loc[test_idx, input_cols]
y_test = df.loc[test_idx, TARGET_COL].values

# Scale
X_test_scaled = scaler.transform(X_test)

# ─── EVALUATE ──────────────────────────────────────────────────────────────────

y_pred = model.predict(X_test_scaled)

acc    = accuracy_score(y_test, y_pred)
f1m    = f1_score(y_test, y_pred, average="macro")
f1mi   = f1_score(y_test, y_pred, average="micro")

print("=== Saved Model Evaluation ===")
print(f"Test samples     : {len(test_idx)}")
print(f"Accuracy         : {acc:.4f}")
print(f"F1 (macro)       : {f1m:.4f}")
print(f"F1 (micro)       : {f1mi:.4f}")

# ─── INSPECT A FEW PREDICTIONS ─────────────────────────────────────────────────

print("\nSample predictions (idx, true → pred):")
for i, idx in enumerate(test_idx[:10]):
    print(f"  {idx:5d} {df.at[idx, TARGET_COL]:3d} → {y_pred[i]:3d}")
