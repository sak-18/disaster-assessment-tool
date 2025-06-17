import os
import json
import joblib
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from generate_counterfactuals import run_scm_counterfactual_simulation  # adjust import

# ------------------ CONFIG ------------------ #
DATA_PATH       = "../data/data_features.csv"
GROUPINGS_PATH  = "../assets/groupings/feature_groupings.csv"
DAG_PATH        = "../assets/dags/dag_structures.json"
OUTPUT_BASE     = "../assets/full_features_v6"
TARGET_COL      = "Property_Damage_GT"
DAG_KEY         = "DAG_2_Infrastructure_Mediator"

# ------------------ LOAD & PREP ------------------ #
df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
trans_cols = [c for c in df if c.startswith("transition_")]
df[trans_cols] = df[trans_cols].div(df["county_area_m2"].replace(0, np.nan), axis=0).fillna(0)

groupings      = pd.read_csv(GROUPINGS_PATH)
valid_feats    = set(groupings["Feature"])
input_features = [c for c in df.columns if c in valid_feats and c != TARGET_COL]
df_model       = df[input_features + [TARGET_COL]].copy()

le = LabelEncoder()
df_model[TARGET_COL] = le.fit_transform(df_model[TARGET_COL])

train_idx = np.loadtxt(os.path.join(OUTPUT_BASE, "final_train_indices.txt"), dtype=int)
test_idx  = np.loadtxt(os.path.join(OUTPUT_BASE, "final_test_indices.txt"),  dtype=int)

# ------------------ SAMPLE PER CLASS ------------------ #
# group test indices by their original (decoded) label
class_to_idxs = defaultdict(list)
for idx in test_idx:
    lbl = le.inverse_transform([int(df_model.at[idx, TARGET_COL])])[0]
    class_to_idxs[lbl].append(idx)

# from each class, sample up to 20
sample_indices = []
for lbl, idxs in class_to_idxs.items():
    n = min(20, len(idxs))
    sample_indices += random.sample(idxs, n)

# prepare counters
flip_counts_null   = {lbl: 0 for lbl in class_to_idxs}
flip_counts_random = {lbl: 0 for lbl in class_to_idxs}

# ------------------ RUN INTERVENTIONS ------------------ #
random.seed(42)
for idx in sample_indices:
    row = df_model.iloc[idx]
    orig_lbl = le.inverse_transform([int(row[TARGET_COL])])[0]
    sample_dict = row.to_dict()

    # 1) NULL intervention
    cf_null, _ = run_scm_counterfactual_simulation(
        original_sample=sample_dict,
        interventions_raw={},      # no change
        dag_key=DAG_KEY
    )
    if cf_null != orig_lbl:
        flip_counts_null[orig_lbl] += 1

    # 2) RANDOM small intervention
    k = random.randint(1, 3)
    feats = random.sample(input_features, k)
    interventions = {
        f: float(random.uniform(df_model[f].min(), df_model[f].max()))
        for f in feats
    }
    cf_rand, _ = run_scm_counterfactual_simulation(
        original_sample=sample_dict,
        interventions_raw=interventions,
        dag_key=DAG_KEY
    )
    if cf_rand != orig_lbl:
        flip_counts_random[orig_lbl] += 1

# ------------------ REPORT ------------------ #
print("Null‐intervention flips per class:")
for lbl, cnt in flip_counts_null.items():
    print(f"  {lbl:10s}: {cnt}/{len(class_to_idxs[lbl])}")

print("\nRandom‐intervention flips per class:")
for lbl, cnt in flip_counts_random.items():
    print(f"  {lbl:10s}: {cnt}/{len(class_to_idxs[lbl])}")
