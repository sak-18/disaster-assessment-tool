import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import RobustScaler
import random

# ------------------ CONFIG ------------------ #
ASSETS_DIR = "../assets/full_features_v6"
RECOURSE_SAVE_DIR = "../recourse_final"
DATA_PATH = "../data/data_features.csv"
GROUPINGS_PATH = "../assets/groupings/feature_groupings.csv"
DAG_PATH = "../assets/dags/dag_structures.json"
TARGET_COL = "Property_Damage_GT"
DESIRED_CLASS = 0
MAX_TOTAL_CHANGE = 2.5
DELTA_FRACTION = 0.3
PERTURBATION = 0.5
MAX_ATTEMPTS = 10

# ------------------ LOADERS ------------------ #
def load_assets():
    model = joblib.load(os.path.join(ASSETS_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(ASSETS_DIR, "scaler.joblib"))
    with open(os.path.join(ASSETS_DIR, "model_config.json")) as f:
        cfg = json.load(f)
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    test_idx = np.loadtxt(os.path.join(ASSETS_DIR, "final_test_indices.txt"), dtype=int)

    input_cols = cfg.get("features")
    if input_cols is None:
        print("⚠️  'features' not found in model_config.json, falling back to groupings.")
        groupings = pd.read_csv(GROUPINGS_PATH)
        valid_feats = set(groupings["Feature"])
        input_cols = [c for c in df.columns if c in valid_feats and c != TARGET_COL]

    return model, scaler, cfg, df, test_idx, input_cols


def load_scm_models(scm_dir):
    scm_models, epsilons = {}, {}
    for fname in os.listdir(scm_dir):
        if fname.endswith(".joblib"):
            node = fname.replace(".joblib", "")
            scm_models[node] = joblib.load(os.path.join(scm_dir, fname))
        elif fname.startswith("eps_test"):
            node = fname.replace("eps_test_", "").replace(".npy", "")
            epsilons[node] = np.load(os.path.join(scm_dir, fname))
    return scm_models, epsilons

def load_feature_groups():
    df = pd.read_csv(GROUPINGS_PATH)
    group_map = defaultdict(list)
    for _, row in df.iterrows():
        group_map[row["Group"]].append(row["Feature"])
    return group_map

# ------------------ DAG EXPANSION & TOPO SORT ------------------ #
def expand_group_dag_to_parents(dag_json, groupings, target_col):
    group_to_features = defaultdict(list)
    for _, row in groupings.iterrows():
        group_to_features[row["Group"]].append(row["Feature"])
    dag_parents = defaultdict(list)
    for src_group, tgt_groups in dag_json.items():
        src_feats = group_to_features.get(src_group, [])
        for tgt_group in tgt_groups:
            tgt_feats = group_to_features.get(tgt_group, [])
            for tgt_feat in tgt_feats:
                dag_parents[tgt_feat].extend(src_feats)
    dag_parents.setdefault(target_col, [])
    for src_group, tgt_groups in dag_json.items():
        if target_col in tgt_groups:
            dag_parents[target_col].extend(group_to_features.get(src_group, []))
    for node in dag_parents:
        dag_parents[node] = list(set(dag_parents[node]))
    return dag_parents

def topological_sort(parents_dict):
    all_nodes = set(parents_dict.keys()) | {p for ps in parents_dict.values() for p in ps}
    in_deg = {node: 0 for node in all_nodes}
    for children in parents_dict.values():
        for child in children:
            in_deg[child] += 1
    queue = [node for node, deg in in_deg.items() if deg == 0]
    sorted_nodes = []
    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for child, parents in parents_dict.items():
            if node in parents:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
    return [n for n in sorted_nodes if n in parents_dict]

# ------------------ UTILS ------------------ #
def is_valid_change(x_orig_scaled, x_cf_scaled, scaler, feat_indices):
    x_orig_raw = scaler.inverse_transform([x_orig_scaled])[0]
    x_cf_raw = scaler.inverse_transform([x_cf_scaled])[0]
    if np.any(x_cf_raw[feat_indices] < 0):
        return False
    max_deltas = DELTA_FRACTION * np.maximum(np.abs(x_orig_raw), 1e-5)
    deltas = np.abs(x_cf_raw - x_orig_raw)
    if np.any(deltas > max_deltas):
        return False
    normed_total_change = np.sum(deltas / (max_deltas + 1e-5))
    return normed_total_change <= MAX_TOTAL_CHANGE

def simulate_forward(x_cf_df, epsilons_i, scm_models, dag, intervention_set):
    x_new = x_cf_df.copy()
    for node in topological_sort(dag):
        if node in intervention_set:
            continue
        parents = dag.get(node, [])
        if not all(p in x_new.columns for p in parents):
            continue
        pred = scm_models[node].predict(x_new[parents].values.reshape(1, -1))[0]
        x_new[node] = pred + epsilons_i.get(node, 0)
    return x_new

# ------------------ MAIN ------------------ #
def run_groupwise_recourse(case=2):
    os.makedirs(RECOURSE_SAVE_DIR, exist_ok=True)
    model, scaler, cfg, df, test_idx, input_cols = load_assets()
    group_map = load_feature_groups()

    if case == 1:
        scm_models, epsilons, dag_parents = None, None, {}
    else:
        # ✅ Correct key mapping to dag_structures.json
        dag_key = {
            2: "DAG_1_Independent",
            3: "DAG_3_Flood_Driven"
        }[case]
        with open(DAG_PATH) as f:
            all_dags = json.load(f)
        dag_struct = all_dags[dag_key]

        scm_dir = os.path.join(ASSETS_DIR, f"scm_{dag_key.lower()}")
        scm_models, epsilons = load_scm_models(scm_dir)

        groupings = pd.read_csv(GROUPINGS_PATH)
        dag_parents = expand_group_dag_to_parents(dag_struct, groupings, TARGET_COL)

    for i, idx in enumerate(test_idx[:1]):  # loop more by increasing the range
        x_orig = df.loc[idx, input_cols]
        x_orig_scaled = scaler.transform([x_orig])[0]
        eps_i = {k: epsilons[k][i] for k in epsilons} if epsilons else {}

        for k in range(1, 4):  # groupwise 1, 2, 3
            for attempt in range(MAX_ATTEMPTS):
                intervention_set = []
                x_cf_scaled = x_orig_scaled.copy()

                for group, features in group_map.items():
                    feats = [f for f in features if f in input_cols]
                    if len(feats) < k:
                        continue
                    chosen = random.sample(feats, k)
                    for feat in chosen:
                        j = input_cols.index(feat)
                        delta = random.choice([-PERTURBATION, PERTURBATION])
                        x_cf_scaled[j] += delta
                        intervention_set.append(feat)

                feat_indices = [input_cols.index(f) for f in intervention_set]
                if not is_valid_change(x_orig_scaled, x_cf_scaled, scaler, feat_indices):
                    continue

                x_cf_df = pd.DataFrame([x_cf_scaled], columns=input_cols)

                if case == 1:
                    x_cf_final = x_cf_df
                else:
                    x_cf_sim = simulate_forward(x_cf_df.copy(), eps_i, scm_models, dag_parents, set(intervention_set))
                    x_cf_final = scaler.transform(x_cf_sim[input_cols])

                y_pred = model.predict(x_cf_final)[0]
                y_orig = model.predict([x_orig_scaled])[0]

                if y_pred != y_orig and y_pred == DESIRED_CLASS:
                    print(f"✓ Recourse for instance {idx}, groupwise k={k}, attempt {attempt}, case={case}")
                    changes = x_cf_df[input_cols].iloc[0] - x_orig
                    print("Changed features:", changes[changes != 0])

                    out_data = {
                        "instance_index": int(idx),
                        "case": case,
                        "original_prediction": int(y_orig),
                        "recourse_prediction": int(y_pred),
                        "intervention_set": intervention_set,
                        "changed_features": {
                            feat: {
                                "original": float(x_orig[feat]),
                                "cf": float(x_cf_df[feat].iloc[0])
                            }
                            for feat in intervention_set
                            if x_orig[feat] != x_cf_df[feat].iloc[0]
                        }
                    }
                    save_path = os.path.join(RECOURSE_SAVE_DIR, f"case_{case}_instance_{idx}.json")
                    with open(save_path, "w") as f:
                        json.dump(out_data, f, indent=2)
                    print(f"✓ Saved to {save_path}")
                    return

    print("No valid recourse found.")

if __name__ == "__main__":
    run_groupwise_recourse(case=2)
