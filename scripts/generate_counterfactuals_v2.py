import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

# ------------------ CONFIG ------------------ #
DATA_PATH = "../data/data_features.csv"
GROUPINGS_PATH = "../assets/groupings/feature_groupings.csv"
DAG_PATH = "../assets/dags/dag_structures.json"
OUTPUT_BASE = "../assets/full_features_v6"
TARGET_COL = "Property_Damage_GT"

# ------------------ PREPROCESSING ------------------ #
def compute_transitions(df):
    # Identify transition columns and normalize them by county area
    transition_cols = [c for c in df.columns if c.startswith("transition_")]
    # Keep raw areas for user interventions
    df = df.copy()
    for col in transition_cols:
        df[col + "_raw"] = df[col]
    df[transition_cols] = df[transition_cols].div(df["county_area_m2"].replace(0, np.nan), axis=0)
    return df


def load_data_and_features():
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    df = compute_transitions(df)
    df.fillna(0, inplace=True)
    groupings = pd.read_csv(GROUPINGS_PATH)

    valid_feats = set(groupings["Feature"])
    input_feats = [c for c in df.columns if c in valid_feats]
    model_df = df[input_feats + [TARGET_COL]].copy()

    # Label and split
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(model_df[TARGET_COL])
    X = model_df[input_feats]

    train_idx = np.loadtxt(os.path.join(OUTPUT_BASE, "final_train_indices.txt"), dtype=int)
    test_idx  = np.loadtxt(os.path.join(OUTPUT_BASE, "final_test_indices.txt"), dtype=int)

    with open(DAG_PATH) as f:
        dags = json.load(f)

    return df, X, y, train_idx, test_idx, label_enc, groupings, dags, input_feats

# ------------------ SCM COMPONENTS ------------------ #
def load_scm_components(model_dir):
    models, configs, epsilons = {}, {}, {}
    model_path = os.path.join(model_dir, "models")
    for fn in os.listdir(model_path):
        if fn.endswith(".joblib") and fn != "scaler.joblib":
            node = fn.replace(".joblib", "")
            models[node] = joblib.load(os.path.join(model_path, fn))
        elif fn.endswith("_model_config.json"):
            node = fn.replace("_model_config.json", "")
            with open(os.path.join(model_path, fn)) as f:
                configs[node] = json.load(f)

    eps_dir = os.path.join(model_dir, "eps")
    for fn in os.listdir(eps_dir):
        if fn.startswith("eps_test_") and fn.endswith(".npy"):
            node = fn.replace("eps_test_", "").replace(".npy", "")
            epsilons[node] = np.load(os.path.join(eps_dir, fn))

    scaler = joblib.load(os.path.join(model_path, "scaler.joblib"))
    return models, configs, epsilons, scaler

# ------------------ TOPSORT ------------------ #
def expand_group_dag_to_parents(dag_json, groupings, target_col):
    from collections import defaultdict
    feat_map = defaultdict(list)
    for _, r in groupings.iterrows():
        feat_map[r['Group']].append(r['Feature'])

    parents = defaultdict(list)
    for src, tgts in dag_json.items():
        src_feats = feat_map.get(src, [])
        for tgt in tgts:
            for f in feat_map.get(tgt, []):
                parents[f].extend(src_feats)
    # include target_col parents
    parents.setdefault(target_col, [])
    for src, tgts in dag_json.items():
        if target_col in tgts:
            parents[target_col].extend(feat_map.get(src, []))

    # dedupe
    for k in parents:
        parents[k] = list(set(parents[k]))
    return parents


def topological_sort(parents_dict):
    nodes = set(parents_dict) | {p for ps in parents_dict.values() for p in ps}
    indegree = {n: 0 for n in nodes}
    for c, ps in parents_dict.items():
        for p in ps:
            indegree[c] += 1
    queue = [n for n, d in indegree.items() if d == 0]
    order = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for c, ps in parents_dict.items():
            if n in ps:
                indegree[c] -= 1
                if indegree[c] == 0:
                    queue.append(c)
    return [n for n in order if n in parents_dict or n == TARGET_COL]

# ------------------ PREDICTION ------------------ #
def evaluate_and_predict_scm(X, y, parents, train_idx, test_idx, model_dir):
    X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_tr, y_te = y[train_idx], y[test_idx]
    scaler = joblib.load(os.path.join(model_dir, "models", "scaler.joblib"))

    X_tr[:] = scaler.transform(X_tr)
    X_te[:] = scaler.transform(X_te)

    order = topological_sort(parents)
    for node in order:
        m = joblib.load(os.path.join(model_dir, "models", f"{node}.joblib"))
        cfg = json.load(open(os.path.join(model_dir, "models", f"{node}_model_config.json")))
        feats = cfg['input_features']
        if node == TARGET_COL:
            p = m.predict(X_te[feats])
            print(f"SCM {node} -> acc: {accuracy_score(y_te,p):.3f}, f1: {f1_score(y_te,p,average='macro'):.3f}")
        X_te[node] = m.predict(X_te[feats])
    return X_te, scaler

# ------------------ INTERVENTION HELPERS ------------------ #
def normalize_interventions(interventions_raw, sample_df):
    # Convert user-provided raw area deltas to normalized values
    mapping = {}
    area = sample_df['county_area_m2'].values[0]
    for var, raw_delta in interventions_raw.items():
        mapping[var] = raw_delta / area
    return mapping


def apply_abduction(instance, models, eps, cfgs, topo, interventions):
    inst = instance.copy()
    for node in topo:
        if node in interventions or node not in eps:
            continue
        parents = cfgs[node]['input_features']
        pred = models[node].predict(inst[parents])[0]
        inst[node] = pred + eps[node][0] if node != TARGET_COL else pred
    return inst

# ------------------ MAIN SIMULATION ------------------ #
def run_scm_counterfactual(original_raw, interventions_raw, dag_key):
    # Load and preprocess
    df, X, y, tr_idx, te_idx, lbl_enc, groupings, dags, input_feats = load_data_and_features()
    dag = dags[dag_key]
    parents = expand_group_dag_to_parents(dag, groupings, TARGET_COL)
    model_dir = os.path.join(OUTPUT_BASE, f"scm_{dag_key.lower()}")

    # Evaluate SCM and get scaler
    _, scaler = evaluate_and_predict_scm(X, y, parents, tr_idx, te_idx, model_dir)

    # Normalize interventions from raw space to normalized
    sample = pd.DataFrame([original_raw])
    print("Original raw values:")
    for v in interventions_raw:
        print(f" - {v}: {original_raw[v]}")

    norm_int = normalize_interventions(interventions_raw, sample)
    # Scale sample
    sample_scaled = sample[input_feats].copy()
    sample_scaled[input_feats] = scaler.transform(sample[input_feats])
    # Compute scaled interventions
    scaled_int = {}
    for var, val in norm_int.items():
        temp = sample_scaled.copy()
        temp[var] = val
        scaled_int[var] = val

    # Load SCM components
    models, configs, eps, _ = load_scm_components(model_dir)
    topo_cfg = topological_sort({k: v['input_features'] for k, v in configs.items()})

    # Do intervention and abduction-prediction
    cf = sample_scaled.copy()
    for v, newv in scaled_int.items(): cf[v] = newv
    cf = apply_abduction(cf, models, eps, configs, topo_cfg, scaled_int)

    orig_label = original_raw[TARGET_COL]
    cf_label = lbl_enc.inverse_transform([int(cf[TARGET_COL])])[0]
    print(f"Original Prediction: {orig_label} | Counterfactual Prediction: {cf_label}")
    return cf_label, cf

# ------------------ USAGE ------------------ #
if __name__ == "__main__":
    # Example: pick a test instance and raw deltas
    df_all = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    df_all = compute_transitions(df_all)
    df_all.fillna(0, inplace=True)
    valid_feats = pd.read_csv(GROUPINGS_PATH)["Feature"].tolist()
    input_feats = [c for c in df_all.columns if c in valid_feats]
    test_idx = np.loadtxt(os.path.join(OUTPUT_BASE, "final_test_indices.txt"), dtype=int)
    test_df = df_all[input_feats + [TARGET_COL] + ['county_area_m2']].iloc[test_idx]

    original = test_df.iloc[10].to_dict()
    # User provides raw area deltas (m^2) rather than normalized fractions
    interventions = {
        'transition_6_0': 12000,
        'transition_6_1': 500000
    }

    run_scm_counterfactual(original, interventions, dag_key="DAG_1_Independent")
