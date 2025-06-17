import os
import json
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

# ------------------ CONFIG ------------------ #
DATA_PATH = "../data/data_features.csv"
GROUPINGS_PATH = "../assets/groupings/feature_groupings.csv"
DAG_PATH = "../assets/dags/dag_structures.json"
OUTPUT_BASE = "../assets/full_features_v6"
TARGET_COL = "Property_Damage_GT"

# ------------------ DAG HELPERS ------------------ #
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

    # Handle direct links to target_col
    dag_parents.setdefault(target_col, [])
    for src_group, tgt_groups in dag_json.items():
        if target_col in tgt_groups:
            dag_parents[target_col].extend(group_to_features.get(src_group, []))

    # Deduplicate
    for node in dag_parents:
        dag_parents[node] = list(set(dag_parents[node]))

    return dag_parents

def topological_sort(parents_dict):
    all_nodes = set(parents_dict.keys()) | {p for ps in parents_dict.values() for p in ps}
    in_deg = {node: 0 for node in all_nodes}
    for child, parents in parents_dict.items():
        for parent in parents:
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

    return [n for n in sorted_nodes if n in parents_dict or n == TARGET_COL]

# ------------------ EVALUATION ------------------ #
def evaluate_loaded_scm_models(X, y, parents_dict, train_idx, test_idx, model_dir):
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    original_X_test = X_test.copy()
    node_order = topological_sort(parents_dict)
    all_metrics = {}

    # Load and apply saved scaler
    scaler_path = os.path.join(model_dir, "models", "scaler.joblib")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        X_train[X_train.columns] = scaler.transform(X_train[X_train.columns])
        X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
    else:
        print("[Warning] Scaler not found. Evaluation may be inconsistent.")

    for node in node_order:
        model_path = os.path.join(model_dir, "models", f"{node}.joblib")
        config_path = os.path.join(model_dir, "models", f"{node}_model_config.json")
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"Model or config for node {node} not found.")
            continue

        model = joblib.load(model_path)
        with open(config_path) as f:
            model_config = json.load(f)
        parents = model_config["input_features"]

        if node == TARGET_COL:
            y_tr, y_te = y_train, y_test
        else:
            y_tr, y_te = X_train[node], original_X_test[node]

        preds_tr = model.predict(X_train[parents])
        preds_te = model.predict(X_test[parents])

        if node == TARGET_COL:
            train_metrics = {
                "accuracy": accuracy_score(y_tr, preds_tr),
                "f1_macro": f1_score(y_tr, preds_tr, average="macro"),
                "precision_macro": precision_score(y_tr, preds_tr, average="macro", zero_division=0),
                "recall_macro": recall_score(y_tr, preds_tr, average="macro", zero_division=0)
            }
            test_metrics = {
                "accuracy": accuracy_score(y_te, preds_te),
                "f1_macro": f1_score(y_te, preds_te, average="macro"),
                "precision_macro": precision_score(y_te, preds_te, average="macro", zero_division=0),
                "recall_macro": recall_score(y_te, preds_te, average="macro", zero_division=0)
            }
        else:
            train_metrics = {"r2": round(model.score(X_train[parents], y_tr), 4)}
            test_metrics = {"r2": round(model.score(X_test[parents], y_te), 4)}

        all_metrics[node] = {"train": train_metrics, "test": test_metrics}

        if node == TARGET_COL:
            print(
                f"[LOADED SCM {node}] "
                f"acc_tr={train_metrics['accuracy']:.4f}, "
                f"f1_tr={train_metrics['f1_macro']:.4f}, "
                f"acc_te={test_metrics['accuracy']:.4f}, "
                f"f1_te={test_metrics['f1_macro']:.4f}"
            )

        X_test[node] = preds_te  # Needed for downstream nodes

    return all_metrics

# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    # Load & preprocess data
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    groupings = pd.read_csv(GROUPINGS_PATH)
    transition_cols = [c for c in df.columns if c.startswith("transition_")]
    df[transition_cols] = df[transition_cols].div(df["county_area_m2"].replace(0, np.nan), axis=0)
    df = df.fillna(0)

    valid_feats = set(groupings["Feature"])
    input_features = [c for c in df.columns if c in valid_feats and c != TARGET_COL]
    df_model = df[input_features + [TARGET_COL]].copy()
    y = LabelEncoder().fit_transform(df_model[TARGET_COL])
    X = df_model[input_features]

    # Load train/test indices
    train_idx = np.loadtxt(os.path.join(OUTPUT_BASE, "final_train_indices.txt"), dtype=int)
    test_idx = np.loadtxt(os.path.join(OUTPUT_BASE, "final_test_indices.txt"), dtype=int)

    # Load DAGs
    with open(DAG_PATH) as f:
        dag_structs = json.load(f)

    # Evaluate each DAG model set
    for dag_key in dag_structs:
        model_dir = os.path.join(OUTPUT_BASE, f"scm_{dag_key.lower()}")
        parents_dict = expand_group_dag_to_parents(dag_structs[dag_key], groupings, TARGET_COL)
        print(f"\n=== Evaluating Models for DAG: {dag_key} ===")
        evaluate_loaded_scm_models(X, y, parents_dict, train_idx, test_idx, model_dir)
