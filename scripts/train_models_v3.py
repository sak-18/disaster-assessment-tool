import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------ CONFIG ------------------ #
DATA_PATH = "../data/data_features.csv"
GROUPINGS_PATH = "../assets/groupings/feature_groupings.csv"
DAG_PATH = "../assets/dags/dag_structures.json"
OUTPUT_BASE = "../assets/full_features_v3"
TARGET_COL = "Property_Damage_GT"

# ------------------ MODEL CONFIGS ------------------ #
model_configs = {
    "XGBoost_Base": XGBClassifier(eval_metric="mlogloss", n_estimators=50,
                                  max_depth=6, learning_rate=0.1, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0, random_state=42),
    "XGBoost_Regularized": XGBClassifier(eval_metric="mlogloss", n_estimators=50,
                                         max_depth=6, learning_rate=0.1, subsample=0.8,
                                         colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.5, random_state=42),
    "RandomForest_Base": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42),
    "RandomForest_Regularized": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
    "MLP_2Layer_Base": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.0001, max_iter=3000, random_state=42),
    "MLP_2Layer_Regularized": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.01, max_iter=3000, random_state=42),
    "MLP_5Layer_Base": MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32, 16), alpha=0.0001, max_iter=3000, random_state=42),
    "MLP_5Layer_Regularized": MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32, 16), alpha=0.01, max_iter=3000, random_state=42),
    "LogisticRegression_Base": LogisticRegression(penalty=None, solver="saga", max_iter=3000, random_state=42),
    "LogisticRegression_L1": LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=3000, random_state=42)
}

def expand_group_dag_to_parents(dag_json, groupings, target_col):
    group_to_features = defaultdict(list)
    for _, row in groupings.iterrows():
        group_to_features[row["Group"]].append(row["Feature"])

    dag_parents = defaultdict(list)

    # Step 1: Expand all group-to-group DAG edges normally
    for src_group, tgt_groups in dag_json.items():
        src_feats = group_to_features.get(src_group, [])
        for tgt_group in tgt_groups:
            tgt_feats = group_to_features.get(tgt_group, [])
            for tgt_feat in tgt_feats:
                dag_parents[tgt_feat].extend(src_feats)

    # Step 2: Explicitly handle target_col
    dag_parents.setdefault(target_col, [])

    # Find all groups that directly point to Property_Damage_GT in the DAG
    for src_group, tgt_groups in dag_json.items():
        if target_col in tgt_groups:
            src_feats = group_to_features.get(src_group, [])
            dag_parents[target_col].extend(src_feats)

    # Step 3: Deduplicate
    for node in dag_parents:
        dag_parents[node] = list(set(dag_parents[node]))

    return dag_parents


def topological_sort(parents_dict):
    all_nodes = set(parents_dict.keys()) | {p for ps in parents_dict.values() for p in ps}
    in_deg = {node: 0 for node in all_nodes}
    for children in parents_dict.values():
        for child in children:
            in_deg[child] += 1
    queue = [node for node in all_nodes if in_deg[node] == 0]
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

def train_predictive_models(X, y, input_cols, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_score = -1
    best_meta = {}
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    for name, model in model_configs.items():
        for train_idx, test_idx in skf.split(X_scaled, y):
            model.fit(X_scaled[train_idx], y[train_idx])
            preds = model.predict(X_scaled[test_idx])
            score = f1_score(y[test_idx], preds, average="macro")
            if score > best_score:
                best_model = clone(model)
                best_score = score
                best_meta = {
                    "model_name": name,
                    "train_idx": train_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                    "score": score
                }

    joblib.dump(best_model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    np.savetxt(os.path.join(output_dir, "final_train_indices.txt"), best_meta["train_idx"], fmt="%s")
    np.savetxt(os.path.join(output_dir, "final_test_indices.txt"), best_meta["test_idx"], fmt="%s")
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(best_meta, f, indent=4)

    return best_meta, best_model, scaler

def train_scm(X, y, parents_dict, train_idx, test_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    models = {}
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]

    node_order = topological_sort(parents_dict)
    for node in node_order:
        if node == TARGET_COL:
            parent_feats = parents_dict.get(node, [])
            model = clone(RandomForestClassifier(n_estimators=50, random_state=42))
            model.fit(X_train[parent_feats], y_train)
            y_pred = model.predict(X_test[parent_feats])
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            joblib.dump(model, os.path.join(output_dir, "target_model.joblib"))
        else:
            parents = parents_dict.get(node, [])
            if not parents:
                continue
            model = clone(RandomForestClassifier(n_estimators=50, random_state=42))
            model.fit(X_train[parents], X_train[node])
            X_test[node] = model.predict(X_test[parents])
            models[node] = model
            joblib.dump(model, os.path.join(output_dir, f"{node}.joblib"))

    with open(os.path.join(output_dir, "scm_metrics.json"), "w") as f:
        json.dump({"macro_f1": macro_f1}, f, indent=4)

    print(f"[SCM] {output_dir.split('/')[-1]} macro-F1: {macro_f1:.4f}")

# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    groupings = pd.read_csv(GROUPINGS_PATH)
    valid_features = set(groupings["Feature"])
    transition_cols = [c for c in df.columns if c.startswith("transition_")]
    df[transition_cols] = df[transition_cols].div(df["county_area_m2"].replace(0, np.nan), axis=0)
    df = df.fillna(0)
    input_features = [col for col in df.columns if col in valid_features and col != TARGET_COL]
    df_model = df[input_features + [TARGET_COL]].copy()
    y = LabelEncoder().fit_transform(df_model[TARGET_COL])
    X = df_model[input_features]

    meta, model, scaler = train_predictive_models(X, y, input_features, OUTPUT_BASE)
    train_idx, test_idx = np.array(meta["train_idx"]), np.array(meta["test_idx"])

    with open(DAG_PATH) as f:
        dag_structs = json.load(f)

    for dag_key in ["DAG_2_Infrastructure_Mediator", "DAG_3_Flood_Driven"]:
        parents_dict = expand_group_dag_to_parents(dag_structs[dag_key], groupings, TARGET_COL)
        print(f"[INFO] Parents of {TARGET_COL} from {dag_key}:", parents_dict.get(TARGET_COL, []))
        train_scm(X, y, parents_dict, train_idx, test_idx, os.path.join(OUTPUT_BASE, f"scm_{dag_key.lower()}"))



    # --- Plotting model performances ---
    scores = [(meta["model_name"], meta["score"])]
    for dag_key in ["DAG_2_Infrastructure_Mediator", "DAG_3_Flood_Driven"]:
        scm_path = os.path.join(OUTPUT_BASE, f"scm_{dag_key.lower()}/scm_metrics.json")
        if os.path.exists(scm_path):
            with open(scm_path) as f:
                scm_f1 = json.load(f)["macro_f1"]
                scores.append((f"SCM_{dag_key.split('_')[1]}", scm_f1))

    labels, values = zip(*scores)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=["steelblue" if not l.startswith("SCM") else "darkorange" for l in labels])
    plt.ylabel("Macro-F1 Score")
    plt.title("Test Macro-F1 Scores: Best Predictive Model vs SCMs")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE, "model_comparison_barplot.png"))
    plt.show()

    print("Pipeline complete.")
