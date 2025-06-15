import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.stats import kstest
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings for logistic regression
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------ CONFIG ------------------ #
DATA_PATH = "../data/data_features.csv"
GROUPINGS_PATH = "../assets/groupings/feature_groupings.csv"
DAG_PATH = "../assets/dags/dag_structures.json"
OUTPUT_BASE = "../assets/full_features_v6"
TARGET_COL = "Property_Damage_GT"
PRIMARY_METRIC = "accuracy"  # Options: accuracy, f1_macro, f1_micro, precision_macro, recall_macro

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

# ------------------ PHASE 1: Training the SCM ------------------ #

def train_scm(X, y, parents_dict, train_idx, test_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Split data
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Keep a copy of the original test features for ground truth
    original_X_test = X_test.copy()

    # Determine node ordering
    node_order = topological_sort(parents_dict)

    models = {}
    all_metrics = {}

    for node in node_order:
        parents = parents_dict.get(node, [])
        if not parents:
            continue

        # Prepare targets
        if node == TARGET_COL:
            y_train_node = y_train
            y_test_node = y_test
        else:
            y_train_node = X_train[node]
            y_test_node = original_X_test[node]

        # Train model for this node
        model = clone(RandomForestClassifier(n_estimators=50, random_state=42))
        model.fit(X_train[parents], y_train_node)

        # Predictions
        preds_train = model.predict(X_train[parents])
        preds_test = model.predict(X_test[parents])

        # Compute residuals (epsilons) and save
        eps_train = y_train_node - preds_train
        eps_test  = y_test_node  - preds_test
        np.save(os.path.join(output_dir, f"eps_train_{node}.npy"), eps_train)
        np.save(os.path.join(output_dir, f"eps_test_{node}.npy"),  eps_test)

        # Evaluate metrics on test
        metrics_node = {
            "accuracy":         accuracy_score(y_test_node, preds_test),
            "f1_macro":         f1_score(y_test_node, preds_test, average="macro"),
            "precision_macro":  precision_score(y_test_node, preds_test, average="macro", zero_division=0),
            "recall_macro":     recall_score(y_test_node, preds_test, average="macro", zero_division=0)
        }
        all_metrics[node] = metrics_node

        # Print metrics in one line
        print(
            f"[SCM {node}] "
            f"accuracy={metrics_node['accuracy']:.4f}, "
            f"f1_macro={metrics_node['f1_macro']:.4f}, "
            f"precision_macro={metrics_node['precision_macro']:.4f}, "
            f"recall_macro={metrics_node['recall_macro']:.4f}"
        )

        # Save model for this node
        joblib.dump(model, os.path.join(output_dir, f"{node}.joblib"))

        # Update X_test for downstream nodes
        X_test[node] = preds_test

        # Store model
        models[node] = model

    # Persist all node‐level SCM metrics
    with open(os.path.join(output_dir, "scm_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)

    return models, all_metrics

# ------------------ HELPER FUNCTIONS ------------------ #

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
    return [n for n in sorted_nodes if n in parents_dict or n == TARGET_COL]

# ------------------ PREDICTIVE MODEL TRAINING ------------------ #
def train_predictive_models(X, y, input_cols, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model, best_score, best_meta = None, -1, {}
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    for name, model in model_configs.items():
        for train_idx, test_idx in skf.split(X_scaled, y):
            model.fit(X_scaled[train_idx], y[train_idx])
            preds = model.predict(X_scaled[test_idx])
            metrics = {
                "accuracy": accuracy_score(y[test_idx], preds),
                "f1_macro": f1_score(y[test_idx], preds, average="macro"),
                "f1_micro": f1_score(y[test_idx], preds, average="micro"),
                "precision_macro": precision_score(y[test_idx], preds, average="macro", zero_division=0),
                "recall_macro": recall_score(y[test_idx], preds, average="macro", zero_division=0)
            }
            if metrics[PRIMARY_METRIC] > best_score:
                best_model, best_score = clone(model), metrics[PRIMARY_METRIC]
                best_meta = {"model_name": name, "train_idx": train_idx.tolist(),
                             "test_idx": test_idx.tolist(), "metrics": metrics}

    joblib.dump(best_model, os.path.join(output_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.joblib"))
    np.savetxt(os.path.join(output_dir, "final_train_indices.txt"), best_meta["train_idx"], fmt="%s")
    np.savetxt(os.path.join(output_dir, "final_test_indices.txt"), best_meta["test_idx"], fmt="%s")
    with open(os.path.join(output_dir, "model_config.json"), "w") as f:
        json.dump(best_meta, f, indent=4)

    print(f"\n=== Best Model Based on `{PRIMARY_METRIC}`: {best_meta['model_name']} ===")
    for k, v in best_meta["metrics"].items():
        print(f"{k}: {v:.4f}")
    print("===========================================\n")
    return best_meta, best_model, scaler

# ------------------ MAIN PIPELINE ------------------ #
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

    # Predictive model training & CV split
    meta, model, scaler = train_predictive_models(X, y, input_features, OUTPUT_BASE)
    train_idx, test_idx = np.array(meta["train_idx"]), np.array(meta["test_idx"])

    # Phase 1: SCM Training for specific DAGs
    with open(DAG_PATH) as f:
        dag_structs = json.load(f)

    for dag_key in dag_structs:
        parents_dict = expand_group_dag_to_parents(
            dag_structs[dag_key], groupings, TARGET_COL
        )
        print(f"[INFO] Parents of {TARGET_COL} from {dag_key}:", 
            parents_dict.get(TARGET_COL, []))
        train_scm(
            X, y, parents_dict,
            train_idx, test_idx,
            os.path.join(OUTPUT_BASE, f"scm_{dag_key.lower()}")
        )


    # Plot comparison
    scores = [(meta["model_name"], meta["metrics"][PRIMARY_METRIC])]
    for dag_key in dag_structs:
        scm_path = os.path.join(OUTPUT_BASE, f"scm_{dag_key.lower()}/scm_metrics.json")
        if os.path.exists(scm_path):
            with open(scm_path) as f:
                scm_m = json.load(f)
                if PRIMARY_METRIC in scm_m:
                    scores.append((f"SCM_{dag_key.split('_')[1]}", scm_m[PRIMARY_METRIC]))

    labels, values = zip(*scores)
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=["steelblue" if not l.startswith("SCM") else "darkorange" for l in labels])
    plt.ylabel(f"{PRIMARY_METRIC.replace('_', ' ').title()} Score")
    plt.title(f"Test {PRIMARY_METRIC.replace('_', ' ').title()} Scores: Best Predictive Model vs SCMs")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE, f"model_comparison_barplot_{PRIMARY_METRIC}.png"))
    plt.show()

    print("Pipeline complete.")
