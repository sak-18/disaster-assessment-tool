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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "eps"), exist_ok=True)

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    original_X_test = X_test.copy()

    node_order = topological_sort(parents_dict)
    # print(f"Node Order: {node_order}")
    all_metrics = {}
    models = {}

    for node in node_order:
        parents = parents_dict.get(node, [])
        if not parents:
            continue

        # Define targets
        if node == TARGET_COL:
            y_tr, y_te = y_train, y_test
            model = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            y_tr, y_te = X_train[node], original_X_test[node]
            model = RandomForestRegressor(n_estimators=50, random_state=42)

        # Train model
        model.fit(X_train[parents], y_tr)

        # Predictions
        preds_tr = model.predict(X_train[parents])
        preds_te = model.predict(X_test[parents])

        # Residuals (epsilons)
        np.save(os.path.join(output_dir, "eps", f"eps_train_{node}.npy"), y_tr - preds_tr)
        np.save(os.path.join(output_dir, "eps", f"eps_test_{node}.npy"), y_te - preds_te)

        # Evaluate metrics
        if node == TARGET_COL:
            train_metrics = {
                "accuracy":        accuracy_score(y_tr, preds_tr),
                "f1_macro":        f1_score(y_tr, preds_tr, average="macro"),
                "precision_macro": precision_score(y_tr, preds_tr, average="macro", zero_division=0),
                "recall_macro":    recall_score(y_tr, preds_tr, average="macro", zero_division=0)
            }
            test_metrics = {
                "accuracy":        accuracy_score(y_te, preds_te),
                "f1_macro":        f1_score(y_te, preds_te, average="macro"),
                "precision_macro": precision_score(y_te, preds_te, average="macro", zero_division=0),
                "recall_macro":    recall_score(y_te, preds_te, average="macro", zero_division=0)
            }
        else:
            train_metrics = {"r2": round(model.score(X_train[parents], y_tr), 4)}
            test_metrics = {"r2": round(model.score(X_test[parents], y_te), 4)}

        # Save model
        model_path = os.path.join(output_dir, "models", f"{node}.joblib")
        joblib.dump(model, model_path)
        models[node] = model

        # Save config
        config = {
            "model_name":      node,
            "type":            str(type(model)).split("'")[1],
            "hyperparameters": model.get_params(),
            "train_metrics":   train_metrics,
            "test_metrics":    test_metrics,
            "input_features":  parents,
            "target_column":   node,
            "preprocessing": {
                "scaler":         "RobustScaler",
                "normalization":  "transition_*/county_area_m2",
                "fillna":         0
            }
        }
        with open(os.path.join(output_dir, "models", f"{node}_model_config.json"), "w") as fp:
            json.dump(config, fp, indent=4)

        # Print summary for TARGET_COL only
        if node == TARGET_COL:
            print(
                f"[SCM {node}] "
                f"acc_tr={train_metrics['accuracy']:.4f}, "
                f"f1_tr={train_metrics['f1_macro']:.4f}, "
                f"precision_macro_tr={train_metrics['precision_macro']:.4f}, "
                f"recall_macro_tr={train_metrics['recall_macro']:.4f}, "
                f"acc_te={test_metrics['accuracy']:.4f}, "
                f"f1_te={test_metrics['f1_macro']:.4f}, "
                f"precision_macro_te={test_metrics['precision_macro']:.4f}, "
                f"recall_macro_te={test_metrics['recall_macro']:.4f}"
            )

        # Update downstream
        X_test[node] = preds_te
        all_metrics[node] = {"train": train_metrics, "test": test_metrics}

    with open(os.path.join(output_dir, "scm_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)

    return models, all_metrics


# ------------------ HELPER FUNCTIONS ------------------ #

def expand_group_dag_to_parents(dag_json, groupings, target_col):
    from collections import defaultdict

    # Step 1: group → features
    group_to_features = defaultdict(list)
    feature_to_group = {}
    for _, row in groupings.iterrows():
        feat, group = row["Feature"], row["Group"]
        group_to_features[group].append(feat)
        feature_to_group[feat] = group

    dag_parents = defaultdict(list)

    # Step 2: Expand all group DAG edges
    for src_group, tgt_groups in dag_json.items():
        src_feats = group_to_features.get(src_group, [])
        for tgt_group in tgt_groups:
            tgt_feats = group_to_features.get(tgt_group, [])
            for tgt_feat in tgt_feats:
                dag_parents[tgt_feat].extend(src_feats)

            # Special case: target_col is referred to directly as a group
            if tgt_group == target_col:
                for tgt_feat in [target_col]:
                    dag_parents[tgt_feat].extend(src_feats)

    # Step 3: Deduplicate
    for node in dag_parents:
        dag_parents[node] = list(set(dag_parents[node]))

    # Make sure target_col is initialized
    dag_parents.setdefault(target_col, [])

    return dag_parents


def topological_sort(parents_dict):
    # Step 1: Collect all unique nodes
    all_nodes = set(parents_dict.keys()) | {p for ps in parents_dict.values() for p in ps}

    # Step 2: Build in-degree map (how many edges point *to* each node)
    in_deg = {node: 0 for node in all_nodes}
    for child, parents in parents_dict.items():
        for parent in parents:
            in_deg[child] += 1

    # Step 3: Start with nodes that have in-degree 0 (i.e., no parents)
    queue = [node for node in all_nodes if in_deg[node] == 0]
    sorted_nodes = []

    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)

        # Reduce in-degree of children that depend on this node
        for potential_child, parents in parents_dict.items():
            if node in parents:
                in_deg[potential_child] -= 1
                if in_deg[potential_child] == 0:
                    queue.append(potential_child)

    # Optional: filter to only keep relevant (observed) nodes
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
        # print(f"[INFO] Parents of {TARGET_COL} from {dag_key}:", 
            # parents_dict.get(TARGET_COL, []))
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
