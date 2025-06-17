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

# ------------------ MODEL LOADING ------------------ #
def load_scm_components(model_dir):
    models, configs, epsilons = {}, {}, {}
    for file in os.listdir(os.path.join(model_dir, "models")):
        if file.endswith(".joblib") and file != "scaler.joblib":
            node = file.replace(".joblib", "")
            models[node] = joblib.load(os.path.join(model_dir, "models", file))
        elif file.endswith("_model_config.json"):
            node = file.replace("_model_config.json", "")
            with open(os.path.join(model_dir, "models", file)) as f:
                configs[node] = json.load(f)

    eps_dir = os.path.join(model_dir, "eps")
    for file in os.listdir(eps_dir):
        if file.startswith("eps_test_") and file.endswith(".npy"):
            node = file.replace("eps_test_", "").replace(".npy", "")
            epsilons[node] = np.load(os.path.join(eps_dir, file))

    scaler = joblib.load(os.path.join(model_dir, "models", "scaler.joblib"))
    return models, configs, epsilons, scaler

# ------------------ SCM EVALUATION ------------------ #
def evaluate_loaded_scm_models(X, y, parents_dict, train_idx, test_idx, model_dir):
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    node_order = topological_sort(parents_dict)

    scaler = joblib.load(os.path.join(model_dir, "models", "scaler.joblib"))
    X_train[X_train.columns] = scaler.transform(X_train[X_train.columns])
    X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

    for node in node_order:
        model = joblib.load(os.path.join(model_dir, "models", f"{node}.joblib"))
        with open(os.path.join(model_dir, "models", f"{node}_model_config.json")) as f:
            model_config = json.load(f)
        parents = model_config["input_features"]

        if node == TARGET_COL:
            preds = model.predict(X_test[parents])
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            print(f"[LOADED SCM {node}] acc_te={acc:.4f}, f1_te={f1:.4f}")
        X_test[node] = model.predict(X_test[parents])

    return X_test, scaler

# ------------------ INTERVENTION & CF GENERATION ------------------ #
def do_intervention(instance, interventions):
    for var, val in interventions.items():
        instance[var] = val
    return instance

def abduction_action_prediction(instance, models, epsilons, configs, top_order, interventions):
    for node in top_order:
        if node in interventions or node not in epsilons:
            continue
        parents = configs[node]["input_features"]
        pred = models[node].predict(instance[parents])[0]
        instance[node] = pred + epsilons[node][0] if node != TARGET_COL else pred
    return instance

# ------------------ MAIN FUNCTION ------------------ #
def run_scm_counterfactual_simulation(original_sample, interventions_raw, dag_key, dag_path=DAG_PATH):
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    groupings = pd.read_csv(GROUPINGS_PATH)
    transition_cols = [c for c in df.columns if c.startswith("transition_")]
    df[transition_cols] = df[transition_cols].div(df["county_area_m2"].replace(0, np.nan), axis=0)
    df = df.fillna(0)

    valid_feats = set(groupings["Feature"])
    input_features = [c for c in df.columns if c in valid_feats and c != TARGET_COL]
    df_model = df[input_features + [TARGET_COL]].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_model[TARGET_COL])
    X = df_model[input_features]

    train_idx = np.loadtxt(os.path.join(OUTPUT_BASE, "final_train_indices.txt"), dtype=int)
    test_idx = np.loadtxt(os.path.join(OUTPUT_BASE, "final_test_indices.txt"), dtype=int)

    with open(dag_path) as f:
        dag_structs = json.load(f)

    model_dir = os.path.join(OUTPUT_BASE, f"scm_{dag_key.lower()}")
    parents_dict = expand_group_dag_to_parents(dag_structs[dag_key], groupings, TARGET_COL)

    print(f"\n=== Evaluating SCM Models for {dag_key} ===")
    _, scaler = evaluate_loaded_scm_models(X, y, parents_dict, train_idx, test_idx, model_dir)

    models, configs, epsilons, scaler = load_scm_components(model_dir)
    top_order = topological_sort({k: v["input_features"] for k, v in configs.items()})

    original = pd.DataFrame([original_sample])
    print("--- Original Unscaled Feature Values ---")
    for var in interventions_raw:
        print(f"{var}: {original[var].values[0]}")

    print("\n--- Raw Intervention Values ---")
    for k, v in interventions_raw.items():
        print(f"{k}: new_value={v:.2f}")

    scaled_input = original[input_features].copy()
    scaled_input[input_features] = scaler.transform(scaled_input[input_features])

    scaled_interventions = {}
    for var, new_val in interventions_raw.items():
        temp = original[input_features].copy()
        temp[var] = new_val
        temp_scaled = scaler.transform([temp.values[0]])
        idx = list(scaler.feature_names_in_).index(var)
        scaled_interventions[var] = temp_scaled[0][idx]

    counterfactual = do_intervention(scaled_input.copy(), scaled_interventions)
    counterfactual = abduction_action_prediction(counterfactual, models, epsilons, configs, top_order, scaled_interventions)

    print("\n--- Scaled Interventions Applied ---")
    for var in scaled_interventions:
        print(f"{var}: original_scaled={scaled_input[var].values[0]:.4f}, new_scaled={scaled_interventions[var]:.4f}")

    original_label = original[TARGET_COL].values[0]
    counterfactual_numeric = counterfactual[TARGET_COL].values[0]
    counterfactual_label = label_encoder.inverse_transform([counterfactual_numeric])[0]

    print(f"\nOriginal Prediction: {original_label}")
    print(f"Counterfactual Prediction: {counterfactual_label}")

    return counterfactual_label, counterfactual

# ------------------ USAGE EXAMPLE ------------------ #
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    groupings = pd.read_csv(GROUPINGS_PATH)
    transition_cols = [c for c in df.columns if c.startswith("transition_")]
    df[transition_cols] = df[transition_cols].div(df["county_area_m2"].replace(0, np.nan), axis=0)
    df = df.fillna(0)

    valid_feats = set(groupings["Feature"])
    input_features = [c for c in df.columns if c in valid_feats and c != TARGET_COL]
    df_model = df[input_features + [TARGET_COL]].copy()

    test_idx = np.loadtxt(os.path.join(OUTPUT_BASE, "final_test_indices.txt"), dtype=int)
    input_test = df_model.iloc[test_idx].copy()

    original = input_test.iloc[[10]].copy()
    sample_dict = original.iloc[0].to_dict()
    interventions = {
        "transition_6_0": 0.0003477482118824513,
        "transition_6_1": 0.053124150408457846
    }

    run_scm_counterfactual_simulation(
        original_sample=sample_dict,
        interventions_raw=interventions,
        # dag_key="DAG_2_Infrastructure_Mediator"
        dag_key="DAG_1_Independent"
    )
