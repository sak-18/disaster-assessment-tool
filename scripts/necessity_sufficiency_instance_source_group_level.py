import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ------------------ Configuration ------------------
FEATURE_SET    = "full_features"
ASSETS_DIR     = f"../assets/{FEATURE_SET}"
DATA_PATH      = "../data/data_features.csv"
GROUPINGS_PATH = "../assets/groupings/feature_groupings.csv"
DAG_PATH       = "../assets/dags/dag_structures.json"
OUTPUT_BASE    = "../assets/importance_scores"
DAG_KEYS       = ["DAG_1_Independent", "DAG_2_Infrastructure_Mediator", "DAG_3_Flood_Driven"]
N_CF           = 10
TARGET_COL = "Property_Damage_GT"
LOAD_BASE = "../assets/full_features_v6"

def compute_transitions(df):
    # Identify transition columns and normalize them by county area
    transition_cols = [c for c in df.columns if c.startswith("transition_")]
    
    df = df.copy()
    
    # Preserve raw transition values
    for col in transition_cols:
        df[col + "_raw"] = df[col]
    
    # Use max(1, area) to avoid division by zero
    area = np.maximum(df["county_area_m2"].values, 1)
    df[transition_cols] = df[transition_cols].div(area, axis=0)
    
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

    train_idx = np.loadtxt(os.path.join(LOAD_BASE, "final_train_indices.txt"), dtype=int)
    test_idx  = np.loadtxt(os.path.join(LOAD_BASE, "final_test_indices.txt"), dtype=int)

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
            #print(f"SCM {node} -> acc: {accuracy_score(y_te,p):.3f}, f1: {f1_score(y_te,p,average='macro'):.3f}")
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
    model_dir = os.path.join(LOAD_BASE, f"scm_{dag_key.lower()}")

    # Evaluate SCM and get scaler
    _, scaler = evaluate_and_predict_scm(X, y, parents, tr_idx, te_idx, model_dir)

    # Normalize interventions from raw space to normalized
    sample = pd.DataFrame([original_raw])
    # print("Original raw values:")
    # for v in interventions_raw:
    #     print(f" - {v}: {original_raw[v]}")

    norm_int = normalize_interventions(interventions_raw, sample)
    # Scale sample
    available_feats = [f for f in input_feats if f in sample.columns]
    sample_scaled = sample[available_feats].copy()
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
    #print(f"Original Prediction: {orig_label} | Counterfactual Prediction: {cf_label}")
    return cf_label, cf

# ------------------ Source Groups ------------------
def get_source_groups(input_cols):
    return {
        "Transition": [f for f in input_cols if f.startswith("transition_")],
        "Reddit":     [f for f in input_cols if f.startswith("Reddit_")],
        "News":       [f for f in input_cols if f.startswith("News_")]
    }

# ------------------ Group Evaluation Function ------------------
def group_intervention_scores(groups, sample, original_label, df, input_cols, dag_key):
    group_nec = {}
    for group_name, feats in groups.items():
        feats = [f for f in feats if f in input_cols]
        num_flips = 0
        for _ in range(N_CF):
            subset_size = random.randint(1, len(feats))
            subset = random.sample(feats, subset_size)
            interventions = {}
            for f in subset:
                feat_min, feat_max = df[f].min(), df[f].max()
                if not np.isfinite(feat_min) or not np.isfinite(feat_max) or feat_min == feat_max:
                    feat_min, feat_max = 0.0, 1.0
                interventions[f] = np.random.uniform(feat_min, feat_max)

            cf_label, _ = run_scm_counterfactual(sample, interventions, dag_key)
            if cf_label is not None and cf_label != original_label:
                num_flips += 1
        group_nec[f"necessity_{group_name}"] = num_flips / N_CF
    return group_nec


# ------------------ Load Data ------------------
df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
df["county_area_m2"] = df["county_area_m2"].replace(0, np.nan)
df.loc[:, df.columns.str.startswith("transition_")] /= df["county_area_m2"]

# ------------------ Load Groupings ------------------
groupings = pd.read_csv(GROUPINGS_PATH)
feat_groups = groupings.groupby("Group")["Feature"].apply(list).to_dict()

# ------------------ Main Evaluation ------------------
for DAG_KEY in DAG_KEYS:
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, DAG_KEY)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    instance_log_path = os.path.join(OUTPUT_DIR, "instance_necessity_log.csv")
    source_log_path   = os.path.join(OUTPUT_DIR, "source_necessity_log.csv")
    group_log_path    = os.path.join(OUTPUT_DIR, "group_necessity_log.csv")

    instance_log = open(instance_log_path, "w")
    source_log   = open(source_log_path, "w")
    group_log    = open(group_log_path, "w")

    instance_log.write("Instance_Index,Feature,Necessity_Score\n")
    source_log.write("Instance_Index,SourceGroup,Necessity_Score\n")
    group_log.write("Instance_Index,FeatureGroup,Necessity_Score\n")


    model = joblib.load(os.path.join(ASSETS_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(ASSETS_DIR, "scaler.joblib"))
    with open(os.path.join(ASSETS_DIR, "model_config.json")) as f:
        config = json.load(f)

    input_cols = config["input_features"]
    target_col = config.get("target_column", config.get("target_col", ""))

    # Load full feature set from groupings
    groupings = pd.read_csv(GROUPINGS_PATH)
    valid_feats = set(groupings["Feature"])
    input_feats = [c for c in df.columns if c in valid_feats]

    # Use grouped features instead of just model-used ones
    df_model = df[input_feats + [target_col] + ['county_area_m2']].fillna(0)

    # Load test indices
    test_idx = np.loadtxt(os.path.join(ASSETS_DIR, "final_test_indices.txt"), dtype=int)

    # Source groups (Reddit/News/Transitions) based on input_feats now
    source_groups = get_source_groups(input_feats)

    nec_scores_all = []
    source_nec_scores_all = []
    featgroup_nec_scores_all = []


    for idx in tqdm(test_idx, desc=f"Evaluating {DAG_KEY}"):
        sample = df_model.iloc[idx].to_dict()
        original_label = sample[target_col]

        nec_scores = {}
        suff_scores = {}

        for feat in input_cols:
            num_flips = 0
            for _ in range(N_CF):
                feat_min = df[feat].min()
                feat_max = df[feat].max()

                # Fallback to standard range if min/max are invalid
                if not np.isfinite(feat_min) or not np.isfinite(feat_max) or feat_min == feat_max:
                    feat_min, feat_max = 0.0, 1.0  # or other sensible defaults
                new_val = np.random.uniform(feat_min, feat_max)
                cf_label, _ = run_scm_counterfactual(sample, {feat: new_val}, DAG_KEY)
                if cf_label is not None and cf_label != original_label:
                    num_flips += 1
            print(f"[Necessity] Inst: {idx} | Feat: {feat} | Δ: {new_val:.3f} | Orig: {original_label} → CF: {cf_label}")
            nec_scores[f"necessity_{feat}"] = num_flips / N_CF

            # num_preserves = 0
            # for _ in range(N_CF):
            #     interventions = {}
            #     for f2 in input_cols:
            #         if f2 == feat:
            #             continue
            #         feat_min, feat_max = df[f2].min(), df[f2].max()
            #         if not np.isfinite(feat_min) or not np.isfinite(feat_max) or feat_min == feat_max:
            #             feat_min, feat_max = 0.0, 1.0  # safe fallback
            #         interventions[f2] = np.random.uniform(feat_min, feat_max)

            #     cf_label, _ = run_scm_counterfactual(sample, interventions, DAG_KEY)
            #     if cf_label is not None and cf_label == original_label:
            #         num_preserves += 1
            # print(f"[Sufficiency] Inst: {idx} | Held: {feat} | Δ others | Orig: {original_label} → CF: {cf_label}")
            # suff_scores[f"sufficiency_{feat}"] = num_preserves / N_CF

        src_nec = group_intervention_scores(source_groups, sample, original_label, df, input_cols, DAG_KEY)
        fg_nec = group_intervention_scores(feat_groups, sample, original_label, df, input_cols, DAG_KEY)

        nec_scores_all.append(nec_scores)
        #suff_scores_all.append(suff_scores)
        source_nec_scores_all.append(src_nec)
        #source_suff_scores_all.append(src_suff)
        featgroup_nec_scores_all.append(fg_nec)
        #featgroup_suff_scores_all.append(fg_suff)

        # Instance-level logs
        for feat, score in nec_scores.items():
            instance_log.write(f"{idx},{feat},{score:.4f}\n")

        # Source-level logs
        for group, score in src_nec.items():
            source_log.write(f"{idx},{group},{score:.4f}\n")

        # Feature-group level logs
        for group, score in fg_nec.items():
            group_log.write(f"{idx},{group},{score:.4f}\n")

    outputs = [
        ("instance_necessity_scores.csv", nec_scores_all),
        ("source_necessity_scores.csv", source_nec_scores_all),
        ("group_necessity_scores.csv", featgroup_nec_scores_all)
    ]


    for filename, data in outputs:
        df_out = pd.concat([meta_df.reset_index(drop=True), pd.DataFrame(data)], axis=1)
        df_out.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

    print(f"\n✓ Scores saved for {DAG_KEY} under:")
    print(f"  {OUTPUT_DIR}")
