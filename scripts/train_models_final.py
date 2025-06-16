#!/usr/bin/env python

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

# suppress convergence warnings from logistic regression
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ------------------ CONFIG ------------------ #
DATA_PATH       = "../data/data_features.csv"
GROUPINGS_PATH  = "../assets/groupings/feature_groupings.csv"
DAG_PATH        = "../assets/dags/dag_structures.json"
OUTPUT_BASE     = "../assets/full_features_final"
TARGET_COL      = "Property_Damage_GT"
PRIMARY_METRIC  = "accuracy"  # accuracy, f1_macro, f1_micro, precision_macro, recall_macro

model_configs = {
    "XGBoost_Base": XGBClassifier(eval_metric="mlogloss", n_estimators=50,
                                  max_depth=6, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  reg_lambda=0.0, reg_alpha=0.0,
                                  random_state=42),
    "XGBoost_Regularized": XGBClassifier(eval_metric="mlogloss", n_estimators=50,
                                         max_depth=6, learning_rate=0.1,
                                         subsample=0.8, colsample_bytree=0.8,
                                         reg_lambda=1.0, reg_alpha=0.5,
                                         random_state=42),
    "RandomForest_Base": RandomForestClassifier(n_estimators=100,
                                                max_depth=10,
                                                min_samples_split=2,
                                                random_state=42),
    "RandomForest_Regularized": RandomForestClassifier(n_estimators=100,
                                                       max_depth=10,
                                                       min_samples_split=5,
                                                       random_state=42),
    "MLP_2Layer_Base": MLPClassifier(hidden_layer_sizes=(128, 64),
                                     alpha=0.0001,
                                     max_iter=3000,
                                     random_state=42),
    "MLP_2Layer_Regularized": MLPClassifier(hidden_layer_sizes=(128, 64),
                                            alpha=0.01,
                                            max_iter=3000,
                                            random_state=42),
    "MLP_5Layer_Base": MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32, 16),
                                     alpha=0.0001,
                                     max_iter=3000,
                                     random_state=42),
    "MLP_5Layer_Regularized": MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32, 16),
                                            alpha=0.01,
                                            max_iter=3000,
                                            random_state=42),
    "LogisticRegression_Base": LogisticRegression(penalty=None,
                                                  solver="saga",
                                                  max_iter=3000,
                                                  random_state=42),
    "LogisticRegression_L1": LogisticRegression(penalty="l1",
                                                solver="saga",
                                                C=1.0,
                                                max_iter=3000,
                                                random_state=42)
}

# ------------------ HELPERS ------------------ #

def build_model_config(model,
                       model_name: str,
                       X_train, y_train,
                       X_test,  y_test,
                       input_features: list,
                       target_column: str,
                       preprocessing: dict):
    """Assemble the JSON summary for a trained model."""
    hyperparams = model.get_params()
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    train_f1 = f1_score(y_train, y_pred_train, average="macro")
    test_f1  = f1_score(y_test,  y_pred_test,  average="macro")

    return {
        "model_name":      model_name,
        "type":            f"{model.__module__}.{model.__class__.__name__}",
        "hyperparameters": hyperparams,
        "train_macro_f1":  round(train_f1, 4),
        "test_macro_f1":   round(test_f1,  4),
        "input_features":  input_features,
        "target_column":   target_column,
        "preprocessing":   preprocessing
    }

def expand_group_dag_to_parents(dag_json, groupings, target_col):
    grp2feat = defaultdict(list)
    for _, row in groupings.iterrows():
        grp2feat[row["Group"]].append(row["Feature"])
    parents = defaultdict(list)
    for src, tgts in dag_json.items():
        src_feats = grp2feat.get(src, [])
        for tgt in tgts:
            for f in grp2feat.get(tgt, []):
                parents[f].extend(src_feats)
    # direct to target
    parents.setdefault(target_col, [])
    for src, tgts in dag_json.items():
        if target_col in tgts:
            parents[target_col].extend(grp2feat.get(src, []))
    # dedupe
    for k in parents:
        parents[k] = list(set(parents[k]))
    return parents

def topological_sort(parents_dict):
    nodes = set(parents_dict.keys()) | {p for ps in parents_dict.values() for p in ps}
    indeg = {n:0 for n in nodes}
    for child, ps in parents_dict.items():
        indeg[child] += len(ps)
    queue = [n for n,d in indeg.items() if d==0]
    order = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for ch, ps in parents_dict.items():
            if n in ps:
                indeg[ch] -= 1
                if indeg[ch]==0:
                    queue.append(ch)
    return [n for n in order if n in parents_dict or n==TARGET_COL]

def train_scm(X, y, parents_dict, train_idx, test_idx, outdir):
    os.makedirs(outdir, exist_ok=True)
    X_tr, X_te = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_tr, y_te = y[train_idx], y[test_idx]
    orig_X_te = X_te.copy()
    order = topological_sort(parents_dict)
    metrics_all = {}
    for node in order:
        ps = parents_dict.get(node, [])
        if not ps: continue
        ytr = y_tr if node==TARGET_COL else X_tr[node]
        yte = y_te if node==TARGET_COL else orig_X_te[node]
        mdl = clone(RandomForestClassifier(n_estimators=50, random_state=42))
        mdl.fit(X_tr[ps], ytr)
        p_tr = mdl.predict(X_tr[ps])
        p_te = mdl.predict(X_te[ps])
        np.save(os.path.join(outdir, f"eps_train_{node}.npy"), ytr - p_tr)
        np.save(os.path.join(outdir, f"eps_test_{node}.npy"),  yte - p_te)
        m = {
            "accuracy":        accuracy_score(yte, p_te),
            "f1_macro":        f1_score(yte, p_te, average="macro"),
            "precision_macro": precision_score(yte, p_te, average="macro", zero_division=0),
            "recall_macro":    recall_score(yte, p_te, average="macro", zero_division=0)
        }
        metrics_all[node] = m
        print(f"[SCM {node}] acc={m['accuracy']:.4f}, f1={m['f1_macro']:.4f}")
        joblib.dump(mdl, os.path.join(outdir, f"{node}.joblib"))
        X_te[node] = p_te
    with open(os.path.join(outdir, "scm_metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=4)
    return

# ------------------ PREDICTIVE TRAINING ------------------ #
def train_predictive_models(X, y, feats, outdir):
    os.makedirs(outdir, exist_ok=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_score, best_meta = -1, {}
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    for name, mdl in model_configs.items():
        for tr, te in skf.split(Xs, y):
            mdl.fit(Xs[tr], y[tr])
            pred = mdl.predict(Xs[te])
            m = {
                "accuracy":        accuracy_score(y[te], pred),
                "f1_macro":        f1_score(y[te], pred, average="macro"),
                "precision_macro": precision_score(y[te], pred, average="macro", zero_division=0),
                "recall_macro":    recall_score(y[te], pred, average="macro", zero_division=0)
            }
            if m[PRIMARY_METRIC] > best_score:
                best_score = m[PRIMARY_METRIC]
                best_meta  = {
                    "name":    name,
                    "tr":      tr.tolist(),
                    "te":      te.tolist(),
                    "metrics": m
                }

    # --- RETRAIN THE WINNER ON THE SELECTED TRAIN SPLIT ---
    best_name = best_meta["name"]
    best_tr   = np.array(best_meta["tr"])
    best_te   = np.array(best_meta["te"])
    best_model = clone(model_configs[best_name])
    best_model.fit(Xs[best_tr], y[best_tr])

    # save model & scaler
    joblib.dump(best_model, os.path.join(outdir, "model.pkl"),  compress=3)
    joblib.dump(scaler,    os.path.join(outdir, "scaler.pkl"), compress=3)

    # save the JSON summary (now using a truly fitted best_model!)
    preproc = {
        "scaler":        scaler.__class__.__name__,
        "normalization": "transition_*/county_area_m2",
        "fillna":        0
    }
    cfg = build_model_config(
        model          = best_model,
        model_name     = best_name,
        X_train        = Xs[best_tr],
        y_train        = y[best_tr],
        X_test         = Xs[best_te],
        y_test         = y[best_te],
        input_features = feats,
        target_column  = TARGET_COL,
        preprocessing  = preproc
    )
    with open(os.path.join(outdir, "model_summary.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    # save indices
    np.savetxt(os.path.join(outdir, "train_idx.txt"), best_tr, fmt="%d")
    np.savetxt(os.path.join(outdir, "test_idx.txt"),  best_te, fmt="%d")

    print(f"Best model: {best_name}  {PRIMARY_METRIC}={best_score:.4f}")
    return best_meta, best_model, scaler

# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, dtype={"FIPS":str})
    grp = pd.read_csv(GROUPINGS_PATH)
    # normalize transitions
    trans_cols = [c for c in df if c.startswith("transition_")]
    df[trans_cols] = df[trans_cols].div(df["county_area_m2"].replace(0, np.nan), axis=0)
    df.fillna(0, inplace=True)
    feats = [c for c in df.columns if c in set(grp["Feature"]) and c!=TARGET_COL]
    X = df[feats]
    y = LabelEncoder().fit_transform(df[TARGET_COL])

    meta, mdl, scaler = train_predictive_models(X, y, feats, OUTPUT_BASE)
    tr, te = np.array(meta["tr"]), np.array(meta["te"])

    # SCMs
    with open(DAG_PATH) as f:
        dags = json.load(f)
    for k, v in dags.items():
        pdict = expand_group_dag_to_parents(v, grp, TARGET_COL)
        print(f"[INFO] SCM parents for {k}: {pdict.get(TARGET_COL, [])}")
        train_scm(X, y, pdict, tr, te, os.path.join(OUTPUT_BASE, f"scm_{k.lower()}"))

    # plot comparison
    scores = [(meta["name"], meta["metrics"][PRIMARY_METRIC])]
    for k in dags:
        path = os.path.join(OUTPUT_BASE, f"scm_{k.lower()}/scm_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                sm = json.load(f)
            if PRIMARY_METRIC in sm:
                scores.append((f"SCM_{k}", sm[PRIMARY_METRIC]))

    labs, vals = zip(*scores)
    plt.figure(figsize=(10,5))
    bars = plt.bar(labs, vals, color=["steelblue" if not l.startswith("SCM") else "darkorange" for l in labs])
    for b, v in zip(bars, vals):
        plt.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom")
    plt.ylabel(PRIMARY_METRIC.replace("_"," ").title()); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_BASE, f"comparison_{PRIMARY_METRIC}.png"))
    plt.show()

    print("Done.")
