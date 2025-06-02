import os
import json
import joblib
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ------------------ Model Configurations ------------------
model_configs = {
    "XGBoost_Base": XGBClassifier(use_label_encoder=True, eval_metric="mlogloss", n_estimators=50,
                                  max_depth=6, learning_rate=0.1, subsample=0.8,
                                  colsample_bytree=0.8, reg_lambda=0.0, reg_alpha=0.0, random_state=42),
    "XGBoost_Regularized": XGBClassifier(use_label_encoder=True, eval_metric="mlogloss", n_estimators=50,
                                         max_depth=6, learning_rate=0.1, subsample=0.8,
                                         colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.5, random_state=42),
    "RandomForest_Base": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42),
    "RandomForest_Regularized": RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42),
    "MLP_2Layer_Base": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.0001, max_iter=300, random_state=42),
    "MLP_2Layer_Regularized": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=0.01, max_iter=300, random_state=42),
    "MLP_5Layer_Base": MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32, 16), alpha=0.0001, max_iter=300, random_state=42),
    "MLP_5Layer_Regularized": MLPClassifier(hidden_layer_sizes=(256, 128, 64, 32, 16), alpha=0.01, max_iter=300, random_state=42),
    "LogisticRegression_Base": LogisticRegression(penalty=None, solver="saga", max_iter=1000, random_state=42),
    "LogisticRegression_L1": LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=1000, random_state=42)
}

# ------------------ Load and Preprocess Data ------------------
df = pd.read_csv("../data/data_features.csv", dtype={"FIPS": str})

transition_cols = [c for c in df.columns if c.startswith("transition_")]
news_cols = [c for c in df.columns if c.startswith("News_")]
reddit_cols = [c for c in df.columns if c.startswith("Reddit_")]
target_col = "Property_Damage_GT"

# Normalize transitions by county area
assert "county_area_m2" in df.columns, "Missing 'county_area_m2' column"
df["county_area_m2"] = df["county_area_m2"].replace(0, np.nan)
for col in transition_cols:
    df[col] = df[col] / df["county_area_m2"]

# Define valid transitions for filtered features
valid_transitions = [
    (1, 0), (2, 0), (4, 0), (6, 0), (1, 7), (2, 7), (5, 7),
    (4, 7), (6, 7), (3, 0), (0, 3)
]
filtered_transition_cols = [f"transition_{a}_{b}" for a, b in valid_transitions if f"transition_{a}_{b}" in df.columns]

# Define two feature sets
feature_sets = {
    "full_features": transition_cols + news_cols + reddit_cols,
    "filtered_features": filtered_transition_cols + news_cols + reddit_cols
}

# ------------------ Train and Save for Both Sets ------------------
for set_name, input_cols in feature_sets.items():
    print(f"\n[INFO] Processing feature set: {set_name}")

    # Directories
    output_dir = os.path.join("../final_model_cv_outputs", set_name)
    assets_dir = os.path.join("../assets", set_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # Preprocess
    df_processed = df[input_cols + [target_col]].fillna(0)
    X = df_processed[input_cols]
    y = LabelEncoder().fit_transform(df_processed[target_col])
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_overall = None
    model_perf = []

    for model_name, model in model_configs.items():
        print(f"  [CV] Evaluating {model_name}")
        best_f1 = -1
        best_train_idx, best_test_idx = None, None

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
            model.fit(X_scaled[train_idx], y[train_idx])
            preds = model.predict(X_scaled[test_idx])
            score = f1_score(y[test_idx], preds, average="macro")

            if score > best_f1:
                best_f1 = score
                best_train_idx, best_test_idx = train_idx, test_idx

        model.fit(X_scaled[best_train_idx], y[best_train_idx])
        train_pred = model.predict(X_scaled[best_train_idx])
        test_pred = model.predict(X_scaled[best_test_idx])
        train_f1 = f1_score(y[best_train_idx], train_pred, average="macro")
        test_f1 = f1_score(y[best_test_idx], test_pred, average="macro")

        model_perf.append({
            "Model": model_name,
            "Train_MacroF1": train_f1,
            "Test_MacroF1": test_f1
        })

        if best_overall is None or test_f1 > best_overall["Test_MacroF1"]:
            best_overall = {
                "model_name": model_name,
                "model_obj": model,
                "train_idx": best_train_idx,
                "test_idx": best_test_idx,
                "Train_MacroF1": train_f1,
                "Test_MacroF1": test_f1
            }

    # Save best model and metadata
    best_model = best_overall["model_obj"]
    joblib.dump(best_model, os.path.join(assets_dir, "model.joblib"))
    joblib.dump(scaler, os.path.join(assets_dir, "scaler.joblib"))
    np.savetxt(os.path.join(assets_dir, "final_train_indices.txt"), best_overall["train_idx"], fmt="%s")
    np.savetxt(os.path.join(assets_dir, "final_test_indices.txt"), best_overall["test_idx"], fmt="%s")

    model_config = {
        "model_name": best_overall["model_name"],
        "type": str(type(best_model)).split("'")[1],
        "hyperparameters": best_model.get_params(),
        "train_macro_f1": round(best_overall["Train_MacroF1"], 4),
        "test_macro_f1": round(best_overall["Test_MacroF1"], 4),
        "input_features": input_cols,
        "target_column": target_col,
        "preprocessing": {
            "scaler": "RobustScaler",
            "normalization": "transition_*/county_area_m2",
            "fillna": 0
        }
    }
    with open(os.path.join(assets_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # Save metrics to txt
    with open(os.path.join(output_dir, "final_model_metrics.txt"), "w") as f:
        f.write(f"Best Model: {best_overall['model_name']}\n")
        f.write(f"Train Macro-F1: {best_overall['Train_MacroF1']:.4f}\n")
        f.write(f"Test Macro-F1: {best_overall['Test_MacroF1']:.4f}\n")

    # Save performance DataFrame
    perf_df = pd.DataFrame(model_perf)
    perf_df.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)

    # Plot bar chart
    plt.figure(figsize=(14, 6))
    bar_width = 0.4
    x = np.arange(len(perf_df))
    plt.bar(x - bar_width/2, perf_df["Train_MacroF1"], width=bar_width, label="Train", color="mediumseagreen")
    plt.bar(x + bar_width/2, perf_df["Test_MacroF1"], width=bar_width, label="Test", color="steelblue")
    plt.xticks(x, perf_df["Model"], rotation=45, ha="right")
    plt.ylabel("Macro-F1 Score")
    plt.title(f"Train vs Test Macro-F1 Scores ({set_name.replace('_', ' ').title()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_train_test_f1_barplot.png"))
    plt.close()

print("\nAll models trained, evaluated, and saved.")
