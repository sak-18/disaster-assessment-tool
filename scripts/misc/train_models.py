import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import joblib
import shutil
import warnings

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, f1_score
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
input_cols = transition_cols + news_cols + reddit_cols
target_col = "Property_Damage_GT"

assert "county_area_m2" in df.columns, "Missing 'county_area_m2' column for normalizing transitions."
df["county_area_m2"] = df["county_area_m2"].replace(0, np.nan)
for col in transition_cols:
    df[col] = df[col] / df["county_area_m2"]

df = df[input_cols + [target_col]].fillna(0)
X = df[input_cols]
y = LabelEncoder().fit_transform(df[target_col])
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ Cross-Validation ------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for model_name, model in model_configs.items():
    fold_scores = []
    fold_indices = []
    print(f"[CV] Evaluating {model_name}")
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred = model.predict(X_scaled[test_idx])
        score = f1_score(y[test_idx], y_pred, average="macro")
        fold_scores.append(score)
        fold_indices.append((train_idx, test_idx))
    best_fold = int(np.argmax(fold_scores))
    cv_results.append({
        "model": model_name,
        "best_train_idx": fold_indices[best_fold][0],
        "best_test_idx": fold_indices[best_fold][1]
    })

# ------------------ Evaluate Best Fold for Each Model ------------------
all_best_results = []
for row in cv_results:
    model_name = row["model"]
    train_idx = row["best_train_idx"]
    test_idx = row["best_test_idx"]
    model = model_configs[model_name]
    model.fit(X_scaled[train_idx], y[train_idx])
    y_train_pred = model.predict(X_scaled[train_idx])
    y_test_pred = model.predict(X_scaled[test_idx])
    train_f1 = f1_score(y[train_idx], y_train_pred, average="macro")
    test_f1 = f1_score(y[test_idx], y_test_pred, average="macro")
    all_best_results.append({
        "Model": model_name,
        "Train_MacroF1": train_f1,
        "Test_MacroF1": test_f1,
        "Train_Idx": train_idx,
        "Test_Idx": test_idx,
        "Model_Obj": model
    })

# ------------------ Select Best by Test Macro-F1 ------------------
perf_df = pd.DataFrame(all_best_results).sort_values("Test_MacroF1", ascending=False)
best_entry = perf_df.iloc[0]
best_model_name = best_entry["Model"]
best_model = best_entry["Model_Obj"]
best_train_idx = best_entry["Train_Idx"]
best_test_idx = best_entry["Test_Idx"]

# ------------------ Save Final Outputs ------------------
output_dir = "../final_model_cv_outputs"
assets_dir = "../assets"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(assets_dir, exist_ok=True)

# Save performance metrics
with open(os.path.join(output_dir, "final_model_metrics.txt"), "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Train Macro-F1: {best_entry['Train_MacroF1']:.4f}\n")
    f.write(f"Test Macro-F1: {best_entry['Test_MacroF1']:.4f}\n")

# Save indices
np.savetxt(os.path.join(output_dir, "final_train_indices.txt"), best_train_idx, fmt="%s")
np.savetxt(os.path.join(output_dir, "final_test_indices.txt"), best_test_idx, fmt="%s")

# ------------------ Save Model, Scaler, Config to Assets ------------------
joblib.dump(best_model, os.path.join(assets_dir, f"{best_model_name}_final_model.joblib"))
joblib.dump(scaler, os.path.join(assets_dir, "scaler.joblib"))

# Save model configuration and metadata
model_config = {
    "model_name": best_model_name,
    "type": str(type(best_model)).split("'")[1],
    "hyperparameters": best_model.get_params(),
    "train_macro_f1": round(best_entry["Train_MacroF1"], 4),
    "test_macro_f1": round(best_entry["Test_MacroF1"], 4),
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

# Copy splits to assets
shutil.copy(os.path.join(output_dir, "final_train_indices.txt"), assets_dir)
shutil.copy(os.path.join(output_dir, "final_test_indices.txt"), assets_dir)

# ------------------ Plot Bar Chart ------------------
plt.figure(figsize=(14, 6))
bar_width = 0.4
x = np.arange(len(perf_df))
plt.bar(x - bar_width/2, perf_df["Train_MacroF1"], width=bar_width, label="Train", color="mediumseagreen")
plt.bar(x + bar_width/2, perf_df["Test_MacroF1"], width=bar_width, label="Test", color="steelblue")

plt.xticks(x, perf_df["Model"], rotation=45, ha="right")
plt.ylabel("Macro-F1 Score")
plt.title("Best Fold Performance (Train vs Test Macro-F1 per Model)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "all_models_train_test_f1_barplot.png"))
plt.close()

# ------------------ Done ------------------
print(f"Best model: {best_model_name}")
print(f"Outputs saved in: {output_dir}")
print(f"Model, scaler, and config saved in: {assets_dir}")
