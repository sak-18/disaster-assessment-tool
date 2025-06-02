#!/usr/bin/env python
"""
recourse_and_lollipop.py (v6)
=============================
Generate counter‑factual recourses for a trained classifier and visualise
feature changes with a lollipop chart.

Changelog
---------
* **v6** – Silence the common scikit‑learn warning:
  "X has feature names, but <Estimator> was fitted without feature names".
  The model itself is fine; this simply hides the message to keep logs clean.
* Retains compatibility with pandas ≥ 2.0 and dice‑ml ≥ 0.10.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# ------------------------------------------------------------------
# GLOBAL WARNING FILTERS
# ------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r"X has feature names, but .* was fitted without feature names",
)

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
ASSETS_DIR: str = "../assets/full_features"   # path to model, scaler, splits
DATA_PATH: str = "../data/data_features.csv"  # full feature CSV
N_POOL: int = 50                              # candidate CFs DiCE should sample

# ------------------------------------------------------------------
# INTERNAL HELPERS
# ------------------------------------------------------------------

def _load_assets():
    model = joblib.load(os.path.join(ASSETS_DIR, "model.joblib"))
    scaler = joblib.load(os.path.join(ASSETS_DIR, "scaler.joblib"))
    with open(os.path.join(ASSETS_DIR, "model_config.json")) as f:
        cfg = json.load(f)
    return model, scaler, cfg


def _prepare_dice_objects(df_train: pd.DataFrame, cfg: dict, model):
    import dice_ml

    target = cfg["target_column"]
    input_cols = cfg["input_features"]

    cat_feats = [c for c in input_cols if str(df_train[c].dtype) == "object"]
    cont_feats = [c for c in input_cols if c not in cat_feats]

    dice_data = dice_ml.Data(
        dataframe=df_train[input_cols + [target]],
        continuous_features=cont_feats,
        categorical_features=cat_feats,
        outcome_name=target,
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    return dice_data, dice_model


def _load_indices(path: str):
    return pd.read_csv(path, header=None).iloc[:, 0].tolist()

# ------------------------------------------------------------------
# PUBLIC: generate_recourse
# ------------------------------------------------------------------

def generate_recourse(
    instance_idx: int,
    desired_class: int,
    *,
    instance_is_test: bool = True,
    features_to_vary: str | list[str] = "all",
    max_changes: int = 10,
    n_pool: int = N_POOL,
    proximity_w: float = 0.5,
    diversity_w: float = 1.0,
):
    """Return a feasible counterfactual and a tidy deltas frame."""
    import dice_ml
    from dice_ml import Dice

    model, scaler, cfg = _load_assets()
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    X_cols = cfg["input_features"]

    idx_file = os.path.join(
        ASSETS_DIR,
        "final_test_indices.txt" if instance_is_test else "final_train_indices.txt",
    )
    test_idx = _load_indices(idx_file)
    df_train = df.drop(index=test_idx)

    dice_data, dice_model = _prepare_dice_objects(df_train, cfg, model)
    original_row = df.loc[instance_idx, X_cols]

    explainer = Dice(dice_data, dice_model, method="random")
    cf_res = explainer.generate_counterfactuals(
        query_instances=original_row.to_frame().T,
        total_CFs=n_pool,
        desired_class=desired_class,
        features_to_vary=features_to_vary,
        proximity_weight=proximity_w,
        diversity_weight=diversity_w,
    )

    cf_df = cf_res.cf_examples_list[0].final_cfs_df[X_cols].copy()
    cf_df["n_changes"] = cf_df.apply(lambda r: (r != original_row).sum(), axis=1)
    feasible = cf_df[cf_df["n_changes"] <= max_changes].copy()

    if feasible.empty:
        raise ValueError(
            f"No counterfactual within ≤{max_changes} changed features. "
            "Increase `n_pool` or relax the cap."
        )

    denom = np.abs(original_row) + 1e-8
    feasible["avg_percent_change"] = ((feasible[X_cols] - original_row).abs() / denom).mean(axis=1)
    best_cf_row = feasible.sort_values("avg_percent_change").iloc[0]

    changed = [c for c in X_cols if original_row[c] != best_cf_row[c]]
    deltas = (
        pd.DataFrame({
            "feature": changed,
            "original": original_row[changed].values,
            "cf": best_cf_row[changed].values,
        })
        .sort_values("feature")
        .reset_index(drop=True)
    )

    return best_cf_row, deltas

# ------------------------------------------------------------------
# PUBLIC: plot_lollipop
# ------------------------------------------------------------------

def plot_lollipop(
    deltas: pd.DataFrame,
    *,
    title: str = "Counterfactual Recourse",
    figsize: tuple[int, int] = (7, 5),
    savepath: str | None = None,
):
    plt.figure(figsize=figsize)
    y_pos = np.arange(len(deltas))

    for i, (_, row) in enumerate(deltas.iterrows()):
        plt.plot([row["original"], row["cf"]], [i, i], lw=1.5)
        plt.scatter(row["original"], i, s=45, zorder=3)
        plt.scatter(row["cf"], i, s=45, zorder=3)

    plt.yticks(y_pos, deltas["feature"])
    plt.xlabel("Value")
    plt.title(title)
    plt.grid(axis="x", ls="--", alpha=0.4)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()

# ------------------------------------------------------------------
# Self‑test
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Running self‑test on instance 0 …")
    try:
        cf_row, delta_df = generate_recourse(0, desired_class=2, max_changes=10)
        print("Found CF with", len(delta_df), "changes:\n", delta_df.head())
        plot_lollipop(delta_df, title="Recourse example for row 0 → class 2")
    except Exception as e:
        print("⚠️", e)
