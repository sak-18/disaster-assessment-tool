#!/usr/bin/env python
"""
recourse_and_lollipop.py (v9)
============================
Generate counter‑factual recourses for a trained classifier and visualise
feature changes with a lollipop chart.

**v9 – adaptive search when no CF is found**
-------------------------------------------
Sometimes, for a specific instance and tight feature‑count bounds, DiCE’s
random sampler just doesn’t stumble upon a valid counterfactual in the
first batch. To spare you from manually fiddling with `n_pool`, the
function now supports an **adaptive retry strategy**:

* New kwargs: `adaptive=True`, `max_attempts=4`, `pool_growth=2`.
* If no CF is found on the first try, it doubles `total_CFs` (`n_pool`)
  and tries again – up to `max_attempts` times.
* If still empty, it raises the same ValueError so your code flow is
  unchanged – you just have a much higher chance of success.

Everything else (min/max feature flips, minimal‑distance selection, etc.)
remains exactly the same.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Literal

# ---------------------------------------------------------------
# GLOBAL WARNING FILTERS
# ---------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message=r"X has feature names, but .* was fitted without feature names",
)

# ---------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------
ASSETS_DIR: str = "../assets/full_features"
DATA_PATH: str = "../data/data_features.csv"
N_POOL_DEFAULT: int = 150  # starting pool size

# ---------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------

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

# ---------------------------------------------------------------
# PUBLIC: generate_recourse
# ---------------------------------------------------------------

def generate_recourse(
    instance_idx: int,
    desired_class: int,
    *,
    instance_is_test: bool = True,
    features_to_vary: str | list[str] = "all",
    min_changes: int = 1,
    max_changes: int = 10,
    n_pool: int = N_POOL_DEFAULT,
    proximity_w: float = 0.5,
    diversity_w: float = 1.0,
    distance_metric: Literal["scaled_l1", "avg_percent_change"] = "scaled_l1",
    # ↓ NEW adaptive‑search knobs
    adaptive: bool = True,
    max_attempts: int = 4,
    pool_growth: int = 2,
):
    """Return a counterfactual that changes *between* `min_changes` and `max_changes` features.

    If no CF is found, the function optionally retries with an enlarged
    `n_pool` (`total_CFs` in DiCE) up to `max_attempts` times.
    """
    if min_changes < 1 or min_changes > max_changes:
        raise ValueError("min_changes must satisfy 1 ≤ min_changes ≤ max_changes")

    import dice_ml
    from dice_ml import Dice

    # ---------- load data & objects
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

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
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
        feasible = cf_df[(cf_df["n_changes"] >= min_changes) & (cf_df["n_changes"] <= max_changes)].copy()

        if not feasible.empty:
            break  # success!

        if not adaptive:
            raise ValueError(
                f"No counterfactual with {min_changes}–{max_changes} changed features. "
                "Try increasing `n_pool` or widening the range."
            )
        # adaptive: grow pool and retry
        n_pool *= pool_growth

    if feasible.empty:
        raise ValueError(
            f"No counterfactual found after {max_attempts} attempts (max pool={n_pool})."
        )

    # ---------- minimal‑distance CF among feasible ones
    if distance_metric == "scaled_l1":
        scale = getattr(scaler, "scale_", np.ones(len(X_cols)))
        feasible["dist"] = feasible[X_cols].sub(original_row).abs().div(scale).sum(axis=1)
    elif distance_metric == "avg_percent_change":
        denom = np.abs(original_row) + 1e-8
        feasible["dist"] = (feasible[X_cols] - original_row).abs().div(denom).mean(axis=1)
    else:
        raise ValueError("Unsupported distance_metric: " + distance_metric)

    best_cf_row = feasible.sort_values("dist").iloc[0]

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

# ---------------------------------------------------------------
# plot_lollipop (unchanged)
# ---------------------------------------------------------------

def plot_lollipop(deltas: pd.DataFrame, *, title: str = "Counterfactual Recourse", figsize=(7, 5), savepath=None):
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

# ---------------------------------------------------------------
# Self‑test
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Self‑test: inst 0 → class 2 (3–10 changes, adaptive search)…")
    try:
        cf, deltas = generate_recourse(0, desired_class=2, min_changes=3, max_changes=10)
        print("Changes", len(deltas), "features; distance =", cf["dist"])
        print(deltas.head())
        plot_lollipop(deltas, title="Self‑test recourse (adaptive)")
    except Exception as e:
        print("⚠️", e)
