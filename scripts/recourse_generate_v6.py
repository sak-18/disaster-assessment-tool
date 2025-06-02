#!/usr/bin/env python
"""
recourse_and_lollipop.py (v12)
================================
Generate counter‑factual recourses for a trained classifier and visualise
feature changes with a lollipop chart.

**v12 – exclude irrelevant features**
-------------------------------------
*   Features ending in ``_Fatalities`` or ``_Injuries`` are **never varied**
    when ``features_to_vary="all"`` (default).  They also won’t appear in
    the lollipop plot since they can’t change.
*   Everything else from v11 (visual styling, INSTANCE_IDX, adaptive
    search, auto‑saving) is unchanged.

If you *do* want to vary additional columns, pass an explicit list to
``features_to_vary``; the function will still filter out any names with
those suffixes to keep property‑damage‑irrelevant features untouched.
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
# CONFIG (edit here!)
# ---------------------------------------------------------------
ASSETS_DIR: str = "../assets/full_features"
DATA_PATH: str = "../data/data_features.csv"
N_POOL_DEFAULT: int = 150  # starting pool size
INSTANCE_IDX: int = 396      # <-- choose which row to explain
EXCLUDE_SUFFIXES: tuple[str, ...] = ("_Fatalities", "_Injuries")

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

def _filter_irrelevant(cols: list[str]) -> list[str]:
    return [c for c in cols if not any(c.endswith(suf) for suf in EXCLUDE_SUFFIXES)]


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
    adaptive: bool = True,
    max_attempts: int = 4,
    pool_growth: int = 2,
):
    """Return a counterfactual complying with the requested change budget.

    Any column whose name ends with one of ``EXCLUDE_SUFFIXES`` is ignored
    when varying features to keep the focus on property‑damage drivers.
    """

    if min_changes < 1 or min_changes > max_changes:
        raise ValueError("min_changes must satisfy 1 ≤ min_changes ≤ max_changes")

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

    # ---------- determine feature set to vary
    if features_to_vary == "all":
        vary_cols = _filter_irrelevant(X_cols)
    else:
        vary_cols = _filter_irrelevant(list(features_to_vary))

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        explainer = Dice(dice_data, dice_model, method="random")
        cf_res = explainer.generate_counterfactuals(
            query_instances=original_row.to_frame().T,
            total_CFs=n_pool,
            desired_class=desired_class,
            features_to_vary=vary_cols,
            proximity_weight=proximity_w,
            diversity_weight=diversity_w,
        )

        cf_df = cf_res.cf_examples_list[0].final_cfs_df[X_cols].copy()
        cf_df["n_changes"] = cf_df.apply(lambda r: (r != original_row).sum(), axis=1)
        feasible = cf_df[(cf_df["n_changes"] >= min_changes) & (cf_df["n_changes"] <= max_changes)].copy()

        if not feasible.empty:
            break

        if not adaptive:
            raise ValueError(
                f"No counterfactual with {min_changes}–{max_changes} changed features. "
                "Try increasing `n_pool` or widening the range."
            )
        n_pool *= pool_growth

    if feasible.empty:
        raise ValueError(
            f"No counterfactual found after {max_attempts} attempts (max pool={n_pool})."
        )

    # ---------- pick minimal-distance CF
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
# plot_lollipop (unchanged from v11)
# ---------------------------------------------------------------

def plot_lollipop(
    deltas: pd.DataFrame,
    *,
    title: str = "Counterfactual Recourse",
    figsize: tuple[int, int] = (7, 5),
    savepath: str | None = None,
    show: bool = True,
):
    plt.figure(figsize=figsize)
    y_pos = np.arange(len(deltas))
    for i, (_, row) in enumerate(deltas.iterrows()):
        plt.annotate(
            "",
            xy=(row["cf"], i),
            xytext=(row["original"], i),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )
        plt.scatter(row["original"], i, s=45, color="black", zorder=3)
        plt.scatter(row["cf"], i, s=45, color="red", zorder=3)
    plt.yticks(y_pos, deltas["feature"])
    plt.xlabel("Value")
    plt.title(title)
    plt.grid(axis="x", ls="--", alpha=0.4)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

# ---------------------------------------------------------------
# Self‑test
# ---------------------------------------------------------------
if __name__ == "__main__":
    print(
        f"Self‑test: instance {INSTANCE_IDX} → class 0 (3–10 changes, adaptive search)…"
    )
    try:
        cf, deltas = generate_recourse(
            INSTANCE_IDX,
            desired_class=0,
            min_changes=3,
            max_changes=10,
        )
        print("Changes", len(deltas), "features; distance =", cf["dist"])
        print(deltas.head())
        out_png = f"../lollipop_{INSTANCE_IDX}.png"
        plot_lollipop(
            deltas,
            title=f"Recourse for instance {INSTANCE_IDX}",
            savepath=out_png,
            show=False,
        )
        print(f"✓ Lollipop chart saved to '{out_png}'.")
    except Exception as e:
        print("⚠️", e)
