#!/usr/bin/env python
"""
recourse_and_lollipop.py (v13)
================================
Generate counterfactual recourses for a trained classifier and visualise
feature changes with a lollipop chart. Now attempts recourses to both
alternative classes and checks whether the desired class was reached.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from typing import Literal

warnings.filterwarnings("ignore", message=r"X has feature names, but .* was fitted without feature names")

# ------------------ CONFIG ------------------
ASSETS_DIR: str = "../assets/full_features"
DATA_PATH: str = "../data/data_features.csv"
N_POOL_DEFAULT: int = 150
EXCLUDE_SUFFIXES: tuple[str, ...] = ("_Fatalities", "_Injuries")

# ------------------ HELPERS ------------------

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

def _filter_irrelevant(cols: list[str]) -> list[str]:
    return [c for c in cols if not any(c.endswith(suf) for suf in EXCLUDE_SUFFIXES)]

# ------------------ RECOURSE FUNCTION ------------------

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
    import dice_ml
    from dice_ml import Dice

    if min_changes < 1 or min_changes > max_changes:
        raise ValueError("min_changes must satisfy 1 ≤ min_changes ≤ max_changes")

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

    vary_cols = _filter_irrelevant(X_cols) if features_to_vary == "all" else _filter_irrelevant(list(features_to_vary))

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
            raise ValueError("No feasible CFs. Try increasing pool size or range.")
        n_pool *= pool_growth

    if feasible.empty:
        raise ValueError("No counterfactual found after max attempts.")

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

# ------------------ VISUALIZATION ------------------

def plot_lollipop(deltas: pd.DataFrame, *, title: str = "Counterfactual Recourse", figsize: tuple[int, int] = (7, 5), savepath: str | None = None, show: bool = True):
    plt.figure(figsize=figsize)
    y_pos = np.arange(len(deltas))
    for i, (_, row) in enumerate(deltas.iterrows()):
        plt.annotate("", xy=(row["cf"], i), xytext=(row["original"], i),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
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

# ------------------ MAIN ------------------

if __name__ == "__main__":
    print("Running recourse generation...")
    os.makedirs("../lollipop_charts", exist_ok=True)

    model, scaler, cfg = _load_assets()
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    X_cols = cfg["input_features"]
    idx_file = os.path.join(ASSETS_DIR, "final_test_indices.txt")
    test_indices = _load_indices(idx_file)

    output_data = []

    for idx in test_indices:
        try:
            original_row = df.loc[idx, X_cols]
            original_scaled = scaler.transform([original_row])[0]
            original_pred = model.predict([original_scaled])[0]

            for desired_class in {0, 1, 2} - {original_pred}:
                try:
                    cf_row, deltas = generate_recourse(
                        instance_idx=idx,
                        desired_class=desired_class,
                        min_changes=3,
                        max_changes=10,
                        instance_is_test=True
                    )
                    cf_scaled = scaler.transform([cf_row[X_cols].values])[0]
                    cf_pred = model.predict([cf_scaled])[0]

                    if cf_pred != desired_class:
                        print(f"⚠️ CF for {idx} failed to reach class {desired_class} (got {cf_pred})")
                        continue

                    output_data.append({
                        "instance_idx": int(idx),
                        "original_prediction": int(original_pred),
                        "desired_class": int(desired_class),
                        "counterfactual_prediction": int(cf_pred),
                        "changed_features": deltas.to_dict(orient="records")
                    })

                    plot_lollipop(
                        deltas,
                        title=f"Recourse for instance {idx} to class {desired_class}",
                        savepath=f"../lollipop_charts/lollipop_{idx}_to_{desired_class}.png",
                        show=False
                    )
                except Exception as e_inner:
                    print(f"⚠️ Failed CF for instance {idx} to class {desired_class}: {e_inner}")

        except Exception as e:
            print(f"⚠️ Failed on instance {idx}: {e}")

    with open("../recourse_results.json", "w") as f:
        json.dump(output_data, f, indent=2)

    print("✓ Recourse generation complete.")
