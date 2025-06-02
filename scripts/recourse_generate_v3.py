#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
recourse_dice_multi.py  (robust, warning-free)
----------------------------------------------
• Works for any multi-class scikit-learn model.
• Uses DiCE to generate counter-factuals, changing ONLY the features
  listed in MODIFIABLE.
• No warnings are suppressed; instead we give the model the
  `feature_names_in_` attribute it expects.

Edit the CONFIG block and run:
    python recourse_dice_multi.py
"""

# ───────────── CONFIG ─────────────
ROW_IDX      = 2
METHOD       = "genetic"          # "genetic", "random", "kdtree"
TOTAL_CFS    = 2

ASSETS_DIR   = "../assets/full_features"
DATA_PATH    = "../data/data_features.csv"

MODIFIABLE = [
    "transition_0_4",
    "Reddit_Buildings",
    "Reddit_Fatalities",
    "News_Power Lines",
    "News_Agriculture",
]
# ──────────────────────────────────

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import dice_ml
from dice_ml import Dice


def ensure_feature_names(model, names):
    """
    Add feature_names_in_ to a scikit-learn estimator if it doesn't have it.
    That removes the 'X has feature names' warning when DataFrames are used.
    """
    if not hasattr(model, "feature_names_in_"):
        model.feature_names_in_ = np.array(names, dtype=object)
    return model


def main():
    # 1 ▸ load artefacts
    assets = Path(ASSETS_DIR)
    model  = joblib.load(assets / "model.joblib")
    scaler = joblib.load(assets / "scaler.joblib")
    with open(assets / "model_config.json") as f:
        cfg = json.load(f)
    input_cols = cfg["input_features"]
    target_col = cfg["target_column"]

    # attach feature names to silence future warnings globally
    model = ensure_feature_names(model, input_cols)

    # 2 ▸ load data → numeric → scale → add target for DiCE meta-data
    df_raw = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    df_raw[input_cols] = (
        df_raw[input_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    )

    X_scaled = pd.DataFrame(
        scaler.transform(df_raw[input_cols]),
        columns=input_cols,
        index=df_raw.index,
    )
    X_scaled[target_col] = df_raw[target_col].values

    # 3 ▸ wrap for DiCE
    data_dice = dice_ml.Data(
        dataframe=X_scaled,
        continuous_features=input_cols,
        outcome_name=target_col,
    )
    model_dice = dice_ml.Model(model=model, backend="sklearn", model_type="classifier")
    dice = Dice(data_dice, model_dice, method=METHOD)

    # 4 ▸ factual row (features only)
    query_feats = X_scaled.loc[[ROW_IDX], input_cols]  # DataFrame
    orig_pred   = int(model.predict(query_feats)[0])   # warning-free now
    print(f"\nRow {ROW_IDX} – original prediction: class {orig_pred}")

    # 5 ▸ loop over alternative classes
    for tgt in model.classes_:
        if tgt == orig_pred:
            continue

        exps = dice.generate_counterfactuals(
            query_instances=query_feats,
            total_CFs=TOTAL_CFS,
            desired_class=int(tgt),
            features_to_vary=MODIFIABLE,
        )
        cf_df = exps.cf_examples_list[0].final_cfs_df
        if cf_df.empty:
            print(f"  ✗  no counter-factuals for class {tgt}")
            continue

        print(f"\n  ✓  {len(cf_df)} counter-factual(s) for class {tgt}")
        for i, cf_row in enumerate(cf_df.itertuples(index=False), 1):
            cf = pd.Series(cf_row, index=input_cols)
            new_pred = int(model.predict(cf.to_frame().T)[0])
            print(f"    CF {i}: new prediction = {new_pred}")
            for f in MODIFIABLE:
                before = query_feats.iloc[0][f]
                after  = cf[f]
                delta  = after - before
                print(f"      {f:20s}  {before:+.3f} → {after:+.3f}   Δ {delta:+.3f}")


if __name__ == "__main__":
    main()
