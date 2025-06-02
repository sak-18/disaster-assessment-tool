#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
recourse_minimal.py
-------------------
Finds counter-factuals for one row by changing ONLY the features in
MODIFIABLE.  For every class different from the current prediction it
moves along the straight line toward that class’s mean and binary-searches
the smallest positive step that flips the model.

No Pandas FutureWarnings.  No external plotting.  Pure, readable code.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib

# ────────────────────────── PATHS / CONSTANTS ──────────────────────────
ASSETS_DIR = Path("../assets/full_features")
DATA_PATH  = Path("../data/data_features.csv")

MODIFIABLE = [
    "transition_0_4",
    "Reddit_Buildings",
    "Reddit_Fatalities",
    "News_Power Lines",
    "News_Agriculture",
]
EPS        = 1e-3      # binary-search precision
MAX_EVALS  = 60        # model calls per target class
# ────────────────────────────────────────────────────────────────────────


def to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert every column to numeric; non-convertible entries → NaN."""
    return df.apply(pd.to_numeric, errors="coerce")


def predict(model, scaler, sample: pd.Series, cols) -> int:
    """Predict label for ONE unscaled Series (all NaN already filled)."""
    x_scaled = scaler.transform(sample[cols].to_frame().T)
    return int(model.predict(x_scaled)[0])


def minimal_on_ray(
    factual: pd.Series,
    mean_target: pd.Series,
    model,
    scaler,
    cols,
    target_cls: int,
) -> tuple[float | None, pd.Series | None]:
    """
    Smallest α ≥ 0 so that x_cf = factual + α·gap flips to target_cls.
    All values are clipped at zero (non-negative constraint).
    """
    gap = mean_target - factual[MODIFIABLE]

    # 1) find an alpha that flips (exponential growth)
    alpha_hi = 1.0
    for _ in range(MAX_EVALS):
        cand = factual.copy()
        cand.loc[MODIFIABLE] = (factual[MODIFIABLE] + alpha_hi * gap).clip(lower=0)
        if predict(model, scaler, cand, cols) == target_cls:
            break
        alpha_hi *= 2.0
    else:
        return None, None  # never flipped

    alpha_lo = 0.0
    # 2) binary search
    for _ in range(MAX_EVALS):
        if alpha_hi - alpha_lo < EPS:
            break
        alpha_mid = 0.5 * (alpha_lo + alpha_hi)
        cand = factual.copy()
        cand.loc[MODIFIABLE] = (factual[MODIFIABLE] + alpha_mid * gap).clip(lower=0)
        if predict(model, scaler, cand, cols) == target_cls:
            alpha_hi = alpha_mid
        else:
            alpha_lo = alpha_mid

    counter = factual.copy()
    counter.loc[MODIFIABLE] = (factual[MODIFIABLE] + alpha_hi * gap).clip(lower=0)
    return alpha_hi, counter


def main(row_idx: int):
    # ── load model + scaler ──
    model  = joblib.load(ASSETS_DIR / "model.joblib")
    scaler = joblib.load(ASSETS_DIR / "scaler.joblib")
    with open(ASSETS_DIR / "model_config.json") as f:
        cfg = json.load(f)
    input_cols = cfg["input_features"]
    target_col = cfg["target_column"]

    # ── load data ──
    df = pd.read_csv(DATA_PATH, dtype={"FIPS": str})
    df[input_cols] = to_numeric_df(df[input_cols]).fillna(0)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)

    factual = df.iloc[row_idx][input_cols]
    orig_cls = predict(model, scaler, factual, input_cols)
    print(f"\nRow {row_idx} original prediction: class {orig_cls}")

    # ── precompute simple class means ──
    means = {
        c: df[df[target_col] == c][MODIFIABLE].mean().fillna(0)
        for c in model.classes_
    }

    # ── attempt recourses to every other class ──
    for tgt in model.classes_:
        if tgt == orig_cls:
            continue

        alpha, cf = minimal_on_ray(factual, means[tgt], model, scaler, input_cols, tgt)
        if cf is None:
            print(f"  ✗  no recourse to class {tgt}")
            continue

        print(f"\n  ✓  recourse to class {tgt} with α = {alpha:.4f}")
        for f in MODIFIABLE:
            print(f"    {f:20s}  {factual[f]:.3f} → {cf[f]:.3f}   Δ {cf[f]-factual[f]:+.3f}")


# ────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find minimal non-negative recourses.")
    parser.add_argument("--idx", type=int, required=True, help="Row index in CSV.")
    args = parser.parse_args()
    main(args.idx)
