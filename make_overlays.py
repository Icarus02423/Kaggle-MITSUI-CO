#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot overlays: training TRUE labels vs TEST predictions on the same timeline (split by a vertical line).
Required files (same folder as this script by default, or change paths below):
  - train_labels.csv             # TRUE labels for training period: ['date_id','target_0',...,'target_423']
  - submission_price_only.csv    # Your predictions on test (columns: 'date_id', target_*)
  - test.csv                     # Must contain ['date_id','is_scored']
  - target_pairs.csv             # ['target','lag','pair']

Outputs:
  - overlays/vis_overlay_<target>.png    # line chart with left=true (train), right=pred (test)
  - overlays/index_overlays.csv          # list of generated figures with lag/pair info

Usage:
  python make_overlays.py
"""

import os
import re
import sys
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------- Config ----------------------
TRAIN_LABELS = "train_labels.csv"
SUBMISSION   = "submission_price_only.csv"
TEST_FILE    = "test.csv"
TARGET_PAIRS = "target_pairs.csv"

OUT_DIR      = "overlays"
MAX_PER_LAG  = 5     # how many targets per lag to plot (chosen by prediction variance over time)
CLASSIC_KEYS = [
    "US_Stock_GLD_adj_close",
    "US_Stock_GDX_adj_close",
    "JPX_Gold_Standard_Futures_Close",
    "LME_CA_Close - LME_ZS_Close",
    "LME_AH_Close - LME_ZS_Close",
    "US_Stock_XLE_adj_close",
    "US_Stock_XOM_adj_close",
    "US_Stock_IEF_adj_close",
    "US_Stock_AGG_adj_close",
    "FX_EURUSD",
    "FX_USDJPY",
    "US_Stock_VT_adj_close",
]

# ---------------------- Helpers ----------------------
def load_required():
    missing = [p for p in [TRAIN_LABELS, SUBMISSION, TEST_FILE, TARGET_PAIRS] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}. Place them in the same folder or update paths in the script.")

    labels = pd.read_csv(TRAIN_LABELS)
    sub    = pd.read_csv(SUBMISSION)
    test   = pd.read_csv(TEST_FILE)[["date_id","is_scored"]]
    tp     = pd.read_csv(TARGET_PAIRS)
    return labels, sub, test, tp

def pick_targets(tp: pd.DataFrame, sub_cols, max_per_lag=5):
    # 1) try to pick classic ones if present
    picked = []
    used = set()
    for key in CLASSIC_KEYS:
        m = tp[tp["pair"].str.contains(key, na=False)].sort_values(["lag"])
        for _, r in m.iterrows():
            if r["target"] in sub_cols and r["target"] not in used:
                picked.append(r)
                used.add(r["target"])
                break

    # 2) fill the rest by top-variance per lag based on predictions
    return picked

def build_overlay(labels: pd.DataFrame, sub: pd.DataFrame, test: pd.DataFrame, tp: pd.DataFrame,
                  max_per_lag=5, out_dir="overlays"):
    os.makedirs(out_dir, exist_ok=True)

    # Keep only scored test dates
    scored_dates = test.loc[test["is_scored"].astype(bool), "date_id"].values
    sub_scored = sub[sub["date_id"].isin(scored_dates)].copy()

    # Create long-form predictions (to measure per-lag variance)
    pred_long = sub_scored.melt(id_vars=["date_id"], var_name="target", value_name="pred")
    pred_long = pred_long.merge(tp, on="target", how="left")

    # Per-lag variance leaderboard
    picks = []
    for lag_val in sorted(pred_long["lag"].dropna().unique().astype(int)):
        lag_df = pred_long[pred_long["lag"] == lag_val]
        wide = lag_df.pivot(index="date_id", columns="target", values="pred")
        var_rank = wide.var(skipna=True).sort_values(ascending=False)
        for tgt in var_rank.index[:max_per_lag]:
            row = tp.loc[tp["target"] == tgt].iloc[0]
            picks.append({"target": tgt, "lag": int(row["lag"]), "pair": row["pair"]})

    # Also try to ensure classics are included
    classics = []
    for key in CLASSIC_KEYS:
        m = tp[tp["pair"].str.contains(key, na=False)].sort_values(["lag"])
        for _, r in m.iterrows():
            if r["target"] not in [p["target"] for p in picks] and r["target"] in sub.columns:
                classics.append({"target": r["target"], "lag": int(r["lag"]), "pair": r["pair"]})
                break
    picks = classics + picks  # classics first
    # drop duplicates while preserving order
    seen = set(); unique_picks = []
    for p in picks:
        if p["target"] not in seen:
            unique_picks.append(p); seen.add(p["target"])
    picks = unique_picks

    # Plot overlays
    max_train_date = labels["date_id"].max()
    out_rows = []
    for p in picks:
        tgt, lag, pair = p["target"], p["lag"], p["pair"]
        if tgt not in labels.columns or tgt not in sub.columns:
            continue

        y_train = labels[["date_id", tgt]].dropna().sort_values("date_id")
        y_testp = sub_scored[["date_id", tgt]].dropna().sort_values("date_id")

        if y_train.empty or y_testp.empty:
            continue

        # Overlay line chart
        plt.figure(figsize=(12, 3.8))
        plt.plot(y_train["date_id"].values, y_train[tgt].values, label="TRUE (train)", linewidth=1)
        plt.plot(y_testp["date_id"].values, y_testp[tgt].values, label="PRED (test)", linewidth=1)
        plt.axvline(max_train_date, linestyle="--", color="gray", linewidth=1)
        plt.title(f"{tgt} | lag={lag} | {pair}\nLeft: TRUE (train) | Right: PRED (test)")
        plt.xlabel("date_id"); plt.legend(); plt.tight_layout()
        out_png = Path(out_dir) / f"vis_overlay_{tgt}.png"
        plt.savefig(out_png, dpi=150); plt.close()
        out_rows.append({"target": tgt, "lag": lag, "pair": pair, "figure": str(out_png)})

    # Index
    idx = pd.DataFrame(out_rows)
    idx.to_csv(Path(out_dir) / "index_overlays.csv", index=False)
    print(f"Done. Figures in: {out_dir}/  | Index: {out_dir}/index_overlays.csv  | Total: {len(idx)}")

def main():
    labels, sub, test, tp = load_required()
    build_overlay(labels, sub, test, tp, max_per_lag=MAX_PER_LAG, out_dir=OUT_DIR)

if __name__ == "__main__":
    main()
