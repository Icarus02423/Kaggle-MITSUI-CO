#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping, log_evaluation

# ---------------------- Configuration ----------------------
TRAIN_FEATURES_PATH = "train.csv"
TRAIN_LABELS_PATH   = "train_labels.csv"
TEST_FEATURES_PATH  = "test.csv"
TARGET_PAIRS_PATH   = "target_pairs.csv"
PRICE_CLUSTERS_PATH = "price_clusters.csv"
CLUSTER_BETA_PATH   = "price_cluster_beta.csv"
ALPHA_RESIDUALS_PATH= "price_alpha_residuals.csv"

OUTPUT_SUBMISSION   = "submission_price_only.csv"
OUTPUT_CV_REPORT    = "training_cv_report.csv"

W_SHORT, W_MED, W_LONG = 5, 20, 60
N_FOLDS, EMBARGO_DAYS = 5, 5
RANDOM_SEED = 42

LGB_PARAMS = dict(
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=2000,
    min_data_in_leaf=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    random_state=RANDOM_SEED,
    objective="regression",
    verbosity=-1,             # ← 静默，抑制“best gain: -inf”的刷屏
)

# ---------------------- Utils ----------------------
def parse_pair(s):
    return [p.strip() for p in s.split("-") if p.strip()]

def to_int_date_id(s: pd.Series) -> pd.Series:
    if np.issubdtype(s.dtype, np.integer): return s.astype(int)
    try:
        dt = pd.to_datetime(s)
        ns = dt.astype("int64")           # 替换 view('int64')
        return ns - ns.min()
    except Exception:
        return pd.Series(np.arange(len(s)), index=s.index)

def zscore(s, w): m=s.rolling(w).mean(); v=s.rolling(w).std(); return (s-m)/(v+1e-9)
def rolling_slope(s, w):
    idx=np.arange(w)
    def _sl(x):
        y=np.asarray(x); x0=idx; xm=x0.mean(); ym=y.mean()
        num=((x0-xm)*(y-ym)).sum(); den=((x0-xm)**2).sum()+1e-9
        return num/den
    return s.rolling(w).apply(_sl, raw=False)
def rolling_sharpe(s,w): m=s.rolling(w).mean(); v=s.rolling(w).std(); return m/(v+1e-9)
def max_drawdown(s,w): rm=s.rolling(w).max(); return s/(rm+1e-12)-1.0

# ---------------------- Load ----------------------
print("Loading files...")
tp = pd.read_csv(TARGET_PAIRS_PATH)
train_feats = pd.read_csv(TRAIN_FEATURES_PATH)
test_feats  = pd.read_csv(TEST_FEATURES_PATH)
clusters = pd.read_csv(PRICE_CLUSTERS_PATH)
f = pd.read_csv(CLUSTER_BETA_PATH)
a = pd.read_csv(ALPHA_RESIDUALS_PATH)

if not os.path.exists(TRAIN_LABELS_PATH):
    raise FileNotFoundError("Missing train_labels.csv (must contain target_0..target_423).")
train_labels = pd.read_csv(TRAIN_LABELS_PATH)

if "date_id" not in train_feats.columns or "date_id" not in test_feats.columns:
    raise ValueError("train.csv/test.csv 必须包含 date_id。")

if not np.issubdtype(f["date_id"].dtype, np.integer): f["date_id"]=to_int_date_id(f["date_id"])
if not np.issubdtype(a["date_id"].dtype, np.integer): a["date_id"]=to_int_date_id(a["date_id"])
# 去重 & 排序，避免后续对齐异常
f = f.drop_duplicates(subset="date_id").sort_values("date_id").reset_index(drop=True)
a = a.drop_duplicates(subset="date_id").sort_values("date_id").reset_index(drop=True)

valid_dates = set(train_feats["date_id"]).union(set(test_feats["date_id"]))
f = f[f["date_id"].isin(valid_dates)].copy()
a = a[a["date_id"].isin(valid_dates)].copy()

# 仅保留“价格系”targets
price_instruments = set(clusters["instrument"])
def all_in_price(pair_str): return all(n in price_instruments for n in parse_pair(pair_str))
tp_price = tp[tp["pair"].apply(all_in_price)].reset_index(drop=True)
print(f"Targets total={len(tp)}, price-only={len(tp_price)}")

ins_to_cluster = dict(zip(clusters["instrument"], clusters["cluster"]))

# ---------------------- Feature builders ----------------------
def build_single_features(ins, date_idx):
    if ins not in a.columns: raise KeyError(f"Alpha not found: {ins}")
    if ins not in ins_to_cluster: raise KeyError(f"Instrument not in price_clusters: {ins}")
    clu = ins_to_cluster[ins]; fcol=f"cluster_{clu}"
    if fcol not in f.columns: raise KeyError(f"Cluster beta not found: {fcol}")

    df = pd.DataFrame({"date_id": date_idx})
    df = df.merge(f[["date_id", fcol]], on="date_id", how="left")
    df = df.merge(a[["date_id", ins]], on="date_id", how="left")
    fc = df[fcol].astype(float); al=df[ins].astype(float)

    feats = pd.DataFrame({"date_id": df["date_id"]})
    feats["f_lag1"]=fc.shift(1)
    feats[f"f_ma{W_SHORT}"]=fc.rolling(W_SHORT).mean().shift(1)
    feats[f"f_ma{W_MED}"]=fc.rolling(W_MED).mean().shift(1)
    feats[f"f_vol{W_MED}"]=fc.rolling(W_MED).std().shift(1)
    feats[f"f_slope{W_MED}"]=rolling_slope(fc,W_MED).shift(1)
    feats[f"f_sharpe{W_MED}"]=rolling_sharpe(fc,W_MED).shift(1)
    feats[f"f_dd{W_LONG}"]=max_drawdown(fc,W_LONG).shift(1)

    feats["a_lag1"]=al.shift(1)
    feats[f"a_ma{W_SHORT}"]=al.rolling(W_SHORT).mean().shift(1)
    feats[f"a_z{W_MED}"]=zscore(al,W_MED).shift(1)
    feats[f"a_z{W_LONG}"]=zscore(al,W_LONG).shift(1)
    feats[f"a_persist{W_SHORT}"]=((al.shift(1)>0).rolling(W_SHORT).mean())
    return feats

def build_pair_features(ins_i, ins_j, date_idx):
    ci,cj = ins_to_cluster[ins_i], ins_to_cluster[ins_j]
    fi_name,fj_name = f"cluster_{ci}", f"cluster_{cj}"
    for col in {fi_name, fj_name}:
        if col not in f.columns: raise KeyError(f"Cluster beta not found: {col}")
    for ins in [ins_i,ins_j]:
        if ins not in a.columns: raise KeyError(f"Alpha not found: {ins}")

    base = pd.DataFrame({"date_id": date_idx})
    betas = f[["date_id", fi_name]].rename(columns={fi_name:"fi"})
    if fj_name != fi_name:
        betas = betas.merge(f[["date_id", fj_name]].rename(columns={fj_name:"fj"}),
                            on="date_id", how="left")
    else:
        betas["fj"] = betas["fi"]
    alphas = a[["date_id", ins_i, ins_j]].rename(columns={ins_i:"ai", ins_j:"aj"})
    df = base.merge(betas, on="date_id", how="left").merge(alphas, on="date_id", how="left")

    fi,fj = df["fi"].astype(float), df["fj"].astype(float)
    ai,aj = df["ai"].astype(float), df["aj"].astype(float)
    ri, rj = ai+fi, aj+fj

    feats = pd.DataFrame({"date_id": df["date_id"]})
    diff_f, diff_a = fi-fj, ai-aj
    feats["f_diff_lag1"]=diff_f.shift(1)
    feats["a_diff_lag1"]=diff_a.shift(1)
    feats[f"fi_vol{W_MED}"]=fi.rolling(W_MED).std().shift(1)
    feats[f"fj_vol{W_MED}"]=fj.rolling(W_MED).std().shift(1)
    feats[f"a_diff_z{W_MED}"]=zscore(diff_a,W_MED).shift(1)
    feats[f"corr_ij{W_LONG}"]=ri.rolling(W_LONG).corr(rj).shift(1)
    return feats

# ---------------------- Train & Predict ----------------------
cv_rows, pred_frames = [], []
train_dates = np.sort(train_feats["date_id"].unique())
test_dates  = np.sort(test_feats.loc[test_feats.get("is_scored", True).astype(bool), "date_id"].unique())
print(f"Training on {len(train_dates)} train dates; predicting {len(test_dates)} test dates")

for _, row in tp_price.iterrows():
    tgt, lag = row["target"], int(row["lag"])
    parts = parse_pair(row["pair"]); is_pair = (len(parts)==2)
    full_dates = np.sort(np.union1d(train_dates, test_dates))
    X_full = build_pair_features(parts[0], parts[1], full_dates) if is_pair \
        else build_single_features(parts[0], full_dates)

    if tgt not in train_labels.columns:
        print(f"[WARN] {tgt} not found in train_labels; skip."); continue

    X_train = pd.merge(pd.DataFrame({"date_id": train_dates}), X_full, on="date_id", how="left")
    y_train = pd.merge(pd.DataFrame({"date_id": train_dates}),
                       train_labels[["date_id", tgt]], on="date_id", how="left")[tgt]

    mask = X_train.drop(columns=["date_id"]).notna().all(axis=1) & y_train.notna()
    X_train = X_train.loc[mask].reset_index(drop=True)
    y_train = y_train.loc[mask].reset_index(drop=True)

    # === 关键：删掉常数/全空/非有限的特征，并统一特征列 ===
    X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    feature_cols = [c for c in X_train.columns if c!="date_id"]
    const_cols = [c for c in feature_cols if X_train[c].nunique(dropna=True) <= 1]
    feature_cols = [c for c in feature_cols if c not in const_cols]
    if len(feature_cols) == 0:
        print(f"[WARN] all-constant features for {tgt}; skip."); continue

    tscv = TimeSeriesSplit(n_splits=N_FOLDS)
    fold_maes = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train)):
        if EMBARGO_DAYS > 0:
            va_start = va_idx[0]
            max_train_date = X_train.loc[va_start, "date_id"] - EMBARGO_DAYS
            tr_idx = np.where(X_train["date_id"].values <= max_train_date)[0]
            if len(tr_idx) == 0: continue

        X_tr = X_train[feature_cols].iloc[tr_idx]; y_tr = y_train.iloc[tr_idx]
        X_va = X_train[feature_cols].iloc[va_idx]; y_va = y_train.iloc[va_idx]

        model = LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l1",
            callbacks=[early_stopping(200), log_evaluation(0)],  # ← 200 轮无提升即停；不打印
        )
        p = model.predict(X_va)
        fold_maes.append(mean_absolute_error(y_va, p))

    cv_rows.append({
        "target": tgt, "n_train": len(X_train), "folds": len(fold_maes),
        "mae_mean": (np.nanmean(fold_maes) if fold_maes else np.nan),
        "mae_std":  (np.nanstd(fold_maes)  if fold_maes else np.nan)
    })

    # final fit（用同一套 feature_cols）
    final_model = LGBMRegressor(**LGB_PARAMS)
    final_model.fit(X_train[feature_cols], y_train)

    X_test = pd.merge(pd.DataFrame({"date_id": test_dates}), X_full, on="date_id", how="left")
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test[["date_id"] + feature_cols].fillna(0.0)   # 与训练列对齐
    yhat = final_model.predict(X_test[feature_cols])
    pred_frames.append(pd.DataFrame({"date_id": test_dates, tgt: yhat}))

# ---------------------- Aggregate ----------------------
if not pred_frames:
    raise RuntimeError("No predictions generated. Check train_labels & filtering.")

pred = pred_frames[0]
for k in range(1, len(pred_frames)):
    pred = pred.merge(pred_frames[k], on="date_id", how="left")

test_scored = test_feats.loc[test_feats.get("is_scored", True).astype(bool), ["date_id"]]
pred = test_scored.merge(pred, on="date_id", how="left")
pred.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Saved predictions to: {OUTPUT_SUBMISSION} (shape={pred.shape})")

cv = pd.DataFrame(cv_rows).sort_values("mae_mean")
cv.to_csv(OUTPUT_CV_REPORT, index=False)
print(f"Saved CV report to: {OUTPUT_CV_REPORT} (rows={len(cv)})")
print("Done.")
