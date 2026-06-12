"""Consensus Alpha Backtest.

Compares 3 strategies on the same 54-week walk-forward:
  1. submit1 (LGB-only, 197 features, equal weight)
  2. Phase 6B (Bull/NonBull regime split, soft blend, Phase 5 optimizer)
  3. Consensus Alpha (0.6*rank(P6B) + 0.4*rank(S1) + consensus_bonus)

All three use:
  - Same validation dates
  - Same Phase 5 portfolio optimizer (for Phase 6B and Consensus)
  - Same forward return measurement

submit1 uses its own equal-weight top-5 selection (matching original design).

Usage: python src/backtest_consensus.py
Output: output/consensus_comparison.csv, output/consensus_equity.csv
"""

import os, random, warnings
from itertools import groupby
from typing import Dict, List, Optional, Set

import joblib, lightgbm as lgb
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from features_phase6b import engineer_phase6b_features
from features_phase4 import discretize_labels_tail
from portfolio_optimizer import PortfolioOptimizer
from features import _get_sector

warnings.filterwarnings("ignore")

CONSENSUS_BONUS = 0.05
P6B_WEIGHT = 0.6
S1_WEIGHT = 0.4


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _compute_forward_returns(df):
    df = df.copy()
    results = []
    for sid, g in df.groupby("股票代码", sort=False):
        g = g.sort_values("日期").copy()
        o1 = g["开盘"].shift(-1); o5 = g["开盘"].shift(-5)
        g["forward_ret"] = (o5 - o1) / o1.replace(0, np.nan)
        results.append(g)
    return pd.concat(results, ignore_index=True)


def _get_weekly_groups(dates):
    dt = pd.DatetimeIndex(dates); iso = dt.isocalendar()
    return pd.Series(list(zip(iso["year"].values, iso["week"].values)), index=dates.index)


def _train_one_model(train_df, feature_cols, regime_name, params, early_stopping_rounds=100):
    df_r = train_df.sort_values(["日期", "股票代码"]).copy()
    all_dates = sorted(df_r["日期"].unique())
    n_val = max(1, int(len(all_dates) * 0.2))
    if len(all_dates) < 10: return None
    val_dates = set(all_dates[-n_val:]); tr_dates = set(all_dates[:-n_val])
    sub_tr = df_r[df_r["日期"].isin(tr_dates)].copy()
    sub_vl = df_r[df_r["日期"].isin(val_dates)].copy()
    sub_tr = discretize_labels_tail(sub_tr); sub_vl = discretize_labels_tail(sub_vl)
    tr_g = sub_tr.groupby("日期", sort=False).size().values
    vl_g = sub_vl.groupby("日期", sort=False).size().values
    Xt = sub_tr[feature_cols].fillna(0).values.astype(np.float32)
    yt = sub_tr["relevance"].values.astype(np.int32)
    Xv = sub_vl[feature_cols].fillna(0).values.astype(np.float32)
    yv = sub_vl["relevance"].values.astype(np.int32)
    return lgb.train(params=params, train_set=lgb.Dataset(Xt, yt, group=tr_g),
                     valid_sets=[lgb.Dataset(Xv, yv, group=vl_g, reference=lgb.Dataset(Xt, yt, group=tr_g))],
                     valid_names=["train", "val"],
                     callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)])


def compute_metrics(equity_df, rf=0.02):
    rets = equity_df["week_return"].values; bench = equity_df["benchmark_return"].values
    n = len(rets)
    if n == 0: return {"n_weeks": 0}
    cum = np.cumprod(1 + rets) - 1; bench_cum = np.cumprod(1 + bench) - 1
    avg, total = float(np.mean(rets)), float(cum[-1])
    wrf = rf / 52; ex = rets - wrf
    sharpe = float(ex.mean() / ex.std() * np.sqrt(52)) if ex.std() > 1e-12 else 0
    dd_std = np.std(rets[rets < 0]) if (rets < 0).any() else 0
    sortino = float(ex.mean() / dd_std * np.sqrt(52)) if dd_std > 1e-12 else 0
    eq = np.concatenate([[1.0], 1 + rets]); peak = np.maximum.accumulate(eq)
    mdd = float(np.min((eq - peak) / peak))
    te = np.std(rets - bench)
    ir = float((avg - np.mean(bench)) / te * np.sqrt(52)) if te > 1e-12 else 0
    hit = float(np.mean(rets > bench))
    signs = np.sign(rets); runs = [(k, len(list(g))) for k, g in groupby(signs)]
    return {"n_weeks": n, "total_return": round(total, 4), "benchmark_return": round(float(bench_cum[-1]), 4),
            "excess_return": round(total - float(bench_cum[-1]), 4), "avg_weekly": round(avg, 6),
            "sharpe": round(sharpe, 4), "sortino": round(sortino, 4), "max_dd": round(mdd, 4),
            "info_ratio": round(ir, 4), "win_rate": round(float(np.mean(rets > 0)), 4),
            "beat_rate": round(hit, 4), "max_win": max((r for k, r in runs if k > 0), default=0),
            "max_loss": max((r for k, r in runs if k < 0), default=0)}


def main():
    cfg = CONFIG; set_seed(cfg["seed"])
    os.makedirs("output", exist_ok=True)

    # ---- Load submit1 scores ----
    s1_path = "model/submit1_scores.csv"
    if not os.path.exists(s1_path):
        print(f"[WARN] {s1_path} not found — run extract_all_scores.py first")
        print("  Continuing with Phase 6B only...")
        s1_scores = None
    else:
        print("Loading submit1 scores...")
        s1_scores = pd.read_csv(s1_path, dtype={"股票代码": str})
        s1_scores["股票代码"] = s1_scores["股票代码"].str.zfill(6)
        s1_scores["日期"] = pd.to_datetime(s1_scores["日期"])
        print(f"  {len(s1_scores)} rows, {s1_scores['日期'].nunique()} dates")

    # ---- Load data ----
    print("\nLoading data...")
    df = pd.read_csv("data/train.csv", dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6); df["日期"] = pd.to_datetime(df["日期"])
    sc = df.groupby("股票代码").size()
    df = df[df["股票代码"].isin(sc[sc >= cfg["min_history_days"]].index)]

    # ---- Feature engineering ----
    print("Phase 6B feature engineering...")
    df, feature_cols = engineer_phase6b_features(df)
    df = df.dropna(subset=["label"]); df[feature_cols] = df[feature_cols].fillna(0.0)
    df = _compute_forward_returns(df)
    fwd = df["forward_ret"]
    if ((fwd < -1.0) | (fwd > 10.0)).any():
        df["forward_ret"] = df["forward_ret"].clip(-1.0, 10.0)

    # ---- Folds ----
    folds = []
    for i, s in enumerate(cfg["wf_fold_splits"]):
        tr = df["日期"] <= pd.Timestamp(s["train_end"])
        vl = (df["日期"] >= pd.Timestamp(s["val_start"])) & (df["日期"] <= pd.Timestamp(s["val_end"]))
        folds.append((tr, vl, f"F{i+1}"))

    # ---- Optimizer ----
    pc = cfg["portfolio"]
    opt = PortfolioOptimizer(temperature=pc["temperature"], max_per_industry=pc["max_per_industry"],
                             corr_threshold=pc["corr_threshold"], min_score_ratio=pc["min_score_ratio"],
                             top_n=pc["top_n"], select_k=pc["select_k"], corr_lookback=pc["corr_lookback"])

    params = cfg["lgb_params"].copy()
    all_weeks = []

    for fi, (tr_m, vl_m, fn) in enumerate(folds):
        print(f"\n{'='*50}\nFold {fi+1}: {fn}\n{'='*50}")
        tr_df = df[tr_m].copy(); vl_df = df[vl_m].copy()
        vl_df["wl"] = _get_weekly_groups(vl_df["日期"])

        # Train Phase 6B
        scaler = StandardScaler(); scaler.fit(tr_df[feature_cols].fillna(0).values)
        bt = tr_df[tr_df["regime"] == "bull"].copy(); nt = tr_df[tr_df["regime"] == "nonbull"].copy()
        for d in [bt, nt]:
            if len(d) > 0: d[feature_cols] = scaler.transform(d[feature_cols].fillna(0).values)
        print(f"Training bull ({len(bt):,} rows)...")
        bm = _train_one_model(bt, feature_cols, "bull", params, cfg["early_stopping_rounds"])
        print(f"Training nonbull ({len(nt):,} rows)...")
        nm = _train_one_model(nt, feature_cols, "nonbull", params, cfg["early_stopping_rounds"])
        if bm is None or nm is None: continue

        for wl, wdf in vl_df.groupby("wl", sort=True):
            pd_date = wdf["日期"].min(); pred = wdf[wdf["日期"] == pd_date].copy()
            if len(pred) < 5: continue
            bs = float(pred["bull_strength"].iloc[0])
            bench_ret = float(pred["forward_ret"].mean())

            # Ensure sector_id
            if "sector_id" not in pred.columns:
                pred["sector_id"] = pred["股票代码"].apply(_get_sector)

            # ---- Phase 6B prediction ----
            mc = set(feature_cols) - set(pred.columns)
            for c in mc: pred[c] = 0.0
            Xp = scaler.transform(pred[feature_cols].fillna(0).values.astype(np.float32))
            br = pd.Series(bm.predict(Xp)).rank(pct=True).values
            nr = pd.Series(nm.predict(Xp)).rank(pct=True).values
            p6b_scores = bs * br + (1 - bs) * nr
            pred["p6b_score"] = p6b_scores

            # ---- submit1 prediction ----
            has_s1 = False
            if s1_scores is not None:
                s1_day = s1_scores[s1_scores["日期"] == pd_date]
                if len(s1_day) > 0:
                    s1_map = dict(zip(s1_day["股票代码"], s1_day["submit1_score"]))
                    pred["s1_score"] = pred["股票代码"].map(s1_map).fillna(0.0)
                    pred["s1_rank"] = pred["s1_score"].rank(pct=True)
                    has_s1 = True

            # ---- Phase 6B optimize ----
            pred["score"] = pred["p6b_score"]
            try:
                r6 = opt.optimize(pred, df, verbose=False)
            except Exception:
                r6 = pd.DataFrame(columns=["stock_id", "weight"])
            r6_ret = _portfolio_return(r6, pred)

            # ---- Consensus Alpha ----
            if has_s1:
                pred["p6b_rank"] = pred["p6b_score"].rank(pct=True)
                # Top-20 bonus
                p6b_top20 = set(pred.nlargest(20, "p6b_score")["股票代码"])
                s1_top20 = set(pred.nlargest(20, "s1_score")["股票代码"])
                consensus_in_both = p6b_top20 & s1_top20
                bonus = np.array([CONSENSUS_BONUS if s in consensus_in_both else 0.0
                                  for s in pred["股票代码"]])
                pred["consensus_score"] = (P6B_WEIGHT * pred["p6b_rank"] +
                                           S1_WEIGHT * pred["s1_rank"] + bonus)
                pred["score"] = pred["consensus_score"]
                try:
                    rc = opt.optimize(pred, df, verbose=False)
                except Exception:
                    rc = pd.DataFrame(columns=["stock_id", "weight"])
                rc_ret = _portfolio_return(rc, pred)

                # submit1 equal-weight top-5
                s1_top5 = pred.nlargest(5, "s1_score")
                s1_ret = _equal_weight_return(s1_top5)
            else:
                rc_ret = r6_ret; s1_ret = 0.0

            all_weeks.append({
                "year": int(wl[0]), "week": int(wl[1]), "date": pd_date, "fold": fn,
                "bs": round(bs, 4), "regime": str(pred["regime"].iloc[0]),
                "p6b_return": round(r6_ret, 6), "consensus_return": round(rc_ret if has_s1 else r6_ret, 6),
                "s1_return": round(s1_ret, 6), "benchmark_return": round(bench_ret, 6),
            })

    # ---- Results ----
    if len(all_weeks) == 0:
        print("No weeks!"); return

    eq = pd.DataFrame(all_weeks).sort_values(["year", "week"]).reset_index(drop=True)
    for col in ["p6b", "consensus", "s1"]:
        rcol = f"{col}_return"
        if rcol in eq.columns:
            eq[f"{col}_cum"] = (1 + eq[rcol]).cumprod() - 1
    eq["bench_cum"] = (1 + eq["benchmark_return"]).cumprod() - 1

    # Metrics
    results = {}
    for name, rcol in [("Phase 6B", "p6b_return"), ("Consensus Alpha", "consensus_return"),
                        ("submit1", "s1_return")]:
        if rcol in eq.columns and eq[rcol].notna().any():
            tmp = pd.DataFrame({"week_return": eq[rcol].values,
                                "benchmark_return": eq["benchmark_return"].values})
            results[name] = compute_metrics(tmp)

    # Save
    eq.to_csv("output/consensus_equity.csv", index=False)
    print(f"\nSaved: output/consensus_equity.csv ({len(eq)} weeks)")

    # Report
    rows = []
    for name, m in results.items():
        for k, v in m.items():
            rows.append({"strategy": name, "metric": k, "value": v})
    pd.DataFrame(rows).to_csv("output/consensus_comparison.csv", index=False)

    # Print
    print(f"\n{'='*70}")
    print(f"Consensus Alpha — 3-Way Comparison ({len(eq)} weeks)")
    print(f"{'='*70}")
    print(f"{'Strategy':<18s} {'Return':>10s} {'Excess':>10s} {'Sharpe':>8s} {'MaxDD':>8s} {'Hit':>7s} {'Weeks':>6s}")
    print(f"{'─'*70}")
    for name, m in results.items():
        print(f"{name:<18s} {m['total_return']:>9.2%} {m['excess_return']:>9.2%} "
              f"{m['sharpe']:>7.2f} {m['max_dd']:>7.2%} {m['beat_rate']:>6.1%} {m['n_weeks']:>5d}")
    print(f"{'─'*70}")
    print(f"{'Benchmark':<18s} {eq['bench_cum'].iloc[-1]:>9.2%}")
    print("Done!")


def _portfolio_return(result, pred):
    if len(result) == 0: return 0.0
    rets = []
    for _, r in result.iterrows():
        m = pred[pred["股票代码"] == r["stock_id"]]
        if len(m) > 0:
            fr = m["forward_ret"].values[0]
            rets.append(fr if not pd.isna(fr) else 0.0)
        else:
            rets.append(0.0)
    return float(np.dot(result["weight"].values, rets))


def _equal_weight_return(top5):
    if len(top5) == 0: return 0.0
    rets = []
    for _, r in top5.iterrows():
        fr = r.get("forward_ret", 0)
        rets.append(fr if not pd.isna(fr) else 0.0)
    return float(np.mean(rets)) if rets else 0.0


if __name__ == "__main__":
    main()
