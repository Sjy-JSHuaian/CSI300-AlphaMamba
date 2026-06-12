"""Phase 7: Safe Mode Backtest.

Wraps Phase 6B regime-split models with risk management layers:

  1. Bull Gate:       bull_strength < 0.6  → 100% cash
  2. Dynamic Exposure: exposure = bull_strength (scale portfolio weight)
  3. Turnover Penalty:  λ = 0.02 on new positions
  4. Persistence Bonus: +0.01 on existing positions

Key hypothesis from Phase X:
  - Bull model:  Sharpe 0.61, Info Ratio 0.93  (WORKS)
  - NonBull model: Sharpe -0.95  (FATAL)
  → Gate out NonBull, scale exposure, reduce churn.

Usage: python src/backtest_phase7.py
Output: output/equity_curve_phase7.csv, output/report_phase7.csv
"""

import os
import random
import warnings
from itertools import groupby
from typing import Dict, List, Optional, Set

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import CONFIG
from features_phase6b import engineer_phase6b_features
from features_phase4 import discretize_labels_tail
from portfolio_optimizer import PortfolioOptimizer
from features import _get_sector

warnings.filterwarnings("ignore")


# ======================================================================
# Helpers
# ======================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _compute_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw T+5 forward return per stock per date."""
    df = df.copy()
    results = []
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()
        open_t1 = group["开盘"].shift(-1)
        open_t5 = group["开盘"].shift(-5)
        group["forward_ret"] = (open_t5 - open_t1) / open_t1.replace(0, np.nan)
        results.append(group)
    return pd.concat(results, ignore_index=True)


def _get_weekly_groups(dates: pd.Series) -> pd.Series:
    """Assign each date to an ISO week label."""
    dt_idx = pd.DatetimeIndex(dates)
    iso = dt_idx.isocalendar()
    return pd.Series(
        list(zip(iso["year"].values, iso["week"].values)),
        index=dates.index,
    )


def _train_one_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    regime_name: str,
    params: dict,
    early_stopping_rounds: int = 100,
) -> Optional[lgb.Booster]:
    """Train a single LightGBM Ranker on one regime's data."""
    df_r = train_df.sort_values(["日期", "股票代码"]).copy()
    all_dates = sorted(df_r["日期"].unique())
    n_val_dates = max(1, int(len(all_dates) * 0.2))

    if len(all_dates) < 10 or n_val_dates < 1:
        return None

    val_dates = set(all_dates[-n_val_dates:])
    train_dates_set = set(all_dates[:-n_val_dates])

    sub_train = df_r[df_r["日期"].isin(train_dates_set)].copy()
    sub_val = df_r[df_r["日期"].isin(val_dates)].copy()

    sub_train = discretize_labels_tail(sub_train)
    sub_val = discretize_labels_tail(sub_val)

    train_groups = sub_train.groupby("日期", sort=False).size().values
    val_groups = sub_val.groupby("日期", sort=False).size().values

    X_train = sub_train[feature_cols].fillna(0).values.astype(np.float32)
    y_train = sub_train["relevance"].values.astype(np.int32)
    X_val = sub_val[feature_cols].fillna(0).values.astype(np.float32)
    y_val = sub_val["relevance"].values.astype(np.int32)

    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_ds)

    model = lgb.train(
        params=params,
        train_set=train_ds,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(0),
        ],
    )
    return model


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(equity_df: pd.DataFrame, risk_free_rate: float = 0.02) -> dict:
    """Compute performance metrics from weekly equity curve."""
    rets = equity_df["week_return"].values
    bench_rets = equity_df["benchmark_return"].values
    n = len(rets)
    if n == 0:
        return {"n_weeks": 0}

    cumulative = np.cumprod(1 + rets) - 1
    bench_cumulative = np.cumprod(1 + bench_rets) - 1

    avg_ret = float(np.mean(rets))
    total_ret = float(cumulative[-1])
    bench_total = float(bench_cumulative[-1])
    weekly_rf = risk_free_rate / 52
    excess_rets = rets - weekly_rf

    sharpe = float(excess_rets.mean() / excess_rets.std() * np.sqrt(52)) if excess_rets.std() > 1e-12 else 0.0

    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 0.0
    sortino = float(excess_rets.mean() / downside_std * np.sqrt(52)) if downside_std > 1e-12 else 0.0

    equity_no_init = np.concatenate([[1.0], 1 + rets])
    peak_equity = np.maximum.accumulate(equity_no_init)
    drawdowns = (equity_no_init - peak_equity) / peak_equity
    max_dd = float(np.min(drawdowns))

    tracking_error = np.std(rets - bench_rets)
    info_ratio = float((avg_ret - np.mean(bench_rets)) / tracking_error * np.sqrt(52)) if tracking_error > 1e-12 else 0.0

    win_rate = float(np.mean(rets > 0))
    beat_rate = float(np.mean(rets > bench_rets))

    signs = np.sign(rets)
    runs = [(k, len(list(g))) for k, g in groupby(signs)]
    max_win_streak = max((r for k, r in runs if k > 0), default=0)
    max_loss_streak = max((r for k, r in runs if k < 0), default=0)

    return {
        "n_weeks": n,
        "avg_weekly_return": round(avg_ret, 6),
        "weekly_volatility": round(float(np.std(rets)), 6),
        "total_return": round(total_ret, 4),
        "benchmark_total_return": round(bench_total, 4),
        "excess_return": round(total_ret - bench_total, 4),
        "win_rate": round(win_rate, 4),
        "beat_benchmark_rate": round(beat_rate, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "information_ratio": round(info_ratio, 4),
        "max_drawdown": round(max_dd, 4),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    # Phase 7 parameters
    p7 = cfg.get("phase7", {})
    BULL_GATE = p7.get("bull_gate", 0.6)
    TURNOVER_PENALTY = p7.get("turnover_penalty", 0.02)
    PERSISTENCE_BONUS = p7.get("persistence_bonus", 0.01)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)

    # ---- 1. Load data ----
    print("=" * 60)
    print("Phase 7: Safe Mode Backtest")
    print("=" * 60)
    print(f"  Bull Gate:        bull_strength >= {BULL_GATE}")
    print(f"  Dynamic Exposure: weight *= bull_strength")
    print(f"  Turnover Penalty: lambda = {TURNOVER_PENALTY}")
    print(f"  Persistence Bonus: +{PERSISTENCE_BONUS}")

    print("\n[1/6] Loading data...")
    data_path = os.path.join(cfg["data_path"], "train.csv")
    df = pd.read_csv(data_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    stock_counts = df.groupby("股票代码").size()
    valid_stocks = stock_counts[stock_counts >= cfg["min_history_days"]].index
    df = df[df["股票代码"].isin(valid_stocks)]

    # ---- 2. Feature engineering ----
    print("\n[2/6] Phase 6B feature engineering...")
    df, feature_cols = engineer_phase6b_features(df)
    df = df.dropna(subset=["label"])
    df[feature_cols] = df[feature_cols].fillna(0.0)

    df = _compute_forward_returns(df)
    fwd = df["forward_ret"]
    if ((fwd < -1.0) | (fwd > 10.0)).any():
        print(f"  [WARN] Extreme forward_ret detected — capping to [-1, 10]")
        df["forward_ret"] = df["forward_ret"].clip(-1.0, 10.0)

    # ---- 3. Fold splits ----
    print("\n[3/6] Setting up folds...")
    fold_splits = cfg["wf_fold_splits"]
    df = df.sort_values("日期")
    folds = []
    for i, split in enumerate(fold_splits):
        train_end = pd.Timestamp(split["train_end"])
        val_start = pd.Timestamp(split["val_start"])
        val_end = pd.Timestamp(split["val_end"])
        train_mask = df["日期"] <= train_end
        val_mask = (df["日期"] >= val_start) & (df["日期"] <= val_end)
        fold_name = f"F{i + 1}"
        folds.append((train_mask, val_mask, fold_name))

    # ---- 4. Portfolio optimizer (with Phase 7 params) ----
    portfolio_cfg = cfg["portfolio"]
    optimizer = PortfolioOptimizer(
        temperature=portfolio_cfg["temperature"],
        max_per_industry=portfolio_cfg["max_per_industry"],
        corr_threshold=portfolio_cfg["corr_threshold"],
        min_score_ratio=portfolio_cfg["min_score_ratio"],
        top_n=portfolio_cfg["top_n"],
        select_k=portfolio_cfg["select_k"],
        corr_lookback=portfolio_cfg["corr_lookback"],
        turnover_penalty=TURNOVER_PENALTY,
        persistence_bonus=PERSISTENCE_BONUS,
    )

    # ---- 5. Walk-forward backtest ----
    print(f"\n[4/6] Running Phase 7 backtest ({len(folds)} folds)...")

    params = cfg["lgb_params"].copy()
    all_weeks: List[Dict] = []

    for fold_idx, (train_mask, val_mask, fold_name) in enumerate(folds):
        print(f"\n{'─' * 50}")
        print(f"Fold {fold_idx + 1}/{len(folds)}: {fold_name}")
        print(f"{'─' * 50}")

        train_df = df[train_mask].copy()
        val_df_full = df[val_mask].copy()

        # Train models
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols].fillna(0).values)

        bull_train = train_df[train_df["regime"] == "bull"].copy()
        nonbull_train = train_df[train_df["regime"] == "nonbull"].copy()

        for regime_df in [bull_train, nonbull_train]:
            if len(regime_df) > 0:
                regime_df[feature_cols] = scaler.transform(
                    regime_df[feature_cols].fillna(0).values
                )

        print(f"  Training bull model ({len(bull_train):,} rows, {bull_train['日期'].nunique()} days)...")
        bull_model = _train_one_model(bull_train, feature_cols, "bull", params,
                                      early_stopping_rounds=cfg["early_stopping_rounds"])

        print(f"  Training nonbull model ({len(nonbull_train):,} rows, {nonbull_train['日期'].nunique()} days)...")
        nonbull_model = _train_one_model(nonbull_train, feature_cols, "nonbull", params,
                                         early_stopping_rounds=cfg["early_stopping_rounds"])

        if bull_model is None or nonbull_model is None:
            print(f"  [WARN] Missing model(s), skipping fold")
            continue

        # Predict each week
        val_df_full = val_df_full.copy()
        val_df_full["week_label"] = _get_weekly_groups(val_df_full["日期"])
        weekly_groups = val_df_full.groupby("week_label", sort=True)

        prev_portfolio: Optional[Set[str]] = None
        n_fold_weeks = len(weekly_groups)

        for wi, (week_label, week_df) in enumerate(weekly_groups):
            pred_date = week_df["日期"].min()
            pred_day = week_df[week_df["日期"] == pred_date].copy()

            if len(pred_day) < 5:
                continue

            bull_strength = float(pred_day["bull_strength"].iloc[0])

            # ── Bull Gate ──
            if bull_strength < BULL_GATE:
                # 100% cash — zero return for this week
                bench_return = float(pred_day["forward_ret"].mean())
                all_weeks.append({
                    "year": int(week_label[0]),
                    "week": int(week_label[1]),
                    "date": pred_date,
                    "fold": fold_name,
                    "bull_strength": round(bull_strength, 4),
                    "regime": str(pred_day["regime"].iloc[0]),
                    "mode": "CASH",
                    "n_stocks_selected": 0,
                    "exposure": 0.0,
                    "week_return": 0.0,
                    "benchmark_return": round(bench_return, 6),
                    "excess_return": round(-bench_return, 6),
                    "turnover": 0.0,
                    "stocks": "",
                })
                continue

            # ── Predict & Soft Blend ──
            X_pred = pred_day[feature_cols].fillna(0).values.astype(np.float32)
            X_pred = scaler.transform(X_pred)

            bull_raw = bull_model.predict(X_pred)
            nonbull_raw = nonbull_model.predict(X_pred)
            bull_rank = pd.Series(bull_raw).rank(pct=True).values
            nonbull_rank = pd.Series(nonbull_raw).rank(pct=True).values
            final_scores = bull_strength * bull_rank + (1.0 - bull_strength) * nonbull_rank

            pred_day["score"] = final_scores
            if "sector_id" not in pred_day.columns:
                pred_day["sector_id"] = pred_day["股票代码"].apply(_get_sector)

            # ── Portfolio Optimization (with prev_portfolio for turnover penalty) ──
            try:
                result = optimizer.optimize(
                    pred_day, df,
                    prev_portfolio=prev_portfolio,
                    verbose=False,
                )
            except Exception:
                continue

            if len(result) == 0:
                prev_portfolio = None
                continue

            selected_stocks = result["stock_id"].tolist()
            selected_weights = result["weight"].values

            # ── Dynamic Exposure ──
            exposure = bull_strength
            selected_weights = selected_weights * exposure

            # ── Compute weekly return ──
            fwd_rets = []
            for sid in selected_stocks:
                match = pred_day[pred_day["股票代码"] == sid]
                if len(match) > 0:
                    fr = match["forward_ret"].values[0]
                    fwd_rets.append(fr if not pd.isna(fr) else 0.0)
                else:
                    fwd_rets.append(0.0)

            week_return = float(np.dot(selected_weights, fwd_rets))
            bench_return = float(pred_day["forward_ret"].mean())

            # Turnover
            current_set = set(selected_stocks)
            if prev_portfolio is not None and len(prev_portfolio) > 0:
                turnover = 1.0 - len(current_set & prev_portfolio) / max(
                    len(current_set), len(prev_portfolio)
                )
            else:
                turnover = 1.0
            prev_portfolio = current_set

            all_weeks.append({
                "year": int(week_label[0]),
                "week": int(week_label[1]),
                "date": pred_date,
                "fold": fold_name,
                "bull_strength": round(bull_strength, 4),
                "regime": str(pred_day["regime"].iloc[0]),
                "mode": "INVESTED",
                "n_stocks_selected": len(result),
                "exposure": round(exposure, 4),
                "week_return": round(week_return, 6),
                "benchmark_return": round(bench_return, 6),
                "excess_return": round(week_return - bench_return, 6),
                "turnover": round(turnover, 4),
                "stocks": ",".join(selected_stocks),
            })

        print(f"  Fold complete: {n_fold_weeks} weeks")

    # ---- 6. Metrics ----
    print(f"\n[5/6] Computing metrics...")
    if len(all_weeks) == 0:
        print("  ERROR: No valid weeks")
        return

    equity_df = pd.DataFrame(all_weeks).sort_values(["year", "week"]).reset_index(drop=True)
    equity_df["cumulative_return"] = (1 + equity_df["week_return"]).cumprod() - 1
    equity_df["benchmark_cumulative"] = (1 + equity_df["benchmark_return"]).cumprod() - 1
    equity_df["excess_cumulative"] = equity_df["cumulative_return"] - equity_df["benchmark_cumulative"]

    # Overall
    overall = compute_metrics(equity_df)

    # By mode
    cash_mask = equity_df["mode"] == "CASH"
    invested_mask = equity_df["mode"] == "INVESTED"
    cash_weeks = cash_mask.sum()
    invested_weeks = invested_mask.sum()

    invested_metrics = compute_metrics(equity_df[invested_mask]) if invested_weeks > 0 else {}
    bull_weeks_mask = invested_mask & (equity_df["regime"] == "bull")
    nonbull_weeks_mask = invested_mask & (equity_df["regime"] == "nonbull")
    bull_metrics = compute_metrics(equity_df[bull_weeks_mask]) if bull_weeks_mask.sum() > 0 else {}
    nonbull_metrics = compute_metrics(equity_df[nonbull_weeks_mask]) if nonbull_weeks_mask.sum() > 0 else {}

    # ---- 7. Save ----
    print(f"\n[6/6] Saving results...")

    equity_path = os.path.join("output", "equity_curve_phase7.csv")
    equity_df.to_csv(equity_path, index=False)
    print(f"  Equity curve: {equity_path} ({len(equity_df)} weeks)")

    report_path = os.path.join("output", "report_phase7.csv")
    report_rows = []

    def add_rows(section, metrics_dict):
        for k, v in metrics_dict.items():
            report_rows.append({"section": section, "metric": k, "value": v})

    add_rows("OVERALL", overall)
    add_rows("INVESTED", invested_metrics)
    add_rows("BULL", bull_metrics)
    add_rows("NONBULL", nonbull_metrics)

    # Phase 7 specific
    report_rows.append({"section": "PHASE7", "metric": "bull_gate", "value": BULL_GATE})
    report_rows.append({"section": "PHASE7", "metric": "turnover_penalty", "value": TURNOVER_PENALTY})
    report_rows.append({"section": "PHASE7", "metric": "persistence_bonus", "value": PERSISTENCE_BONUS})
    report_rows.append({"section": "PHASE7", "metric": "cash_weeks", "value": cash_weeks})
    report_rows.append({"section": "PHASE7", "metric": "invested_weeks", "value": invested_weeks})
    report_rows.append({"section": "PHASE7", "metric": "cash_ratio", "value": round(cash_weeks / len(equity_df), 4)})
    report_rows.append({"section": "PHASE7", "metric": "avg_turnover", "value": round(equity_df["turnover"].mean(), 4)})

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)
    print(f"  Report: {report_path}")

    # ---- Print Summary ----
    print(f"\n{'=' * 60}")
    print(f"Phase 7 Safe Mode Results")
    print(f"{'=' * 60}")
    print(f"  Period:          {equity_df['date'].min().date()} to {equity_df['date'].max().date()}")
    print(f"  Total Weeks:     {len(equity_df)}")
    print(f"  Cash Weeks:      {cash_weeks} ({cash_weeks/len(equity_df):.0%})")
    print(f"  Invested Weeks:  {invested_weeks} ({invested_weeks/len(equity_df):.0%})")
    print(f"")
    print(f"  Total Return:    {overall['total_return']:.2%}")
    print(f"  Benchmark:       {overall['benchmark_total_return']:.2%}")
    print(f"  Excess:          {overall['excess_return']:.2%}")
    print(f"  Sharpe:          {overall['sharpe_ratio']:.2f}")
    print(f"  Sortino:         {overall['sortino_ratio']:.2f}")
    print(f"  Max DD:          {overall['max_drawdown']:.2%}")
    print(f"  Avg Turnover:    {equity_df['turnover'].mean():.1%}")
    print(f"  Win Rate:        {overall['win_rate']:.1%}")
    print(f"")

    if invested_metrics:
        print(f"  --- When Invested ---")
        print(f"  Invested Return: {invested_metrics.get('total_return', 0):.2%}")
        print(f"  Invested Sharpe: {invested_metrics.get('sharpe_ratio', 0):.2f}")

    if bull_metrics:
        print(f"")
        print(f"  --- Bull Weeks (invested) ---")
        print(f"  Bull Return:     {bull_metrics.get('total_return', 0):.2%}")
        print(f"  Bull Sharpe:     {bull_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Bull Win Rate:   {bull_metrics.get('win_rate', 0):.1%}")

    if nonbull_metrics:
        print(f"")
        print(f"  --- NonBull Weeks (invested) ---")
        print(f"  NonBull Return:  {nonbull_metrics.get('total_return', 0):.2%}")
        print(f"  NonBull Sharpe:  {nonbull_metrics.get('sharpe_ratio', 0):.2f}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
