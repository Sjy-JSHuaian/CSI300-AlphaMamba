"""Phase X: Historical Weekly Backtest.

Walk-forward backtest using Phase 6B methodology:
  - Binary Bull/NonBull regime models
  - Soft blending by bull_strength
  - Phase 5 portfolio optimization

Uses 2-fold expanding-window splits from config:
  F1: Train <=2024-12-31, Predict 2025H1
  F2: Train <=2025-06-30, Predict 2025H2+

For each week: predict on Monday, hold 5 trading days, rebalance.

Output:
  - output/equity_curve.csv: weekly PnL, cumulative returns
  - output/report.csv: summary statistics (Sharpe, MDD, win rate, etc.)

Usage: python src/backtest.py
"""

import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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
    """Compute raw T+5 forward return per stock per date.

    forward_ret = (Open[T+5] - Open[T+1]) / Open[T+1]

    This is the actual holding-period return used for portfolio PnL.
    """
    df = df.copy()
    results = []
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()
        open_t1 = group["开盘"].shift(-1)
        open_t5 = group["开盘"].shift(-5)
        group["forward_ret"] = (open_t5 - open_t1) / open_t1.replace(0, np.nan)
        results.append(group)
    return pd.concat(results, ignore_index=True)


def _train_one_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    regime_name: str,
    params: dict,
    early_stopping_rounds: int = 100,
) -> Optional[lgb.Booster]:
    """Train a single LightGBM Ranker on one regime's data.

    Uses time-based 80/20 split within training data for early stopping.
    """
    df_r = train_df.sort_values(["日期", "股票代码"]).copy()

    all_dates = sorted(df_r["日期"].unique())
    n_val_dates = max(1, int(len(all_dates) * 0.2))

    if len(all_dates) < 10 or n_val_dates < 1:
        print(f"  [SKIP] {regime_name}: only {len(all_dates)} training days")
        return None

    val_dates = set(all_dates[-n_val_dates:])
    train_dates_set = set(all_dates[:-n_val_dates])

    sub_train = df_r[df_r["日期"].isin(train_dates_set)].copy()
    sub_val = df_r[df_r["日期"].isin(val_dates)].copy()

    print(f"  [{regime_name.upper()}] "
          f"train={len(sub_train):,} rows/{len(train_dates_set)} days, "
          f"val={len(sub_val):,} rows/{len(val_dates)} days")

    # Discretize labels
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

    ndcg = model.best_score.get("val", {}).get("ndcg@5", 0.0)
    print(f"  [{regime_name.upper()}] "
          f"NDCG@5={ndcg:.4f}, Best Iter={model.best_iteration}")

    return model


def _get_weekly_groups(dates: pd.Series) -> pd.Series:
    """Assign each date to a week label: (iso_year, iso_week)."""
    dt_idx = pd.DatetimeIndex(dates)
    iso = dt_idx.isocalendar()
    return pd.Series(
        list(zip(iso["year"].values, iso["week"].values)),
        index=dates.index,
    )


# ======================================================================
# Metrics
# ======================================================================

def compute_metrics(
    equity_df: pd.DataFrame,
    risk_free_rate: float = 0.02,
) -> dict:
    """Compute performance metrics from weekly equity curve.

    Args:
        equity_df: DataFrame with columns [week_return, benchmark_return]
        risk_free_rate: Annual risk-free rate (default 2%)

    Returns:
        Dict of metric name → value
    """
    rets = equity_df["week_return"].values
    bench_rets = equity_df["benchmark_return"].values
    n = len(rets)

    if n == 0:
        return {"n_weeks": 0}

    # Cumulative return
    cumulative = np.cumprod(1 + rets) - 1
    bench_cumulative = np.cumprod(1 + bench_rets) - 1

    # Basic stats
    avg_ret = float(np.mean(rets))
    total_ret = float(cumulative[-1])
    bench_total = float(bench_cumulative[-1])
    excess_total = total_ret - bench_total

    # Win rate
    win_rate = float(np.mean(rets > 0))

    # Sharpe ratio (annualized from weekly)
    weekly_rf = risk_free_rate / 52
    excess_rets = rets - weekly_rf
    sharpe = float(excess_rets.mean() / excess_rets.std() * np.sqrt(52)) if excess_rets.std() > 1e-12 else 0.0

    # Sortino ratio (downside deviation only)
    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 0 else 0.0
    sortino = float(excess_rets.mean() / downside_std * np.sqrt(52)) if downside_std > 1e-12 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(np.concatenate([[1.0], 1 + rets]))
    equity_no_init = np.concatenate([[1.0], 1 + rets])
    peak_equity = np.maximum.accumulate(equity_no_init)
    drawdowns = (equity_no_init - peak_equity) / peak_equity
    max_dd = float(np.min(drawdowns))

    # Calmar ratio
    calmar = float(total_ret / abs(max_dd)) if abs(max_dd) > 1e-12 else 0.0

    # Information ratio
    tracking_error = np.std(rets - bench_rets)
    info_ratio = float((avg_ret - np.mean(bench_rets)) / tracking_error * np.sqrt(52)) if tracking_error > 1e-12 else 0.0

    # Hit rate (beat benchmark)
    beat_rate = float(np.mean(rets > bench_rets))

    # Max consecutive wins / losses
    signs = np.sign(rets)
    from itertools import groupby
    runs = [len(list(g)) for _, g in groupby(signs)]
    max_win_streak = max((r for s, r in zip([list(g)[0] for _, g in groupby(signs)], runs) if s > 0), default=0)
    max_loss_streak = max((r for s, r in zip([list(g)[0] for _, g in groupby(signs)], runs) if s < 0), default=0)

    # Volatility
    weekly_vol = float(np.std(rets))
    annual_vol = float(weekly_vol * np.sqrt(52))

    return {
        "n_weeks": n,
        "avg_weekly_return": round(avg_ret, 6),
        "weekly_volatility": round(weekly_vol, 6),
        "annual_volatility": round(annual_vol, 4),
        "total_return": round(total_ret, 4),
        "benchmark_total_return": round(bench_total, 4),
        "excess_return": round(excess_total, 4),
        "win_rate": round(win_rate, 4),
        "beat_benchmark_rate": round(beat_rate, 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "calmar_ratio": round(calmar, 4),
        "information_ratio": round(info_ratio, 4),
        "max_drawdown": round(max_dd, 4),
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


# ======================================================================
# Main backtest
# ======================================================================

def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)

    # ---- 1. Load data ----
    print("=" * 60)
    print("Phase X: Historical Weekly Backtest")
    print("=" * 60)

    print("\n[1/6] Loading data...")
    data_path = os.path.join(cfg["data_path"], "train.csv")
    df = pd.read_csv(data_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    print(f"  Raw: {len(df):,} rows, {df['股票代码'].nunique()} stocks")
    print(f"  Range: {df['日期'].min().date()} to {df['日期'].max().date()}")

    stock_counts = df.groupby("股票代码").size()
    valid_stocks = stock_counts[stock_counts >= cfg["min_history_days"]].index
    df = df[df["股票代码"].isin(valid_stocks)]
    print(f"  After filter: {df['股票代码'].nunique()} stocks")

    # ---- 2. Feature engineering (ONCE on full data) ----
    print("\n[2/6] Phase 6B feature engineering (full dataset)...")
    df, feature_cols = engineer_phase6b_features(df)
    df = df.dropna(subset=["label"])
    df[feature_cols] = df[feature_cols].fillna(0.0)

    # Compute forward returns for PnL calculation
    print("  Computing forward returns (T+5 horizon)...")
    df = _compute_forward_returns(df)

    # Sanity-check and cap extreme forward returns (data errors)
    fwd = df["forward_ret"]
    extreme_mask = (fwd < -1.0) | (fwd > 10.0)
    if extreme_mask.any():
        n_extreme = extreme_mask.sum()
        print(f"  [WARN] {n_extreme} extreme forward_ret values detected "
              f"(min={fwd.min():.4f}, max={fwd.max():.4f}) — capping to [-1, 10]")
        df["forward_ret"] = df["forward_ret"].clip(-1.0, 10.0)

    print(f"  Features: {len(feature_cols)}, Samples: {len(df):,}")

    # ---- 3. Define fold splits ----
    print("\n[3/6] Setting up walk-forward folds...")
    fold_splits = cfg["wf_fold_splits"]
    df = df.sort_values("日期")

    folds = []
    for i, split in enumerate(fold_splits):
        train_end = pd.Timestamp(split["train_end"])
        val_start = pd.Timestamp(split["val_start"])
        val_end = pd.Timestamp(split["val_end"])

        train_mask = df["日期"] <= train_end
        val_mask = (df["日期"] >= val_start) & (df["日期"] <= val_end)

        fold_name = (
            f"F{i + 1}_train-{split['train_end'][:4]}"
            f"_val-{val_start.strftime('%Y%m')}-{val_end.strftime('%Y%m')}"
        )
        folds.append((train_mask, val_mask, fold_name, val_start, val_end))
        print(f"  {fold_name}")

    # ---- 4. Portfolio optimizer config ----
    portfolio_cfg = cfg["portfolio"]
    optimizer = PortfolioOptimizer(
        temperature=portfolio_cfg["temperature"],
        max_per_industry=portfolio_cfg["max_per_industry"],
        corr_threshold=portfolio_cfg["corr_threshold"],
        min_score_ratio=portfolio_cfg["min_score_ratio"],
        top_n=portfolio_cfg["top_n"],
        select_k=portfolio_cfg["select_k"],
        corr_lookback=portfolio_cfg["corr_lookback"],
    )

    # ---- 5. Run walk-forward weekly backtest ----
    print(f"\n[4/6] Running weekly backtest ({len(folds)} folds)...")

    params = cfg["lgb_params"].copy()
    all_weeks: List[Dict] = []
    total_weeks = 0

    for fold_idx, (train_mask, val_mask, fold_name, val_start, val_end) in enumerate(folds):
        print(f"\n{'─' * 50}")
        print(f"Fold {fold_idx + 1}/{len(folds)}: {fold_name}")
        print(f"{'─' * 50}")

        train_df = df[train_mask].copy()
        val_df_full = df[val_mask].copy()

        # ---- 5a. Train regime models on THIS fold's training data ----
        print(f"  Training data: {len(train_df):,} rows, "
              f"{train_df['日期'].nunique()} days "
              f"({train_df['日期'].min().date()} to {train_df['日期'].max().date()})")

        # Fit scaler on training data only
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols].fillna(0).values)

        # Split training data by regime
        bull_train = train_df[train_df["regime"] == "bull"].copy()
        nonbull_train = train_df[train_df["regime"] == "nonbull"].copy()

        # Scale
        for regime_df in [bull_train, nonbull_train]:
            if len(regime_df) > 0:
                regime_df[feature_cols] = scaler.transform(
                    regime_df[feature_cols].fillna(0).values
                )

        # Train models
        print(f"  Bull training data:    {len(bull_train):,} rows, "
              f"{bull_train['日期'].nunique()} days")
        print(f"  NonBull training data: {len(nonbull_train):,} rows, "
              f"{nonbull_train['日期'].nunique()} days")

        bull_model = _train_one_model(
            bull_train, feature_cols, "bull", params,
            early_stopping_rounds=cfg["early_stopping_rounds"],
        )
        nonbull_model = _train_one_model(
            nonbull_train, feature_cols, "nonbull", params,
            early_stopping_rounds=cfg["early_stopping_rounds"],
        )

        if bull_model is None or nonbull_model is None:
            print(f"  [WARN] Missing model(s), skipping fold")
            continue

        # ---- 5b. Predict each WEEK in validation period ----
        val_df_full = val_df_full.copy()
        val_df_full["week_label"] = _get_weekly_groups(val_df_full["日期"])

        weekly_groups = val_df_full.groupby("week_label", sort=True)
        n_fold_weeks = len(weekly_groups)

        prev_portfolio: Optional[set] = None
        turnover_rates: List[float] = []

        for wi, (week_label, week_df) in enumerate(weekly_groups):
            # Use the FIRST trading day of each week for prediction
            pred_date = week_df["日期"].min()
            pred_day = week_df[week_df["日期"] == pred_date].copy()

            if len(pred_day) < 5:
                continue

            # Scale features
            X_pred = pred_day[feature_cols].fillna(0).values.astype(np.float32)
            X_pred = scaler.transform(X_pred)

            # Predict with both models
            bull_raw = bull_model.predict(X_pred)
            nonbull_raw = nonbull_model.predict(X_pred)

            # Rank normalize + soft blend
            bull_rank = pd.Series(bull_raw).rank(pct=True).values
            nonbull_rank = pd.Series(nonbull_raw).rank(pct=True).values

            w = float(pred_day["bull_strength"].iloc[0])
            final_scores = w * bull_rank + (1.0 - w) * nonbull_rank

            pred_day["score"] = final_scores

            # Ensure sector_id
            if "sector_id" not in pred_day.columns:
                pred_day["sector_id"] = pred_day["股票代码"].apply(_get_sector)

            # Portfolio optimization
            try:
                result = optimizer.optimize(pred_day, df, verbose=False)
            except Exception:
                continue

            if len(result) == 0:
                continue

            # ---- 5c. Compute weekly return ----
            selected_stocks = result["stock_id"].tolist()
            selected_weights = result["weight"].values

            # Map to forward returns
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

            # Turnover vs previous week
            current_set = set(selected_stocks)
            if prev_portfolio is not None and len(prev_portfolio) > 0:
                turnover = 1.0 - len(current_set & prev_portfolio) / max(
                    len(current_set), len(prev_portfolio)
                )
            else:
                turnover = 1.0  # First week = full turnover
            turnover_rates.append(turnover)
            prev_portfolio = current_set

            all_weeks.append({
                "year": int(week_label[0]),
                "week": int(week_label[1]),
                "date": pred_date,
                "fold": fold_name,
                "bull_strength": round(w, 4),
                "regime": str(pred_day["regime"].iloc[0]),
                "n_stocks_selected": len(result),
                "week_return": round(week_return, 6),
                "benchmark_return": round(bench_return, 6),
                "excess_return": round(week_return - bench_return, 6),
                "turnover": round(turnover, 4),
                "stocks": ",".join(selected_stocks),
            })

            if (wi + 1) % 20 == 0:
                print(f"  Progress: {wi + 1}/{n_fold_weeks} weeks done")

        print(f"  Fold complete: {len(weekly_groups)} weeks, "
              f"{len([w for w in all_weeks if w['fold'] == fold_name])} valid predictions")

        total_weeks += n_fold_weeks

    # ---- 6. Compute metrics & save ----
    print(f"\n[5/6] Computing performance metrics...")

    if len(all_weeks) == 0:
        print("  ERROR: No valid weeks in backtest")
        return

    equity_df = pd.DataFrame(all_weeks).sort_values(["year", "week"]).reset_index(drop=True)

    # Cumulative returns
    equity_df["cumulative_return"] = (1 + equity_df["week_return"]).cumprod() - 1
    equity_df["benchmark_cumulative"] = (1 + equity_df["benchmark_return"]).cumprod() - 1
    equity_df["excess_cumulative"] = (
        equity_df["cumulative_return"] - equity_df["benchmark_cumulative"]
    )

    # Overall metrics
    overall_metrics = compute_metrics(equity_df)

    # Per-fold metrics
    fold_metrics = {}
    for fold in equity_df["fold"].unique():
        fold_df = equity_df[equity_df["fold"] == fold]
        fold_metrics[fold] = compute_metrics(fold_df)

    # Per-regime metrics
    regime_metrics = {}
    for regime in equity_df["regime"].unique():
        regime_df = equity_df[equity_df["regime"] == regime]
        if len(regime_df) >= 3:
            regime_metrics[regime] = compute_metrics(regime_df)

    # ---- 7. Save results ----
    print(f"\n[6/6] Saving results...")

    # Equity curve
    equity_path = os.path.join("output", "equity_curve.csv")
    equity_df.to_csv(equity_path, index=False)
    print(f"  Equity curve: {equity_path} ({len(equity_df)} weeks)")

    # Report
    report_path = os.path.join("output", "report.csv")
    report_rows = []

    # Overall
    report_rows.append({"section": "OVERALL", "metric": "n_weeks", "value": overall_metrics["n_weeks"]})
    for k, v in overall_metrics.items():
        if k != "n_weeks":
            report_rows.append({"section": "OVERALL", "metric": k, "value": v})

    # Per fold
    for fold, metrics in fold_metrics.items():
        for k, v in metrics.items():
            report_rows.append({"section": f"FOLD_{fold}", "metric": k, "value": v})

    # Per regime
    for regime, metrics in regime_metrics.items():
        for k, v in metrics.items():
            report_rows.append({"section": f"REGIME_{regime}", "metric": k, "value": v})

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)
    print(f"  Report: {report_path}")

    # ---- Print summary ----
    print(f"\n{'=' * 60}")
    print(f"Backtest Results")
    print(f"{'=' * 60}")
    print(f"  Period:       {equity_df['date'].min().date()} to {equity_df['date'].max().date()}")
    print(f"  Weeks:        {overall_metrics['n_weeks']}")
    print(f"  Win Rate:     {overall_metrics['win_rate']:.2%}")
    print(f"  Beat Benchmark: {overall_metrics['beat_benchmark_rate']:.2%}")
    print(f"  Total Return: {overall_metrics['total_return']:.2%}")
    print(f"  Benchmark:    {overall_metrics['benchmark_total_return']:.2%}")
    print(f"  Excess:       {overall_metrics['excess_return']:.2%}")
    print(f"  Sharpe:       {overall_metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino:      {overall_metrics['sortino_ratio']:.2f}")
    print(f"  Max DD:       {overall_metrics['max_drawdown']:.2%}")
    print(f"  Info Ratio:   {overall_metrics['information_ratio']:.2f}")
    print(f"  Avg Turnover: {equity_df['turnover'].mean():.2%}")
    print(f"  Max Win Streak:  {overall_metrics['max_win_streak']} weeks")
    print(f"  Max Loss Streak: {overall_metrics['max_loss_streak']} weeks")

    if fold_metrics:
        print(f"\n  Per-Fold:")
        for fold, m in fold_metrics.items():
            print(f"    {fold}: Return={m['total_return']:.2%}, "
                  f"Sharpe={m['sharpe_ratio']:.2f}, "
                  f"Win={m['win_rate']:.2%}, "
                  f"n={m['n_weeks']}w")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
