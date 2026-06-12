"""Phase 3: Shock + Market Regime + Cross-stock Interaction Features.

Extends Phase 2 (83 shape features) with ~21 new features capturing
structural market information rather than statistical descriptions.

Categories:
  1. Shock/Event (~10): vol shocks, volume spikes, gaps, abnormal returns
  2. Market Regime (~5): trend, vol regime, breadth, index slope
  3. Cross-stock Interaction (~6): beta, sector rank delta, leader correlation

Total: ~104 features.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from features_phase2 import engineer_phase2_features


# ======================================================================
# Shock / Event Features
# ======================================================================

def _add_shock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility, volume, price shock features (per-stock).

    Computes:
      vol_shock5   = std(ret1, 5)  / std(ret1, 20)
      vol_shock20  = std(ret1, 20) / std(ret1, 60)
      volume_shock5  = volume / mean(volume, 20)
      volume_shock20 = volume / mean(volume, 60)
      gap         = (open - prev_close) / prev_close
      gap_abs     = |gap|
      high_low_ratio = (high - low) / close
      close_position = (close - low) / (high - low)
      abnormal_ret   = ret1 / std(ret1, 20)
      amihud      = |ret1| / (volume * close)
    """
    df = df.copy()
    results = []

    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()

        ret1 = group["ret1"].astype(float)
        volume = group["成交量"].astype(float)
        close = group["收盘"].astype(float)
        high = group["最高"].astype(float)
        low = group["最低"].astype(float)
        open_ = group["开盘"].astype(float)
        prev_close = close.shift(1)

        # Rolling standard deviations (min_periods ensures early values work)
        std5 = ret1.rolling(5, min_periods=3).std()
        std20 = ret1.rolling(20, min_periods=5).std()
        std60 = ret1.rolling(60, min_periods=10).std()

        # Vol shock
        group["vol_shock5"] = (std5 / (std20 + 1e-12)).values
        group["vol_shock20"] = (std20 / (std60 + 1e-12)).values

        # Volume shock
        vol_mean20 = volume.rolling(20, min_periods=5).mean()
        vol_mean60 = volume.rolling(60, min_periods=10).mean()
        group["volume_shock5"] = (volume / (vol_mean20 + 1e-12)).values
        group["volume_shock20"] = (volume / (vol_mean60 + 1e-12)).values

        # Gap (overnight surprise)
        gap_series = (open_ - prev_close) / (prev_close + 1e-12)
        group["gap"] = gap_series.values
        group["gap_abs"] = gap_series.abs().values

        # Intraday pressure
        group["high_low_ratio"] = ((high - low) / (close + 1e-12)).values
        group["close_position"] = ((close - low) / (high - low + 1e-12)).values

        # Abnormal return (standardized)
        group["abnormal_ret"] = (ret1 / (std20 + 1e-6)).values

        # Amihud illiquidity proxy
        group["amihud"] = (ret1.abs() / (volume * close + 1e-12)).values

        results.append(group)

    return pd.concat(results, ignore_index=True)


# ======================================================================
# Market Regime Features (date-level → broadcast to all stocks)
# ======================================================================

def _add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market regime features computed at the index level.

    All features are identical for all stocks on the same date.

    Computes:
      idx300_ret1    = equal-weight avg daily return
      idx300_close   = equal-weight avg close price
      idx_trend20    = rolling 20-day mean of idx300_ret20
      idx_slope20    = linear slope of idx300_close over 20 days
      vol_regime     = idx_vol20 / idx_vol60
      market_breadth = rolling 20-day mean of (ret1 > 0) fraction
    """
    df = df.copy()

    # Market-level daily series
    if "idx300_ret1" not in df.columns:
        df["idx300_ret1"] = df.groupby("日期")["ret1"].transform("mean")
    if "idx300_close" not in df.columns:
        df["idx300_close"] = df.groupby("日期")["收盘"].transform("mean")

    # Build date-level dataframe
    date_cols = ["日期", "idx300_ret1", "idx300_ret20", "idx300_close"]
    date_df = df[date_cols].drop_duplicates("日期").sort_values("日期").copy()

    # idx_trend20: persistent market trend
    date_df["idx_trend20"] = (
        date_df["idx300_ret20"]
        .rolling(20, min_periods=5).mean()
    )

    # idx_slope20: direction of market
    def _slope(x):
        x = x[~np.isnan(x)]
        if len(x) < 5:
            return 0.0
        return float(np.polyfit(np.arange(len(x)), x, 1)[0])

    date_df["idx_slope20"] = (
        date_df["idx300_close"]
        .rolling(20, min_periods=5).apply(_slope, raw=True)
    )

    # Vol regime: market vol expansion/contraction
    date_df["_idx_vol20"] = (
        date_df["idx300_ret1"]
        .rolling(20, min_periods=5).std()
    )
    date_df["_idx_vol60"] = (
        date_df["idx300_ret1"]
        .rolling(60, min_periods=10).std()
    )
    date_df["vol_regime"] = (
        date_df["_idx_vol20"] / (date_df["_idx_vol60"] + 1e-12)
    )

    # Market breadth: fraction of stocks advancing
    breadth_raw = df.groupby("日期")["ret1"].apply(lambda x: (x > 0).mean())
    date_df["_breadth_raw"] = date_df["日期"].map(breadth_raw)
    date_df["market_breadth"] = (
        date_df["_breadth_raw"]
        .rolling(20, min_periods=5).mean()
    )

    # Merge back to main df
    merge_cols = [
        "日期", "idx300_ret1", "idx300_close",
        "idx_trend20", "idx_slope20", "vol_regime", "market_breadth",
    ]
    df = df.merge(date_df[merge_cols], on="日期", how="left")

    return df


# ======================================================================
# Cross-stock Interaction Features
# ======================================================================

def _add_leader_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling correlation of each stock with its sector leader.

    Leader = stock with highest 20-day average volume in each sector/date.
    """
    df = df.copy()

    if "sector_id" not in df.columns:
        df["corr_leader20"] = 0.0
        return df

    # 20-day average volume per stock
    df["_avg_vol20"] = df.groupby("股票代码")["成交量"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )

    # Leader per (date, sector): stock with max _avg_vol20
    leader_map = {}
    for (date, sector), group in df.groupby(["日期", "sector_id"], sort=False):
        valid = group.dropna(subset=["_avg_vol20"])
        if len(valid) == 0:
            continue
        leader_stock = valid.loc[valid["_avg_vol20"].idxmax(), "股票代码"]
        leader_map[(date, sector)] = leader_stock

    # Build leader_ret1 lookup
    leader_ret = {}
    for (date, sector), leader_stock in leader_map.items():
        mask = (df["日期"] == date) & (df["股票代码"] == leader_stock)
        match = df.loc[mask, "ret1"]
        if len(match) > 0:
            leader_ret[(date, sector)] = match.values[0]

    # Map leader return to each row
    df["_leader_ret1"] = df.apply(
        lambda row: leader_ret.get((row["日期"], row["sector_id"]), np.nan),
        axis=1,
    )

    # Rolling correlation per stock with leader
    df["corr_leader20"] = 0.0
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期")
        idx = group.index
        s_ret = group["ret1"].astype(float)
        l_ret = group["_leader_ret1"].astype(float)
        corr = s_ret.rolling(20, min_periods=10).corr(l_ret)
        df.loc[idx, "corr_leader20"] = corr.fillna(0).values

    # Cleanup
    df.drop(columns=["_avg_vol20", "_leader_ret1"], inplace=True, errors="ignore")

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-stock interaction features.

    Computes:
      beta_20, beta_60      — rolling market beta
      sector_rank_delta5/20 — change in within-sector rank
      corr_leader20         — correlation with sector leader
    """
    df = df.copy()

    # Ensure market return exists
    if "idx300_ret1" not in df.columns:
        df["idx300_ret1"] = df.groupby("日期")["ret1"].transform("mean")

    # Beta per stock (rolling cov/var)
    results = []
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()

        ret1 = group["ret1"].astype(float)
        idx_ret = group["idx300_ret1"].astype(float)

        # Rolling beta 20
        cov20 = ret1.rolling(20, min_periods=10).cov(idx_ret)
        var20 = idx_ret.rolling(20, min_periods=10).var()
        group["beta_20"] = (cov20 / (var20 + 1e-12)).values

        # Rolling beta 60
        cov60 = ret1.rolling(60, min_periods=20).cov(idx_ret)
        var60 = idx_ret.rolling(60, min_periods=20).var()
        group["beta_60"] = (cov60 / (var60 + 1e-12)).values

        results.append(group)

    df = pd.concat(results, ignore_index=True)

    # Sector rank delta
    rank_col = "ind_rank_ret20"
    if rank_col in df.columns:
        for lag in [5, 20]:
            col = f"sector_rank_delta{lag}"
            df[col] = df.groupby("股票代码")[rank_col].transform(
                lambda x: x - x.shift(lag)
            )
    else:
        # Fallback: use rank_alpha20
        for lag in [5, 20]:
            col = f"sector_rank_delta{lag}"
            if "rank_alpha20" in df.columns:
                df[col] = df.groupby("股票代码")["rank_alpha20"].transform(
                    lambda x: x - x.shift(lag)
                )

    # Leader correlation
    df = _add_leader_correlation(df)

    return df


# ======================================================================
# Main Phase 3 Pipeline
# ======================================================================

# Phase 3 feature names (added on top of Phase 2's 83)
PHASE3_SHOCK_FEATURES = [
    "vol_shock5", "vol_shock20",
    "volume_shock5", "volume_shock20",
    "gap", "gap_abs",
    "high_low_ratio", "close_position",
    "abnormal_ret", "amihud",
]

PHASE3_REGIME_FEATURES = [
    "idx300_ret1", "idx300_close",
    "idx_trend20", "idx_slope20",
    "vol_regime", "market_breadth",
]

PHASE3_INTERACTION_FEATURES = [
    "beta_20", "beta_60",
    "sector_rank_delta5", "sector_rank_delta20",
    "corr_leader20",
]


def engineer_phase3_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Phase 3 feature engineering pipeline.

    Extends Phase 2 (83 shape features) with:
      - Shock/Event features (volatility, volume, gap, abnormal return)
      - Market Regime features (trend, vol regime, breadth)
      - Cross-stock Interaction (beta, sector delta, leader correlation)

    Total: ~104 features.

    Args:
        df: Raw dataframe with OHLCV columns.

    Returns:
        (df_with_features, list_of_feature_column_names)
    """
    print("=" * 60)
    print("Phase 3: Shock + Regime + Interaction Pipeline")
    print("=" * 60)

    # ---- Step 1: Phase 2 base features (83 features) ----
    print("[Phase 2 base] Running Phase 2 shape feature pipeline...")
    df, phase2_cols = engineer_phase2_features(df)
    print(f"  Phase 2 features: {len(phase2_cols)}")

    # ---- Step 2: Shock features ----
    print("[Phase 3-A] Adding shock/event features...")
    df = _add_shock_features(df)
    shock_cols = [c for c in PHASE3_SHOCK_FEATURES if c in df.columns]
    print(f"  Shock features: {len(shock_cols)}")

    # ---- Step 3: Market regime features ----
    print("[Phase 3-B] Adding market regime features...")
    df = _add_regime_features(df)
    regime_cols = [c for c in PHASE3_REGIME_FEATURES if c in df.columns]
    print(f"  Regime features: {len(regime_cols)}")

    # ---- Step 4: Cross-stock interaction features ----
    print("[Phase 3-C] Adding cross-stock interaction features...")
    df = _add_interaction_features(df)
    interaction_cols = [c for c in PHASE3_INTERACTION_FEATURES if c in df.columns]
    print(f"  Interaction features: {len(interaction_cols)}")

    # ---- Step 5: Combine feature columns ----
    phase3_new_cols = shock_cols + regime_cols + interaction_cols
    feature_cols = phase2_cols + [c for c in phase3_new_cols if c not in phase2_cols]

    # Fill NaN in new features
    for c in phase3_new_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    print(f"\n  Phase 2 features:  {len(phase2_cols)}")
    print(f"  Phase 3 new:      {len(phase3_new_cols)}")
    print(f"  Total features:   {len(feature_cols)}")
    print("=" * 60)

    return df, feature_cols
