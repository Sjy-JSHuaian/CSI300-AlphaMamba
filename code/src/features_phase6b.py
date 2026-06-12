"""Phase 6B: Bull / Non-Bull Binary Regime + Soft Blend.

Key insight from Phase 6A:
  - Bull regime alpha is structurally different from non-bull
  - Bear (24 days) is too few for a separate ranking model
  - Market state is continuous → soft blending > hard switching

Design:
  1. Binary regime (bull / nonbull) for training data split
  2. bull_strength = sigmoid(idx_ret20 / idx_vol20) ∈ (0, 1)
  3. At inference: final_score = w * bull_score + (1-w) * nonbull_score
     where w = bull_strength (today's market "bullishness")

This is NOT parameter tuning — it's a structural upgrade from
hard 3-class switching to continuous 2-model blending.

Usage:
    from features_phase6b import engineer_phase6b_features

    df, feature_cols = engineer_phase6b_features(df)
    # df now has 'regime' ('bull'/'nonbull') and 'bull_strength' (0-1)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from features_phase4 import engineer_phase4_features


# ======================================================================
# Bull strength computation
# ======================================================================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: 1 / (1 + exp(-x))."""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def compute_bull_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Compute continuous bull strength and binary regime for each date.

    bull_strength = sigmoid(idx_ret20 / idx_vol20)

    This is essentially the 20-day "information ratio" mapped to (0, 1):
      - Strong uptrend, low vol → near 1.0 (bull)
      - Strong downtrend, high vol → near 0.0 (bear/sideway)
      - Flat market → near 0.5 (uncertain)

    Binary split for training:
      - regime = 'bull'    if bull_strength >= 0.5
      - regime = 'nonbull' otherwise

    Requires idx300_close and idx300_ret1 (computed from stock data if missing).

    Args:
        df: DataFrame with at minimum [日期, 收盘] columns.
            Should have idx300_close, idx300_ret1 if Phase 3 has run.

    Returns:
        Same DataFrame with new columns: ['regime', 'bull_strength']
    """
    df = df.copy()

    # Ensure index-level data exists
    if "idx300_close" not in df.columns:
        df["idx300_close"] = df.groupby("日期")["收盘"].transform("mean")

    if "idx300_ret1" not in df.columns:
        if "ret1" not in df.columns:
            # Compute daily return from close
            results = []
            for sid, g in df.groupby("股票代码", sort=False):
                g = g.sort_values("日期").copy()
                g["ret1"] = g["收盘"].pct_change()
                results.append(g)
            df = pd.concat(results, ignore_index=True)
        df["idx300_ret1"] = df.groupby("日期")["ret1"].transform("mean")

    # Build date-level index time series
    date_df = (
        df[["日期", "idx300_close", "idx300_ret1"]]
        .drop_duplicates("日期")
        .sort_values("日期")
        .copy()
    )

    # 20-day return
    date_df["idx_ret20"] = date_df["idx300_close"].pct_change(20)

    # 20-day volatility (std of daily returns)
    date_df["idx_vol20"] = (
        date_df["idx300_ret1"].rolling(20, min_periods=10).std()
    )

    # Bull strength: trend-per-unit-risk → sigmoid → (0, 1)
    ratio = date_df["idx_ret20"].values / (date_df["idx_vol20"].values + 1e-12)
    date_df["bull_strength"] = _sigmoid(ratio)

    # Binary regime for training data split
    date_df["regime"] = np.where(
        date_df["bull_strength"] >= 0.5, "bull", "nonbull"
    )

    # Merge back to main dataframe
    df = df.merge(
        date_df[["日期", "regime", "bull_strength"]], on="日期", how="left"
    )

    # Report distribution
    regime_counts = date_df["regime"].value_counts()
    total = len(date_df)
    bs_mean = date_df["bull_strength"].mean()
    bs_std = date_df["bull_strength"].std()

    print(f"\n  Bull/NonBull Regime Distribution ({total} trading days):")
    for r in ["bull", "nonbull"]:
        n = regime_counts.get(r, 0)
        print(f"    {r:<8s}: {n:>5d} days ({100 * n / total:>5.1f}%)")
    print(f"    bull_strength: mean={bs_mean:.3f}, std={bs_std:.3f}")

    return df


# ======================================================================
# Phase 6B feature engineering pipeline
# ======================================================================

def engineer_phase6b_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Phase 6B feature pipeline: Phase 4 features + binary regime + bull_strength.

    Steps:
      1. Run Phase 4 pipeline (ensemble label + 102 features)
      2. Compute bull_strength and binary regime from index data
      3. Return 102 features + regime + bull_strength columns

    Args:
        df: Raw dataframe with OHLCV columns

    Returns:
        (df_with_features, list_of_102_feature_column_names)
    """
    print("=" * 60)
    print("Phase 6B: Bull / Non-Bull + Soft Blend Pipeline")
    print("=" * 60)

    # Step 1: Phase 4 features
    df, feature_cols = engineer_phase4_features(df)

    # Step 2: Bull strength + binary regime
    print("\n[Phase 6B] Computing bull_strength = sigmoid(ret20 / vol20)...")
    df = compute_bull_strength(df)

    print(f"\n  Features: {len(feature_cols)} "
          f"(regime + bull_strength are split/blend keys, not features)")
    print(f"  Samples:  {len(df)}")
    print("=" * 60)

    return df, feature_cols
