"""Phase 6A: Regime-Split Modeling.

Adds market regime classification to the Phase 4 feature pipeline.
Trains separate LightGBM Rankers for bull/bear/sideway regimes.

Regime definitions (from CSI 300 equal-weight index):
  - Bull:   idx300_slope20 > 0  AND  idx300_vol20 < vol_threshold
  - Bear:   idx300_slope20 < 0  AND  idx300_vol20 > vol_threshold
  - Sideway: everything else

Usage:
    from features_phase6 import engineer_phase6_features

    df, feature_cols = engineer_phase6_features(df)
    # df now has a 'regime' column in addition to 102 Phase 4 features
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from features_phase4 import engineer_phase4_features


# ======================================================================
# Market regime classification
# ======================================================================

def _compute_slope(x: np.ndarray) -> float:
    """Linear slope of a 1D array."""
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return 0.0
    return float(np.polyfit(np.arange(len(x), dtype=np.float64), x, 1)[0])


def compute_market_regime(
    df: pd.DataFrame,
    vol_threshold: float = 0.02,
) -> pd.DataFrame:
    """Compute market regime (bull/bear/sideway) from index-level data.

    Requires idx300_close and idx300_ret1 columns (produced by Phase 3/4).

    Computed per-date:
      - idx300_ret20 = pct_change(close, 20)
      - idx300_vol20 = rolling std(ret1, 20)
      - idx300_slope20 = rolling linear slope(close, 20)

    Classification:
      - Bull:   slope20 > 0 AND vol20 < vol_threshold
      - Bear:   slope20 < 0 AND vol20 > vol_threshold
      - Sideway: all other cases

    Args:
        df: DataFrame with columns [日期, idx300_close, idx300_ret1]
        vol_threshold: Daily volatility threshold for bull/bear distinction

    Returns:
        Same DataFrame with new 'regime' column ('bull'/'bear'/'sideway')
    """
    df = df.copy()

    # Build date-level index time series
    date_cols = ["日期", "idx300_close", "idx300_ret1"]
    available = [c for c in date_cols if c in df.columns]

    if "idx300_close" not in df.columns or "idx300_ret1" not in df.columns:
        print("  [WARN] idx300_close/idx300_ret1 not found, computing from stock data...")
        if "idx300_close" not in df.columns:
            df["idx300_close"] = df.groupby("日期")["收盘"].transform("mean")
        if "idx300_ret1" not in df.columns:
            if "ret1" not in df.columns:
                # Compute ret1 from close
                results = []
                for sid, g in df.groupby("股票代码", sort=False):
                    g = g.sort_values("日期").copy()
                    g["ret1"] = g["收盘"].pct_change()
                    results.append(g)
                df = pd.concat(results, ignore_index=True)
            df["idx300_ret1"] = df.groupby("日期")["ret1"].transform("mean")

    date_df = (
        df[["日期", "idx300_close", "idx300_ret1"]]
        .drop_duplicates("日期")
        .sort_values("日期")
        .copy()
    )

    # 20-day return
    date_df["idx300_ret20"] = date_df["idx300_close"].pct_change(20)

    # 20-day rolling volatility (std of daily returns)
    date_df["idx300_vol20"] = (
        date_df["idx300_ret1"]
        .rolling(20, min_periods=10)
        .std()
    )

    # 20-day linear slope of close
    date_df["idx300_slope20"] = (
        date_df["idx300_close"]
        .rolling(20, min_periods=10)
        .apply(_compute_slope, raw=True)
    )

    # Classify regime
    def _classify(row) -> str:
        slope = row.get("idx300_slope20")
        vol = row.get("idx300_vol20")
        if pd.isna(slope) or pd.isna(vol):
            return "sideway"
        if slope > 0 and vol < vol_threshold:
            return "bull"
        elif slope < 0 and vol > vol_threshold:
            return "bear"
        else:
            return "sideway"

    date_df["regime"] = date_df.apply(_classify, axis=1)

    # Merge back to main dataframe
    df = df.merge(date_df[["日期", "regime"]], on="日期", how="left")

    # Report regime distribution
    regime_counts = date_df["regime"].value_counts()
    total = len(date_df)
    print(f"\n  Market Regime Distribution ({total} trading days):")
    for regime in ["bull", "bear", "sideway"]:
        n = regime_counts.get(regime, 0)
        pct = 100 * n / total if total > 0 else 0
        print(f"    {regime:<8s}: {n:>5d} days ({pct:>5.1f}%)")

    return df


# ======================================================================
# Phase 6 feature engineering pipeline
# ======================================================================

def engineer_phase6_features(
    df: pd.DataFrame,
    vol_threshold: float = 0.02,
) -> Tuple[pd.DataFrame, List[str]]:
    """Phase 6A feature engineering: Phase 4 features + regime classification.

    Steps:
      1. Run Phase 4 pipeline (ensemble label + Phase 3 features)
      2. Compute market regime from idx300 data
      3. Return 102 features + regime column

    Args:
        df: Raw dataframe with OHLCV columns
        vol_threshold: Volatility threshold for bull/bear classification

    Returns:
        (df_with_features_and_regime, list_of_feature_column_names)
    """
    print("=" * 60)
    print("Phase 6A: Regime-Split Feature Pipeline")
    print("=" * 60)

    # Step 1: Phase 4 features (102-dim)
    df, feature_cols = engineer_phase4_features(df)

    # Step 2: Regime classification
    print("\n[Phase 6A] Computing market regime...")
    df = compute_market_regime(df, vol_threshold=vol_threshold)

    print(f"\n  Total features: {len(feature_cols)} (regime is a split key, not a feature)")
    print(f"  Total samples:  {len(df)}")
    print("=" * 60)

    return df, feature_cols
