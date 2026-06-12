"""Phase 2: Temporal Shape Features Ranker.

Produces ~85 features focused on the SHAPE of past 60-day trajectories
for 6 core factors. Low-dimensional, captures trend/reversal/cyclical patterns.

Categories:
  1. Shape features (60): 6 factors x 10 stats over 60-day window
  2. Cross-sectional (8): rank_alpha, rank_ret, rank_delta
  3. Industry strength (8)
  4. Base factors (8): raw values of core factors + key references
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from features import (
    compute_labels, add_basic_features, add_return_features,
    add_momentum_features, add_volume_features,
    add_market_state_features, add_alpha_features,
    _get_sector,
)


# ---------------------------------------------------------------------------
# Shape statistics for a single window
# ---------------------------------------------------------------------------

def _window_stats(x: np.ndarray) -> dict:
    """Compute all 10 shape statistics for a window array.

    Args:
        x: 1D numpy array (should be length ~60).

    Returns:
        dict with keys: slope, max, min, range, ac1, ac5, ac10,
                        fft1, fft2, up_ratio
    """
    n = len(x)
    if n < 10:
        xmax = float(np.max(x)) if n > 0 else 0.0
        xmin = float(np.min(x)) if n > 0 else 0.0
        return {
            'slope60': 0.0, 'max60': xmax, 'min60': xmin, 'range60': xmax - xmin,
            'ac1': 0.0, 'ac5': 0.0, 'ac10': 0.0,
            'fft1': 0.0, 'fft2': 0.0,
            'up60': 0.5,
        }

    # 1. Linear slope
    t = np.arange(n, dtype=np.float64)
    slope = float(np.polyfit(t, x, 1)[0])

    # 2-4. Max, min, range
    xmax = float(np.max(x))
    xmin = float(np.min(x))
    xrange = xmax - xmin

    # 5-7. Autocorrelation at lags 1, 5, 10
    def _autocorr(arr, lag):
        if n <= lag + 3:
            return 0.0
        a1 = arr[:n - lag]
        a2 = arr[lag:]
        s1, s2 = np.std(a1), np.std(a2)
        if s1 < 1e-12 or s2 < 1e-12:
            return 0.0
        c = np.corrcoef(a1, a2)[0, 1]
        return float(c) if not np.isnan(c) else 0.0

    ac1 = _autocorr(x, 1)
    ac5 = _autocorr(x, 5)
    ac10 = _autocorr(x, 10)

    # 8-9. FFT dominant frequencies (skip DC)
    fft = np.fft.rfft(x)
    mag = np.abs(fft).copy()
    mag[0] = 0.0       # kill DC
    if len(mag) >= 3:
        top = np.argsort(mag)[-2:][::-1]
        fft1 = float(top[0] / n)
        fft2 = float(top[1] / n)
    elif len(mag) == 2:
        fft1 = float(1.0 / n)
        fft2 = 0.0
    else:
        fft1 = 0.0
        fft2 = 0.0

    # 10. Up-ratio
    up_ratio = float(np.mean(np.diff(x) > 0)) if n >= 2 else 0.5

    return {
        'slope60': slope, 'max60': xmax, 'min60': xmin, 'range60': xrange,
        'ac1': ac1, 'ac5': ac5, 'ac10': ac10,
        'fft1': fft1, 'fft2': fft2,
        'up60': up_ratio,
    }


# ---------------------------------------------------------------------------
# Per-stock shape feature computation
# ---------------------------------------------------------------------------

# Map raw column names -> short keys for feature naming
_COL_TO_KEY = {
    "ret1": "ret1",
    "成交量": "volume",
    "换手率": "turnover",
    "rsi_14": "rsi",
    "macd": "macd",
    "alpha20": "alpha20",
}

_STAT_NAMES = ["slope60", "max60", "min60", "range60",
               "ac1", "ac5", "ac10", "fft1", "fft2", "up60"]


def _compute_shape_features_for_stock(group: pd.DataFrame,
                                      factor_cols: List[str],
                                      window: int = 60) -> pd.DataFrame:
    """Compute shape features for a single stock across all dates.

    For each factor column, compute all 10 shape statistics over a
    trailing window of `window` days. Results are written back as
    new columns named {key}_{stat}.
    """
    group = group.sort_values("日期").copy()
    n = len(group)

    for col in factor_cols:
        if col not in group.columns:
            continue
        key = _COL_TO_KEY.get(col, col)
        series = group[col].values.astype(np.float64)

        # Pre-allocate
        stats_arrays = {
            stat: np.full(n, np.nan, dtype=np.float64)
            for stat in _STAT_NAMES
        }

        for i in range(window - 1, n):
            w = series[i - window + 1: i + 1]
            nan_frac = np.isnan(w).mean()
            if nan_frac > 0.33:
                continue
            # Fill internal NaN
            w = pd.Series(w).interpolate().bfill().ffill().values
            if len(w) < 10:
                continue

            s = _window_stats(w)
            for stat in _STAT_NAMES:
                stats_arrays[stat][i] = s[stat]

        # Write back to dataframe
        for stat in _STAT_NAMES:
            group[f"{key}_{stat}"] = stats_arrays[stat]

    return group


# ---------------------------------------------------------------------------
# Rank delta (momentum in cross-sectional rank)
# ---------------------------------------------------------------------------

def _add_rank_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank momentum features: change in alpha rank over time."""
    df = df.copy()
    for lag in [5, 20]:
        col = f"rank_alpha20_delta{lag}"
        if "rank_alpha20" in df.columns:
            df[col] = df.groupby("股票代码")["rank_alpha20"].transform(
                lambda x: x - x.shift(lag)
            )
    return df


# ---------------------------------------------------------------------------
# Main Phase 2 pipeline
# ---------------------------------------------------------------------------

def engineer_phase2_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Phase 2 feature engineering pipeline.

    Focus: Shape of past 60-day trajectories for 6 core factors.
    Target: ~85 low-dimensional features.

    Core factors: ret1, volume(成交量), turnover(换手率), RSI14(rsi_14),
                  MACD(macd), alpha20

    Shape stats (10 per factor, 60-day window):
        slope, max, min, range, ac1, ac5, ac10, fft1, fft2, up_ratio

    Args:
        df: Raw dataframe with columns [股票代码, 日期, 开盘, 收盘, 最高,
            最低, 成交量, 成交额, 振幅, 换手率, 涨跌幅]

    Returns:
        (df_with_features, list_of_feature_column_names)
    """
    print("=" * 60)
    print("Phase 2: Temporal Shape Features Pipeline")
    print("=" * 60)

    # ---- Step 1: Labels (skip if pre-computed, e.g. from Phase 4) ----
    if "label" not in df.columns:
        print("[1/8] Computing labels...")
        df = compute_labels(df)
    else:
        print("[1/8] Using pre-computed labels (multi-horizon ensemble)...")

    # ---- Step 2: Base factors ----
    print("[2/8] Base factors (returns, RSI, MACD, volume)...")
    df = add_basic_features(df)
    df = add_return_features(df, [1, 5, 10, 20, 60])
    df = add_momentum_features(df)     # rsi_14, macd, etc.
    df = add_volume_features(df)       # volume_ma, volume_ratio, etc.

    # ---- Step 3: Market state + Alpha ----
    print("[3/8] Market state & alpha features...")
    df = add_market_state_features(df)
    df = add_alpha_features(df)        # alpha5/10/20/60

    # ---- Step 4: 60-day shape features (THE CORE) ----
    print("[4/8] Computing 60-day shape features "
          "(slope/max/min/range/ac/fft/up_ratio)...")

    CORE_FACTORS = [
        "ret1",       # daily return
        "成交量",      # volume
        "换手率",      # turnover
        "rsi_14",     # RSI
        "macd",       # MACD
        "alpha20",    # 20-day excess return
    ]

    results = []
    n_stocks = df["股票代码"].nunique()
    for i, (stock_id, group) in enumerate(df.groupby("股票代码", sort=False)):
        group = _compute_shape_features_for_stock(group, CORE_FACTORS, window=60)
        results.append(group)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_stocks} stocks...")

    df = pd.concat(results, ignore_index=True)

    shape_cols = [c for c in df.columns
                  if any(c.endswith(f"_{s}") for s in _STAT_NAMES)
                  and any(c.startswith(_COL_TO_KEY.get(f, f))
                  for f in CORE_FACTORS)]
    print(f"  Shape features: {len(shape_cols)}")

    # ---- Step 5: Cross-sectional rank features ----
    print("[5/8] Cross-sectional rank features...")

    # Rank returns
    if "ret20" in df.columns:
        df["rank_ret20"] = df.groupby("日期")["ret20"].rank(pct=True)
    if "vol20" in df.columns:
        df["rank_volatility"] = df.groupby("日期")["vol20"].rank(pct=True)

    # Rank alpha
    for w in [5, 10, 20, 60]:
        col = f"alpha{w}"
        if col in df.columns:
            df[f"rank_{col}"] = df.groupby("日期")[col].rank(pct=True)

    # Rank delta (momentum of rank)
    df = _add_rank_delta_features(df)

    # ---- Step 6: Industry strength ----
    print("[6/8] Industry strength features...")
    if "sector_id" not in df.columns:
        df["sector_id"] = df["股票代码"].apply(_get_sector)

    for w in [5, 20, 60]:
        alpha_col = f"alpha{w}"
        ret_col = f"ret{w}"

        if alpha_col in df.columns:
            grand_mean = df.groupby("日期")[alpha_col].transform("mean")
            sector_mean = df.groupby(
                ["日期", "sector_id"]
            )[alpha_col].transform("mean")
            df[f"ind_strength{w}"] = sector_mean - grand_mean
            df[f"ind_rel_str{w}"] = df[alpha_col] - sector_mean

        if ret_col in df.columns:
            df[f"ind_rank_ret{w}"] = df.groupby(
                ["日期", "sector_id"]
            )[ret_col].rank(pct=True)

    # ---- Step 7: Collect feature columns ----
    print("[7/8] Collecting feature columns (whitelist approach)...")

    # Build the exact feature set we want
    FEATURE_WHITELIST = []

    # A. Shape features: {key}_{stat} for 6 factors x 10 stats
    for col in CORE_FACTORS:
        key = _COL_TO_KEY.get(col, col)
        for stat in _STAT_NAMES:
            FEATURE_WHITELIST.append(f"{key}_{stat}")

    # B. Cross-sectional rank features
    FEATURE_WHITELIST.extend([
        "rank_alpha5", "rank_alpha10", "rank_alpha20", "rank_alpha60",
        "rank_ret20", "rank_volatility",
        "rank_alpha20_delta5", "rank_alpha20_delta20",
    ])

    # C. Industry strength
    FEATURE_WHITELIST.extend([
        "ind_strength5", "ind_strength20", "ind_strength60",
        "ind_rel_str5", "ind_rel_str20",
        "ind_rank_ret5", "ind_rank_ret20", "ind_rank_ret60",
    ])

    # D. Base / reference factors
    FEATURE_WHITELIST.extend([
        "ret1", "成交量", "换手率", "rsi_14", "macd", "alpha20",
        "ret20", "vol20", "sector_id",
    ])

    # Keep only columns that actually exist
    feature_cols = [c for c in FEATURE_WHITELIST if c in df.columns]

    # ---- Step 8: Clean up ----
    print("[8/8] Cleaning NaN and finalizing...")

    if "label" in df.columns:
        before = len(df)
        df = df.dropna(subset=["label"])
        print(f"  Dropped {before - len(df)} rows with NaN labels")

    # Fill NaN in features
    df[feature_cols] = df[feature_cols].fillna(0.0)

    print(f"\n  Total features: {len(feature_cols)}")
    print(f"  Total samples:  {len(df)}")
    print(f"  Feature breakdown:")
    print(f"    Shape:      {len([c for c in feature_cols if any(c.endswith('_'+s) for s in _STAT_NAMES)])}")
    print(f"    Rank:       {len([c for c in feature_cols if c.startswith('rank_')])}")
    print(f"    Industry:   {len([c for c in feature_cols if c.startswith('ind_')])}")
    print(f"    Base:       {len([c for c in feature_cols if c in ['ret1','成交量','换手率','rsi_14','macd','alpha20','ret20','vol20','sector_id']])}")
    print("=" * 60)

    return df, feature_cols
