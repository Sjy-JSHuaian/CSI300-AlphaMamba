"""Feature engineering pipeline for Phase 1 LightGBM Ranker.

Produces ~260 features organized by category:
  1. Basic price       (9)
  2. Returns           (7)
  3. Volatility        (4)
  4. Trend             (6)
  5. Momentum (TA-Lib) (9)
  6. Volume            (4)
  7. Market state      (8)
  8. Alpha factors     (4)
  9. Temporal summary  (165)
 10. Cross-sectional   (5)
 11. Rank alpha        (4)
 12. Sector            (7)
 13. Industry strength (15)
 14. Excess return     (4)
 15. Price patterns    (6)
 16. Higher-order stats(4)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def _safe_divide(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    """Divide two series, replacing inf/nan with fill value."""
    result = a / b.replace(0, np.nan)
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result.fillna(fill, inplace=True)
    return result


# ---------------------------------------------------------------------------
# Label computation
# ---------------------------------------------------------------------------

def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute competition label: (Open_T+5 - Open_T+1) / Open_T+1.

    Computed per-stock. NaN where T+1 or T+5 has no open (suspension).
    """
    df = df.copy()
    results = []
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()
        group["open_t1"] = group["开盘"].shift(-1)
        group["open_t5"] = group["开盘"].shift(-5)
        group["label"] = (group["open_t5"] - group["open_t1"]) / group["open_t1"].replace(0, np.nan)
        group.drop(columns=["open_t1", "open_t5"], inplace=True)
        results.append(group)
    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# 1. Basic price features
# ---------------------------------------------------------------------------

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric types for OHLCV columns."""
    df = df.copy()
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "换手率", "涨跌幅"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Additional derived basic features
    df["amount"] = df.get("成交额", 0)
    df["turnover"] = df.get("换手率", 0)
    df["amplitude"] = df.get("振幅", 0)
    return df


# ---------------------------------------------------------------------------
# 2. Return features
# ---------------------------------------------------------------------------

def add_return_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """retN = close / close.shift(N) - 1, computed per-stock."""
    df = df.copy()
    for w in windows:
        df[f"ret{w}"] = df.groupby("股票代码")["收盘"].transform(
            lambda x: x / x.shift(w) - 1
        )
    return df


# ---------------------------------------------------------------------------
# 3. Volatility features
# ---------------------------------------------------------------------------

def add_volatility_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Rolling std of ret1."""
    df = df.copy()
    if "ret1" not in df.columns:
        df["ret1"] = df.groupby("股票代码")["收盘"].transform(lambda x: x.pct_change(1))
    for w in windows:
        df[f"vol{w}"] = df.groupby("股票代码")["ret1"].transform(
            lambda x: x.rolling(w, min_periods=max(5, w // 2)).std()
        )
    return df


# ---------------------------------------------------------------------------
# 4. Trend features (MA + EMA)
# ---------------------------------------------------------------------------

def add_trend_features(
    df: pd.DataFrame,
    ma_windows: List[int],
    ema_windows: List[int],
) -> pd.DataFrame:
    """Moving averages and EMA crossover signals."""
    df = df.copy()
    close = df["收盘"]
    for w in ma_windows:
        df[f"ma{w}"] = df.groupby("股票代码")["收盘"].transform(
            lambda x: x.rolling(w, min_periods=max(3, w // 2)).mean()
        )
        df[f"ma_ratio{w}"] = _safe_divide(close, df[f"ma{w}"]) - 1
    for w in ema_windows:
        df[f"ema{w}"] = df.groupby("股票代码")["收盘"].transform(
            lambda x: x.ewm(span=w, adjust=False).mean()
        )
    if 12 in ema_windows and 26 in ema_windows:
        df["ema_cross"] = df["ema12"] - df["ema26"]
    return df


# ---------------------------------------------------------------------------
# 5. Momentum features (TA-Lib)
# ---------------------------------------------------------------------------

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """TA-Lib: RSI, MACD, KDJ, CCI, ATR, WILLR. Computed per-stock."""
    try:
        import talib
    except ImportError:
        raise ImportError("TA-Lib required. Install: pip install TA-Lib")

    df = df.copy()
    results = []
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()
        close = group["收盘"].astype(float).values
        high = group["最高"].astype(float).values
        low = group["最低"].astype(float).values

        n = len(group)

        group["rsi_14"] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        group["macd"] = macd
        group["macd_signal"] = macd_signal
        group["macd_hist"] = macd - macd_signal

        k, d = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
        group["kdj_k"] = k
        group["kdj_d"] = d
        group["kdj_j"] = 3 * k - 2 * d

        group["cci_14"] = talib.CCI(high, low, close, timeperiod=14)
        group["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
        group["atr_ratio"] = group["atr_14"] / (close + 1e-12)
        group["willr_14"] = talib.WILLR(high, low, close, timeperiod=14)

        results.append(group)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# 6. Volume features
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """OBV, volume ratio, volume MAs, volume change. Per-stock."""
    try:
        import talib
    except ImportError:
        raise ImportError("TA-Lib required.")

    df = df.copy()
    results = []
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()
        close = group["收盘"].astype(float).values
        volume = group["成交量"].astype(float).values

        group["obv"] = talib.OBV(close, volume)
        vol_s = pd.Series(volume, index=group.index)
        group["volume_ma5"] = vol_s.rolling(5, min_periods=3).mean().values
        group["volume_ma20"] = vol_s.rolling(20, min_periods=5).mean().values
        group["volume_ratio"] = _safe_divide(
            pd.Series(group["volume_ma5"].values, index=group.index),
            pd.Series(group["volume_ma20"].values, index=group.index),
        ).values
        group["volume_change"] = vol_s.pct_change(1).values

        results.append(group)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# 7. Cross-sectional rank features
# ---------------------------------------------------------------------------

def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily percentile ranks (0~1). Captures relative strength within universe."""
    df = df.copy()

    rank_map = {
        "ret20": "rank_ret20",
        "ret60": "rank_ret60",
        "成交量": "rank_volume",
        "换手率": "rank_turnover",
        "vol20": "rank_volatility",
    }

    for src, dst in rank_map.items():
        if src in df.columns:
            df[dst] = df.groupby("日期")[src].rank(pct=True)

    return df


# ---------------------------------------------------------------------------
# 7b. Sector / Industry features
# ---------------------------------------------------------------------------

def _get_sector( stock_code: str) -> int:
    """Infer sector from CSI 300 stock code prefix.

    Returns:
        0 = Shanghai Main (600/601/603/605)
        1 = Shenzhen Main (000/001/002/003)
        2 = STAR Market (688) — 科创板
        3 = ChiNext (300/301) — 创业板
    """
    code = str(stock_code).zfill(6)
    prefix = int(code[:3])
    if 600 <= prefix <= 609:
        return 0
    if 0 <= prefix <= 3:
        return 1
    if prefix == 688:
        return 2
    if 300 <= prefix <= 301:
        return 3
    # Fallback
    if prefix < 600:
        return 1
    return 0


def add_sector_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sector encoding + within-sector rank features.

    Adds:
      - sector_id: label-encoded sector
      - sector_rank_{factor}: percentile rank within same sector
      - sector_ret{N}: sector average return
    """
    df = df.copy()

    # 1. Sector ID
    df["sector_id"] = df["股票代码"].apply(_get_sector)

    # 2. Within-sector ranks (captures relative strength vs sector peers)
    rank_factors = {
        "ret20": "sector_rank_ret20",
        "ret60": "sector_rank_ret60",
        "成交量": "sector_rank_volume",
        "vol20": "sector_rank_volatility",
    }
    for src, dst in rank_factors.items():
        if src in df.columns:
            df[dst] = df.groupby(["日期", "sector_id"])[src].rank(pct=True)

    # 3. Sector average returns (macro context per sector)
    for w in [5, 20]:
        ret_col = f"ret{w}"
        if ret_col in df.columns:
            df[f"sector_ret{w}"] = df.groupby(["日期", "sector_id"])[ret_col].transform("mean")

    return df


# ---------------------------------------------------------------------------
# 7c. Excess return rank features
# ---------------------------------------------------------------------------

def add_excess_return_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional ranks of excess returns.

    Requires market state features (idx300_ret*) to be computed first.
    """
    df = df.copy()

    if "ret20" in df.columns and "idx300_ret20" in df.columns:
        df["excess_ret20"] = df["ret20"] - df["idx300_ret20"]
        df["rank_excess_ret20"] = df.groupby("日期")["excess_ret20"].rank(pct=True)

    if "ret60" in df.columns and "idx300_ret60" in df.columns:
        df["excess_ret60"] = df["ret60"] - df["idx300_ret60"]
        df["rank_excess_ret60"] = df.groupby("日期")["excess_ret60"].rank(pct=True)

    return df


# ---------------------------------------------------------------------------
# 8. Market state features
# ---------------------------------------------------------------------------

def add_market_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Index proxy returns computed from stock universe.

    - idx300: equal-weighted all stocks (HS300 proxy)
    - sh_proxy: top 70% by volume stocks (Shanghai proxy)
    - zz500_proxy: bottom 30% by volume stocks (ZZ500 proxy)
    - excess_ret20: stock return minus market return
    """
    df = df.copy()

    # HS300 proxy: equal-weight avg of all stocks
    for w in [5, 10, 20, 60]:
        ret_col = f"ret{w}"
        if ret_col in df.columns:
            df[f"idx300_ret{w}"] = df.groupby("日期")[ret_col].transform("mean")

    # Volume-based proxy split
    if "成交量" in df.columns:
        stock_avg_vol = df.groupby("股票代码")["成交量"].transform("mean")
        vol_70 = stock_avg_vol.quantile(0.7)

        # Shanghai proxy: top 70% by volume
        sh_mask = stock_avg_vol >= vol_70
        for w in [5, 20]:
            ret_col = f"ret{w}"
            if ret_col in df.columns:
                # Compute the mean of ret for stocks in sh_mask for each date
                df_ret = df[ret_col].copy()
                df_ret[~sh_mask] = np.nan
                df[f"sh_ret{w}"] = df_ret.groupby(df["日期"]).transform("mean")

        # ZZ500 proxy: bottom 30% by volume
        zz_mask = stock_avg_vol < vol_70
        for w in [5, 20]:
            ret_col = f"ret{w}"
            if ret_col in df.columns:
                df_ret = df[ret_col].copy()
                df_ret[~zz_mask] = np.nan
                df[f"zz500_ret{w}"] = df_ret.groupby(df["日期"]).transform("mean")

    # Excess return
    if "ret20" in df.columns and "idx300_ret20" in df.columns:
        df["excess_ret20"] = df["ret20"] - df["idx300_ret20"]

    return df


# ---------------------------------------------------------------------------
# 9. Price pattern features
# ---------------------------------------------------------------------------

def add_price_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Price spread and candlestick pattern features."""
    df = df.copy()
    df["high_low_spread"] = df["最高"] - df["最低"]
    df["open_close_spread"] = df["开盘"] - df["收盘"]
    df["high_close_spread"] = df["最高"] - df["收盘"]
    df["low_close_spread"] = df["最低"] - df["收盘"]
    df["high_low_pct"] = _safe_divide(df["high_low_spread"], df["开盘"])
    df["body_pct"] = _safe_divide((df["收盘"] - df["开盘"]).abs(), df["开盘"])
    return df


# ---------------------------------------------------------------------------
# 10. Higher-order statistics
# ---------------------------------------------------------------------------

def add_higher_order_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling skewness and kurtosis of returns and volume changes."""
    df = df.copy()

    cols_to_add = []
    if "ret1" in df.columns:
        cols_to_add.extend(["skew_ret20", "kurt_ret20"])
    if "volume_change" in df.columns:
        cols_to_add.extend(["skew_vol20", "kurt_vol20"])

    for col in cols_to_add:
        df[col] = 0.0

    # Compute per-stock, assign back by index
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期")
        idx = group.index
        if "ret1" in group.columns:
            ret = group["ret1"]
            df.loc[idx, "skew_ret20"] = ret.rolling(20, min_periods=10).skew().values
            df.loc[idx, "kurt_ret20"] = ret.rolling(20, min_periods=10).kurt().values
        if "volume_change" in group.columns:
            vc = group["volume_change"]
            df.loc[idx, "skew_vol20"] = vc.rolling(20, min_periods=10).skew().values
            df.loc[idx, "kurt_vol20"] = vc.rolling(20, min_periods=10).kurt().values

    return df


# ---------------------------------------------------------------------------
# 11. Alpha features (excess return vs market)
# ---------------------------------------------------------------------------

def add_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute alpha (excess return) for 5/10/20/60 day windows.

    alpha{w} = ret{w} - idx300_ret{w}

    Requires return features and market state features computed first.
    """
    df = df.copy()
    for w in [5, 10, 20, 60]:
        ret_col = f"ret{w}"
        idx_col = f"idx300_ret{w}"
        if ret_col in df.columns and idx_col in df.columns:
            df[f"alpha{w}"] = df[ret_col] - df[idx_col]
    return df


# ---------------------------------------------------------------------------
# 12. Temporal summary features
# ---------------------------------------------------------------------------

def _compute_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling linear regression slope per-stock."""
    try:
        import talib
        return talib.LINEARREG_SLOPE(series.values, timeperiod=window)
    except Exception:
        # Fallback: numpy polyfit slope
        def _slope(x):
            x = x[~np.isnan(x)]
            if len(x) < max(3, window // 2):
                return 0.0
            t = np.arange(len(x))
            return np.polyfit(t, x, 1)[0]
        return series.rolling(window, min_periods=max(3, window // 2)).apply(_slope, raw=True)


def add_temporal_summary_features(
    df: pd.DataFrame,
    windows: list = None,
) -> pd.DataFrame:
    """Compute temporal summary statistics over rolling windows.

    For each base factor in factor_map, compute mean/std/max/min/slope
    over each window in `windows`. All computations are per-stock.

    Base factors:
        ret1, ret5, ret10, ret20, volume(成交量), turnover(换手率),
        rsi_14, macd, alpha5, alpha20, alpha60

    Statistics: mean, std, max, min, slope
    Windows: 5, 20, 60

    Naming: {factor_key}_ts_{stat}{w} (e.g., ret1_ts_mean5, alpha20_ts_slope60)
    """
    if windows is None:
        windows = [5, 20, 60]

    df = df.copy()

    # Map friendly keys → actual column names
    factor_map = {
        "ret1": "ret1",
        "ret5": "ret5",
        "ret10": "ret10",
        "ret20": "ret20",
        "volume": "成交量",
        "turnover": "换手率",
        "rsi": "rsi_14",
        "macd": "macd",
        "alpha5": "alpha5",
        "alpha20": "alpha20",
        "alpha60": "alpha60",
    }

    # Only compute for factors that exist in the dataframe
    active_factors = {
        key: col for key, col in factor_map.items() if col in df.columns
    }
    print(f"  Temporal summary: {len(active_factors)} factors × "
          f"5 stats × {len(windows)} windows = "
          f"{len(active_factors) * 5 * len(windows)} features")

    # Compute per-stock
    results = []
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()

        for factor_key, col_name in active_factors.items():
            series = group[col_name].astype(float)
            for w in windows:
                roll = series.rolling(w, min_periods=max(3, w // 2))
                group[f"{factor_key}_ts_mean{w}"] = roll.mean().values
                group[f"{factor_key}_ts_std{w}"] = roll.std().values
                group[f"{factor_key}_ts_max{w}"] = roll.max().values
                group[f"{factor_key}_ts_min{w}"] = roll.min().values
                group[f"{factor_key}_ts_slope{w}"] = _compute_slope(series, w)

        results.append(group)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# 13. Rank alpha features
# ---------------------------------------------------------------------------

def add_rank_alpha_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile ranks of alpha factors.

    Adds rank_alpha5, rank_alpha10, rank_alpha20, rank_alpha60.
    """
    df = df.copy()
    for w in [5, 10, 20, 60]:
        col = f"alpha{w}"
        if col in df.columns:
            df[f"rank_{col}"] = df.groupby("日期")[col].rank(pct=True)
    return df


# ---------------------------------------------------------------------------
# 14. Industry strength features
# ---------------------------------------------------------------------------

def add_industry_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """Industry/sector strength factors built on board classification.

    Adds:
      - ind_strength{w}: sector average alpha minus overall market average
        (how strong is this sector relative to others)
      - ind_rel_str{w}: stock alpha minus sector average alpha
        (how strong is this stock within its sector)
      - ind_rank_ret{w}: within-sector percentile rank of returns
    """
    df = df.copy()

    if "sector_id" not in df.columns:
        df["sector_id"] = df["股票代码"].apply(_get_sector)

    windows = [5, 20, 60]

    for w in windows:
        alpha_col = f"alpha{w}"
        ret_col = f"ret{w}"

        # Industry strength: sector avg alpha vs grand mean alpha
        if alpha_col in df.columns:
            grand_mean = df.groupby("日期")[alpha_col].transform("mean")
            sector_mean = df.groupby(["日期", "sector_id"])[alpha_col].transform("mean")
            df[f"ind_strength{w}"] = sector_mean - grand_mean

            # Stock relative strength within sector
            df[f"ind_rel_str{w}"] = df[alpha_col] - sector_mean

        # Within-sector return rank
        if ret_col in df.columns:
            df[f"ind_rank_ret{w}"] = df.groupby(
                ["日期", "sector_id"]
            )[ret_col].rank(pct=True)

    return df


# ---------------------------------------------------------------------------
# Label discretization (for LightGBM lambdarank)
# ---------------------------------------------------------------------------

def discretize_labels(df: pd.DataFrame, n_bins: int = 30) -> pd.DataFrame:
    """Convert continuous label to discrete integer bins per day.

    LightGBM lambdarank expects integer relevance scores. We discretize
    within each day: rank the continuous labels, then bin into n_bins groups.
    """
    df = df.copy()
    # Pre-allocate relevance column
    df["relevance"] = 1

    for date, group in df.groupby("日期", sort=False):
        if len(group) < 2:
            df.loc[group.index, "relevance"] = 1
            continue
        ranks = group["label"].rank(method="dense", ascending=True)
        max_rank = ranks.max()
        if max_rank <= n_bins:
            df.loc[group.index, "relevance"] = ranks.astype(int).values - 1
        else:
            df.loc[group.index, "relevance"] = np.digitize(
                ranks, np.linspace(0, max_rank, n_bins + 1)
            ) - 1

    df["relevance"] = df["relevance"].fillna(0).astype(int).clip(lower=0)
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    compute_labels_flag: bool = True,
    windows_return: List[int] = None,
    windows_vol: List[int] = None,
    windows_trend: List[int] = None,
    ema_windows: List[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Main feature engineering pipeline.

    Args:
        df: Raw dataframe with columns [股票代码, 日期, 开盘, 收盘, 最高,
            最低, 成交量, 成交额, 振幅, 换手率, 涨跌幅]
        compute_labels_flag: Whether to compute labels (False for pure inference).

    Returns:
        (df_with_features, list_of_feature_column_names)
    """
    if windows_return is None:
        windows_return = [1, 3, 5, 10, 20, 60, 120]
    if windows_vol is None:
        windows_vol = [5, 10, 20, 60]
    if windows_trend is None:
        windows_trend = [5, 10, 20, 60]
    if ema_windows is None:
        ema_windows = [12, 26]

    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)

    # Step 1: Compute labels first
    if compute_labels_flag:
        print("[1/16] Computing labels...")
        df = compute_labels(df)

    # Step 2: Basic
    print("[2/16] Basic price features...")
    df = add_basic_features(df)

    # Step 3: Returns
    print("[3/16] Return features...")
    df = add_return_features(df, windows_return)

    # Step 4: Volatility
    print("[4/16] Volatility features...")
    df = add_volatility_features(df, windows_vol)

    # Step 5: Trend
    print("[5/16] Trend features (MA, EMA)...")
    df = add_trend_features(df, windows_trend, ema_windows)

    # Step 6: Momentum (TA-Lib)
    print("[6/16] Momentum features (TA-Lib)...")
    df = add_momentum_features(df)

    # Step 7: Volume
    print("[7/16] Volume features...")
    df = add_volume_features(df)

    # Step 8: Market state (must be before alpha)
    print("[8/16] Market state features...")
    df = add_market_state_features(df)

    # Step 9: Alpha (excess returns vs market)
    print("[9/16] Alpha features (excess vs market)...")
    df = add_alpha_features(df)

    # Step 10: Temporal summary features
    print("[10/16] Temporal summary features (mean/std/max/min/slope)...")
    df = add_temporal_summary_features(df, windows=[5, 20, 60])

    # Step 11: Cross-sectional ranks
    print("[11/16] Cross-sectional rank features...")
    df = add_cross_sectional_features(df)

    # Step 12: Rank alpha
    print("[12/16] Rank alpha features...")
    df = add_rank_alpha_features(df)

    # Step 13: Sector features
    print("[13/16] Sector features...")
    df = add_sector_features(df)

    # Step 14: Industry strength
    print("[14/16] Industry strength features...")
    df = add_industry_strength_features(df)

    # Step 15: Excess return ranks (after market state)
    print("[15/16] Excess return rank features...")
    df = add_excess_return_ranks(df)

    # Step 16: Price patterns + higher-order
    print("[16/16] Price pattern & higher-order features...")
    df = add_price_pattern_features(df)
    df = add_higher_order_features(df)

    # Collect feature columns (exclude ID/label columns)
    exclude_cols = {
        "股票代码", "日期", "instrument", "datetime",
        "label", "open_t1", "open_t5", "relevance",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Drop rows with NaN labels
    if "label" in df.columns:
        before = len(df)
        df = df.dropna(subset=["label"])
        print(f"Dropped {before - len(df)} rows with NaN labels (suspension/delisting)")

    # Fill remaining NaN in features
    df[feature_cols] = df[feature_cols].fillna(0.0)

    print(f"Total features: {len(feature_cols)}")
    print(f"Total samples:  {len(df)}")
    print("=" * 60)

    return df, feature_cols
