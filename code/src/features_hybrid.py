"""Hybrid feature engineering: safe rolling windows + cross-sectional ranks + sector.

Approach:
  1. Engineer 197 base features (158 Alpha + 39 TA-Lib) — reuses submit1 pipeline
  2. Build 20-day rolling windows per stock (POINT-IN-TIME: window ends at T,
     label uses T+1→T+5, ZERO overlap)
  3. Flatten windows → 20 × 197 = 3940 sequence features
  4. Add cross-sectional rank features at window end date (5 features)
  5. Add sector features (sector_id one-hot + sector ranks)
  6. Compute labels

This combines submit1's temporal pattern learning with our cross-sectional
alpha factors, while strictly avoiding target-window leakage.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

from features import (
    add_cross_sectional_features,
    add_sector_features,
    add_excess_return_ranks,
    add_market_state_features,
    compute_labels,
)


# ---------------------------------------------------------------------------
# Step 1: Base 197 features (from submit1)
# ---------------------------------------------------------------------------

def _engineer_39_features(df: pd.DataFrame) -> pd.DataFrame:
    """39 TA-Lib technical indicators (per-stock, from submit1)."""
    try:
        import talib
    except ImportError:
        raise ImportError("TA-Lib required.")

    df = df.copy()
    open_ = df["开盘"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    close = df["收盘"].astype(float)
    volume = df["成交量"].astype(float)

    df["sma_5"] = talib.SMA(close, timeperiod=5)
    df["sma_20"] = talib.SMA(close, timeperiod=20)
    df["ema_12"] = talib.EMA(close, timeperiod=12)
    df["ema_26"] = talib.EMA(close, timeperiod=26)
    df["ema_60"] = talib.EMA(close, timeperiod=60)

    macd_line, macd_signal_line, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal_line

    df["rsi"] = talib.RSI(close, timeperiod=14)
    df["kdj_k"], df["kdj_d"] = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowd_period=3)
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    df["boll_mid"], boll_upper, boll_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df["boll_std"] = (boll_upper - df["boll_mid"]) / 2

    df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    df["obv"] = talib.OBV(close, volume)

    df["volume_change"] = volume.pct_change()
    df["volume_ma_5"] = talib.SMA(volume, timeperiod=5)
    df["volume_ma_20"] = talib.SMA(volume, timeperiod=20)
    df["volume_ratio"] = df["volume_ma_5"] / (df["volume_ma_20"] + 1e-12)

    df["return_1"] = close.pct_change(1)
    df["return_5"] = close.pct_change(5)
    df["return_10"] = close.pct_change(10)
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_20"] = df["return_1"].rolling(20).std()

    df["high_low_spread"] = high - low
    df["open_close_spread"] = open_ - close
    df["high_close_spread"] = high - close
    df["low_close_spread"] = low - close

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def _engineer_158_features(df: pd.DataFrame) -> pd.DataFrame:
    """158 Alpha factors (from submit1)."""
    try:
        import talib
    except ImportError:
        raise ImportError("TA-Lib required.")

    df = df.copy()
    open_ = df["开盘"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    close = df["收盘"].astype(float)
    volume = df["成交量"].astype(float)
    vwap = df["成交额"] / (volume + 1e-12)

    features = []
    feature_names = []

    # 1. K-line features (9)
    features.extend([
        (close - open_) / (open_ + 1e-12),
        (high - low) / (open_ + 1e-12),
        (close - open_) / (high - low + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (open_ + 1e-12),
        (high - pd.concat([open_, close], axis=1).max(axis=1)) / (high - low + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (open_ + 1e-12),
        (pd.concat([open_, close], axis=1).min(axis=1) - low) / (high - low + 1e-12),
        (2 * close - high - low) / (open_ + 1e-12),
        (2 * close - high - low) / (high - low + 1e-12),
    ])
    feature_names.extend(["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"])

    # 2. Price features (4)
    features.extend([open_ / (close + 1e-12), high / (close + 1e-12),
                     low / (close + 1e-12), vwap / (close + 1e-12)])
    feature_names.extend(["OPEN0", "HIGH0", "LOW0", "VWAP0"])

    windows = [5, 10, 20, 30, 60]

    # 3. ROC (5)
    for w in windows:
        features.append(close.shift(w) / (close + 1e-12))
        feature_names.append(f"ROC{w}")

    # 4. MA (5)
    for w in windows:
        features.append(talib.SMA(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f"MA{w}")

    # 5. STD (5)
    for w in windows:
        features.append(talib.STDDEV(close, timeperiod=w) / (close + 1e-12))
        feature_names.append(f"STD{w}")

    # 6. Regression features (15)
    for w in windows:
        slope = talib.LINEARREG_SLOPE(close, timeperiod=w)
        features.append(slope / (close + 1e-12))
        feature_names.append(f"BETA{w}")
        n_idx = min(w, len(close))
        time_period_series = pd.Series(range(n_idx), index=close.index[:n_idx])
        rolling_corr = close.rolling(w).corr(time_period_series)
        features.append(rolling_corr ** 2)
        feature_names.append(f"RSQR{w}")
        intercept = talib.LINEARREG_INTERCEPT(close, timeperiod=w)
        predicted = slope * (w - 1) + intercept
        resi = close - predicted
        features.append(resi / (close + 1e-12))
        feature_names.append(f"RESI{w}")

    # 7. MAX (5)
    for w in windows:
        features.append(talib.MAX(high, timeperiod=w) / (close + 1e-12))
        feature_names.append(f"MAX{w}")

    # 8. MIN (5)
    for w in windows:
        features.append(talib.MIN(low, timeperiod=w) / (close + 1e-12))
        feature_names.append(f"MIN{w}")

    # 9. Quantile (10)
    for w in windows:
        features.append(close.rolling(w).quantile(0.8) / (close + 1e-12))
        feature_names.append(f"QTLU{w}")
    for w in windows:
        features.append(close.rolling(w).quantile(0.2) / (close + 1e-12))
        feature_names.append(f"QTLD{w}")

    # 10. Rank (5)
    for w in windows:
        features.append(close.rolling(w).rank(pct=True))
        feature_names.append(f"RANK{w}")

    # 11. RSV (5)
    for w in windows:
        min_low = low.rolling(w).min()
        max_high = high.rolling(w).max()
        features.append((close - min_low) / (max_high - min_low + 1e-12))
        feature_names.append(f"RSV{w}")

    # 12-17: More features (IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD,
    #                       SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD)
    for w in windows:
        features.append(high.rolling(w).apply(np.argmax, raw=True) / w)
        feature_names.append(f"IMAX{w}")
    for w in windows:
        features.append(low.rolling(w).apply(np.argmin, raw=True) / w)
        feature_names.append(f"IMIN{w}")
    for w in windows:
        imax = high.rolling(w).apply(np.argmax, raw=True)
        imin = low.rolling(w).apply(np.argmin, raw=True)
        features.append((imax - imin) / w)
        feature_names.append(f"IMXD{w}")

    log_volume = np.log(volume + 1)
    for w in windows:
        features.append(talib.CORREL(close, log_volume, timeperiod=w))
        feature_names.append(f"CORR{w}")

    close_ret = close / close.shift(1)
    volume_ret = volume / (volume.shift(1) + 1e-12)
    log_volume_ret = np.log(volume_ret + 1)
    for w in windows:
        corr_df = pd.concat([close_ret, log_volume_ret], axis=1).fillna(0)
        features.append(talib.CORREL(corr_df.iloc[:, 0], corr_df.iloc[:, 1], timeperiod=w))
        feature_names.append(f"CORD{w}")

    close_diff_pos = (close > close.shift(1))
    close_diff_neg = (close < close.shift(1))
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean())
        feature_names.append(f"CNTP{w}")
    for w in windows:
        features.append(close_diff_neg.rolling(w).mean())
        feature_names.append(f"CNTN{w}")
    for w in windows:
        features.append(close_diff_pos.rolling(w).mean() - close_diff_neg.rolling(w).mean())
        feature_names.append(f"CNTD{w}")

    close_diff_abs = (close - close.shift(1)).abs()
    close_diff_up = (close - close.shift(1)).clip(lower=0)
    close_diff_down = -(close - close.shift(1)).clip(upper=0)
    for w in windows:
        sum_up = close_diff_up.rolling(w).sum()
        sum_abs = close_diff_abs.rolling(w).sum()
        features.append(sum_up / (sum_abs + 1e-12))
        feature_names.append(f"SUMP{w}")
    for w in windows:
        sum_down = close_diff_down.rolling(w).sum()
        features.append(sum_down / (sum_abs + 1e-12))
        feature_names.append(f"SUMN{w}")
    for w in windows:
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f"SUMD{w}")

    for w in windows:
        features.append(talib.SMA(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f"VMA{w}")
    for w in windows:
        features.append(talib.STDDEV(volume, timeperiod=w) / (volume + 1e-12))
        feature_names.append(f"VSTD{w}")

    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in windows:
        mean_vwr = vol_weighted_ret.rolling(w).mean()
        std_vwr = vol_weighted_ret.rolling(w).std()
        features.append(std_vwr / (mean_vwr + 1e-12))
        feature_names.append(f"WVMA{w}")

    volume_diff_abs = (volume - volume.shift(1)).abs()
    volume_diff_up = (volume - volume.shift(1)).clip(lower=0)
    volume_diff_down = -(volume - volume.shift(1)).clip(upper=0)
    for w in windows:
        sum_abs = volume_diff_abs.rolling(w).sum()
        features.append(volume_diff_up.rolling(w).sum() / (sum_abs + 1e-12))
        feature_names.append(f"VSUMP{w}")
    for w in windows:
        features.append(volume_diff_down.rolling(w).sum() / (sum_abs + 1e-12))
        feature_names.append(f"VSUMN{w}")
    for w in windows:
        sum_up = volume_diff_up.rolling(w).sum()
        sum_down = volume_diff_down.rolling(w).sum()
        features.append((sum_up - sum_down) / (sum_abs + 1e-12))
        feature_names.append(f"VSUMD{w}")

    feature_df = pd.concat(features, axis=1)
    feature_df.columns = feature_names
    df = pd.concat([df, feature_df], axis=1)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df


def engineer_197_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full 158+39 = 197 base features."""
    df = df.copy()
    df = df.sort_values(["股票代码", "日期"]).reset_index(drop=True)

    # Run 158 and 39 per stock
    results = []
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()
        group = _engineer_158_features(group)
        group = _engineer_39_features(group)
        results.append(group)

    return pd.concat(results, ignore_index=True)


# ---------------------------------------------------------------------------
# Step 2: Safe rolling windows (point-in-time, no leakage)
# ---------------------------------------------------------------------------

BASE_COLS_197 = [
    "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌额", "换手率", "涨跌幅",
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
    "OPEN0", "HIGH0", "LOW0", "VWAP0",
    "ROC5", "ROC10", "ROC20", "ROC30", "ROC60",
    "MA5", "MA10", "MA20", "MA30", "MA60",
    "STD5", "STD10", "STD20", "STD30", "STD60",
    "BETA5", "BETA10", "BETA20", "BETA30", "BETA60",
    "RSQR5", "RSQR10", "RSQR20", "RSQR30", "RSQR60",
    "RESI5", "RESI10", "RESI20", "RESI30", "RESI60",
    "MAX5", "MAX10", "MAX20", "MAX30", "MAX60",
    "MIN5", "MIN10", "MIN20", "MIN30", "MIN60",
    "QTLU5", "QTLU10", "QTLU20", "QTLU30", "QTLU60",
    "QTLD5", "QTLD10", "QTLD20", "QTLD30", "QTLD60",
    "RANK5", "RANK10", "RANK20", "RANK30", "RANK60",
    "RSV5", "RSV10", "RSV20", "RSV30", "RSV60",
    "IMAX5", "IMAX10", "IMAX20", "IMAX30", "IMAX60",
    "IMIN5", "IMIN10", "IMIN20", "IMIN30", "IMIN60",
    "IMXD5", "IMXD10", "IMXD20", "IMXD30", "IMXD60",
    "CORR5", "CORR10", "CORR20", "CORR30", "CORR60",
    "CORD5", "CORD10", "CORD20", "CORD30", "CORD60",
    "CNTP5", "CNTP10", "CNTP20", "CNTP30", "CNTP60",
    "CNTN5", "CNTN10", "CNTN20", "CNTN30", "CNTN60",
    "CNTD5", "CNTD10", "CNTD20", "CNTD30", "CNTD60",
    "SUMP5", "SUMP10", "SUMP20", "SUMP30", "SUMP60",
    "SUMN5", "SUMN10", "SUMN20", "SUMN30", "SUMN60",
    "SUMD5", "SUMD10", "SUMD20", "SUMD30", "SUMD60",
    "VMA5", "VMA10", "VMA20", "VMA30", "VMA60",
    "VSTD5", "VSTD10", "VSTD20", "VSTD30", "VSTD60",
    "WVMA5", "WVMA10", "WVMA20", "WVMA30", "WVMA60",
    "VSUMP5", "VSUMP10", "VSUMP20", "VSUMP30", "VSUMP60",
    "VSUMN5", "VSUMN10", "VSUMN20", "VSUMN30", "VSUMN60",
    "VSUMD5", "VSUMD10", "VSUMD20", "VSUMD30", "VSUMD60",
    "sma_5", "sma_20", "ema_12", "ema_26", "rsi", "macd", "macd_signal",
    "volume_change", "obv", "volume_ma_5", "volume_ma_20", "volume_ratio",
    "kdj_k", "kdj_d", "kdj_j", "boll_mid", "boll_std", "atr_14", "ema_60",
    "volatility_10", "volatility_20", "return_1", "return_5", "return_10",
    "high_low_spread", "open_close_spread", "high_close_spread", "low_close_spread",
]


def build_safe_windows(
    df: pd.DataFrame,
    seq_len: int = 20,
    min_history: int = 120,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Build safe rolling windows — ZERO overlap with target.

    For each stock on date T:
      - Window: features from [T-seq_len+1, T] (historical only)
      - Label: (Open_T+5 - Open_T+1) / Open_T+1 (future only)
      - No overlap between window data and label data.

    Args:
        df: DataFrame with 197 features already computed + labels.
        seq_len: Number of days in each rolling window.

    Returns:
        (windows_df, seq_feature_cols, extra_feature_cols)
        windows_df has columns: [股票代码, 日期, label, seq_0, seq_1, ..., seq_N]
    """
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"])

    # Determine which base columns exist
    base_cols = [c for c in BASE_COLS_197 if c in df.columns]
    print(f"  Base columns available for windowing: {len(base_cols)}/{len(BASE_COLS_197)}")

    # Filter stocks with enough history
    stock_counts = df.groupby("股票代码").size()
    valid_stocks = stock_counts[stock_counts >= min_history].index
    df = df[df["股票代码"].isin(valid_stocks)]

    all_windows = []  # List of dicts: {股票代码, 日期, label, seq_0..seq_N}

    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").reset_index(drop=True)
        n = len(group)

        if n < seq_len + 5:  # Need seq_len history + 5 future for label
            continue

        feature_vals = group[base_cols].values.astype(np.float32)  # (T, F)
        labels = group["label"].values.astype(np.float32)
        dates = group["日期"].values

        # Build windows: window ends at index i (date T), label at i
        # Label was already computed as (Open_T+5 - Open_T+1) / Open_T+1
        for i in range(seq_len - 1, n):
            # Window: [i - seq_len + 1, i] — purely historical
            window = feature_vals[i - seq_len + 1 : i + 1]  # (seq_len, F)

            # Label at this position (already point-in-time)
            label = labels[i]
            if np.isnan(label):
                continue

            # Verify: ensure future data is available for this label
            # (label was computed with shift(-1) and shift(-5) on open price)
            # The label at index i already encodes T+1→T+5 info

            row = {
                "股票代码": stock_id,
                "日期": dates[i],
                "label": label,
            }
            # Flatten window into seq_0..seq_N features
            for d in range(seq_len):
                for f_idx, col_name in enumerate(base_cols):
                    row[f"seq_d{d}_f{f_idx}"] = window[d, f_idx]

            all_windows.append(row)

    windows_df = pd.DataFrame(all_windows)
    seq_feature_cols = [c for c in windows_df.columns if c.startswith("seq_")]

    print(f"  Windows built: {len(windows_df)} samples")
    print(f"  Sequence features: {len(seq_feature_cols)} ({seq_len} days × {len(base_cols)} features)")

    return windows_df, seq_feature_cols


# ---------------------------------------------------------------------------
# Main hybrid pipeline
# ---------------------------------------------------------------------------

def engineer_hybrid_features(
    df: pd.DataFrame,
    seq_len: int = 20,
    min_history: int = 120,
) -> Tuple[pd.DataFrame, List[str]]:
    """Full hybrid feature pipeline.

    1. Compute 197 base features (submit1 style)
    2. Compute labels (point-in-time)
    3. Build safe rolling windows → flattened sequence features
    4. Add cross-sectional rank features
    5. Add sector features
    6. Add excess return rank features

    Returns:
        (df_with_all_features, list_of_feature_column_names)
    """
    print("=" * 60)
    print("Hybrid Feature Pipeline (safe windows + cross-sectional + sector)")
    print("=" * 60)

    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"])

    # Step 1: Compute labels
    print("[1/7] Computing labels...")
    df = compute_labels(df)

    # Step 2: Engineer 197 base features
    print("[2/7] Engineering 197 base features (this may take a few minutes)...")
    df = engineer_197_features(df)

    # Step 3: Build safe rolling windows
    print("[3/7] Building safe rolling windows...")
    windows_df, seq_cols = build_safe_windows(df, seq_len=seq_len, min_history=min_history)

    # After building windows, we have a compact DataFrame
    # Merge back to get full context for cross-sectional features
    # Keep only window end dates for point-in-time features
    df_pt = windows_df[["股票代码", "日期", "label"]].copy()

    # Step 4: Add cross-sectional rank features (point-in-time at window end)
    print("[4/7] Adding cross-sectional rank features...")
    # We need ret20, ret60, etc. at the window end dates
    # Get these from the original df
    pt_cols_needed = ["股票代码", "日期"] + [c for c in df.columns if c in {
        "ret1", "ret5", "ret10", "ret20", "ret60", "ret120",
        "成交量", "换手率", "vol20", "vol10", "vol60",
    }]
    cols_avail = [c for c in pt_cols_needed if c in df.columns]
    df_pt = df_pt.merge(df[cols_avail], on=["股票代码", "日期"], how="left")

    df_pt = add_cross_sectional_features(df_pt)

    # Step 5: Add market state features
    print("[5/7] Adding market state features...")
    df_pt = add_market_state_features(df_pt)
    df_pt = add_excess_return_ranks(df_pt)

    # Step 6: Add sector features
    print("[6/7] Adding sector features...")
    df_pt = add_sector_features(df_pt)

    # Step 7: Merge point-in-time features back to windows
    print("[7/7] Merging all features...")
    # Columns to merge from df_pt: all except 股票代码, 日期, label (already in windows_df)
    pt_feature_cols = [c for c in df_pt.columns if c not in {"股票代码", "日期", "label"}]
    # Also exclude the base columns that are already in seq features
    pt_feature_cols = [c for c in pt_feature_cols if c not in set(BASE_COLS_197)]

    windows_df = windows_df.merge(
        df_pt[["股票代码", "日期"] + pt_feature_cols],
        on=["股票代码", "日期"],
        how="left",
    )

    # Drop rows with NaN labels
    before = len(windows_df)
    windows_df = windows_df.dropna(subset=["label"])
    print(f"  Dropped {before - len(windows_df)} rows with NaN labels")

    # Fill NaN features
    all_feature_cols = seq_cols + pt_feature_cols
    windows_df[all_feature_cols] = windows_df[all_feature_cols].fillna(0.0)

    print(f"\n  Sequence features:  {len(seq_cols)}")
    print(f"  Point-in-time features: {len(pt_feature_cols)}")
    print(f"  Total features:     {len(all_feature_cols)}")
    print(f"  Total samples:      {len(windows_df)}")
    print("=" * 60)

    return windows_df, all_feature_cols
