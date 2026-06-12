"""Phase 4: Multi-Horizon Label Ensemble + Rank Smoothing + Tail Amplification.

Reuses Phase 3's 102-dim feature space with improved training targets:
  1. Multi-horizon ensemble label (T+3, T+5, T+10) — cross-sectional ranks
  2. Tail-amplified discretization — top 10% → 20, top 10-20% → 10
  3. Phase 3 feature pipeline (unchanged)

Key insight: RankIC ↑177% from Phase 3 but WF NDCG@5 didn't follow.
The fix is NOT more features — it's better labels that smooth the
4-day noise window using longer-horizon information.

Usage:
    from features_phase4 import engineer_phase4_features, discretize_labels_tail

    df, feature_cols = engineer_phase4_features(df)
    df = discretize_labels_tail(df)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# Reuse Phase 3 pipeline (will auto-detect pre-computed labels)
from features_phase3 import engineer_phase3_features


# ======================================================================
# Multi-Horizon Ensemble Label
# ======================================================================

def compute_ensemble_label(df: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-horizon rank-ensemble label.

    Horizons: T+3, T+5, T+10
    Formula per horizon:
        label_h = (Open[T+h] - Open[T+1]) / Open[T+1]
    Ensemble:
        label = 0.2 * rank(label_3) + 0.5 * rank(label_5) + 0.3 * rank(label_10)

    All ranks are cross-sectional percentile (0-1) per date.
    This uses longer-horizon structure (T+10 trend info) while respecting
    that the competition target is T+5.

    Returns df with new 'label' column (0-1 continuous, higher = better).
    """
    print("=" * 60)
    print("Phase 4: Multi-Horizon Label Ensemble")
    print("=" * 60)

    df = df.copy()
    HORIZONS = [3, 5, 10]
    WEIGHTS = {3: 0.2, 5: 0.5, 10: 0.3}

    # Compute per-horizon labels per stock
    results = []
    for stock_id, group in df.groupby("股票代码", sort=False):
        group = group.sort_values("日期").copy()
        open_t1 = group["开盘"].shift(-1)

        for h in HORIZONS:
            open_th = group["开盘"].shift(-h)
            group[f"_lh{h}"] = (open_th - open_t1) / open_t1.replace(0, np.nan)

        results.append(group)

    df = pd.concat(results, ignore_index=True)

    # Cross-sectional rank each horizon (0-1, higher = stronger return)
    for h in HORIZONS:
        df[f"_rh{h}"] = df.groupby("日期")[f"_lh{h}"].rank(pct=True)

    # Weighted ensemble
    df["label"] = (
        WEIGHTS[3] * df["_rh3"].fillna(0.5) +
        WEIGHTS[5] * df["_rh5"].fillna(0.5) +
        WEIGHTS[10] * df["_rh10"].fillna(0.5)
    )

    # Cleanup temp columns
    for h in HORIZONS:
        df.drop(columns=[f"_lh{h}", f"_rh{h}"], inplace=True, errors="ignore")

    before = len(df)
    df = df.dropna(subset=["label"])
    print(f"  Dropped {before - len(df)} rows with NaN ensemble label")
    print(f"  Label range: [{df['label'].min():.3f}, {df['label'].max():.3f}]")
    print(f"  Weights: T+3={WEIGHTS[3]}, T+5={WEIGHTS[5]}, T+10={WEIGHTS[10]}")

    return df


# ======================================================================
# Tail-Amplified Discretization
# ======================================================================

def discretize_labels_tail(df: pd.DataFrame) -> pd.DataFrame:
    """Discretize continuous ensemble label with tail amplification.

    Strategy — make the model focus on distinguishing winners:
      - Top 10% per day     → relevance = 20
      - Top 10-20%          → relevance = 10
      - Top 20-30%          → relevance = 5
      - Bottom 70%          → relevance = 0-4 (linear bins)

    LightGBM LambdaRank uses these integer relevance scores for
    pairwise ranking optimization with eval_at=[5].
    """
    df = df.copy()
    df["relevance"] = 0

    for date, group in df.groupby("日期", sort=False):
        idx = group.index
        n = len(group)
        if n < 5:
            df.loc[idx, "relevance"] = 1
            continue

        ranks = group["label"].rank(pct=True)
        rel = np.zeros(n, dtype=np.int32)

        for i, r in enumerate(ranks):
            if r >= 0.90:       # Top 10%
                rel[i] = 20
            elif r >= 0.80:     # Top 10-20%
                rel[i] = 10
            elif r >= 0.70:     # Top 20-30%
                rel[i] = 5
            else:
                rel[i] = max(0, min(4, int(r * 5)))

        df.loc[idx, "relevance"] = rel

    df["relevance"] = df["relevance"].fillna(0).astype(int)
    print(f"  Tail-amplified discretization: "
          f"max={df['relevance'].max()}, "
          f"top10%_count={(df['relevance'] >= 20).sum()}")
    return df


# ======================================================================
# Main Phase 4 Pipeline
# ======================================================================

def engineer_phase4_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Phase 4 pipeline: ensemble label + Phase 3 features.

    1. Compute multi-horizon rank-ensemble label
    2. Run Phase 3 feature pipeline (detects pre-computed label, skips recompute)
    3. Return 102-dim feature set with ensemble label

    The tail-amplified discretization is applied separately during training.

    Returns:
        (df_with_features, list_of_feature_column_names)
    """
    # Step 1: Multi-horizon ensemble label
    df = compute_ensemble_label(df)

    # Step 2: Phase 3 features (will skip label computation — already present)
    df, feature_cols = engineer_phase3_features(df)

    return df, feature_cols
