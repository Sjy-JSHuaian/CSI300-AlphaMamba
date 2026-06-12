"""Phase 6B: Bull / Non-Bull Soft Blending Inference.

Detects today's bull_strength (continuous, 0-1), predicts with both
Bull and NonBull models, then blends predictions:

    final_score = w * bull_score + (1-w) * nonbull_score

where w = bull_strength (today's market "bullishness").

This is a structural upgrade over Phase 6A's hard switching:
  - Market state is continuous, not discrete
  - Both models contribute, weighted by confidence
  - Naturally handles transitions (weak bull, weak bear, etc.)

Usage: python src/predict_phase6b.py
Output: output/result.csv
"""

import os
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CONFIG
from features_phase6b import engineer_phase6b_features
from portfolio_optimizer import PortfolioOptimizer

warnings.filterwarnings("ignore")


def main():
    cfg = CONFIG

    regime_dir = os.path.join(cfg["output_dir"], "regime6b")
    scaler_path = os.path.join(regime_dir, "scaler.pkl")
    feature_cols_path = os.path.join(regime_dir, "feature_cols.pkl")
    regime_stats_path = os.path.join(regime_dir, "regime_stats.pkl")

    # Check that regime models exist
    model_paths = {}
    for regime in ["bull", "nonbull"]:
        p = os.path.join(regime_dir, f"lgb_ranker_{regime}.txt")
        if os.path.exists(p):
            model_paths[regime] = p

    if len(model_paths) < 2:
        raise FileNotFoundError(
            f"Need both bull and nonbull models in {regime_dir}/. "
            f"Found: {list(model_paths.keys())}. Run train_lgb_phase6b.py first."
        )

    for p in [scaler_path, feature_cols_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing artifact: {p}. Run train_lgb_phase6b.py first."
            )

    # ---- 1. Load shared artifacts ----
    print("Loading Phase 6B artifacts...")
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_cols_path)
    regime_stats = (
        joblib.load(regime_stats_path)
        if os.path.exists(regime_stats_path)
        else {}
    )

    print(f"  Features: {len(feature_cols)}")
    for regime, info in regime_stats.items():
        print(f"    {regime}: NDCG@5={info['ndcg']:.4f}, "
              f"n={info['n_samples']:,}, days={info['n_days']}, "
              f"best_iter={info['best_iter']}")

    # ---- 2. Load models ----
    print("  Loading regime models...")
    models = {}
    for regime, path in model_paths.items():
        models[regime] = lgb.Booster(model_file=path)
        print(f"    {regime}: {os.path.basename(path)}")

    # ---- 3. Load and preprocess data ----
    data_path = os.path.join(cfg["data_path"], "train.csv")
    df = pd.read_csv(data_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    # Filter stocks with minimum history
    stock_counts = df.groupby("股票代码").size()
    valid_stocks = stock_counts[stock_counts >= cfg["min_history_days"]].index
    df = df[df["股票代码"].isin(valid_stocks)]
    print(f"Stocks after filter: {df['股票代码'].nunique()}")

    # ---- 4. Phase 6B feature engineering ----
    df, _ = engineer_phase6b_features(df)
    df = df.dropna(subset=["label"])

    # ---- 5. Get prediction date & bull_strength ----
    latest_date = df["日期"].max()
    pred_day = df[df["日期"] == latest_date].copy()

    # Today's bull_strength (computed during feature engineering)
    bull_strength = pred_day["bull_strength"].iloc[0] if len(pred_day) > 0 else 0.5

    print(f"\nPrediction date:  {latest_date.date()}")
    print(f"Bull strength:    {bull_strength:.4f} "
          f"({'STRONG BULL' if bull_strength > 0.8 else 'BULL' if bull_strength > 0.5 else 'WEAK/FLAT' if bull_strength > 0.2 else 'BEARISH'})")
    print(f"Stocks available: {len(pred_day)}")

    if len(pred_day) < cfg["top_k"]:
        raise ValueError(
            f"Only {len(pred_day)} stocks available on {latest_date.date()}, "
            f"need at least {cfg['top_k']}"
        )

    # ---- 6. Scale features ----
    missing_cols = set(feature_cols) - set(pred_day.columns)
    if missing_cols:
        print(f"Warning: {len(missing_cols)} missing feature columns, filling with 0")
        for col in missing_cols:
            pred_day[col] = 0.0

    X_pred = pred_day[feature_cols].fillna(0.0).values.astype(np.float32)
    X_pred = scaler.transform(X_pred)

    # ---- 7. Predict & Soft Blend ----
    bull_raw = models["bull"].predict(X_pred)
    nonbull_raw = models["nonbull"].predict(X_pred)

    # Cross-sectional rank normalization: put both models on same scale (0-1)
    # This is essential — raw scores from different models are NOT comparable
    bull_rank = pd.Series(bull_raw).rank(pct=True).values
    nonbull_rank = pd.Series(nonbull_raw).rank(pct=True).values

    # Soft regime blend on ranked scores
    w = bull_strength
    final_scores = w * bull_rank + (1.0 - w) * nonbull_rank

    pred_day = pred_day.copy()
    pred_day["score"] = final_scores
    pred_day["bull_score"] = bull_rank
    pred_day["nonbull_score"] = nonbull_rank

    # NaN guard
    pred_day = pred_day.dropna(subset=["score"])

    # Tie-breaking
    if pred_day["score"].duplicated().any():
        rng = np.random.RandomState(cfg["seed"])
        pred_day["score"] = pred_day["score"] + rng.uniform(-1e-8, 1e-8, len(pred_day))

    # Blending diagnostics
    bull_corr = np.corrcoef(bull_raw, nonbull_raw)[0, 1]
    bull_top5 = set(pred_day.nlargest(5, "bull_score")["股票代码"].values)
    nonbull_top5 = set(pred_day.nlargest(5, "nonbull_score")["股票代码"].values)
    blended_top5 = set(pred_day.nlargest(5, "score")["股票代码"].values)
    overlap = len(bull_top5 & nonbull_top5)

    print(f"\n  Blend weight:    {w:.4f} * bull + {1-w:.4f} * nonbull")
    print(f"  Bull vs NonBull score correlation: {bull_corr:.4f}")
    print(f"  Top-5 overlap (bull vs nonbull):   {overlap}/5")
    print(f"  Blended retains {len(blended_top5 & bull_top5)} from bull, "
          f"{len(blended_top5 & nonbull_top5)} from nonbull")

    # ---- 8. Portfolio Optimization (Phase 5) ----
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

    # Ensure sector_id is available
    if "sector_id" not in pred_day.columns:
        from features import _get_sector
        pred_day["sector_id"] = pred_day["股票代码"].apply(_get_sector)

    result = optimizer.optimize(pred_day, df, verbose=True)

    # ---- 9. Output ----
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
    result.to_csv(cfg["output_csv"], index=False)
    print(f"\nSaved to: {cfg['output_csv']}")
    print("Done!")


if __name__ == "__main__":
    main()
