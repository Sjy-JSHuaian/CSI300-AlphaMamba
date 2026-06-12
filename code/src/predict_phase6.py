"""Phase 6A: Regime-Split Inference + Portfolio Optimization.

Detects today's market regime, loads the corresponding LightGBM model,
predicts scores, and runs the Phase 5 portfolio optimizer.

Usage: python src/predict_phase6.py
Output: output/result.csv
"""

import os
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CONFIG
from features_phase6 import engineer_phase6_features
from portfolio_optimizer import PortfolioOptimizer

warnings.filterwarnings("ignore")


def main():
    cfg = CONFIG
    regime_cfg = cfg.get("phase6", {})
    vol_threshold = regime_cfg.get("vol_threshold", 0.02)

    regime_dir = os.path.join(cfg["output_dir"], "regime")
    scaler_path = os.path.join(regime_dir, "scaler.pkl")
    feature_cols_path = os.path.join(regime_dir, "feature_cols.pkl")
    regime_stats_path = os.path.join(regime_dir, "regime_stats.pkl")

    # Check that regime models exist
    available_regimes = []
    for regime in ["bull", "bear", "sideway"]:
        model_path = os.path.join(regime_dir, f"lgb_ranker_{regime}.txt")
        if os.path.exists(model_path):
            available_regimes.append(regime)

    if not available_regimes:
        raise FileNotFoundError(
            f"No regime models found in {regime_dir}/. "
            f"Run train_lgb_phase6.py first."
        )

    for p in [scaler_path, feature_cols_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing artifact: {p}. Run train_lgb_phase6.py first."
            )

    # ---- 1. Load shared artifacts ----
    print("Loading Phase 6 artifacts...")
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_cols_path)
    regime_stats = joblib.load(regime_stats_path) if os.path.exists(regime_stats_path) else {}

    print(f"  Features: {len(feature_cols)}")
    print(f"  Trained regimes: {available_regimes}")
    for regime, info in regime_stats.items():
        print(f"    {regime}: NDCG@5={info['ndcg']:.4f}, "
              f"n={info['n_samples']:,}, days={info['n_days']}")

    # ---- 2. Load and preprocess data ----
    data_path = os.path.join(cfg["data_path"], "train.csv")
    df = pd.read_csv(data_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    # Filter stocks with minimum history
    stock_counts = df.groupby("股票代码").size()
    valid_stocks = stock_counts[stock_counts >= cfg["min_history_days"]].index
    df = df[df["股票代码"].isin(valid_stocks)]
    print(f"Stocks after filter: {df['股票代码'].nunique()}")

    # ---- 3. Phase 6 feature engineering (features + regime) ----
    df, _ = engineer_phase6_features(df, vol_threshold=vol_threshold)

    # Drop rows with NaN labels
    df = df.dropna(subset=["label"])

    # ---- 4. Get prediction date and regime ----
    latest_date = df["日期"].max()
    pred_day = df[df["日期"] == latest_date].copy()

    today_regime = pred_day["regime"].iloc[0] if len(pred_day) > 0 else "sideway"
    print(f"\nPrediction date: {latest_date.date()}")
    print(f"Today's regime:  {today_regime.upper()}")
    print(f"Stocks available: {len(pred_day)}")

    # ---- 5. Select model ----
    # Use today's regime model; fall back to sideway → bull → bear
    fallback_order = [today_regime, "sideway", "bull", "bear"]
    model_regime = None
    model = None

    for r in fallback_order:
        model_path = os.path.join(regime_dir, f"lgb_ranker_{r}.txt")
        if os.path.exists(model_path):
            model_regime = r
            model = lgb.Booster(model_file=model_path)
            break

    if model is None:
        raise FileNotFoundError("No usable regime model found.")

    if model_regime != today_regime:
        print(f"  [WARN] No model for '{today_regime}' — falling back to '{model_regime}'")
    print(f"  Using model: lgb_ranker_{model_regime}.txt")

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

    # ---- 7. Predict ----
    scores = model.predict(X_pred)
    pred_day = pred_day.copy()
    pred_day["score"] = scores

    # NaN guard
    pred_day = pred_day.dropna(subset=["score"])

    # Tie-breaking: add tiny random noise to identical scores
    if pred_day["score"].duplicated().any():
        rng = np.random.RandomState(cfg["seed"])
        pred_day["score"] = pred_day["score"] + rng.uniform(-1e-8, 1e-8, len(pred_day))

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
