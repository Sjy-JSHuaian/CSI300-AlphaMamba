"""Predict top-5 CSI 300 stocks using trained LightGBM Ranker.

Usage: python src/predict.py
Output: output/result.csv
"""

import os
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CONFIG
from features_phase4 import engineer_phase4_features
from portfolio_optimizer import PortfolioOptimizer

warnings.filterwarnings("ignore")


def main():
    cfg = CONFIG

    model_path = os.path.join(cfg["output_dir"], "lgb_ranker.txt")
    scaler_path = os.path.join(cfg["output_dir"], "scaler.pkl")
    feature_cols_path = os.path.join(cfg["output_dir"], "feature_cols.pkl")

    for p in [model_path, scaler_path, feature_cols_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing artifact: {p}. Run train_lgb.py first."
            )

    # ---- 1. Load artifacts ----
    print("Loading model artifacts...")
    model = lgb.Booster(model_file=model_path)
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_cols_path)
    print(f"Model loaded, {len(feature_cols)} features expected")

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

    # ---- 3. Phase 4 feature engineering ----
    df, _ = engineer_phase4_features(df)

    # Drop rows with NaN labels
    df = df.dropna(subset=["label"])

    # ---- 4. Get prediction date ----
    latest_date = df["日期"].max()
    pred_day = df[df["日期"] == latest_date].copy()

    print(f"\nPrediction date: {latest_date.date()}")
    print(f"Stocks available on prediction day: {len(pred_day)}")

    if len(pred_day) < cfg["top_k"]:
        raise ValueError(
            f"Only {len(pred_day)} stocks available on {latest_date.date()}, "
            f"need at least {cfg['top_k']}"
        )

    # ---- 5. Scale features ----
    missing_cols = set(feature_cols) - set(pred_day.columns)
    if missing_cols:
        print(f"Warning: {len(missing_cols)} missing feature columns, filling with 0")
        for col in missing_cols:
            pred_day[col] = 0.0

    X_pred = pred_day[feature_cols].fillna(0.0).values.astype(np.float32)
    X_pred = scaler.transform(X_pred)

    # ---- 6. Predict ----
    scores = model.predict(X_pred)
    pred_day = pred_day.copy()
    pred_day["score"] = scores

    # NaN guard
    pred_day = pred_day.dropna(subset=["score"])

    # Tie-breaking: add tiny random noise to identical scores
    if pred_day["score"].duplicated().any():
        rng = np.random.RandomState(cfg["seed"])
        pred_day["score"] = pred_day["score"] + rng.uniform(-1e-8, 1e-8, len(pred_day))

    # ---- 7. Portfolio Optimization (Phase 5) ----
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

    # Ensure sector_id is available (should be from feature pipeline)
    if "sector_id" not in pred_day.columns:
        from features import _get_sector
        pred_day["sector_id"] = pred_day["股票代码"].apply(_get_sector)

    result = optimizer.optimize(pred_day, df, verbose=True)

    # ---- 8. Output ----
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
    result.to_csv(cfg["output_csv"], index=False)
    print(f"\nSaved to: {cfg['output_csv']}")
    print("Done!")


if __name__ == "__main__":
    main()
