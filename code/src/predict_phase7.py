"""Phase 7: Safe Mode Final Inference.

Production prediction script with all four safety layers:

  1. Bull Gate:       bull_strength < 0.6  → 100% cash (empty output)
  2. Dynamic Exposure: exposure = bull_strength (scale weights)
  3. Turnover Penalty:  λ = 0.02 (new positions penalized)
  4. Persistence Bonus: +0.01 (existing positions rewarded)

Uses Phase 6B pre-trained models (./model/regime6b/).

Layers 3-4 use previous portfolio from output/result.csv if available,
otherwise start fresh.

Usage: python src/predict_phase7.py
Output: output/result.csv
"""

import os
import warnings
from typing import Optional, Set

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from config import CONFIG
from features_phase6b import engineer_phase6b_features
from portfolio_optimizer import PortfolioOptimizer

warnings.filterwarnings("ignore")


# ======================================================================
# Helpers
# ======================================================================

def _load_prev_portfolio(result_path: str) -> Optional[Set[str]]:
    """Load previous portfolio from result.csv for persistence bonus."""
    if not os.path.exists(result_path):
        return None
    try:
        prev = pd.read_csv(result_path)
        if "stock_id" not in prev.columns or len(prev) == 0:
            return None
        stocks = set(prev["stock_id"].astype(str).str.zfill(6).values)
        if "CASH" in stocks:
            return None
        return stocks if len(stocks) > 0 else None
    except Exception:
        return None


# ======================================================================
# Main
# ======================================================================

def main():
    cfg = CONFIG
    p7 = cfg.get("phase7", {})
    BULL_GATE = p7.get("bull_gate", 0.6)
    TURNOVER_PENALTY = p7.get("turnover_penalty", 0.02)
    PERSISTENCE_BONUS = p7.get("persistence_bonus", 0.01)

    regime_dir = os.path.join(cfg["output_dir"], "regime6b")

    # ---- 1. Load artifacts ----
    print("=" * 60)
    print("Phase 7: Safe Mode Prediction")
    print("=" * 60)
    print(f"  Bull Gate:        bs >= {BULL_GATE}")
    print(f"  Dynamic Exposure: weight *= bull_strength")
    print(f"  Turnover Penalty: lambda = {TURNOVER_PENALTY}")
    print(f"  Persistence Bonus: +{PERSISTENCE_BONUS}")

    print("\nLoading Phase 6B artifacts...")
    scaler_path = os.path.join(regime_dir, "scaler.pkl")
    feature_cols_path = os.path.join(regime_dir, "feature_cols.pkl")

    for p in [scaler_path, feature_cols_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}. Run train_lgb_phase6b.py first.")

    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_cols_path)

    # Load both regime models
    models = {}
    for regime in ["bull", "nonbull"]:
        model_path = os.path.join(regime_dir, f"lgb_ranker_{regime}.txt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing: {model_path}")
        models[regime] = lgb.Booster(model_file=model_path)

    # ---- 2. Load & preprocess data ----
    print("Loading data...")
    data_path = os.path.join(cfg["data_path"], "train.csv")
    df = pd.read_csv(data_path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    stock_counts = df.groupby("股票代码").size()
    valid_stocks = stock_counts[stock_counts >= cfg["min_history_days"]].index
    df = df[df["股票代码"].isin(valid_stocks)]
    print(f"  Stocks after filter: {df['股票代码'].nunique()}")

    # ---- 3. Feature engineering ----
    df, _ = engineer_phase6b_features(df)
    df = df.dropna(subset=["label"])

    # ---- 4. Prediction date & bull_strength ----
    latest_date = df["日期"].max()
    pred_day = df[df["日期"] == latest_date].copy()

    bull_strength = float(pred_day["bull_strength"].iloc[0]) if len(pred_day) > 0 else 0.5
    regime_label = str(pred_day["regime"].iloc[0]) if len(pred_day) > 0 else "unknown"

    print(f"\n  Prediction date:  {latest_date.date()}")
    print(f"  Bull strength:    {bull_strength:.4f}")
    print(f"  Regime:           {regime_label}")

    # ---- 5. Bull Gate ----
    if bull_strength < BULL_GATE:
        print(f"\n  [BULL GATE] bull_strength {bull_strength:.4f} < {BULL_GATE}")
        print(f"  → 100% CASH (no positions)")
        os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
        pd.DataFrame(columns=["stock_id", "weight"]).to_csv(
            cfg["output_csv"], index=False, encoding="utf-8"
        )
        print(f"  Saved: {cfg['output_csv']} (empty — all cash)")
        print("Done!")
        return

    # ---- 6. Scale & predict ----
    missing_cols = set(feature_cols) - set(pred_day.columns)
    if missing_cols:
        for col in missing_cols:
            pred_day[col] = 0.0

    X_pred = pred_day[feature_cols].fillna(0.0).values.astype(np.float32)
    X_pred = scaler.transform(X_pred)

    # Predict both models
    bull_raw = models["bull"].predict(X_pred)
    nonbull_raw = models["nonbull"].predict(X_pred)

    # Rank normalize + soft blend
    bull_rank = pd.Series(bull_raw).rank(pct=True).values
    nonbull_rank = pd.Series(nonbull_raw).rank(pct=True).values
    final_scores = bull_strength * bull_rank + (1.0 - bull_strength) * nonbull_rank

    pred_day["score"] = final_scores

    # Debug: score range
    print(f"  Bull/NonBull corr: {np.corrcoef(bull_raw, nonbull_raw)[0, 1]:.4f}")
    print(f"  Score range:       [{final_scores.min():.4f}, {final_scores.max():.4f}]")

    # ---- 7. Portfolio optimization ----
    portfolio_cfg = cfg["portfolio"]
    optimizer = PortfolioOptimizer(
        temperature=portfolio_cfg["temperature"],
        max_per_industry=portfolio_cfg["max_per_industry"],
        corr_threshold=portfolio_cfg["corr_threshold"],
        min_score_ratio=portfolio_cfg["min_score_ratio"],
        top_n=portfolio_cfg["top_n"],
        select_k=portfolio_cfg["select_k"],
        corr_lookback=portfolio_cfg["corr_lookback"],
        turnover_penalty=TURNOVER_PENALTY,
        persistence_bonus=PERSISTENCE_BONUS,
    )

    # Load previous portfolio for persistence bonus
    prev_portfolio = _load_prev_portfolio(cfg["output_csv"])

    # Ensure sector_id
    if "sector_id" not in pred_day.columns:
        from features import _get_sector
        pred_day["sector_id"] = pred_day["股票代码"].apply(_get_sector)

    result = optimizer.optimize(pred_day, df, prev_portfolio=prev_portfolio, verbose=True)

    # ---- 8. Dynamic Exposure ----
    result["weight"] = result["weight"] * bull_strength
    weight_sum = result["weight"].sum()

    print(f"\n  Dynamic Exposure:  weights *= {bull_strength:.4f}")
    print(f"  Adjusted weight sum: {weight_sum:.4f}")
    print(f"  Implied cash:        {1.0 - weight_sum:.4f}")

    # ---- 9. Output ----
    # Round weights to 6 decimal places
    result["weight"] = result["weight"].round(6)
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
    result.to_csv(cfg["output_csv"], index=False, encoding="utf-8")
    print(f"\nSaved: {cfg['output_csv']}")
    print("Done!")


if __name__ == "__main__":
    main()
