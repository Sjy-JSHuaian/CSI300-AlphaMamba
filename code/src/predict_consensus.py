"""Consensus Alpha — Final Prediction.

Blends Phase 6B regime-split models with submit1 LGB model:
  consensus = 0.6 * rank(p6b_score) + 0.4 * rank(submit1_score)
  + 0.05 bonus if stock is in BOTH models' Top-20

Then runs Phase 5 portfolio optimization (no Bull Gate, no Dynamic Exposure).

Usage: python src/predict_consensus.py
Output: output/result.csv
"""

import os, warnings
from typing import Optional, Set
import joblib, lightgbm as lgb, numpy as np, pandas as pd

from config import CONFIG
from features_phase6b import engineer_phase6b_features
from portfolio_optimizer import PortfolioOptimizer

warnings.filterwarnings("ignore")

CONSENSUS_BONUS = 0.05
P6B_WEIGHT = 0.6
S1_WEIGHT = 0.4


def _load_prev_portfolio(path: str) -> Optional[Set[str]]:
    if not os.path.exists(path): return None
    try:
        prev = pd.read_csv(path)
        if "stock_id" not in prev.columns or len(prev) == 0: return None
        stocks = set(prev["stock_id"].astype(str).str.zfill(6).values)
        return stocks if len(stocks) > 0 and "CASH" not in stocks else None
    except Exception:
        return None


def main():
    cfg = CONFIG
    rd = os.path.join(cfg["output_dir"], "regime6b")

    print("=" * 60)
    print("Consensus Alpha Prediction")
    print("=" * 60)
    print(f"  Blend: {P6B_WEIGHT} * rank(P6B) + {S1_WEIGHT} * rank(submit1)")
    print(f"  Bonus: +{CONSENSUS_BONUS} if stock in both Top-20")

    # ---- 1. Load submit1 scores ----
    s1_path = "model/submit1_scores.csv"
    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"{s1_path} not found. Run extract_all_scores.py first.")
    print("\nLoading submit1 scores...")
    s1_scores = pd.read_csv(s1_path, dtype={"股票代码": str})
    s1_scores["股票代码"] = s1_scores["股票代码"].str.zfill(6)
    s1_scores["日期"] = pd.to_datetime(s1_scores["日期"])

    # ---- 2. Load Phase 6B artifacts ----
    print("Loading Phase 6B models...")
    scaler = joblib.load(f"{rd}/scaler.pkl")
    feature_cols = joblib.load(f"{rd}/feature_cols.pkl")
    models = {}
    for regime in ["bull", "nonbull"]:
        models[regime] = lgb.Booster(model_file=f"{rd}/lgb_ranker_{regime}.txt")

    # ---- 3. Load & preprocess data ----
    print("Loading data...")
    df = pd.read_csv("data/train.csv", dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])
    sc = df.groupby("股票代码").size()
    df = df[df["股票代码"].isin(sc[sc >= cfg["min_history_days"]].index)]
    print(f"  {df['股票代码'].nunique()} stocks")

    # ---- 4. Phase 6B features ----
    df, _ = engineer_phase6b_features(df)
    df = df.dropna(subset=["label"])

    latest_date = df["日期"].max()
    pred = df[df["日期"] == latest_date].copy()
    bs = float(pred["bull_strength"].iloc[0])

    print(f"\n  Prediction date:  {latest_date.date()}")
    print(f"  Bull strength:    {bs:.4f}")
    print(f"  Regime:           {pred['regime'].iloc[0]}")

    # ---- 5. Phase 6B predict ----
    mc = set(feature_cols) - set(pred.columns)
    for c in mc: pred[c] = 0.0
    X = scaler.transform(pred[feature_cols].fillna(0).values.astype(np.float32))
    br = pd.Series(models["bull"].predict(X)).rank(pct=True).values
    nr = pd.Series(models["nonbull"].predict(X)).rank(pct=True).values
    p6b = bs * br + (1 - bs) * nr
    pred["p6b_score"] = p6b
    pred["p6b_rank"] = pd.Series(p6b).rank(pct=True).values

    # ---- 6. submit1 score ----
    s1_day = s1_scores[s1_scores["日期"] == latest_date]
    if len(s1_day) == 0:
        print(f"  [WARN] No submit1 scores for {latest_date.date()} — using P6B only")
        pred["s1_score"] = 0.0
        pred["s1_rank"] = 0.5
    else:
        s1_map = dict(zip(s1_day["股票代码"], s1_day["submit1_score"]))
        pred["s1_score"] = pred["股票代码"].map(s1_map).fillna(0.0)
        pred["s1_rank"] = pred["s1_score"].rank(pct=True).fillna(0.5).values

    # ---- 7. Consensus score ----
    p6b_top20 = set(pred.nlargest(20, "p6b_score")["股票代码"])
    s1_top20 = set(pred.nlargest(20, "s1_score")["股票代码"])
    in_both = p6b_top20 & s1_top20
    bonus = np.array([CONSENSUS_BONUS if s in in_both else 0.0 for s in pred["股票代码"]])

    pred["consensus_score"] = (P6B_WEIGHT * pred["p6b_rank"] +
                                S1_WEIGHT * pred["s1_rank"] + bonus)
    pred["score"] = pred["consensus_score"]

    print(f"\n  Consensus Top-20 overlap: {len(in_both)} stocks")
    print(f"  Bonus applied to: {in_both if in_both else '(none)'}")
    print(f"  Score range: [{pred['score'].min():.4f}, {pred['score'].max():.4f}]")

    # ---- 8. Portfolio optimization ----
    if "sector_id" not in pred.columns:
        from features import _get_sector
        pred["sector_id"] = pred["股票代码"].apply(_get_sector)

    pc = cfg["portfolio"]
    opt = PortfolioOptimizer(
        temperature=pc["temperature"], max_per_industry=pc["max_per_industry"],
        corr_threshold=pc["corr_threshold"], min_score_ratio=pc["min_score_ratio"],
        top_n=pc["top_n"], select_k=pc["select_k"], corr_lookback=pc["corr_lookback"],
    )
    prev = _load_prev_portfolio(cfg["output_csv"])
    result = opt.optimize(pred, df, prev_portfolio=prev, verbose=True)

    # ---- 9. Output ----
    result["weight"] = result["weight"].round(6)
    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
    result.to_csv(cfg["output_csv"], index=False, encoding="utf-8")
    print(f"\nSaved: {cfg['output_csv']}")
    print("Done!")


if __name__ == "__main__":
    main()
