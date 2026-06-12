"""Meta Ranker V2 — Final Prediction.

Stacked ensemble:
  Phase 6B + submit1 → Candidate Pool (top-40 each) → Ridge Meta Ranker → softmax → Optimizer

Includes Phase 7 risk: Bull Gate, Dynamic Exposure, Turnover Penalty.

Usage: python src/predict_meta_ranker.py
Output: output/result.csv
"""

import os, warnings
from typing import Optional, Set
import joblib, lightgbm as lgb, numpy as np, pandas as pd
from config import CONFIG
from features_phase6b import engineer_phase6b_features
from portfolio_optimizer import PortfolioOptimizer
from meta_ranker import MetaRanker
from features import _get_sector

warnings.filterwarnings("ignore")

BULL_GATE = 0.6
TURNOVER_PENALTY = 0.02
PERSISTENCE_BONUS = 0.01


def _load_prev(path: str) -> Optional[Set[str]]:
    if not os.path.exists(path): return None
    try:
        prev = pd.read_csv(path)
        if "stock_id" not in prev.columns or len(prev) == 0: return None
        s = set(prev["stock_id"].astype(str).str.zfill(6))
        return s if "CASH" not in str(s) and len(s) > 0 else None
    except Exception:
        return None


def main():
    cfg = CONFIG
    rd = os.path.join(cfg["output_dir"], "regime6b")

    print("=" * 60)
    print("Meta Ranker V2 Prediction (Ridge + top-40 pool)")
    print("=" * 60)

    # ── 1. Load submit1 scores ──
    s1p = "model/submit1_scores.csv"
    s1_df = pd.read_csv(s1p, dtype={"股票代码": str}) if os.path.exists(s1p) else None
    if s1_df is not None:
        s1_df["股票代码"] = s1_df["股票代码"].str.zfill(6)
        s1_df["日期"] = pd.to_datetime(s1_df["日期"])
    print(f"  submit1: {len(s1_df):,} rows" if s1_df is not None else "  [WARN] No submit1 scores")

    # ── 2. Load Phase 6B artifacts ──
    feature_cols = joblib.load(f"{rd}/feature_cols.pkl")

    # ── 3. Load data & features ──
    df = pd.read_csv("data/train.csv", dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].str.zfill(6); df["日期"] = pd.to_datetime(df["日期"])
    sc = df.groupby("股票代码").size()
    df = df[df["股票代码"].isin(sc[sc >= cfg["min_history_days"]].index)]

    df, _ = engineer_phase6b_features(df)
    df = df.dropna(subset=["label"])

    # ── 4. Train Meta Ranker ──
    print("\nTraining Meta Ranker V2...")
    mr = MetaRanker(alpha=1.0, top_n_candidates=40)
    mr.fit(df, feature_cols, s1_df)

    # ── 5. Predict on latest date ──
    latest_date = df["日期"].max()
    pred = df[df["日期"] == latest_date].copy()
    bs = float(pred["bull_strength"].iloc[0])
    print(f"\n  Date: {latest_date.date()}  |  BS: {bs:.4f}  |  Regime: {pred['regime'].iloc[0]}")

    if bs < BULL_GATE:
        print(f"  [BULL GATE] → 100% CASH")
        os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
        pd.DataFrame(columns=["stock_id", "weight"]).to_csv(cfg["output_csv"], index=False, encoding="utf-8")
        return

    pred = mr.predict(pred, df)

    # ── 6. Portfolio optimization ──
    if "sector_id" not in pred.columns:
        pred["sector_id"] = pred["股票代码"].apply(_get_sector)
    pred["score"] = pred["meta_score"]

    pc = cfg["portfolio"]
    opt = PortfolioOptimizer(
        temperature=pc["temperature"], max_per_industry=pc["max_per_industry"],
        corr_threshold=pc["corr_threshold"], min_score_ratio=pc["min_score_ratio"],
        top_n=pc["top_n"], select_k=pc["select_k"], corr_lookback=pc["corr_lookback"],
        turnover_penalty=TURNOVER_PENALTY, persistence_bonus=PERSISTENCE_BONUS,
    )
    prev = _load_prev(cfg["output_csv"])

    # Debug: show candidate pool size
    pool = mr.build_candidate_pool(pred)
    print(f"  Candidate pool: {len(pool)} stocks (P6B + S1 top-40)")

    result = opt.optimize(pred, df, prev_portfolio=prev, verbose=True)

    # Dynamic exposure
    result["weight"] = (result["weight"] * bs).round(6)
    ws = result["weight"].sum()
    if ws > 1.0 + 1e-10:
        result["weight"] = (result["weight"] / ws).round(6)

    os.makedirs(os.path.dirname(cfg["output_csv"]), exist_ok=True)
    result.to_csv(cfg["output_csv"], index=False, encoding="utf-8")
    print(f"\n  Weight sum: {result['weight'].sum():.4f}  |  Exposure: {bs:.4f}")
    print(f"Saved: {cfg['output_csv']}")
    print("Done!")


if __name__ == "__main__":
    main()
