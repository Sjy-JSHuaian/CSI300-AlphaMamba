# backend/services/engine.py
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from typing import Optional, List, Set, Dict
from config import CONFIG
from features_phase6b import engineer_phase6b_features
from portfolio_optimizer import PortfolioOptimizer

warnings.filterwarnings("ignore")


class PredictionEngine:
    """Loads models/data once at startup. Provides predict() for all 4 modes + portfolio optimization."""

    def __init__(self):
        self.cfg = CONFIG
        self.ready = False
        self.models: Dict[str, lgb.Booster] = {}
        self.scaler = None
        self.feature_cols: List[str] = []
        self.df: Optional[pd.DataFrame] = None  # full feature-engineered df
        self.valid_dates: List[str] = []
        self.s1_scores: Optional[pd.DataFrame] = None

    def load(self):
        regime_dir = os.path.join(self.cfg["output_dir"], "regime6b")

        # Load Phase 6B models
        for regime in ["bull", "nonbull"]:
            path = os.path.join(regime_dir, f"lgb_ranker_{regime}.txt")
            if os.path.exists(path):
                self.models[regime] = lgb.Booster(model_file=path)

        # Load scaler and feature columns
        self.scaler = joblib.load(os.path.join(regime_dir, "scaler.pkl"))
        self.feature_cols = joblib.load(os.path.join(regime_dir, "feature_cols.pkl"))

        # Load and preprocess data
        data_path = os.path.join(self.cfg["data_path"], "train.csv")
        df = pd.read_csv(data_path, dtype={"股票代码": str})
        df["股票代码"] = df["股票代码"].str.zfill(6)
        df["日期"] = pd.to_datetime(df["日期"])

        stock_counts = df.groupby("股票代码").size()
        valid_stocks = stock_counts[stock_counts >= self.cfg["min_history_days"]].index
        df = df[df["股票代码"].isin(valid_stocks)]

        df, _ = engineer_phase6b_features(df)
        df = df.dropna(subset=["label"])
        self.df = df
        self.valid_dates = sorted(df["日期"].dt.strftime("%Y-%m-%d").unique())

        # Load submit1 scores if available
        s1_path = "model/submit1_scores.csv"
        if os.path.exists(s1_path):
            s1 = pd.read_csv(s1_path, dtype={"股票代码": str})
            s1["股票代码"] = s1["股票代码"].str.zfill(6)
            s1["日期"] = pd.to_datetime(s1["日期"])
            self.s1_scores = s1

        self.ready = True
        print(f"Engine loaded: {len(self.models)} models, "
              f"{len(self.feature_cols)} features, "
              f"{len(self.valid_dates)} valid dates")

    def _get_pred_day(self, date_str: str) -> pd.DataFrame:
        """Get prediction-day DataFrame with features."""
        target_date = pd.to_datetime(date_str)
        mask = self.df["日期"] == target_date
        if not mask.any():
            raise ValueError(f"No data for date {date_str}")
        return self.df[mask].copy()

    def predict_phase6b(self, date_str: str, params: dict) -> dict:
        """Run Phase 6B Bull/NonBull soft blend prediction."""
        pred = self._get_pred_day(date_str)

        bs = float(pred["bull_strength"].iloc[0])
        regime_label = str(pred["regime"].iloc[0])

        # Scale features
        missing = set(self.feature_cols) - set(pred.columns)
        for col in missing:
            pred[col] = 0.0
        X = pred[self.feature_cols].fillna(0.0).values.astype(np.float32)
        X = self.scaler.transform(X)

        # Predict both regimes
        bull_raw = self.models["bull"].predict(X)
        nonbull_raw = self.models["nonbull"].predict(X)

        # Rank normalize + soft blend
        bull_rank = pd.Series(bull_raw).rank(pct=True).values
        nonbull_rank = pd.Series(nonbull_raw).rank(pct=True).values
        final_scores = bs * bull_rank + (1.0 - bs) * nonbull_rank

        pred = pred.copy()
        pred["score"] = final_scores
        pred["bull_score"] = bull_rank
        pred["nonbull_score"] = nonbull_rank
        pred = pred.dropna(subset=["score"])

        # Tie-breaking
        if pred["score"].duplicated().any():
            rng = np.random.RandomState(self.cfg["seed"])
            pred["score"] = pred["score"] + rng.uniform(-1e-8, 1e-8, len(pred))

        # Build result dict BEFORE portfolio optimization
        bull_top5 = pred.nlargest(5, "bull_score")["股票代码"].tolist()
        nonbull_top5 = pred.nlargest(5, "nonbull_score")["股票代码"].tolist()

        # Score distribution stats
        scores = pred["score"].values
        score_dist = {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

        return {
            "pred_df": pred,
            "bull_strength": round(bs, 4),
            "regime_label": regime_label,
            "bull_top5": bull_top5,
            "nonbull_top5": nonbull_top5,
            "score_distribution": score_dist,
            "full_df": self.df,
        }

    def predict_consensus(self, date_str: str, params: dict) -> dict:
        """Phase 6B + submit1 consensus blend."""
        result = self.predict_phase6b(date_str, params)
        pred = result["pred_df"]

        if self.s1_scores is not None:
            target_date = pd.to_datetime(date_str)
            s1_day = self.s1_scores[self.s1_scores["日期"] == target_date]
            if len(s1_day) > 0:
                s1_map = dict(zip(s1_day["股票代码"], s1_day["submit1_score"]))
                pred["s1_score"] = pred["股票代码"].map(s1_map).fillna(0.0)
                pred["s1_rank"] = pred["s1_score"].rank(pct=True).fillna(0.5).values
            else:
                pred["s1_score"] = 0.0
                pred["s1_rank"] = 0.5
        else:
            pred["s1_score"] = 0.0
            pred["s1_rank"] = 0.5

        # Consensus: 0.6 * p6b_rank + 0.4 * s1_rank + bonus
        p6b_top20 = set(pred.nlargest(20, "score")["股票代码"])
        s1_top20 = set(pred.nlargest(20, "s1_score")["股票代码"])
        in_both = p6b_top20 & s1_top20

        bonus = np.array([0.05 if s in in_both else 0.0 for s in pred["股票代码"]])
        pred["score"] = 0.6 * pred["score"].rank(pct=True) + 0.4 * pred["s1_rank"] + bonus

        scores = pred["score"].values
        result["score_distribution"] = {
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        }
        result["consensus_overlap"] = len(in_both)
        result["pred_df"] = pred
        return result

    def predict_phase7(self, date_str: str, params: dict) -> dict:
        """Phase 7 safe mode — adds bull_gate check."""
        result = self.predict_phase6b(date_str, params)
        bs = result["bull_strength"]
        bull_gate = params.get("bull_gate", 0.6)

        if bs < bull_gate:
            result["gated"] = True
            result["portfolio"] = []
            return result

        result["gated"] = False
        return result

    def predict_meta_ranker(self, date_str: str, params: dict) -> dict:
        """Meta Ranker V2."""
        from meta_ranker import MetaRanker

        result = self.predict_phase6b(date_str, params)
        pred = result["pred_df"]

        mr = MetaRanker(alpha=1.0, top_n_candidates=40)
        mr.fit(self.df, self.feature_cols, self.s1_scores)
        pred = mr.predict(pred, self.df)

        pred["score"] = pred["meta_score"]
        scores = pred["score"].values
        result["score_distribution"] = {
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        }
        result["pred_df"] = pred
        return result

    def _optimize_portfolio(self, pred_df, full_df, params, prev_portfolio=None, verbose=False):
        """Run portfolio optimization on prediction results."""
        opt = PortfolioOptimizer(
            temperature=params.get("temperature", 0.5),
            max_per_industry=params.get("max_per_industry", 2),
            corr_threshold=params.get("corr_threshold", 0.8),
            min_score_ratio=params.get("min_score_ratio", 0.7),
            top_n=params.get("top_n", 10),
            select_k=params.get("select_k", 5),
            corr_lookback=60,
            turnover_penalty=params.get("turnover_penalty", 0.0),
            persistence_bonus=params.get("persistence_bonus", 0.0),
        )
        if "sector_id" not in pred_df.columns:
            from features import _get_sector
            pred_df = pred_df.copy()
            pred_df["sector_id"] = pred_df["股票代码"].apply(_get_sector)

        result = opt.optimize(pred_df, full_df, prev_portfolio=prev_portfolio, verbose=False)
        result["weight"] = result["weight"].round(6)
        return result

    def predict(self, date_str: str, mode: str, params: dict) -> dict:
        """Unified predict interface — dispatches by mode name."""
        mode = mode.lower().replace(" ", "_").replace("-", "_")

        mode_fn = {
            "phase6b": self.predict_phase6b,
            "consensus": self.predict_consensus,
            "meta_ranker": self.predict_meta_ranker,
            "phase7": self.predict_phase7,
        }

        if mode not in mode_fn:
            raise ValueError(f"Unknown mode: {mode}. Options: {list(mode_fn.keys())}")

        result = mode_fn[mode](date_str, params)

        # Phase 7 bull gate: return empty portfolio if gated
        if result.get("gated"):
            return {
                "date": date_str, "mode": mode,
                "bull_strength": result["bull_strength"],
                "regime_label": result.get("regime_label", ""),
                "portfolio": [],
                "bull_top5": result.get("bull_top5", []),
                "nonbull_top5": result.get("nonbull_top5", []),
                "consensus_overlap": result.get("consensus_overlap", 0),
                "score_distribution": result.get("score_distribution", {}),
                "sector_distribution": {},
            }

        pred = result["pred_df"]

        # Ensure sector_id exists for output
        if "sector_id" not in pred.columns:
            from features import _get_sector
            pred = pred.copy()
            pred["sector_id"] = pred["股票代码"].apply(_get_sector)

        # Portfolio optimization
        portfolio = self._optimize_portfolio(pred, result.get("full_df", self.df), params)

        # Sector distribution
        sector_counts = pred.loc[
            pred["股票代码"].isin(portfolio["stock_id"].values), "sector_id"
        ].value_counts().to_dict()

        # Build portfolio list
        score_map = dict(zip(pred["股票代码"], pred["score"]))
        sector_map = dict(zip(pred["股票代码"], pred["sector_id"]))

        return {
            "date": date_str,
            "mode": mode,
            "bull_strength": result["bull_strength"],
            "regime_label": result.get("regime_label", ""),
            "portfolio": [
                {
                    "stock_id": row["stock_id"],
                    "weight": float(row["weight"]),
                    "score": float(score_map.get(row["stock_id"], 0)),
                    "sector_id": int(sector_map.get(row["stock_id"], 0)),
                }
                for _, row in portfolio.iterrows()
            ],
            "bull_top5": result.get("bull_top5", []),
            "nonbull_top5": result.get("nonbull_top5", []),
            "consensus_overlap": result.get("consensus_overlap", 0),
            "score_distribution": result.get("score_distribution", {}),
            "sector_distribution": {str(k): int(v) for k, v in sector_counts.items()},
        }
