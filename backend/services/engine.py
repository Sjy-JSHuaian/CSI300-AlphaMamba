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
    """Loads models/data once, provides predict() for any date/mode."""

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
