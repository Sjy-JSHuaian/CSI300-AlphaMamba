"""Meta Ranker V2 — lightweight stacked ensemble over Phase 6B + submit1.

V2 improvements over V1:
  1. Larger candidate pool: top-40 per model (vs top-20)
  2. Richer features: interaction terms, rank deviations, market context (~35 dims)
  3. Linear model (Ridge) instead of LGB — learns fine-tuning weights, avoids overfitting
  4. Labels from full universe ranking, not just candidate pool

Architecture:
  Phase 6B + submit1 → Candidate Pool (~55 stocks) → Ridge → softmax → Portfolio Optimizer
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Set
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# ======================================================================
# Meta feature spec V2 — ~35 features
# ======================================================================

# Base structural features (pulled from df_full)
STRUCTURAL_COLS = [
    "beta_20", "beta_60",
    "vol_shock5", "vol_shock20",
    "abnormal_ret", "close_position", "gap",
    "rank_ret20", "rank_alpha20",
    "ind_strength20", "sector_id",
    "ret1_slope60", "volume_slope60",
    "rsi_ac1", "macd_ac1",
    "ret1_fft1", "turnover_up60", "alpha20_up60",
    "corr_leader20",
]

STRUCTURAL_FALLBACK = {
    "beta_20": 1.0, "beta_60": 1.0,
    "vol_shock5": 1.0, "vol_shock20": 1.0,
    "abnormal_ret": 0.0, "close_position": 0.5, "gap": 0.0,
    "rank_ret20": 0.5, "rank_alpha20": 0.5,
    "ind_strength20": 0.0, "sector_id": 0,
    "ret1_slope60": 0.0, "volume_slope60": 0.0,
    "rsi_ac1": 0.0, "macd_ac1": 0.0,
    "ret1_fft1": 0.0, "turnover_up60": 0.5, "alpha20_up60": 0.5,
    "corr_leader20": 0.0,
}


def _make_features(candidates: pd.DataFrame, df_full: pd.DataFrame,
                   structural_cols: List[str]) -> pd.DataFrame:
    """Extract ~35 meta features from candidate stocks.

    Feature groups:
      A. Base scores (4): p6b_rank, s1_rank, p6b_raw, s1_raw
      B. Consensus (1): in_both_topN
      C. Market context (2): bull_strength, market_breadth
      D. Interaction (4): p6b*s1_rank, (p6b-s1)^2, p6b-s1, |p6b-s1|
      E. Market-conditioned (2): bs*p6b_rank, bs*s1_rank
      F. Structural (~19): beta, vol_shock, rank, shape, sector
    """
    c = candidates.copy()
    date = c["日期"].iloc[0] if "日期" in c.columns else None
    n = len(c)

    # ── A. Base scores ──
    if "p6b_score" in c.columns:
        c["p6b_rank"] = c["p6b_score"].rank(pct=True).fillna(0.5).values
        c["p6b_raw"] = c["p6b_score"].values
    else:
        c["p6b_rank"] = 0.5; c["p6b_raw"] = 0.0

    if "s1_score" in c.columns and c["s1_score"].notna().any():
        c["s1_rank"] = c["s1_score"].rank(pct=True).fillna(0.5).values
        c["s1_raw"] = c["s1_score"].values
    else:
        c["s1_rank"] = 0.5; c["s1_raw"] = 0.0

    # ── B. Consensus ──
    top_n = min(40, max(10, n // 5))
    if "p6b_score" in c.columns and "s1_score" in c.columns:
        p6b_set = set(c.nlargest(top_n, "p6b_score")["股票代码"])
        s1_set = set(c.nlargest(top_n, "s1_score")["股票代码"])
        c["in_both_top"] = c["股票代码"].apply(
            lambda s: 1.0 if s in p6b_set and s in s1_set else 0.0).values
    else:
        c["in_both_top"] = 0.0

    # ── C. Market context ──
    bs = float(c["bull_strength"].iloc[0]) if "bull_strength" in c.columns else 0.5
    if "bull_strength" not in c.columns:
        c["bull_strength"] = bs

    if date is not None and "market_breadth" in df_full.columns:
        mb = df_full[df_full["日期"] == date]["market_breadth"]
        c["market_breadth"] = mb.iloc[0] if len(mb) > 0 else 0.5
    else:
        c["market_breadth"] = 0.5

    # ── D. Interaction features ──
    c["p6b_x_s1"] = c["p6b_rank"] * c["s1_rank"]
    diff = c["p6b_rank"].values - c["s1_rank"].values
    c["rank_diff"] = diff                      # -1 to +1
    c["rank_diff_abs"] = np.abs(diff)           # 0 to 1
    c["rank_diff_sq"] = diff ** 2               # 0 to 1

    # ── E. Market-conditioned ──
    c["bs_x_p6b"] = bs * c["p6b_rank"]
    c["bs_x_s1"] = bs * c["s1_rank"]

    # ── F. Structural features ──
    if date is not None:
        day = df_full[df_full["日期"] == date]
        for col in structural_cols:
            if col in day.columns:
                val_map = dict(zip(day["股票代码"], day[col]))
                c[col] = c["股票代码"].map(val_map)

    return c


# All generated feature names (order must be deterministic)
def _feature_names(structural_cols: List[str]) -> List[str]:
    base = ["p6b_rank", "s1_rank", "p6b_raw", "s1_raw",
            "in_both_top", "bull_strength", "market_breadth",
            "p6b_x_s1", "rank_diff", "rank_diff_abs", "rank_diff_sq",
            "bs_x_p6b", "bs_x_s1"]
    return base + [c for c in structural_cols]


# ======================================================================
# Meta Ranker V2
# ======================================================================

class MetaRanker:
    """Lightweight second-stage ranker — Ridge regression on meta features.

    Usage:
        mr = MetaRanker()
        mr.fit(df, feature_cols_p6b, s1_scores_df)
        pred = mr.predict(pred_day, df_full)
    """

    def __init__(self, alpha: float = 1.0, top_n_candidates: int = 40):
        self.alpha = alpha
        self.top_n = top_n_candidates
        self.model: Optional[Ridge] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.structural_cols = [c for c in STRUCTURAL_COLS]

    # ------------------------------------------------------------------
    # Candidate pool
    # ------------------------------------------------------------------

    def build_candidate_pool(self, pred_day: pd.DataFrame) -> pd.DataFrame:
        """Union of top-N from P6B and top-N from submit1."""
        tn = min(self.top_n, len(pred_day) // 3)
        p6b_ids = set(pred_day.nlargest(tn, "p6b_score")["股票代码"]) if "p6b_score" in pred_day.columns else set()
        s1_ids = set()
        if "s1_score" in pred_day.columns and pred_day["s1_score"].notna().any():
            s1_ids = set(pred_day.nlargest(tn, "s1_score")["股票代码"])
        pool = p6b_ids | s1_ids
        if len(pool) < 10:
            # Fallback: just use top-N from whatever we have
            score_col = "p6b_score" if "p6b_score" in pred_day.columns else "s1_score"
            return pred_day.nlargest(min(30, len(pred_day)), score_col).copy()
        return pred_day[pred_day["股票代码"].isin(pool)].copy()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, candidates: pd.DataFrame,
                         df_full: pd.DataFrame) -> pd.DataFrame:
        """Extract meta features, fill missing, return feature matrix."""
        c = _make_features(candidates, df_full, self.structural_cols)
        self.feature_names = _feature_names(self.structural_cols)

        # Ensure all features exist
        for col in self.feature_names:
            if col not in c.columns:
                c[col] = STRUCTURAL_FALLBACK.get(col, 0.0)

        # Fill NaN
        for col in self.feature_names:
            if c[col].isna().any():
                c[col] = c[col].fillna(STRUCTURAL_FALLBACK.get(col, 0.0))

        return c

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols_p6b: List[str],
        s1_scores_df: Optional[pd.DataFrame] = None,
        val_ratio: float = 0.15,
    ) -> "MetaRanker":
        """Train Ridge meta ranker on historical candidate pools.

        Args:
            df: Full feature-engineered dataframe
            feature_cols_p6b: Phase 6B feature names (not used in V2 — scores come from pre-computed)
            s1_scores_df: Pre-computed submit1 scores
            val_ratio: Fraction of dates for validation (reporting only)

        Returns:
            self (trained)
        """
        from features_phase6b import engineer_phase6b_features

        print("=" * 50)
        print("Meta Ranker V2 Training")
        print("=" * 50)

        # ── 1. Generate P6B scores on all dates ──
        # Use pre-trained Phase 6B models for efficiency
        print("Generating P6B scores...")
        rd = "model/regime6b"
        import joblib, lightgbm as lgb
        scaler_p6b = joblib.load(f"{rd}/scaler.pkl")
        bm = lgb.Booster(model_file=f"{rd}/lgb_ranker_bull.txt")
        nm = lgb.Booster(model_file=f"{rd}/lgb_ranker_nonbull.txt")

        df = df.sort_values("日期").copy()
        all_dates = sorted(df["日期"].unique())

        for date in all_dates:
            day_idx = df["日期"] == date
            day = df[day_idx]
            mc = set(feature_cols_p6b) - set(day.columns)
            for c in mc: day[c] = 0.0
            X = scaler_p6b.transform(day[feature_cols_p6b].fillna(0).values.astype(np.float32))
            br = pd.Series(bm.predict(X)).rank(pct=True).values
            nr = pd.Series(nm.predict(X)).rank(pct=True).values
            bs = float(day["bull_strength"].iloc[0]) if "bull_strength" in day.columns else 0.5
            df.loc[day_idx, "p6b_score"] = bs * br + (1 - bs) * nr

        # ── 2. Merge submit1 scores ──
        if s1_scores_df is not None:
            print("Merging submit1 scores...")
            df["s1_score"] = np.nan
            s1_map = s1_scores_df.set_index(["日期", "股票代码"])["submit1_score"]
            for date in all_dates:
                day_idx = df["日期"] == date
                for i in df[day_idx].index:
                    sid = df.loc[i, "股票代码"]
                    if (date, sid) in s1_map.index:
                        df.loc[i, "s1_score"] = s1_map.loc[(date, sid)]
        else:
            df["s1_score"] = np.nan

        # ── 3. Build candidate pools for all dates ──
        print(f"Building candidate pools (top-{self.top_n})...")
        rows = []
        for date in all_dates:
            day = df[df["日期"] == date].copy()
            if len(day) < 20:
                continue
            pool = self.build_candidate_pool(day)
            if len(pool) < 10:
                continue
            pool = self.extract_features(pool, df)
            # Label from full universe (the ensemble label is already cross-sectionally meaningful)
            pool["label"] = day.set_index("股票代码").reindex(pool["股票代码"])["label"].values
            rows.append(pool)

        meta = pd.concat(rows, ignore_index=True)
        print(f"  Meta dataset: {len(meta):,} rows, {meta['日期'].nunique()} dates")

        # ── 4. Split and train ──
        meta_dates = sorted(meta["日期"].unique())
        n_val = max(1, int(len(meta_dates) * val_ratio))
        train_dates = set(meta_dates[:-n_val])
        val_dates = set(meta_dates[-n_val:])

        tr = meta[meta["日期"].isin(train_dates)].copy()
        vl = meta[meta["日期"].isin(val_dates)].copy()

        self.feature_names = _feature_names(self.structural_cols)
        # Keep only features that exist
        self.feature_names = [c for c in self.feature_names if c in meta.columns]

        for col in self.feature_names:
            if tr[col].isna().any():
                tr[col] = tr[col].fillna(STRUCTURAL_FALLBACK.get(col, 0.0))
            if vl[col].isna().any():
                vl[col] = vl[col].fillna(STRUCTURAL_FALLBACK.get(col, 0.0))

        self.scaler = StandardScaler()
        Xt = self.scaler.fit_transform(tr[self.feature_names].fillna(0).values.astype(np.float64))
        yt = tr["label"].values.astype(np.float64)
        Xv = self.scaler.transform(vl[self.feature_names].fillna(0).values.astype(np.float64))
        yv = vl["label"].values.astype(np.float64)

        # Ridge regression
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(Xt, yt)

        # Evaluate
        from scipy.stats import spearmanr
        yp = self.model.predict(Xv)
        ic, _ = spearmanr(yp, yv)
        train_ic, _ = spearmanr(self.model.predict(Xt), yt)

        # Top-5 overlap
        print(f"  Train Rank IC: {train_ic:.4f}")
        print(f"  Val Rank IC:   {ic:.4f}")
        print(f"  Coef range:    [{self.model.coef_.min():.4f}, {self.model.coef_.max():.4f}]")

        # Report top features
        coef_abs = np.abs(self.model.coef_)
        top_idx = np.argsort(coef_abs)[-8:][::-1]
        print(f"  Top features:  ", end="")
        for i in top_idx:
            print(f"{self.feature_names[i]}({self.model.coef_[i]:+.3f}) ", end="")
        print()

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, pred_day: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
        """Predict meta scores for a single prediction date."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Meta ranker not trained. Call fit() first.")

        pool = self.build_candidate_pool(pred_day)
        pool = self.extract_features(pool, df_full)

        for col in self.feature_names:
            if col not in pool.columns:
                pool[col] = STRUCTURAL_FALLBACK.get(col, 0.0)
            pool[col] = pool[col].fillna(STRUCTURAL_FALLBACK.get(col, 0.0))

        X = self.scaler.transform(pool[self.feature_names].values.astype(np.float64))
        scores = self.model.predict(X)

        score_map = dict(zip(pool["股票代码"], scores))
        pred_day = pred_day.copy()
        min_score = np.min(scores) if len(scores) > 0 else 0.0
        pred_day["meta_score"] = pred_day["股票代码"].map(score_map).fillna(min_score - 0.1)
        return pred_day


# Backward compatibility alias
MetaRankerV2 = MetaRanker
