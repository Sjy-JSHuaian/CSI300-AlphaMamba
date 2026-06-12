"""Phase 5: Portfolio Optimization Module.

Greedy selection with diversification constraints:
  - Top-N candidates from model predictions
  - 60-day return correlation matrix for diversification
  - Max 2 stocks per industry (board-level: Shanghai/Shenzhen/STAR/ChiNext)
  - Correlation penalty (>0.8 → score reduction)
  - Temperature softmax weights (T=0.5)
  - Quality filter: score ≥ 70% of top-1 score
  - Sum of weights ≤ 1, remainder = cash

Usage:
    from portfolio_optimizer import PortfolioOptimizer

    optimizer = PortfolioOptimizer(temperature=0.5)
    result = optimizer.optimize(pred_day_df, full_history_df)
    result.to_csv("output/result.csv", index=False)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class PortfolioOptimizer:
    """Correlation-aware greedy portfolio optimizer.

    Algorithm:
    1. Select top-N stocks by model score
    2. Compute 60-day pairwise return correlation matrix
    3. Greedy selection:
       - Start with highest-score stock
       - For each remaining iteration, pick the best candidate:
         * Skip if score < 70% of top-1 score
         * Skip if industry already has max_per_industry selected
         * Penalize score if max correlation with selected > corr_threshold
       - Repeat until select_k stocks chosen or no valid candidates remain
    4. Apply temperature softmax (T=0.5) for final weights
    5. Output weights sum ≤ 1, cash = remainder
    """

    def __init__(
        self,
        temperature: float = 0.5,
        max_per_industry: int = 2,
        corr_threshold: float = 0.8,
        min_score_ratio: float = 0.7,
        top_n: int = 10,
        select_k: int = 5,
        corr_lookback: int = 60,
        turnover_penalty: float = 0.0,
        persistence_bonus: float = 0.0,
    ):
        self.temperature = temperature
        self.max_per_industry = max_per_industry
        self.corr_threshold = corr_threshold
        self.min_score_ratio = min_score_ratio
        self.top_n = top_n
        self.select_k = select_k
        self.corr_lookback = corr_lookback
        self.turnover_penalty = turnover_penalty
        self.persistence_bonus = persistence_bonus

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    def _compute_return_correlation(
        self, full_df: pd.DataFrame, stock_ids: List[str]
    ) -> pd.DataFrame:
        """Compute pairwise daily-return correlation over past N trading days.

        Args:
            full_df: DataFrame with columns [股票代码, 日期, ret1]
            stock_ids: List of stock codes to include

        Returns:
            Correlation matrix (DataFrame) indexed by stock_id.
            Missing/insufficient-data entries are filled with 0.0.
        """
        # Most recent trading days
        all_dates = sorted(full_df["日期"].unique())
        recent_dates = all_dates[-self.corr_lookback:]

        # Filter to candidate stocks × recent dates
        mask = (
            full_df["日期"].isin(recent_dates)
            & full_df["股票代码"].isin(stock_ids)
        )
        subset = full_df[mask].copy()

        if len(subset) == 0:
            return pd.DataFrame(0.0, index=stock_ids, columns=stock_ids)

        # Pivot: dates × stocks, values = ret1
        pivot = subset.pivot_table(
            index="日期",
            columns="股票代码",
            values="ret1",
            aggfunc="first",
        )

        # Drop stocks with too few observations (< 10 days)
        valid_mask = pivot.count() >= 10
        valid_stocks = pivot.columns[valid_mask]
        pivot = pivot[valid_stocks]

        # Compute Pearson correlation
        corr = pivot.corr()

        # Ensure all requested stock_ids appear in the matrix
        for sid in stock_ids:
            if sid not in corr.index:
                corr[sid] = 0.0
                corr.loc[sid] = 0.0

        return corr.fillna(0.0)

    # ------------------------------------------------------------------
    # Temperature softmax
    # ------------------------------------------------------------------

    def _temperature_softmax(self, scores: np.ndarray) -> np.ndarray:
        """Convert scores to weights via temperature-scaled softmax.

        Lower T → more concentrated weights on top scores.
        T=0.5 produces moderately concentrated allocation.
        """
        scores = np.array(scores, dtype=np.float64)
        if len(scores) == 0:
            return np.array([])
        scaled = scores / self.temperature
        scaled = scaled - np.max(scaled)  # numerical stability
        exp_scores = np.exp(scaled)
        weights = exp_scores / exp_scores.sum()
        return weights

    # ------------------------------------------------------------------
    # Main optimization entry point
    # ------------------------------------------------------------------

    def optimize(
        self,
        pred_df: pd.DataFrame,
        full_df: pd.DataFrame,
        prev_portfolio: Optional[set] = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run portfolio optimization.

        Args:
            pred_df: Prediction-day DataFrame with columns:
                     [股票代码, score, sector_id (optional)]
            full_df: Full historical DataFrame with columns:
                     [股票代码, 日期, ret1]
            verbose: Print optimization details to stdout.

        Returns:
            DataFrame with columns [stock_id, weight], sorted by weight desc.
        """
        # ── Step 1: Select top-N candidates ──
        top_n = min(self.top_n, len(pred_df))
        candidates = pred_df.nlargest(top_n, "score").copy()

        if len(candidates) == 0:
            raise ValueError("No stocks available for portfolio optimization")

        # Ensure sector_id column exists
        if "sector_id" not in candidates.columns:
            candidates["sector_id"] = 0

        top1_score = candidates["score"].iloc[0]
        score_threshold = top1_score * self.min_score_ratio

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Phase 5: Portfolio Optimization")
            print(f"{'=' * 50}")
            print(f"  Candidates:       {len(candidates)} (top-{top_n})")
            print(f"  Top-1 score:      {top1_score:.4f}")
            print(f"  Score threshold:  {score_threshold:.4f} "
                  f"({self.min_score_ratio * 100:.0f}% of top-1)")

        # ── Step 2: Correlation matrix ──
        stock_ids = candidates["股票代码"].tolist()
        corr_matrix = self._compute_return_correlation(full_df, stock_ids)

        if verbose:
            n_stocks_with_data = sum(
                1 for sid in stock_ids
                if sid in corr_matrix.index and corr_matrix.loc[sid, sid] > 0
            )
            print(f"  Corr matrix:      {n_stocks_with_data}/{len(stock_ids)} "
                  f"stocks with ≥10 days of return data")

        # ── Step 3: Greedy selection ──
        selected: List[Dict] = []  # [{stock_id, score, sector_id}]
        remaining_indices = list(range(len(candidates)))

        # Always start with the highest-score stock
        first_idx = 0
        first_row = candidates.iloc[first_idx]
        selected.append({
            "stock_id": first_row["股票代码"],
            "score": first_row["score"],
            "sector_id": int(first_row["sector_id"]),
        })
        remaining_indices.remove(first_idx)

        # Greedily add remaining stocks
        while len(selected) < self.select_k and remaining_indices:
            best_idx: Optional[int] = None
            best_adjusted_score = -np.inf

            for idx in remaining_indices:
                row = candidates.iloc[idx]
                stock_id = row["股票代码"]
                score = row["score"]
                sector_id = int(row["sector_id"])

                # ---- Quality filter ----
                if score < score_threshold:
                    continue

                # ---- Industry constraint ----
                sector_counts: Dict[int, int] = {}
                for s in selected:
                    sec = s["sector_id"]
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if sector_counts.get(sector_id, 0) >= self.max_per_industry:
                    continue

                # ---- Correlation penalty ----
                max_corr = 0.0
                for s in selected:
                    sel_id = s["stock_id"]
                    try:
                        c = abs(corr_matrix.loc[sel_id, stock_id])
                    except (KeyError, AttributeError):
                        c = 0.0
                    max_corr = max(max_corr, c)

                adjusted_score = score
                if max_corr > self.corr_threshold:
                    # Linear penalty: excess correlation → score reduction
                    excess = (
                        (max_corr - self.corr_threshold)
                        / (1.0 - self.corr_threshold)
                    )
                    adjusted_score = score * (1.0 - excess)

                # Turnover penalty / persistence bonus
                if prev_portfolio is not None:
                    if stock_id in prev_portfolio:
                        adjusted_score += self.persistence_bonus
                    else:
                        adjusted_score -= self.turnover_penalty

                if adjusted_score > best_adjusted_score:
                    best_adjusted_score = adjusted_score
                    best_idx = idx

            if best_idx is None:
                if verbose:
                    print(f"  [WARN] No more valid candidates "
                          f"(selected {len(selected)}/{self.select_k})")
                break

            row = candidates.iloc[best_idx]
            selected.append({
                "stock_id": row["股票代码"],
                "score": row["score"],
                "sector_id": int(row["sector_id"]),
            })
            remaining_indices.remove(best_idx)

        # ── Step 4: Temperature softmax weights ──
        selected_scores = np.array([s["score"] for s in selected])
        weights = self._temperature_softmax(selected_scores)

        # ── Step 5: Build output ──
        result = pd.DataFrame({
            "stock_id": [str(s["stock_id"]).zfill(6) for s in selected],
            "weight": np.round(weights, 6),
        })

        weight_sum = result["weight"].sum()
        cash_weight = max(0.0, 1.0 - weight_sum)

        if verbose:
            print(f"\n  Portfolio ({len(selected)}/{self.select_k} stocks):")
            print(f"  {'stock_id':<10s} {'weight':>8s}  {'score':>8s}  "
                  f"{'sector':>6s}  {'penalty':>8s}")
            print(f"  {'─' * 50}")
            for i, (_, row) in enumerate(result.iterrows()):
                s = selected[i]
                # Show correlation penalty for non-first stocks
                if i == 0:
                    pen_str = "─"
                else:
                    max_c = max(
                        abs(corr_matrix.loc[prev["stock_id"], s["stock_id"]])
                        for prev in selected[:i]
                        if prev["stock_id"] in corr_matrix.index
                        and s["stock_id"] in corr_matrix.columns
                    ) if selected[:i] else 0.0
                    pen_str = f"corr={max_c:.2f}"
                print(f"  {row['stock_id']:<10s} {row['weight']:>8.4f}  "
                      f"{s['score']:>8.4f}  {s['sector_id']:>6d}  {pen_str:>8s}")

            print(f"  {'─' * 50}")
            print(f"  Weight sum: {weight_sum:.4f}")
            print(f"  Cash:       {cash_weight:.4f}")

        return result
