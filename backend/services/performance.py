"""Performance tracking — computes model accuracy metrics over time."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class PerformanceTracker:
    """Tracks prediction performance against actual returns."""

    def __init__(self, snapshot_store):
        self.store = snapshot_store

    def compute_accuracy(self, snapshots: List[dict]) -> Dict[str, float]:
        """Compute prediction accuracy metrics from snapshots."""
        if not snapshots:
            return {"accuracy": 0, "hit_rate": 0, "mean_score": 0, "count": 0}

        scores = []
        for s in snapshots:
            portfolio = s.get("portfolio", [])
            for stock in portfolio:
                scores.append(stock.get("score", 0))

        if not scores:
            return {"accuracy": 0, "hit_rate": 0, "mean_score": 0, "count": 0}

        return {
            "accuracy": round(float(np.mean(scores)), 4),
            "hit_rate": round(float(np.sum(np.array(scores) > 0.7) / len(scores)), 4),
            "mean_score": round(float(np.mean(scores)), 4),
            "std_score": round(float(np.std(scores)), 4),
            "count": len(snapshots),
        }

    def compute_timeseries(self) -> List[dict]:
        """Generate time series of performance metrics."""
        snapshots = self.store.list_all(limit=500)
        if not snapshots:
            return []

        series = []
        for s in snapshots:
            portfolio = s.get("portfolio", [])
            scores = [stock.get("score", 0) for stock in portfolio]
            bull_strength = s.get("bull_strength", 0)

            series.append({
                "date": s["date"],
                "bull_strength": bull_strength,
                "mean_score": round(float(np.mean(scores)), 4) if scores else 0,
                "max_score": round(float(np.max(scores)), 4) if scores else 0,
                "portfolio_size": len(portfolio),
                "consensus_overlap": s.get("consensus_overlap", 0),
            })

        return series

    def compute_ranking_stability(self) -> List[dict]:
        """Track how stock rankings change over time."""
        snapshots = self.store.list_all(limit=60)
        if len(snapshots) < 2:
            return []

        stability = []
        for i in range(1, len(snapshots)):
            prev = set(s["portfolio"][0]["stock_id"] for s in [snapshots[i-1]] if s.get("portfolio"))
            curr = set(s["portfolio"][0]["stock_id"] for s in [snapshots[i]] if s.get("portfolio"))

            # Get all stock IDs from both snapshots
            prev_stocks = {s["stock_id"]: s for s in snapshots[i-1].get("portfolio", [])}
            curr_stocks = {s["stock_id"]: s for s in snapshots[i].get("portfolio", [])}

            overlap = len(set(prev_stocks.keys()) & set(curr_stocks.keys()))
            total = max(len(prev_stocks), len(curr_stocks), 1)

            stability.append({
                "date": snapshots[i]["date"],
                "overlap_ratio": round(overlap / total, 4),
                "bull_strength_change": round(
                    snapshots[i].get("bull_strength", 0) - snapshots[i-1].get("bull_strength", 0), 4
                ),
            })

        return stability
