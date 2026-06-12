"""Snapshot storage — saves every prediction run as immutable historical record."""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict


class SnapshotStore:
    """Manages prediction snapshots on disk."""

    def __init__(self, snapshots_dir: str = "snapshots"):
        self.snapshots_dir = snapshots_dir
        os.makedirs(snapshots_dir, exist_ok=True)

    def save(self, prediction_result: dict) -> str:
        """Save a prediction result as a snapshot. Returns the snapshot path."""
        date_str = prediction_result["date"]
        filepath = os.path.join(self.snapshots_dir, f"{date_str}.json")

        snapshot = {
            "date": date_str,
            "saved_at": datetime.now().isoformat(),
            "mode": prediction_result.get("mode", "phase6b"),
            "market_regime": prediction_result.get("regime_label", ""),
            "bull_strength": prediction_result.get("bull_strength", 0),
            "portfolio": prediction_result.get("portfolio", []),
            "bull_top5": prediction_result.get("bull_top5", []),
            "nonbull_top5": prediction_result.get("nonbull_top5", []),
            "consensus_overlap": prediction_result.get("consensus_overlap", 0),
            "score_distribution": prediction_result.get("score_distribution", {}),
            "sector_distribution": prediction_result.get("sector_distribution", {}),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        return filepath

    def load(self, date_str: str) -> Optional[dict]:
        """Load a snapshot by date."""
        filepath = os.path.join(self.snapshots_dir, f"{date_str}.json")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_dates(self) -> List[str]:
        """List all snapshot dates."""
        if not os.path.exists(self.snapshots_dir):
            return []
        files = [f for f in os.listdir(self.snapshots_dir) if f.endswith(".json")]
        dates = [f.replace(".json", "") for f in files]
        return sorted(dates)

    def list_all(self, limit: int = 100, start_date: str = None, end_date: str = None) -> List[dict]:
        """List snapshots with optional filtering."""
        dates = self.list_dates()
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
        dates = dates[-limit:]

        snapshots = []
        for d in dates:
            s = self.load(d)
            if s:
                snapshots.append(s)
        return snapshots

    def count(self) -> int:
        return len(self.list_dates())

    def latest(self) -> Optional[dict]:
        dates = self.list_dates()
        if not dates:
            return None
        return self.load(dates[-1])
