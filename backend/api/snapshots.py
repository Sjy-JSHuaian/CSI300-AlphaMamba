"""Snapshot and history API endpoints."""
from fastapi import APIRouter, HTTPException, Query
from backend.services.snapshot import SnapshotStore
from backend.services.performance import PerformanceTracker

router = APIRouter(prefix="/api", tags=["snapshots"])

_store = SnapshotStore()
_tracker = PerformanceTracker(_store)


def _get_engine():
    from backend.main import engine
    return engine


@router.get("/snapshots")
async def list_snapshots(
    limit: int = Query(default=100, le=500),
    start_date: str = Query(default=None),
    end_date: str = Query(default=None),
):
    """List prediction snapshots with optional filtering."""
    snapshots = _store.list_all(limit=limit, start_date=start_date, end_date=end_date)
    return {
        "snapshots": snapshots,
        "count": len(snapshots),
        "total": _store.count(),
    }


@router.get("/snapshots/compare")
async def compare_snapshots(date1: str = Query(...), date2: str = Query(...)):
    """Compare two prediction dates."""
    s1 = _store.load(date1)
    s2 = _store.load(date2)

    if not s1:
        raise HTTPException(404, f"No snapshot for {date1}")
    if not s2:
        raise HTTPException(404, f"No snapshot for {date2}")

    stocks1 = {s["stock_id"]: s for s in s1.get("portfolio", [])}
    stocks2 = {s["stock_id"]: s for s in s2.get("portfolio", [])}

    all_stocks = set(stocks1.keys()) | set(stocks2.keys())
    stayed = set(stocks1.keys()) & set(stocks2.keys())
    entered = set(stocks2.keys()) - set(stocks1.keys())
    left = set(stocks1.keys()) - set(stocks2.keys())

    ranking_changes = []
    for sid in stayed:
        rank1 = list(stocks1.keys()).index(sid) + 1
        rank2 = list(stocks2.keys()).index(sid) + 1
        score_change = stocks2[sid]["score"] - stocks1[sid]["score"]
        ranking_changes.append({
            "stock_id": sid,
            "rank_delta": rank2 - rank1,
            "score_delta": round(score_change, 4),
        })

    return {
        "date1": date1, "date2": date2,
        "regime_change": {
            "bull_strength_1": s1.get("bull_strength", 0),
            "bull_strength_2": s2.get("bull_strength", 0),
            "delta": round(s2.get("bull_strength", 0) - s1.get("bull_strength", 0), 4),
        },
        "portfolio_changes": {
            "stayed": list(stayed), "entered": list(entered), "left": list(left),
            "overlap_pct": round(len(stayed) / max(len(all_stocks), 1), 4),
        },
        "ranking_changes": sorted(ranking_changes, key=lambda x: abs(x["rank_delta"]), reverse=True),
        "snapshot1": s1, "snapshot2": s2,
    }


@router.get("/snapshots/{date}")
async def get_snapshot(date: str):
    """Get a specific snapshot by date."""
    snapshot = _store.load(date)
    if snapshot is None:
        raise HTTPException(404, f"No snapshot for date {date}")
    return snapshot


@router.get("/performance")
async def get_performance():
    """Get model performance metrics over time."""
    timeseries = _tracker.compute_timeseries()
    accuracy = _tracker.compute_accuracy(_store.list_all(limit=500))
    stability = _tracker.compute_ranking_stability()

    return {
        "accuracy": accuracy,
        "timeseries": timeseries,
        "stability": stability,
        "total_snapshots": _store.count(),
    }


@router.post("/update")
async def trigger_update():
    """Trigger a new prediction run and save snapshot."""
    engine = _get_engine()
    if engine is None or not engine.ready:
        raise HTTPException(503, "Engine not loaded")

    try:
        # Run prediction on latest date
        latest_date = engine.valid_dates[-1]
        result = engine.predict(latest_date, "phase6b", {})

        # Also run other modes for richer snapshot
        consensus = engine.predict(latest_date, "consensus", {})

        # Save snapshot
        snapshot_path = _store.save(result)

        return {
            "status": "ok",
            "date": latest_date,
            "snapshot_path": snapshot_path,
            "portfolio": result["portfolio"],
            "bull_strength": result["bull_strength"],
            "consensus_overlap": consensus.get("consensus_overlap", 0),
            "total_snapshots": _store.count(),
        }
    except Exception as e:
        raise HTTPException(500, f"Update failed: {str(e)}")


@router.post("/backfill")
async def backfill_snapshots(start_date: str = None, end_date: str = None):
    """Backfill snapshots for all historical dates."""
    engine = _get_engine()
    if engine is None or not engine.ready:
        raise HTTPException(503, "Engine not loaded")

    dates = engine.valid_dates
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]

    created = 0
    skipped = 0
    errors = []

    for date_str in dates:
        if _store.load(date_str):
            skipped += 1
            continue
        try:
            result = engine.predict(date_str, "phase6b", {})
            _store.save(result)
            created += 1
        except Exception as e:
            errors.append({"date": date_str, "error": str(e)})

    return {
        "status": "ok",
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "total_dates": len(dates),
    }
