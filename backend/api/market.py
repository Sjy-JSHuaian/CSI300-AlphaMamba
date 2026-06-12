from fastapi import APIRouter, HTTPException, Query
from backend.schemas.models import MarketStatusResponse

router = APIRouter(prefix="/api/market", tags=["market"])


def _get_engine():
    from backend.main import engine
    return engine


@router.get("/dates")
async def get_dates():
    """Return all valid prediction dates."""
    engine = _get_engine()
    if engine is None:
        raise HTTPException(503, "Engine not loaded")
    return {
        "dates": engine.valid_dates,
        "count": len(engine.valid_dates),
        "first": engine.valid_dates[0] if engine.valid_dates else None,
        "last": engine.valid_dates[-1] if engine.valid_dates else None,
    }


@router.get("/status")
async def get_status(date: str = Query(..., description="YYYY-MM-DD")):
    """Return market status for a given date."""
    engine = _get_engine()
    if engine is None:
        raise HTTPException(503, "Engine not loaded")

    try:
        pred = engine._get_pred_day(date)
    except ValueError:
        raise HTTPException(404, f"No data for date {date}")

    bs = float(pred["bull_strength"].iloc[0])
    regime = str(pred["regime"].iloc[0])
    n = len(pred)

    return MarketStatusResponse(
        date=date,
        bull_strength=round(bs, 4),
        regime_label=regime,
        n_stocks=n,
    )
