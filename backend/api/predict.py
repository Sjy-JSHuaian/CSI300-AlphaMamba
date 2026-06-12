from fastapi import APIRouter, HTTPException
from backend.schemas.models import (
    PredictRequest, CompareRequest
)

router = APIRouter(prefix="/api/predict", tags=["predict"])


def _get_engine():
    from backend.main import engine
    return engine


def _to_response(result: dict) -> dict:
    return {
        "date": result["date"],
        "mode": result["mode"],
        "bull_strength": result["bull_strength"],
        "regime_label": result["regime_label"],
        "portfolio": result["portfolio"],
        "bull_top5": result.get("bull_top5", []),
        "nonbull_top5": result.get("nonbull_top5", []),
        "consensus_overlap": result.get("consensus_overlap", 0),
        "score_distribution": result.get("score_distribution", {}),
        "sector_distribution": result.get("sector_distribution", {}),
    }


@router.post("")
async def predict(req: PredictRequest):
    """Run single-mode prediction."""
    engine = _get_engine()
    if engine is None:
        raise HTTPException(503, "Engine not loaded")

    try:
        params = req.model_dump(exclude={"date", "mode"})
        result = engine.predict(req.date, req.mode, params)
        return _to_response(result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@router.post("/compare")
async def compare(req: CompareRequest):
    """Run multi-mode parallel prediction comparison."""
    engine = _get_engine()
    if engine is None:
        raise HTTPException(503, "Engine not loaded")

    results = {}
    first_result = None
    for mode in req.modes:
        try:
            result = engine.predict(req.date, mode, {})
            results[mode] = _to_response(result)
            if first_result is None:
                first_result = result
        except Exception as e:
            results[mode] = {"error": str(e)}

    return {
        "date": req.date,
        "bull_strength": first_result["bull_strength"] if first_result else 0,
        "regime_label": first_result.get("regime_label", "") if first_result else "",
        "results": results,
    }
