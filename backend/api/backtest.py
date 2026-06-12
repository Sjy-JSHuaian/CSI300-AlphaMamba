import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from backend.schemas.models import BacktestRequest

router = APIRouter(prefix="/api", tags=["backtest"])


def _get_engine():
    from backend.main import engine
    return engine


@router.post("/backtest")
async def run_backtest(req: BacktestRequest):
    """Run backtest over a date range."""
    engine = _get_engine()
    if engine is None:
        raise HTTPException(503, "Engine not loaded")

    dates = [d for d in engine.valid_dates if req.start_date <= d <= req.end_date]
    if not dates:
        raise HTTPException(400, f"No valid dates in range {req.start_date} to {req.end_date}")

    # Simple backtest: run prediction for each date, track cumulative
    daily_rets = []
    for date_str in dates:
        try:
            result = engine.predict(date_str, req.mode, {})
            # Placeholder: actual return tracking requires next-day data
            daily_rets.append({"date": date_str, "return": 0.0})
        except Exception:
            daily_rets.append({"date": date_str, "return": 0.0})

    ret_series = pd.Series(
        [r["return"] for r in daily_rets],
        index=pd.to_datetime([r["date"] for r in daily_rets]),
    )

    # Compute metrics
    if len(ret_series) == 0:
        metrics = {"cumulative_return": 0, "sharpe": 0, "max_drawdown": 0, "win_rate": 0}
    else:
        cumulative = (1 + ret_series).cumprod()
        cr = float(cumulative.iloc[-1] - 1)
        mean_ret = ret_series.mean()
        std_ret = ret_series.std()
        sharpe = float((mean_ret / std_ret) * (252 ** 0.5)) if std_ret > 0 else 0.0
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = float(drawdown.min())
        win_rate = float((ret_series > 0).mean())
        metrics = {
            "cumulative_return": round(cr, 4),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
        }

    daily_cumulative = (1 + ret_series).cumprod()
    daily_returns = [
        {"date": str(idx.date()), "cumulative_return": round(float(val - 1), 6)}
        for idx, val in daily_cumulative.items()
    ]

    monthly = ret_series.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_returns = [
        {"year": int(idx.year), "month": int(idx.month), "return": round(float(val), 4)}
        for idx, val in monthly.items()
    ]

    return {
        "mode": req.mode,
        "metrics": metrics,
        "daily_returns": daily_returns,
        "monthly_returns": monthly_returns,
    }


@router.get("/history")
async def get_history(limit: int = Query(default=30, le=100)):
    """Get recent prediction history (last N valid dates)."""
    engine = _get_engine()
    if engine is None:
        raise HTTPException(503, "Engine not loaded")

    recent_dates = engine.valid_dates[-limit:]
    records = []
    for date_str in recent_dates:
        try:
            result = engine.predict(date_str, "phase6b", {})
            records.append({
                "date": date_str,
                "bull_strength": result["bull_strength"],
                "regime_label": result["regime_label"],
                "stocks": [p["stock_id"] for p in result["portfolio"]],
                "weights": [p["weight"] for p in result["portfolio"]],
            })
        except Exception:
            records.append({"date": date_str, "error": "prediction failed"})

    return {"history": records, "count": len(records)}
