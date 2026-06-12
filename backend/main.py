import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure code/src is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "src"))

# Import routers - these will be created in later tasks
# For now, create placeholders that will be filled in
from backend.api.market import router as market_router
from backend.api.predict import router as predict_router
from backend.api.backtest import router as backtest_router
from backend.api.snapshots import router as snapshots_router
from backend.services.engine import PredictionEngine


# Global engine instance — loaded once at startup
engine: PredictionEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Loading models and data...")
    engine = PredictionEngine()
    engine.load()
    print(f"Engine ready. Available dates: {len(engine.valid_dates)}")
    yield
    print("Shutting down...")


app = FastAPI(
    title="CSI300 AlphaMamba API",
    description="Stock prediction and analysis API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router)
app.include_router(predict_router)
app.include_router(backtest_router)
app.include_router(snapshots_router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine_loaded": engine is not None and engine.ready}
