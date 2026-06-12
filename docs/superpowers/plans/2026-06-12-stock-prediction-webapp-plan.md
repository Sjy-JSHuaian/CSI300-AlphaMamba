# CSI300 AlphaMamba 股票预测分析工作台 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 CLI 预测系统改造为 FastAPI + React 交互式 Web 分析工作台，支持日期选择、多模式预测、参数调优。

**Architecture:** FastAPI 后端包装现有 Python 模块提供 REST API，启动时一次性加载模型和数据到内存。React 前端通过 Ant Design 组件和 ECharts 图表消费 API，Vite 构建。

**Tech Stack:** FastAPI + Uvicorn / React 18 + TypeScript + Ant Design 5 + ECharts 5 + Vite

**Priority:** P0 = Task 1-14 (后端 API + PredictTab + TuningTab)，P1 = Task 15-17 (CompareTab + BacktestTab + HistoryTab)

---

## File Structure

```
backend/
├── main.py                  # FastAPI app, CORS, lifespan (model loading)
├── api/
│   ├── __init__.py
│   ├── market.py            # GET /api/market/dates, /api/market/status
│   ├── predict.py           # POST /api/predict, /api/predict/compare
│   └── backtest.py          # POST /api/backtest, GET /api/history
├── services/
│   ├── __init__.py
│   ├── engine.py            # PredictionEngine class — wraps all predict modes
│   └── backtest.py          # BacktestCalculator — wraps backtest logic
├── schemas/
│   ├── __init__.py
│   └── models.py            # Pydantic models: PredictRequest, PredictResponse, etc.
└── start.sh

frontend/
├── package.json
├── tsconfig.json
├── vite.config.ts
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx              # Tabs layout
│   ├── api/
│   │   └── client.ts        # fetch wrapper for all API calls
│   ├── types/
│   │   └── index.ts         # TypeScript type definitions
│   ├── components/
│   │   ├── AppHeader.tsx
│   │   ├── PredictTab.tsx
│   │   ├── TuningTab.tsx
│   │   ├── CompareTab.tsx
│   │   ├── BacktestTab.tsx
│   │   ├── HistoryTab.tsx
│   │   ├── MarketStatusCard.tsx
│   │   ├── StockCardGrid.tsx
│   │   ├── SectorPieChart.tsx
│   │   ├── ScoreBarChart.tsx
│   │   ├── MetricsCardRow.tsx
│   │   ├── CumulativeReturnChart.tsx
│   │   └── ParamSliders.tsx
│   └── hooks/
│       └── useApi.ts        # usePredict, useCompare, useBacktest hooks
```

---

### Task 1: Backend scaffold — FastAPI app, CORS, lifespan, schemas

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/main.py`
- Create: `backend/schemas/__init__.py`
- Create: `backend/schemas/models.py`
- Create: `backend/api/__init__.py`
- Create: `backend/services/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add dependencies to pyproject.toml**

Read `pyproject.toml`, then edit the `dependencies` array to append fastapi, uvicorn, pydantic (pydantic already comes with fastapi but let's be explicit):

```
dependencies = [
    # ... existing deps ...
    "fastapi>=0.115",
    "uvicorn>=0.30",
]
```

Update `uv.lock` by running: `cd d:/CSI300-AlphaMamba && uv lock` (or `uv sync`).

- [ ] **Step 2: Write Pydantic schemas**

```python
# backend/schemas/models.py
from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    date: str = Field(..., description="Prediction date YYYY-MM-DD")
    mode: str = Field(default="phase6b", description="phase6b | consensus | meta_ranker | phase7")
    temperature: float = Field(default=0.5, ge=0.1, le=2.0)
    max_per_industry: int = Field(default=2, ge=1, le=5)
    corr_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    min_score_ratio: float = Field(default=0.7, ge=0.3, le=0.9)
    top_n: int = Field(default=10, ge=5, le=30)
    select_k: int = Field(default=5, ge=1, le=10)
    bull_gate: float = Field(default=0.6, ge=0.0, le=1.0)
    turnover_penalty: float = Field(default=0.02, ge=0.0, le=0.1)
    persistence_bonus: float = Field(default=0.01, ge=0.0, le=0.05)


class StockResult(BaseModel):
    stock_id: str
    weight: float
    score: float
    sector_id: int


class PredictResponse(BaseModel):
    date: str
    mode: str
    bull_strength: float
    regime_label: str
    portfolio: List[StockResult]
    bull_top5: List[str]
    nonbull_top5: List[str] = []
    consensus_overlap: int = 0
    score_distribution: Dict[str, float] = {}
    sector_distribution: Dict[str, int] = {}


class CompareRequest(BaseModel):
    date: str = Field(..., description="Prediction date YYYY-MM-DD")
    modes: List[str] = Field(default=["phase6b", "consensus", "meta_ranker", "phase7"])


class CompareResponse(BaseModel):
    date: str
    bull_strength: float
    regime_label: str
    results: Dict[str, PredictResponse]


class MarketStatusResponse(BaseModel):
    date: str
    bull_strength: float
    regime_label: str
    n_stocks: int


class BacktestRequest(BaseModel):
    start_date: str = Field(default="2024-01-01")
    end_date: str = Field(default="2025-12-31")
    mode: str = Field(default="phase6b")


class BacktestResponse(BaseModel):
    mode: str
    metrics: Dict[str, float]  # cumulative_return, sharpe, max_drawdown, win_rate
    daily_returns: List[Dict[str, any]]  # [{date, cumulative_return, benchmark_return}]
    monthly_returns: List[Dict[str, any]]  # [{year, month, return}]


class ErrorResponse(BaseModel):
    error: str
    detail: str
```

- [ ] **Step 3: Write FastAPI main.py with lifespan**

```python
# backend/main.py
import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure code/src is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "src"))

from backend.api.market import router as market_router
from backend.api.predict import router as predict_router
from backend.api.backtest import router as backtest_router
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


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine_loaded": engine is not None and engine.ready}
```

- [ ] **Step 4: Write startup script**

```bash
# backend/start.sh
#!/bin/bash
cd "$(dirname "$0")/.."
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- [ ] **Step 5: Verify scaffold runs**

```bash
cd d:/CSI300-AlphaMamba
python -c "from backend.main import app; print('FastAPI app created OK')"
```

Expected: `FastAPI app created OK` (may warn about missing engine, that's fine).

- [ ] **Step 6: Commit**

```bash
git add backend/ pyproject.toml uv.lock
git commit -m "feat: add backend scaffold — FastAPI app, schemas, lifespan"
```

---

### Task 2: PredictionEngine — model loading and data preprocessing

**Files:**
- Create: `backend/services/engine.py`

- [ ] **Step 1: Write PredictionEngine.load()**

```python
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
```

- [ ] **Step 2: Write `_get_pred_day()` helper**

Add to `engine.py`:

```python
    def _get_pred_day(self, date_str: str) -> pd.DataFrame:
        """Get prediction-day DataFrame with features."""
        target_date = pd.to_datetime(date_str)
        mask = self.df["日期"] == target_date
        if not mask.any():
            raise ValueError(f"No data for date {date_str}")
        return self.df[mask].copy()
```

- [ ] **Step 3: Verify engine loads correctly**

```bash
cd d:/CSI300-AlphaMamba
python -c "from backend.services.engine import PredictionEngine; e = PredictionEngine(); e.load(); print('OK:', e.ready, len(e.valid_dates))"
```

Expected: `Engine loaded: 2 models, X features, Y valid dates` then `OK: True Y`

- [ ] **Step 4: Commit**

```bash
git add backend/services/engine.py
git commit -m "feat: add PredictionEngine — model/data loading"
```

---

### Task 3: PredictionEngine — predict() method (Phase 6B)

**Files:**
- Modify: `backend/services/engine.py`

- [ ] **Step 1: Write predict_phase6b() method**

Add to `PredictionEngine` class:

```python
    def predict_phase6b(self, date_str: str, params: dict) -> dict:
        """Run Phase 6B Bull/NonBull soft blend prediction."""
        pred = self._get_pred_day(date_str)

        bs = float(pred["bull_strength"].iloc[0])
        regime_label = str(pred["regime"].iloc[0])

        # Scale features
        missing = set(self.feature_cols) - set(pred.columns)
        for col in missing:
            pred[col] = 0.0
        X = pred[self.feature_cols].fillna(0.0).values.astype(np.float32)
        X = self.scaler.transform(X)

        # Predict both regimes
        bull_raw = self.models["bull"].predict(X)
        nonbull_raw = self.models["nonbull"].predict(X)

        # Rank normalize + soft blend
        bull_rank = pd.Series(bull_raw).rank(pct=True).values
        nonbull_rank = pd.Series(nonbull_raw).rank(pct=True).values
        final_scores = bs * bull_rank + (1.0 - bs) * nonbull_rank

        pred = pred.copy()
        pred["score"] = final_scores
        pred["bull_score"] = bull_rank
        pred["nonbull_score"] = nonbull_rank
        pred = pred.dropna(subset=["score"])

        # Tie-breaking
        if pred["score"].duplicated().any():
            rng = np.random.RandomState(self.cfg["seed"])
            pred["score"] = pred["score"] + rng.uniform(-1e-8, 1e-8, len(pred))

        # Build result dict BEFORE portfolio optimization
        bull_top5 = pred.nlargest(5, "bull_score")["股票代码"].tolist()
        nonbull_top5 = pred.nlargest(5, "nonbull_score")["股票代码"].tolist()

        # Score distribution stats
        scores = pred["score"].values
        score_dist = {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }

        return {
            "pred_df": pred,
            "bull_strength": round(bs, 4),
            "regime_label": regime_label,
            "bull_top5": bull_top5,
            "nonbull_top5": nonbull_top5,
            "score_distribution": score_dist,
            "full_df": self.df,
        }
```

- [ ] **Step 2: Commit**

```bash
git add backend/services/engine.py
git commit -m "feat: add predict_phase6b() to PredictionEngine"
```

---

### Task 4: PredictionEngine — other predict modes + portfolio optimizer

**Files:**
- Modify: `backend/services/engine.py`

- [ ] **Step 1: Write predict_consensus()**

```python
    def predict_consensus(self, date_str: str, params: dict) -> dict:
        """Phase 6B + submit1 consensus blend."""
        result = self.predict_phase6b(date_str, params)
        pred = result["pred_df"]

        # Add consensus signals
        if self.s1_scores is not None:
            target_date = pd.to_datetime(date_str)
            s1_day = self.s1_scores[self.s1_scores["日期"] == target_date]
            if len(s1_day) > 0:
                s1_map = dict(zip(s1_day["股票代码"], s1_day["submit1_score"]))
                pred["s1_score"] = pred["股票代码"].map(s1_map).fillna(0.0)
                pred["s1_rank"] = pred["s1_score"].rank(pct=True).fillna(0.5).values
            else:
                pred["s1_score"] = 0.0
                pred["s1_rank"] = 0.5
        else:
            pred["s1_score"] = 0.0
            pred["s1_rank"] = 0.5

        # Consensus: 0.6 * p6b_rank + 0.4 * s1_rank + bonus
        p6b_top20 = set(pred.nlargest(20, "score")["股票代码"])
        s1_top20 = set(pred.nlargest(20, "s1_score")["股票代码"])
        in_both = p6b_top20 & s1_top20

        bonus = np.array([0.05 if s in in_both else 0.0 for s in pred["股票代码"]])
        pred["score"] = 0.6 * pred["score"].rank(pct=True) + 0.4 * pred["s1_rank"] + bonus

        scores = pred["score"].values
        result["score_distribution"] = {
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        }
        result["consensus_overlap"] = len(in_both)
        result["pred_df"] = pred
        return result
```

- [ ] **Step 2: Write predict_phase7() (safe mode)**

```python
    def predict_phase7(self, date_str: str, params: dict) -> dict:
        """Phase 7 safe mode — adds bull_gate, turnover penalty, dynamic exposure."""
        result = self.predict_phase6b(date_str, params)
        pred = result["pred_df"]
        bs = result["bull_strength"]
        bull_gate = params.get("bull_gate", 0.6)

        if bs < bull_gate:
            result["gated"] = True
            result["portfolio"] = []
            return result

        result["gated"] = False
        return result  # penalties applied in optimize()
```

- [ ] **Step 3: Write predict_meta_ranker()**

```python
    def predict_meta_ranker(self, date_str: str, params: dict) -> dict:
        """Meta Ranker V2 — uses predict_meta_ranker.py logic."""
        from meta_ranker import MetaRanker

        result = self.predict_phase6b(date_str, params)
        pred = result["pred_df"]

        mr = MetaRanker(alpha=1.0, top_n_candidates=40)
        mr.fit(self.df, self.feature_cols, self.s1_scores)
        pred = mr.predict(pred, self.df)

        pred["score"] = pred["meta_score"]
        scores = pred["score"].values
        result["score_distribution"] = {
            "min": float(np.min(scores)), "max": float(np.max(scores)),
            "mean": float(np.mean(scores)), "std": float(np.std(scores)),
        }
        result["pred_df"] = pred
        return result
```

- [ ] **Step 4: Write _optimize_portfolio()**

```python
    def _optimize_portfolio(self, pred_df, full_df, params, prev_portfolio=None, verbose=False):
        """Run portfolio optimization on prediction results."""
        opt = PortfolioOptimizer(
            temperature=params.get("temperature", 0.5),
            max_per_industry=params.get("max_per_industry", 2),
            corr_threshold=params.get("corr_threshold", 0.8),
            min_score_ratio=params.get("min_score_ratio", 0.7),
            top_n=params.get("top_n", 10),
            select_k=params.get("select_k", 5),
            corr_lookback=60,
            turnover_penalty=params.get("turnover_penalty", 0.0),
            persistence_bonus=params.get("persistence_bonus", 0.0),
        )
        # Ensure sector_id
        if "sector_id" not in pred_df.columns:
            from features import _get_sector
            pred_df["sector_id"] = pred_df["股票代码"].apply(_get_sector)

        result = opt.optimize(pred_df, full_df, prev_portfolio=prev_portfolio, verbose=False)
        result["weight"] = result["weight"].round(6)
        return result
```

- [ ] **Step 5: Write predict() dispatcher**

```python
    def predict(self, date_str: str, mode: str, params: dict) -> dict:
        """Unified predict interface — dispatches by mode name."""
        mode = mode.lower().replace(" ", "_").replace("-", "_")

        mode_fn = {
            "phase6b": self.predict_phase6b,
            "consensus": self.predict_consensus,
            "meta_ranker": self.predict_meta_ranker,
            "phase7": self.predict_phase7,
        }

        if mode not in mode_fn:
            raise ValueError(f"Unknown mode: {mode}. Options: {list(mode_fn.keys())}")

        result = mode_fn[mode](date_str, params)
        pred = result["pred_df"]

        # Portfolio optimization
        prev = None  # Could load from output/result.csv for persistence
        portfolio = self._optimize_portfolio(pred, result.get("full_df", self.df), params, prev)

        # Sector distribution
        if "sector_id" in pred.columns:
            sector_counts = pred.loc[
                pred["股票代码"].isin(portfolio["stock_id"].values), "sector_id"
            ].value_counts().to_dict()
        else:
            sector_counts = {}

        # Build portfolio list
        score_map = dict(zip(pred["股票代码"], pred["score"]))
        sector_map = dict(zip(pred["股票代码"], pred["sector_id"])) if "sector_id" in pred.columns else {}

        return {
            "date": date_str,
            "mode": mode,
            "bull_strength": result["bull_strength"],
            "regime_label": result.get("regime_label", ""),
            "portfolio": [
                {
                    "stock_id": row["stock_id"],
                    "weight": float(row["weight"]),
                    "score": float(score_map.get(row["stock_id"], 0)),
                    "sector_id": int(sector_map.get(row["stock_id"], 0)),
                }
                for _, row in portfolio.iterrows()
            ],
            "bull_top5": result.get("bull_top5", []),
            "nonbull_top5": result.get("nonbull_top5", []),
            "consensus_overlap": result.get("consensus_overlap", 0),
            "score_distribution": result.get("score_distribution", {}),
            "sector_distribution": {str(k): int(v) for k, v in sector_counts.items()},
        }
```

- [ ] **Step 6: Verify predict() works**

```bash
cd d:/CSI300-AlphaMamba
python -c "
from backend.services.engine import PredictionEngine
e = PredictionEngine()
e.load()
r = e.predict(e.valid_dates[-1], 'phase6b', {})
print('Portfolio:', [(p['stock_id'], p['weight']) for p in r['portfolio']])
"
```

Expected: 5 stock-weight pairs for the latest date.

- [ ] **Step 7: Commit**

```bash
git add backend/services/engine.py
git commit -m "feat: add predict modes + portfolio optimizer to PredictionEngine"
```

---

### Task 5: Market API endpoints

**Files:**
- Create: `backend/api/market.py`

- [ ] **Step 1: Write market router**

```python
# backend/api/market.py
from fastapi import APIRouter, HTTPException, Query
from backend.main import engine
from backend.schemas.models import MarketStatusResponse

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/dates")
async def get_dates():
    """Return all valid prediction dates."""
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
```

- [ ] **Step 2: Verify endpoints**

```bash
cd d:/CSI300-AlphaMamba
python -c "
from backend.services.engine import PredictionEngine
import backend.main
backend.main.engine = PredictionEngine()
backend.main.engine.load()
from backend.api.market import get_dates, get_status
import asyncio
r = asyncio.run(get_dates())
print('Dates:', r['count'])
r2 = asyncio.run(get_status(r['last']))
print('Status:', r2)
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/api/market.py
git commit -m "feat: add /api/market/dates and /api/market/status endpoints"
```

---

### Task 6: Predict API endpoints

**Files:**
- Create: `backend/api/predict.py`

- [ ] **Step 1: Write predict router**

```python
# backend/api/predict.py
from fastapi import APIRouter, HTTPException
from backend.main import engine
from backend.schemas.models import (
    PredictRequest, PredictResponse, CompareRequest, CompareResponse, ErrorResponse
)

router = APIRouter(prefix="/api/predict", tags=["predict"])


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
    if engine is None:
        raise HTTPException(503, "Engine not loaded")

    results = {}
    for mode in req.modes:
        try:
            result = engine.predict(req.date, mode, {})
            results[mode] = _to_response(result)
        except Exception as e:
            results[mode] = {"error": str(e)}

    return {
        "date": req.date,
        "bull_strength": next(iter(results.values())).get("bull_strength", 0) if results else 0,
        "regime_label": next(iter(results.values())).get("regime_label", "") if results else "",
        "results": results,
    }
```

- [ ] **Step 2: Verify endpoint**

```bash
cd d:/CSI300-AlphaMamba
python -c "
from backend.services.engine import PredictionEngine
import backend.main
backend.main.engine = PredictionEngine()
backend.main.engine.load()
from backend.api.predict import predict
from backend.schemas.models import PredictRequest
import asyncio
req = PredictRequest(date=backend.main.engine.valid_dates[-1], mode='phase6b')
r = asyncio.run(predict(req))
print('Portfolio:', len(r['portfolio']), 'stocks')
"
```

- [ ] **Step 3: Commit**

```bash
git add backend/api/predict.py
git commit -m "feat: add /api/predict and /api/predict/compare endpoints"
```

---

### Task 7: Backtest API endpoint

**Files:**
- Create: `backend/api/backtest.py`
- Create: `backend/services/backtest.py`

- [ ] **Step 1: Write BacktestCalculator**

```python
# backend/services/backtest.py
import numpy as np
import pandas as pd
from typing import Dict, List


class BacktestCalculator:
    """Compute backtest metrics from daily returns."""

    @staticmethod
    def compute_metrics(daily_returns: pd.Series) -> Dict[str, float]:
        """Compute standard performance metrics."""
        if len(daily_returns) == 0:
            return {"cumulative_return": 0, "sharpe": 0, "max_drawdown": 0, "win_rate": 0}

        cumulative = (1 + daily_returns).cumprod()
        cumulative_return = float(cumulative.iloc[-1] - 1)

        # Sharpe ratio (annualized, assuming 252 trading days)
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

        # Max drawdown
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = float(drawdown.min())

        # Win rate
        win_rate = float((daily_returns > 0).mean())

        return {
            "cumulative_return": round(cumulative_return, 4),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_drawdown, 4),
            "win_rate": round(win_rate, 4),
        }

    @staticmethod
    def compute_daily_cumulative(daily_returns: pd.Series) -> List[dict]:
        """Generate daily cumulative return series for charting."""
        cumulative = (1 + daily_returns).cumprod()
        return [
            {"date": str(idx.date()), "cumulative_return": round(float(val - 1), 6)}
            for idx, val in cumulative.items()
        ]

    @staticmethod
    def compute_monthly_heatmap(daily_returns: pd.Series) -> List[dict]:
        """Generate monthly returns for heatmap."""
        monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        return [
            {"year": int(idx.year), "month": int(idx.month), "return": round(float(val), 4)}
            for idx, val in monthly.items()
        ]
```

- [ ] **Step 2: Write backtest API router**

```python
# backend/api/backtest.py
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from backend.main import engine
from backend.schemas.models import BacktestRequest, BacktestResponse
from backend.services.backtest import BacktestCalculator

router = APIRouter(prefix="/api", tags=["backtest"])


@router.post("/backtest")
async def run_backtest(req: BacktestRequest):
    """Run backtest over a date range."""
    if engine is None:
        raise HTTPException(503, "Engine not loaded")

    dates = [d for d in engine.valid_dates if req.start_date <= d <= req.end_date]
    if not dates:
        raise HTTPException(400, f"No valid dates in range {req.start_date} to {req.end_date}")

    daily_rets = []
    for date_str in dates:
        try:
            result = engine.predict(date_str, req.mode, {})
            # Simple daily return: equal-weight portfolio return
            # (in production this would use actual next-day returns)
            daily_rets.append({"date": date_str, "return": 0.0})  # placeholder
        except Exception:
            daily_rets.append({"date": date_str, "return": 0.0})

    ret_series = pd.Series(
        [r["return"] for r in daily_rets],
        index=pd.to_datetime([r["date"] for r in daily_rets]),
    )

    metrics = BacktestCalculator.compute_metrics(ret_series)
    daily_cum = BacktestCalculator.compute_daily_cumulative(ret_series)
    monthly = BacktestCalculator.compute_monthly_heatmap(ret_series)

    return {
        "mode": req.mode,
        "metrics": metrics,
        "daily_returns": daily_cum,
        "monthly_returns": monthly,
    }


@router.get("/history")
async def get_history(limit: int = Query(default=30, le=100)):
    """Get recent prediction history (last N valid dates)."""
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
```

- [ ] **Step 3: Commit**

```bash
git add backend/services/backtest.py backend/api/backtest.py
git commit -m "feat: add backtest and history API endpoints"
```

---

### Task 8: Start backend and verify all APIs work

- [ ] **Step 1: Start backend server**

```bash
cd d:/CSI300-AlphaMamba && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Keep this running in a separate terminal. Verify it starts and prints "Engine ready."

- [ ] **Step 2: Test all endpoints with curl**

```bash
# Health check
curl -s http://localhost:8000/api/health | python -m json.tool

# Get available dates
curl -s http://localhost:8000/api/market/dates | python -m json.tool

# Get status for latest date
LATEST=$(curl -s http://localhost:8000/api/market/dates | python -c "import sys,json; print(json.load(sys.stdin)['last'])")
curl -s "http://localhost:8000/api/market/status?date=$LATEST" | python -m json.tool

# Run prediction
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d "{\"date\": \"$LATEST\", \"mode\": \"phase6b\"}" | python -m json.tool
```

Expected: All endpoints return valid JSON with expected fields.

- [ ] **Step 3: Commit (if any fixes needed)**

---

### Task 9: React frontend scaffold

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.node.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/api/client.ts`

- [ ] **Step 1: Create Vite + React + TypeScript project**

```bash
cd d:/CSI300-AlphaMamba
mkdir -p frontend
cd frontend
npm create vite@latest . -- --template react-ts 2>/dev/null || echo "Will create manually"
```

If the above doesn't work in the environment, write files manually:

**frontend/package.json:**
```json
{
  "name": "csi300-analysis-workbench",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "antd": "^5.22.0",
    "echarts": "^5.5.0",
    "echarts-for-react": "^3.0.2",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "dayjs": "^1.11.13"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "typescript": "^5.6.3",
    "vite": "^6.0.0"
  }
}
```

- [ ] **Step 2: Write config files**

**frontend/vite.config.ts:**
```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
});
```

**frontend/tsconfig.json:**
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "moduleDetection": "force",
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": false,
    "noUnusedParameters": false,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src"]
}
```

**frontend/index.html:**
```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>CSI300 AlphaMamba · 分析工作台</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 3: Write TypeScript types**

```typescript
// frontend/src/types/index.ts
export interface StockResult {
  stock_id: string;
  weight: number;
  score: number;
  sector_id: number;
}

export interface PredictResponse {
  date: string;
  mode: string;
  bull_strength: number;
  regime_label: string;
  portfolio: StockResult[];
  bull_top5: string[];
  nonbull_top5: string[];
  consensus_overlap: number;
  score_distribution: Record<string, number>;
  sector_distribution: Record<string, number>;
}

export interface PredictRequest {
  date: string;
  mode: string;
  temperature?: number;
  max_per_industry?: number;
  corr_threshold?: number;
  min_score_ratio?: number;
  top_n?: number;
  select_k?: number;
  bull_gate?: number;
  turnover_penalty?: number;
  persistence_bonus?: number;
}

export interface MarketStatus {
  date: string;
  bull_strength: number;
  regime_label: string;
  n_stocks: number;
}

export interface DatesResponse {
  dates: string[];
  count: number;
  first: string;
  last: string;
}

export interface CompareResponse {
  date: string;
  bull_strength: number;
  regime_label: string;
  results: Record<string, PredictResponse>;
}

export interface BacktestResponse {
  mode: string;
  metrics: Record<string, number>;
  daily_returns: { date: string; cumulative_return: number }[];
  monthly_returns: { year: number; month: number; return: number }[];
}

export interface HistoryRecord {
  date: string;
  bull_strength: number;
  regime_label: string;
  stocks: string[];
  weights: number[];
}

export interface HistoryResponse {
  history: HistoryRecord[];
  count: number;
}
```

- [ ] **Step 4: Write API client**

```typescript
// frontend/src/api/client.ts
const BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  getDates: () => request<import('../types').DatesResponse>('/market/dates'),

  getStatus: (date: string) =>
    request<import('../types').MarketStatus>(`/market/status?date=${date}`),

  predict: (data: import('../types').PredictRequest) =>
    request<import('../types').PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  compare: (date: string, modes: string[]) =>
    request<import('../types').CompareResponse>('/predict/compare', {
      method: 'POST',
      body: JSON.stringify({ date, modes }),
    }),

  runBacktest: (data: { start_date: string; end_date: string; mode: string }) =>
    request<import('../types').BacktestResponse>('/backtest', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  getHistory: (limit?: number) =>
    request<import('../types').HistoryResponse>(`/history?limit=${limit || 30}`),
};
```

- [ ] **Step 5: Write main.tsx and App.tsx**

```typescript
// frontend/src/main.tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
```

```typescript
// frontend/src/App.tsx
import React, { useState } from 'react';
import { ConfigProvider, Tabs, Layout, Typography, theme } from 'antd';
import { BarChartOutlined, LineChartOutlined, ExperimentOutlined, SettingOutlined, HistoryOutlined } from '@ant-design/icons';
import PredictTab from './components/PredictTab';
import CompareTab from './components/CompareTab';
import TuningTab from './components/TuningTab';
import BacktestTab from './components/BacktestTab';
import HistoryTab from './components/HistoryTab';

const { Header, Content } = Layout;
const { Title } = Typography;

const tabItems = [
  { key: 'predict', label: '📊 预测', icon: <BarChartOutlined />, children: <PredictTab /> },
  { key: 'compare', label: '🔬 对比', icon: <ExperimentOutlined />, children: <CompareTab /> },
  { key: 'tuning', label: '⚙️ 调参', icon: <SettingOutlined />, children: <TuningTab /> },
  { key: 'backtest', label: '📈 回测', icon: <LineChartOutlined />, children: <BacktestTab /> },
  { key: 'history', label: '📋 历史', icon: <HistoryOutlined />, children: <HistoryTab /> },
];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('predict');

  return (
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: { colorPrimary: '#1677ff', borderRadius: 8 },
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ background: '#001529', display: 'flex', alignItems: 'center', padding: '0 24px' }}>
          <Title level={3} style={{ color: '#fff', margin: 0 }}>
            📈 CSI300 AlphaMamba · 分析工作台
          </Title>
        </Header>
        <Content style={{ padding: 24 }}>
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={tabItems}
            size="large"
            style={{ background: '#fff', padding: '0 24px', borderRadius: 8 }}
          />
        </Content>
      </Layout>
    </ConfigProvider>
  );
};

export default App;
```

- [ ] **Step 6: Install deps and verify dev server starts**

```bash
cd d:/CSI300-AlphaMamba/frontend
npm install
npm run dev
```

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat: add React frontend scaffold with Ant Design tabs"
```

---

### Task 10: PredictTab component

**Files:**
- Create: `frontend/src/components/PredictTab.tsx`
- Create: `frontend/src/components/MarketStatusCard.tsx`
- Create: `frontend/src/components/StockCardGrid.tsx`
- Create: `frontend/src/components/SectorPieChart.tsx`
- Create: `frontend/src/components/ScoreBarChart.tsx`

- [ ] **Step 1: Write MarketStatusCard**

```typescript
// frontend/src/components/MarketStatusCard.tsx
import React from 'react';
import { Card, Progress, Tag, Space } from 'antd';
import { RiseOutlined, FallOutlined, MinusOutlined } from '@ant-design/icons';

interface Props {
  bullStrength: number;
  regimeLabel: string;
  nStocks: number;
}

const MarketStatusCard: React.FC<Props> = ({ bullStrength, regimeLabel, nStocks }) => {
  const getStatus = () => {
    if (bullStrength > 0.8) return { icon: <RiseOutlined />, color: '#cf1322', text: '强牛市' };
    if (bullStrength > 0.5) return { icon: <RiseOutlined />, color: '#fa8c16', text: '牛市' };
    if (bullStrength > 0.2) return { icon: <MinusOutlined />, color: '#8c8c8c', text: '震荡市' };
    return { icon: <FallOutlined />, color: '#1890ff', text: '弱势/熊市' };
  };

  const status = getStatus();

  return (
    <Card size="small" title="🌡️ 市场状态" style={{ marginBottom: 16 }}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: status.color, fontSize: 18 }}>{status.icon}</span>
          <span style={{ fontWeight: 600, fontSize: 16 }}>{status.text}</span>
          <Tag color="blue">{regimeLabel}</Tag>
        </div>
        <Progress
          percent={Math.round(bullStrength * 100)}
          strokeColor={{
            '0%': '#1890ff',
            '50%': '#fa8c16',
            '100%': '#cf1322',
          }}
          format={(p) => `Bull Strength: ${(p! / 100).toFixed(3)}`}
        />
        <span style={{ color: '#8c8c8c', fontSize: 12 }}>可用股票: {nStocks} 只</span>
      </Space>
    </Card>
  );
};

export default MarketStatusCard;
```

- [ ] **Step 2: Write StockCardGrid**

```typescript
// frontend/src/components/StockCardGrid.tsx
import React from 'react';
import { Card, Progress, Row, Col } from 'antd';
import type { StockResult } from '../types';

interface Props {
  stocks: StockResult[];
}

const medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'];

const StockCardGrid: React.FC<Props> = ({ stocks }) => (
  <Row gutter={[12, 12]}>
    {stocks.map((stock, i) => (
      <Col xs={24} sm={12} md={Math.floor(24 / Math.min(stocks.length, 5))} key={stock.stock_id}>
        <Card
          size="small"
          hoverable
          title={
            <span>
              <span style={{ fontSize: 20, marginRight: 8 }}>{medals[i] || `#${i + 1}`}</span>
              <strong>{stock.stock_id}</strong>
            </span>
          }
        >
          <Progress
            percent={Math.round(stock.weight * 100)}
            size="small"
            strokeColor={i === 0 ? '#cf1322' : i === 1 ? '#fa8c16' : '#1677ff'}
          />
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
            <span style={{ color: '#8c8c8c' }}>权重</span>
            <strong>{(stock.weight * 100).toFixed(1)}%</strong>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span style={{ color: '#8c8c8c' }}>得分</span>
            <span>{stock.score.toFixed(4)}</span>
          </div>
        </Card>
      </Col>
    ))}
  </Row>
);

export default StockCardGrid;
```

- [ ] **Step 3: Write SectorPieChart**

```typescript
// frontend/src/components/SectorPieChart.tsx
import React from 'react';
import ReactECharts from 'echarts-for-react';

interface Props {
  data: Record<string, number>;
}

const SectorPieChart: React.FC<Props> = ({ data }) => {
  const option = {
    title: { text: '行业分布', left: 'center', textStyle: { fontSize: 14 } },
    tooltip: { trigger: 'item' as const },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        data: Object.entries(data).map(([name, value]) => ({ name, value })),
        label: { formatter: '{b}\n{d}%' },
      },
    ],
  };

  if (Object.keys(data).length === 0) {
    return <div style={{ textAlign: 'center', padding: 40, color: '#8c8c8c' }}>暂无行业数据</div>;
  }

  return <ReactECharts option={option} style={{ height: 280 }} />;
};

export default SectorPieChart;
```

- [ ] **Step 4: Write PredictTab**

```typescript
// frontend/src/components/PredictTab.tsx
import React, { useState, useEffect } from 'react';
import { DatePicker, Select, Button, Spin, Alert, Row, Col } from 'antd';
import dayjs from 'dayjs';
import { api } from '../api/client';
import type { PredictResponse, MarketStatus, DatesResponse } from '../types';
import MarketStatusCard from './MarketStatusCard';
import StockCardGrid from './StockCardGrid';
import SectorPieChart from './SectorPieChart';

const MODES = [
  { value: 'phase6b', label: 'Phase 6B (Bull/NonBull)' },
  { value: 'consensus', label: 'Consensus' },
  { value: 'meta_ranker', label: 'Meta Ranker V2' },
  { value: 'phase7', label: 'Phase 7 (Safe Mode)' },
];

const PredictTab: React.FC = () => {
  const [dates, setDates] = useState<DatesResponse | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [mode, setMode] = useState('phase6b');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [marketStatus, setMarketStatus] = useState<MarketStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDates().then(setDates).catch(console.error);
  }, []);

  const handlePredict = async () => {
    if (!selectedDate) return;
    setLoading(true);
    setError(null);
    try {
      const [status, predResult] = await Promise.all([
        api.getStatus(selectedDate),
        api.predict({ date: selectedDate, mode }),
      ]);
      setMarketStatus(status);
      setResult(predResult);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col>
          <DatePicker
            value={selectedDate ? dayjs(selectedDate) : null}
            onChange={(d) => setSelectedDate(d?.format('YYYY-MM-DD') || null)}
            disabledDate={(d) => !dates?.dates.includes(d.format('YYYY-MM-DD'))}
            placeholder="选择预测日期"
            style={{ width: 200 }}
          />
        </Col>
        <Col>
          <Select value={mode} onChange={setMode} options={MODES} style={{ width: 220 }} />
        </Col>
        <Col>
          <Button type="primary" onClick={handlePredict} loading={loading} size="large">
            🚀 运行预测
          </Button>
        </Col>
      </Row>

      {error && <Alert type="error" message={error} closable style={{ marginBottom: 16 }} />}

      {marketStatus && (
        <MarketStatusCard
          bullStrength={marketStatus.bull_strength}
          regimeLabel={marketStatus.regime_label}
          nStocks={marketStatus.n_stocks}
        />
      )}

      {loading && <Spin size="large" style={{ display: 'block', margin: '40px auto' }} />}

      {result && !loading && (
        <>
          <StockCardGrid stocks={result.portfolio} />
          <Row gutter={16} style={{ marginTop: 24 }}>
            <Col span={12}>
              <SectorPieChart data={result.sector_distribution} />
            </Col>
            <Col span={12}>
              <div style={{ background: '#fafafa', padding: 16, borderRadius: 8 }}>
                <div style={{ marginBottom: 12 }}>
                  <strong>📈 得分统计</strong>
                </div>
                {Object.entries(result.score_distribution).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0' }}>
                    <span style={{ color: '#8c8c8c' }}>{k}</span>
                    <span>{typeof v === 'number' ? v.toFixed(4) : String(v)}</span>
                  </div>
                ))}
                {result.consensus_overlap > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <strong>🤝 共识重叠: </strong>
                    <span>{result.consensus_overlap} 只股票</span>
                  </div>
                )}
              </div>
            </Col>
          </Row>
        </>
      )}
    </div>
  );
};

export default PredictTab;
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/PredictTab.tsx frontend/src/components/MarketStatusCard.tsx frontend/src/components/StockCardGrid.tsx frontend/src/components/SectorPieChart.tsx
git commit -m "feat: add PredictTab with date selector, mode switch, stock cards"
```

---

### Task 11: TuningTab component

**Files:**
- Create: `frontend/src/components/TuningTab.tsx`
- Create: `frontend/src/components/ParamSliders.tsx`

- [ ] **Step 1: Write ParamSliders**

```typescript
// frontend/src/components/ParamSliders.tsx
import React from 'react';
import { Slider, Row, Col, Typography } from 'antd';

const { Text } = Typography;

export interface ParamValues {
  temperature: number;
  max_per_industry: number;
  corr_threshold: number;
  min_score_ratio: number;
  bull_gate: number;
  top_n: number;
  select_k: number;
  turnover_penalty: number;
  persistence_bonus: number;
}

export const DEFAULT_PARAMS: ParamValues = {
  temperature: 0.5,
  max_per_industry: 2,
  corr_threshold: 0.8,
  min_score_ratio: 0.7,
  bull_gate: 0.6,
  top_n: 10,
  select_k: 5,
  turnover_penalty: 0.02,
  persistence_bonus: 0.01,
};

interface Props {
  values: ParamValues;
  onChange: (vals: ParamValues) => void;
}

const ParamSliders: React.FC<Props> = ({ values, onChange }) => {
  const update = (key: keyof ParamValues, val: number) => {
    onChange({ ...values, [key]: val });
  };

  const sliders = [
    { key: 'temperature' as const, label: '温度 (Temperature)', min: 0.1, max: 2.0, step: 0.1, tip: '越低权重越集中' },
    { key: 'max_per_industry' as const, label: '行业上限 (Max/Industry)', min: 1, max: 5, step: 1, tip: '每个行业最多选几只' },
    { key: 'corr_threshold' as const, label: '相关性阈值 (Corr Threshold)', min: 0.5, max: 1.0, step: 0.05, tip: '超过则惩罚得分' },
    { key: 'min_score_ratio' as const, label: '最低得分比 (Min Score Ratio)', min: 0.3, max: 0.9, step: 0.05, tip: '低于Top-1的百分比则不选' },
    { key: 'bull_gate' as const, label: '牛市门槛 (Bull Gate)', min: 0.0, max: 1.0, step: 0.05, tip: 'Bull Strength低于此值空仓' },
    { key: 'top_n' as const, label: '候选池大小 (Top-N)', min: 5, max: 30, step: 1, tip: '初选股票数量' },
    { key: 'select_k' as const, label: '持仓数量 (Select-K)', min: 1, max: 10, step: 1, tip: '最终组合股票数' },
    { key: 'turnover_penalty' as const, label: '换手惩罚', min: 0.0, max: 0.1, step: 0.005, tip: '新开仓得分惩罚' },
    { key: 'persistence_bonus' as const, label: '持仓奖励', min: 0.0, max: 0.05, step: 0.005, tip: '已有持仓得分奖励' },
  ];

  return (
    <div>
      {sliders.map((s) => (
        <Row key={s.key} style={{ marginBottom: 16 }}>
          <Col span={24}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text strong>{s.label}</Text>
              <Text type="secondary">{values[s.key]}</Text>
            </div>
            <Slider
              min={s.min}
              max={s.max}
              step={s.step}
              value={values[s.key]}
              onChange={(v) => update(s.key, v as number)}
            />
            <Text type="secondary" style={{ fontSize: 11 }}>{s.tip}</Text>
          </Col>
        </Row>
      ))}
    </div>
  );
};

export default ParamSliders;
```

- [ ] **Step 2: Write TuningTab**

```typescript
// frontend/src/components/TuningTab.tsx
import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Button, DatePicker, Select, Spin, Alert } from 'antd';
import dayjs from 'dayjs';
import { api } from '../api/client';
import type { PredictResponse, DatesResponse } from '../types';
import ParamSliders, { ParamValues, DEFAULT_PARAMS } from './ParamSliders';
import StockCardGrid from './StockCardGrid';
import SectorPieChart from './SectorPieChart';

const MODES = [
  { value: 'phase6b', label: 'Phase 6B (Bull/NonBull)' },
  { value: 'consensus', label: 'Consensus' },
  { value: 'meta_ranker', label: 'Meta Ranker V2' },
  { value: 'phase7', label: 'Phase 7 (Safe Mode)' },
];

const TuningTab: React.FC = () => {
  const [dates, setDates] = useState<DatesResponse | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [mode, setMode] = useState('phase6b');
  const [params, setParams] = useState<ParamValues>(DEFAULT_PARAMS);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDates().then(setDates).catch(console.error);
  }, []);

  const handleRun = async () => {
    if (!selectedDate) return;
    setLoading(true);
    setError(null);
    try {
      const predResult = await api.predict({ date: selectedDate, mode, ...params });
      setResult(predResult);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setParams(DEFAULT_PARAMS);
    setResult(null);
    setError(null);
  };

  return (
    <Row gutter={24}>
      <Col span={10}>
        <Card title="🎛️ 参数配置" extra={<Button onClick={handleReset} size="small">重置</Button>}>
          <div style={{ marginBottom: 16, display: 'flex', gap: 12 }}>
            <DatePicker
              value={selectedDate ? dayjs(selectedDate) : null}
              onChange={(d) => setSelectedDate(d?.format('YYYY-MM-DD') || null)}
              disabledDate={(d) => !dates?.dates.includes(d.format('YYYY-MM-DD'))}
              placeholder="选择日期"
              style={{ flex: 1 }}
            />
            <Select value={mode} onChange={setMode} options={MODES} style={{ width: 200 }} />
          </div>
          <ParamSliders values={params} onChange={setParams} />
          <Button type="primary" onClick={handleRun} loading={loading} block size="large" style={{ marginTop: 16 }}>
            🔄 应用参数并重跑
          </Button>
        </Card>
      </Col>
      <Col span={14}>
        {error && <Alert type="error" message={error} closable style={{ marginBottom: 16 }} />}
        {loading && <Spin size="large" style={{ display: 'block', margin: '80px auto' }} />}
        {result && !loading && (
          <>
            <Card size="small" title={`📈 结果 · Bull Strength: ${result.bull_strength.toFixed(3)}`} style={{ marginBottom: 16 }}>
              <StockCardGrid stocks={result.portfolio} />
            </Card>
            <SectorPieChart data={result.sector_distribution} />
          </>
        )}
      </Col>
    </Row>
  );
};

export default TuningTab;
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/TuningTab.tsx frontend/src/components/ParamSliders.tsx
git commit -m "feat: add TuningTab with parameter sliders"
```

---

### Task 12: CompareTab component

**Files:**
- Create: `frontend/src/components/CompareTab.tsx`

- [ ] **Step 1: Write CompareTab**

```typescript
// frontend/src/components/CompareTab.tsx
import React, { useState, useEffect } from 'react';
import { DatePicker, Button, Table, Spin, Alert, Row, Col, Card } from 'antd';
import dayjs from 'dayjs';
import { api } from '../api/client';
import type { CompareResponse, DatesResponse } from '../types';

const ALL_MODES = ['phase6b', 'consensus', 'meta_ranker', 'phase7'];
const MODE_LABELS: Record<string, string> = {
  phase6b: 'Phase 6B',
  consensus: 'Consensus',
  meta_ranker: 'Meta Ranker',
  phase7: 'Phase 7',
};

const CompareTab: React.FC = () => {
  const [dates, setDates] = useState<DatesResponse | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDates().then(setDates).catch(console.error);
  }, []);

  const handleCompare = async () => {
    if (!selectedDate) return;
    setLoading(true);
    setError(null);
    try {
      const r = await api.compare(selectedDate, ALL_MODES);
      setResult(r);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const buildTableData = () => {
    if (!result) return [];
    // Collect all unique stock_ids across all mode results
    const allStocks = new Map<string, Record<string, any>>();
    ALL_MODES.forEach((mode) => {
      const modeResult = result.results[mode];
      if (modeResult && modeResult.portfolio) {
        modeResult.portfolio.forEach((s) => {
          if (!allStocks.has(s.stock_id)) {
            allStocks.set(s.stock_id, { stock_id: s.stock_id });
          }
          const entry = allStocks.get(s.stock_id)!;
          entry[`${mode}_score`] = s.score;
          entry[`${mode}_weight`] = s.weight;
        });
      }
    });
    return Array.from(allStocks.values());
  };

  const columns = [
    { title: '股票代码', dataIndex: 'stock_id', key: 'stock_id', fixed: 'left' as const, width: 100 },
    ...ALL_MODES.map((mode) => ({
      title: MODE_LABELS[mode],
      key: mode,
      children: [
        { title: '得分', dataIndex: `${mode}_score`, key: `${mode}_score`, width: 100, render: (v: number) => v?.toFixed(4) || '-' },
        { title: '权重%', dataIndex: `${mode}_weight`, key: `${mode}_weight`, width: 90, render: (v: number) => v ? `${(v * 100).toFixed(1)}%` : '-' },
      ],
    })),
  ];

  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', gap: 12, alignItems: 'center' }}>
        <DatePicker
          value={selectedDate ? dayjs(selectedDate) : null}
          onChange={(d) => setSelectedDate(d?.format('YYYY-MM-DD') || null)}
          disabledDate={(d) => !dates?.dates.includes(d.format('YYYY-MM-DD'))}
          placeholder="选择日期"
        />
        <Button type="primary" onClick={handleCompare} loading={loading}>
          🔬 并行对比
        </Button>
        {result && (
          <span style={{ color: '#8c8c8c' }}>
            Bull Strength: <strong>{result.bull_strength.toFixed(3)}</strong> | {result.regime_label}
          </span>
        )}
      </div>

      {error && <Alert type="error" message={error} closable style={{ marginBottom: 16 }} />}
      {loading && <Spin size="large" style={{ display: 'block', margin: '40px auto' }} />}

      {result && !loading && (
        <>
          <Card title="📊 多模式得分对比" style={{ marginBottom: 16 }}>
            <Table
              dataSource={buildTableData()}
              columns={columns}
              rowKey="stock_id"
              size="small"
              scroll={{ x: 900 }}
              pagination={{ pageSize: 20 }}
            />
          </Card>

          <Row gutter={16}>
            {ALL_MODES.map((mode) => {
              const r = result.results[mode];
              if (!r || 'error' in r) return null;
              return (
                <Col span={6} key={mode}>
                  <Card size="small" title={MODE_LABELS[mode]}>
                    {r.portfolio.slice(0, 5).map((s, i) => (
                      <div key={s.stock_id} style={{ display: 'flex', justifyContent: 'space-between', padding: '2px 0' }}>
                        <span>{['🥇','🥈','🥉','4','5'][i]} {s.stock_id}</span>
                        <span>{(s.weight * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </Card>
                </Col>
              );
            })}
          </Row>
        </>
      )}
    </div>
  );
};

export default CompareTab;
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/components/CompareTab.tsx
git commit -m "feat: add CompareTab — multi-mode parallel comparison"
```

---

### Task 13: BacktestTab + HistoryTab components

**Files:**
- Create: `frontend/src/components/BacktestTab.tsx`
- Create: `frontend/src/components/HistoryTab.tsx`
- Create: `frontend/src/components/MetricsCardRow.tsx`
- Create: `frontend/src/components/CumulativeReturnChart.tsx`

- [ ] **Step 1: Write MetricsCardRow**

```typescript
// frontend/src/components/MetricsCardRow.tsx
import React from 'react';
import { Row, Col, Card, Statistic } from 'antd';
import { RiseOutlined, TrophyOutlined, FallOutlined, PercentageOutlined } from '@ant-design/icons';

interface Props {
  metrics: Record<string, number>;
}

const MetricsCardRow: React.FC<Props> = ({ metrics }) => (
  <Row gutter={16} style={{ marginBottom: 16 }}>
    <Col span={6}>
      <Card><Statistic title="累计收益" value={(metrics.cumulative_return * 100).toFixed(2)} suffix="%" prefix={<RiseOutlined />} /></Card>
    </Col>
    <Col span={6}>
      <Card><Statistic title="夏普比率" value={metrics.sharpe} prefix={<TrophyOutlined />} /></Card>
    </Col>
    <Col span={6}>
      <Card><Statistic title="最大回撤" value={(metrics.max_drawdown * 100).toFixed(2)} suffix="%" prefix={<FallOutlined />} valueStyle={{ color: '#cf1322' }} /></Card>
    </Col>
    <Col span={6}>
      <Card><Statistic title="胜率" value={(metrics.win_rate * 100).toFixed(1)} suffix="%" prefix={<PercentageOutlined />} /></Card>
    </Col>
  </Row>
);

export default MetricsCardRow;
```

- [ ] **Step 2: Write CumulativeReturnChart**

```typescript
// frontend/src/components/CumulativeReturnChart.tsx
import React from 'react';
import ReactECharts from 'echarts-for-react';

interface Props {
  data: { date: string; cumulative_return: number }[];
}

const CumulativeReturnChart: React.FC<Props> = ({ data }) => {
  const option = {
    title: { text: '累计收益曲线', left: 'center' },
    tooltip: { trigger: 'axis' as const },
    xAxis: { type: 'category' as const, data: data.map((d) => d.date), axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: {
      type: 'value' as const,
      axisLabel: { formatter: (v: number) => `${(v * 100).toFixed(0)}%` },
    },
    series: [
      {
        name: '累计收益',
        type: 'line',
        data: data.map((d) => d.cumulative_return),
        smooth: true,
        areaStyle: { opacity: 0.1 },
        lineStyle: { color: '#1677ff' },
      },
    ],
    dataZoom: [{ type: 'inside' }, { type: 'slider' }],
  };

  return <ReactECharts option={option} style={{ height: 350 }} />;
};

export default CumulativeReturnChart;
```

- [ ] **Step 3: Write BacktestTab**

```typescript
// frontend/src/components/BacktestTab.tsx
import React, { useState, useEffect } from 'react';
import { DatePicker, Select, Button, Spin, Alert } from 'antd';
import dayjs from 'dayjs';
import { api } from '../api/client';
import type { BacktestResponse, DatesResponse } from '../types';
import MetricsCardRow from './MetricsCardRow';
import CumulativeReturnChart from './CumulativeReturnChart';

const BacktestTab: React.FC = () => {
  const [dates, setDates] = useState<DatesResponse | null>(null);
  const [startDate, setStartDate] = useState<string | null>(null);
  const [endDate, setEndDate] = useState<string | null>(null);
  const [mode, setMode] = useState('phase6b');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDates().then((d) => {
      setDates(d);
      setStartDate(d.first);
      setEndDate(d.last);
    }).catch(console.error);
  }, []);

  const handleRun = async () => {
    if (!startDate || !endDate) return;
    setLoading(true);
    setError(null);
    try {
      const r = await api.runBacktest({ start_date: startDate, end_date: endDate, mode });
      setResult(r);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div style={{ marginBottom: 16, display: 'flex', gap: 12, alignItems: 'center' }}>
        <DatePicker value={startDate ? dayjs(startDate) : null} onChange={(d) => setStartDate(d?.format('YYYY-MM-DD') || null)} placeholder="起始日期" />
        <span>至</span>
        <DatePicker value={endDate ? dayjs(endDate) : null} onChange={(d) => setEndDate(d?.format('YYYY-MM-DD') || null)} placeholder="结束日期" />
        <Select value={mode} onChange={setMode} options={[
          { value: 'phase6b', label: 'Phase 6B' },
          { value: 'consensus', label: 'Consensus' },
          { value: 'meta_ranker', label: 'Meta Ranker' },
          { value: 'phase7', label: 'Phase 7' },
        ]} style={{ width: 160 }} />
        <Button type="primary" onClick={handleRun} loading={loading}>📈 运行回测</Button>
      </div>

      {error && <Alert type="error" message={error} closable style={{ marginBottom: 16 }} />}
      {loading && <Spin size="large" style={{ display: 'block', margin: '40px auto' }} />}
      {result && !loading && (
        <>
          <MetricsCardRow metrics={result.metrics} />
          <CumulativeReturnChart data={result.daily_returns} />
        </>
      )}
    </div>
  );
};

export default BacktestTab;
```

- [ ] **Step 4: Write HistoryTab**

```typescript
// frontend/src/components/HistoryTab.tsx
import React, { useState, useEffect } from 'react';
import { Table, Spin, Alert, Tag } from 'antd';
import { api } from '../api/client';
import type { HistoryRecord, HistoryResponse } from '../types';

const HistoryTab: React.FC = () => {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getHistory(50)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <Spin size="large" style={{ display: 'block', margin: '40px auto' }} />;
  if (error) return <Alert type="error" message={error} />;

  const columns = [
    { title: '日期', dataIndex: 'date', key: 'date', width: 120 },
    {
      title: '市场状态', key: 'market', width: 180,
      render: (_: any, r: HistoryRecord) => (
        <>
          <Tag color={r.bull_strength > 0.5 ? 'red' : 'blue'}>
            BS: {r.bull_strength.toFixed(3)}
          </Tag>
          <Tag>{r.regime_label}</Tag>
        </>
      ),
    },
    {
      title: '持仓', key: 'holdings',
      render: (_: any, r: HistoryRecord) => (
        r.stocks?.map((s, i) => (
          <Tag key={s} color="green">{s} ({(r.weights[i] * 100).toFixed(1)}%)</Tag>
        ))
      ),
    },
  ];

  return (
    <div>
      <Table
        dataSource={data?.history || []}
        columns={columns}
        rowKey="date"
        size="small"
        pagination={{ pageSize: 20 }}
        expandable={{
          expandedRowRender: (r: HistoryRecord) => (
            <div>
              {r.stocks?.map((s, i) => (
                <div key={s} style={{ padding: '4px 0' }}>
                  {['🥇','🥈','🥉','4','5'][i] || `#${i + 1}`} {s} — {(r.weights[i] * 100).toFixed(1)}%
                </div>
              ))}
            </div>
          ),
        }}
      />
    </div>
  );
};

export default HistoryTab;
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/BacktestTab.tsx frontend/src/components/HistoryTab.tsx frontend/src/components/MetricsCardRow.tsx frontend/src/components/CumulativeReturnChart.tsx
git commit -m "feat: add BacktestTab and HistoryTab components"
```

---

### Task 14: Integration test — full stack verification

- [ ] **Step 1: Start both servers**

Terminal 1:
```bash
cd d:/CSI300-AlphaMamba && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Terminal 2:
```bash
cd d:/CSI300-AlphaMamba/frontend && npm run dev
```

- [ ] **Step 2: Open browser at http://localhost:5173**

Verify:
1. Page loads with 5 tabs visible
2. PredictTab: select date → "运行预测" → stock cards appear with weights
3. TuningTab: adjust sliders → "应用参数并重跑" → results update
4. CompareTab: select date → "并行对比" → comparison table shows
5. BacktestTab: select range → "运行回测" → chart renders
6. HistoryTab: loads historical predictions

- [ ] **Step 3: Fix any issues found**

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete CSI300 AlphaMamba analysis workbench"
git push origin master
```

---

## Execution Order

```
Task 1  → Backend scaffold
Task 2  → PredictionEngine.load()
Task 3  → predict_phase6b()
Task 4  → Other modes + optimizer
Task 5  → Market API
Task 6  → Predict API
Task 7  → Backtest API
Task 8  → Verify backend APIs
Task 9  → React scaffold
Task 10 → PredictTab
Task 11 → TuningTab
Task 12 → CompareTab
Task 13 → BacktestTab + HistoryTab
Task 14 → Integration test
```

Tasks 1-8 build the backend incrementally (each depends on prior).
Tasks 9-13 build the frontend (each depends on Task 9).
Task 14 validates the full stack together.
