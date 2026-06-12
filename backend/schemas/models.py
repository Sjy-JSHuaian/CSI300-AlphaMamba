from typing import Optional, List, Dict, Any
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
    metrics: Dict[str, float]
    daily_returns: List[Dict[str, Any]]
    monthly_returns: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    error: str
    detail: str
