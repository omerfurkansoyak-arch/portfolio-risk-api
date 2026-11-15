"""Request and response models."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field


class PortfolioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class Position(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    quantity: float = Field(description="Units held; negative for a short position")


class Portfolio(BaseModel):
    id: int
    name: str
    positions: list[Position]


class PricePoint(BaseModel):
    date: dt.date
    close: float = Field(gt=0)


class PriceUpload(BaseModel):
    symbol: str = Field(min_length=1, max_length=20)
    prices: list[PricePoint] = Field(min_length=1)


class PriceUploadResult(BaseModel):
    symbol: str
    stored: int


class PositionRisk(BaseModel):
    symbol: str
    exposure: float
    component_var: float = Field(description="Share of parametric VaR; components sum to it")


class RiskReport(BaseModel):
    portfolio_id: int
    method: Literal["historical", "parametric", "monte_carlo"]
    confidence: float
    horizon_days: int
    observations: int
    as_of: dt.date
    market_value: float
    var: float
    expected_shortfall: float
    positions: list[PositionRisk]


class BacktestReport(BaseModel):
    portfolio_id: int
    confidence: float
    window: int
    observations: int
    exceptions: int
    expected_exceptions: float
    kupiec_statistic: float
    p_value: float
    model_rejected_at_5pct: bool
