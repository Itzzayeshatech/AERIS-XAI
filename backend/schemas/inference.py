"""Model inference schemas for anomaly detection and forecasting."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    run_id: str
    flight_id: str
    threshold: float | None = None


class AnomalyWindow(BaseModel):
    start_t_s: float
    end_t_s: float
    score: float
    is_anomaly: bool
    true_label: int | None = None
    pred_label: int


class DetectResult(BaseModel):
    run_id: str
    model_type: str
    flight_id: str
    threshold: float
    total_windows: int
    flagged_windows: int
    windows: list[dict[str, Any]]
    evaluation_metrics: dict[str, Any] | None = None
    disclaimer: str


class ForecastRequest(BaseModel):
    run_id: str
    flight_id: str


class ForecastResult(BaseModel):
    run_id: str
    flight_id: str
    target_columns: list[str]
    forecast_points: list[dict[str, Any]]  # t_s, actual_*, pred_*, error_*, flagged_*
    metrics: dict[str, float]
    disclaimer: str
