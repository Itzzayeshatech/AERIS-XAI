"""AI model training and experiment tracking schemas."""
from __future__ import annotations

import datetime
from typing import Any
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    model_type: str = Field(..., pattern="^(isolation_forest|lstm_autoencoder|telemetry_predictor)$")
    flight_ids: list[str] | None = None  # None means use all available flights
    seed: int = Field(42, ge=0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelRunOut(BaseModel):
    id: int
    run_id: str
    model_type: str
    dataset_version: str
    status: str
    mlflow_run_id: str | None = None
    parameters: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    trained_at: datetime.datetime | None = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class TrainResult(BaseModel):
    run_id: str
    model_type: str
    status: str
    dataset_version: str
    split_counts: dict[str, int]
    metrics: dict[str, Any]
    mlflow_run_id: str | None = None
    message: str
