"""Explainable AI (XAI) schemas."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    run_id: str
    flight_id: str
    window_index: int = Field(0, ge=0)


class FeatureContribution(BaseModel):
    feature_name: str
    attribution_value: float
    feature_value: float | None = None


class ExplainResult(BaseModel):
    run_id: str
    model_type: str
    flight_id: str
    window_index: int
    window_start_t_s: float
    window_end_t_s: float
    anomaly_score: float
    is_flagged: bool
    method: str  # "shap_tree" or "temporal_ablation"
    base_value: float
    top_features: list[dict[str, Any]]
    all_attributions: dict[str, float]
    summary_text: str
