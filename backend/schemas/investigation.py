"""Engineering investigation and feedback schemas."""
from __future__ import annotations

import datetime
from pydantic import BaseModel, Field


class InvestigationFeedbackCreate(BaseModel):
    flight_id: str
    timestamp_s: float = Field(..., ge=0)
    engineer_name: str = Field(..., min_length=2, max_length=64)
    feedback_type: str = Field(
        ...,
        pattern="^(confirmed_anomaly|false_positive|benign_variation|sensor_glitch)$",
    )
    severity: str = Field("medium", pattern="^(low|medium|high|critical)$")
    comments: str = Field(..., min_length=3)


class InvestigationFeedbackOut(BaseModel):
    id: int
    feedback_id: str
    flight_id: str
    timestamp_s: float
    engineer_name: str
    feedback_type: str
    severity: str
    comments: str
    created_at: datetime.datetime

    class Config:
        from_attributes = True
