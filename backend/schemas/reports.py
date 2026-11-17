"""Report generation schemas."""
from __future__ import annotations

import datetime
from typing import Any
from pydantic import BaseModel, Field


class ReportGenerateRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=256)
    report_type: str = Field(
        "flight_summary",
        pattern="^(flight_summary|anomaly_investigation|model_evaluation)$",
    )
    format: str = Field("html", pattern="^(html|pdf)$")
    flight_id: str | None = None
    run_id: str | None = None
    include_plots: bool = True
    executive_summary: str | None = None


class ReportRecordOut(BaseModel):
    id: int
    report_id: str
    title: str
    report_type: str
    format: str
    storage_path: str
    created_by: str
    summary_data: dict[str, Any] | None = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True
