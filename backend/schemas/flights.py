"""Flight schemas for upload, synthetic generation, and series retrieval."""
from __future__ import annotations

import datetime
from typing import Any
from pydantic import BaseModel, Field


class FlightRecordOut(BaseModel):
    id: int
    flight_id: str
    source: str
    filename: str
    row_count: int
    duration_s: float
    has_labels: bool
    seed: int | None = None
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class FlightSummaryOut(BaseModel):
    flight_id: str
    source: str
    row_count: int
    duration_s: float
    has_labels: bool
    phase_distribution: dict[str, int]
    anomaly_counts: dict[str, int] | None = None
    telemetry_ranges: dict[str, dict[str, float]]
    created_at: datetime.datetime


class SeriesPoint(BaseModel):
    t_s: float
    phase: str
    altitude_ft: float | None = None
    airspeed_kts: float | None = None
    vertical_speed_fpm: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    heading_deg: float | None = None
    n1_pct: float | None = None
    egt_c: float | None = None
    fuel_flow_pph: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    is_anomaly: int | None = None
    anomaly_type: str | None = None


class SeriesResponse(BaseModel):
    flight_id: str
    total_points: int
    returned_points: int
    is_downsampled: bool
    series: list[dict[str, Any]]


class FlightGenerateRequest(BaseModel):
    flight_count: int = Field(5, ge=1, le=50)
    duration_rows: int = Field(90, ge=30, le=1000)
    base_seed: int = Field(42, ge=0)
    inject_anomalies: bool = True
    prefix: str = Field("SYN", min_length=2, max_length=10)
