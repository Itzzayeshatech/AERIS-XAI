"""Model Inference router for anomaly detection and forecasting."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.db.models import User
from backend.db.session import get_db
from backend.schemas.inference import DetectRequest, DetectResult, ForecastRequest, ForecastResult
from backend.security.dependencies import get_current_user
from backend.services.inference_service import detect_flight_anomalies, forecast_flight_telemetry

router = APIRouter(prefix="/api/v1/inference", tags=["Model Inference"])


@router.post("/detect", response_model=DetectResult)
def run_anomaly_detection(
    request: DetectRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> DetectResult:
    """Run anomaly detection on a flight using a trained model."""
    return detect_flight_anomalies(request, db)


@router.post("/forecast", response_model=ForecastResult)
def run_telemetry_forecasting(
    request: ForecastRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ForecastResult:
    """Generate predictive telemetry forecast and highlight large residuals."""
    return forecast_flight_telemetry(request, db)
