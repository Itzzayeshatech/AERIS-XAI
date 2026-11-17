"""Explainable AI (XAI) router."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.db.models import User
from backend.db.session import get_db
from backend.schemas.explain import ExplainRequest, ExplainResult
from backend.security.dependencies import get_current_user
from backend.services.explain_service import explain_anomaly_window

router = APIRouter(prefix="/api/v1/explain", tags=["Explainable AI"])


@router.post("", response_model=ExplainResult)
def explain_flight_anomaly(
    request: ExplainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ExplainResult:
    """Generate SHAP or ablation feature attributions for a flagged window."""
    return explain_anomaly_window(request, db)
