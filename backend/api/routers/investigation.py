"""Engineering Investigation router."""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.db.models import InvestigationFeedback, User
from backend.db.session import get_db
from backend.schemas.investigation import InvestigationFeedbackCreate, InvestigationFeedbackOut
from backend.security.dependencies import get_current_user, require_role
from backend.services.investigation_service import (
    add_investigation_feedback,
    get_parameter_correlations,
    list_investigation_feedback,
)

router = APIRouter(prefix="/api/v1/investigation", tags=["Engineering Investigation"])


@router.get("/correlations/{flight_id}")
def get_flight_correlations(
    flight_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve parameter correlation matrix for a flight."""
    return get_parameter_correlations(flight_id, db)


@router.post("/feedback", response_model=InvestigationFeedbackOut, status_code=status.HTTP_201_CREATED)
def submit_investigation_feedback(
    data: InvestigationFeedbackCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["admin", "engineer"])),
) -> InvestigationFeedback:
    """Submit flight analyst finding or anomaly ground-truth classification."""
    return add_investigation_feedback(data, db)


@router.get("/feedback", response_model=list[InvestigationFeedbackOut])
def get_feedback_entries(
    flight_id: str | None = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[InvestigationFeedback]:
    """List investigation feedback entries."""
    return list_investigation_feedback(flight_id, db)
