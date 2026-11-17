"""Digital Flight Replay router."""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.db.models import User
from backend.db.session import get_db
from backend.security.dependencies import get_current_user
from backend.services.replay_service import get_flight_replay_data

router = APIRouter(prefix="/api/v1/replay", tags=["Digital Flight Replay"])


@router.get("/{flight_id}")
def get_replay_stream(
    flight_id: str,
    max_points: int = Query(1500, ge=100, le=5000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve synchronized 2D flight coordinates and cockpit instruments."""
    return get_flight_replay_data(flight_id, db, max_points=max_points)
