"""Engineering investigation service for flight parameter diagnostics and feedback."""
from __future__ import annotations

import uuid
from typing import Any
import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from aeris_data.schema import NUMERIC_FEATURES
from backend.db.models import InvestigationFeedback
from backend.schemas.investigation import InvestigationFeedbackCreate
from backend.services.flight_service import get_flight_record, load_flight_dataframe


def get_parameter_correlations(flight_id: str, db: Session) -> dict[str, Any]:
    """Calculate correlation matrix between numeric flight parameters."""
    record = get_flight_record(flight_id, db)
    df = load_flight_dataframe(record)

    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    corr = df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0)

    return {
        "flight_id": flight_id,
        "features": numeric_cols,
        "matrix": corr.round(3).values.tolist(),
    }


def add_investigation_feedback(
    data: InvestigationFeedbackCreate,
    db: Session,
) -> InvestigationFeedback:
    """Store an engineer's investigation finding or anomaly verification."""
    # Ensure flight exists
    get_flight_record(data.flight_id, db)

    feedback_id = f"fb_{uuid.uuid4().hex[:8]}"
    entry = InvestigationFeedback(
        feedback_id=feedback_id,
        flight_id=data.flight_id,
        timestamp_s=data.timestamp_s,
        engineer_name=data.engineer_name,
        feedback_type=data.feedback_type,
        severity=data.severity,
        comments=data.comments,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def list_investigation_feedback(flight_id: str | None, db: Session) -> list[InvestigationFeedback]:
    """List investigation feedback entries, optionally filtered by flight_id."""
    query = db.query(InvestigationFeedback)
    if flight_id:
        query = query.filter(InvestigationFeedback.flight_id == flight_id)
    return query.order_by(InvestigationFeedback.created_at.desc()).all()
