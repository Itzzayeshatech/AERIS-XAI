"""Model Comparison router."""
from __future__ import annotations

import json
from typing import Any
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.db.models import ModelRun, User
from backend.db.session import get_db
from backend.security.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/models", tags=["Model Comparison"])


@router.get("/compare")
def compare_models(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict[str, Any]]:
    """Return metrics and parameter comparison across all completed training runs."""
    runs = db.query(ModelRun).filter(ModelRun.status == "completed").all()
    comparison = []

    for r in runs:
        metrics = json.loads(r.metrics_json or "{}")
        params = json.loads(r.parameters_json or "{}")

        comparison.append({
            "run_id": r.run_id,
            "model_type": r.model_type,
            "dataset_version": r.dataset_version,
            "status": r.status,
            "trained_at": r.trained_at.isoformat() if r.trained_at else None,
            "f1": metrics.get("f1"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "average_precision": metrics.get("average_precision"),
            "evaluation_status": metrics.get("status"),
            "evaluation_reason": metrics.get("reason"),
            "mae_altitude_ft": metrics.get("mae_altitude_ft"),
            "mae_airspeed_kts": metrics.get("mae_airspeed_kts"),
            "mae_n1_pct": metrics.get("mae_n1_pct"),
            "metrics": metrics,
            "parameters": params,
        })

    return comparison
