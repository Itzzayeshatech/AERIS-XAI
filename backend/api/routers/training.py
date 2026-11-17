"""AI Training Centre router."""
from __future__ import annotations

import json
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from backend.db.models import ModelRun, User
from backend.db.session import get_db
from backend.schemas.training import ModelRunOut, TrainRequest, TrainResult
from backend.security.dependencies import get_current_user, require_role
from backend.services.training_service import get_model_run, list_model_runs, train_model

router = APIRouter(prefix="/api/v1/training", tags=["AI Training Centre"])


@router.post("/run", response_model=TrainResult, status_code=status.HTTP_201_CREATED)
def trigger_model_training(
    request: TrainRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["admin", "engineer"])),
) -> TrainResult:
    """Initiate training for Isolation Forest, LSTM Autoencoder, or Predictor."""
    return train_model(request, db)


@router.get("/runs", response_model=list[ModelRunOut])
def list_all_model_runs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """Retrieve catalog of all trained model runs."""
    runs = list_model_runs(db)
    result = []
    for r in runs:
        result.append({
            "id": r.id,
            "run_id": r.run_id,
            "model_type": r.model_type,
            "dataset_version": r.dataset_version,
            "status": r.status,
            "mlflow_run_id": r.mlflow_run_id,
            "parameters": json.loads(r.parameters_json) if r.parameters_json else {},
            "metrics": json.loads(r.metrics_json) if r.metrics_json else {},
            "trained_at": r.trained_at,
            "created_at": r.created_at,
        })
    return result


@router.get("/runs/{run_id}", response_model=ModelRunOut)
def get_run_details(
    run_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Retrieve parameters and evaluation metrics for a specific run."""
    r = get_model_run(run_id, db)
    return {
        "id": r.id,
        "run_id": r.run_id,
        "model_type": r.model_type,
        "dataset_version": r.dataset_version,
        "status": r.status,
        "mlflow_run_id": r.mlflow_run_id,
        "parameters": json.loads(r.parameters_json) if r.parameters_json else {},
        "metrics": json.loads(r.metrics_json) if r.metrics_json else {},
        "trained_at": r.trained_at,
        "created_at": r.created_at,
    }
