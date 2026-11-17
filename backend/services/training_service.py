"""Training orchestration service for AERIS XAI AI models."""
from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any
import mlflow
import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from aeris_data.schema import label_kind
from backend.config import settings
from backend.db.models import FlightRecord, ModelRun
from backend.schemas.training import TrainRequest, TrainResult
from backend.services.flight_service import load_flight_dataframe
from ml.errors import InsufficientFlightsError
from ml.models.isolation_forest import FlightIsolationForest
from ml.models.lstm_autoencoder import FlightLSTMAutoencoder
from ml.models.predictor import FlightTelemetryPredictor
from ml.preprocess import FlightPreprocessor
from ml.splits import split_flight_ids


def compute_dataset_version(flight_ids: list[str]) -> str:
    """Compute a deterministic SHA-256 hash representing the flight dataset version."""
    sorted_ids = sorted(flight_ids)
    raw = "|".join(sorted_ids).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def train_model(
    request: TrainRequest,
    db: Session,
) -> TrainResult:
    """Execute end-to-end model training, evaluation, and experiment tracking."""
    settings.ensure_directories()

    # 1. Resolve flights to include in dataset
    if request.flight_ids:
        records = db.query(FlightRecord).filter(FlightRecord.flight_id.in_(request.flight_ids)).all()
    else:
        records = db.query(FlightRecord).all()

    flight_ids = [r.flight_id for r in records]
    found_count = len(flight_ids)
    if found_count < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Training requires at least 5 distinct flights for train/val/test split. "
                f"Found {found_count} flight(s). Please generate or upload more flights first."
            ),
        )

    dataset_ver = compute_dataset_version(flight_ids)

    # 2. Load flights and label types for label-aware flight-level splitting.
    # Evaluation labels are used only to select a representative test partition;
    # they are never added to model feature matrices.
    flight_dfs: dict[str, pd.DataFrame] = {}
    flight_labels: dict[str, str] = {}
    for r in records:
        df = load_flight_dataframe(r)
        flight_dfs[r.flight_id] = df
        flight_labels[r.flight_id] = label_kind(df)

    try:
        split = split_flight_ids(flight_ids, seed=request.seed, labels=flight_labels)
    except InsufficientFlightsError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    train_fids = set(split["train"])
    val_fids = set(split["validation"])
    test_fids = set(split["test"])

    # 3. Partition dataframes by split
    train_dfs = [flight_dfs[fid] for fid in train_fids if fid in flight_dfs]
    val_dfs = [flight_dfs[fid] for fid in val_fids if fid in flight_dfs]
    test_dfs = [flight_dfs[fid] for fid in test_fids if fid in flight_dfs]

    train_combined = pd.concat(train_dfs, ignore_index=True)
    val_combined = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    test_combined = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    # 4. Fit FlightPreprocessor strictly on training flights only (no data leakage)
    preprocessor = FlightPreprocessor()
    preprocessor.fit(train_combined)

    run_id = f"{request.model_type}_{uuid.uuid4().hex[:8]}"
    model_dir = settings.DATA_DIR / "models" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save fitted preprocessor alongside model
    import joblib
    joblib.dump(preprocessor, model_dir / "preprocessor.joblib")

    # 5. Train specific model architecture
    metrics: dict[str, Any] = {}
    mlflow_run_id: str | None = None

    # Setup MLflow if available
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)
    except Exception:
        pass

    if request.model_type == "isolation_forest":
        contamination = request.parameters.get("contamination", 0.05)
        n_estimators = request.parameters.get("n_estimators", 100)
        max_samples = request.parameters.get("max_samples", 1.0)

        model = FlightIsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=request.seed,
        )

        X_train, _, _ = model.extract_features(train_combined, preprocessor)
        X_test, _, y_test = model.extract_features(test_combined, preprocessor)

        model.fit(X_train)
        metrics = model.evaluate(
            X_test=X_test,
            y_test=y_test,
            test_flight_ids=list(test_fids),
            test_flight_labels={fid: flight_labels[fid] for fid in test_fids},
        )
        model.save(model_dir / "model.joblib")

    elif request.model_type == "lstm_autoencoder":
        epochs = request.parameters.get("epochs", 20)
        hidden_dim = request.parameters.get("hidden_dim", 48)
        latent_dim = request.parameters.get("latent_dim", 16)
        learning_rate = request.parameters.get("learning_rate", 0.002)

        model = FlightLSTMAutoencoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            random_seed=request.seed,
        )

        X_train, _, _ = model.extract_sequences(train_combined, preprocessor)
        X_val, _, _ = model.extract_sequences(val_combined, preprocessor) if len(val_combined) > 0 else (np.empty((0, 30, len(preprocessor.model_columns()))), [], [])
        X_test, _, y_test = model.extract_sequences(test_combined, preprocessor)

        model.fit(X_train, X_val)
        metrics = model.evaluate(
            X_test=X_test,
            y_test=y_test,
            test_flight_ids=list(test_fids),
            test_flight_labels={fid: flight_labels[fid] for fid in test_fids},
        )
        model.save(model_dir / "model.pt")

    elif request.model_type == "telemetry_predictor":
        deviation_sigma = request.parameters.get("deviation_sigma", 3.0)

        model = FlightTelemetryPredictor(
            random_state=request.seed,
            deviation_sigma=deviation_sigma,
        )

        X_train, Y_train, _ = model.extract_dataset(train_combined, preprocessor)
        X_test, Y_test, _ = model.extract_dataset(test_combined, preprocessor)

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        metrics = model.evaluate(Y_test, Y_pred)
        model.save(model_dir / "model.joblib")

    # 7. Log to MLflow
    try:
        with mlflow.start_run(run_name=run_id) as run:
            mlflow_run_id = run.info.run_id
            mlflow.log_params({
                "model_type": request.model_type,
                "seed": request.seed,
                "dataset_version": dataset_ver,
                **request.parameters,
            })
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and v is not None:
                    mlflow.log_metric(k, float(v))
    except Exception:
        pass

    # 8. Store ModelRun in SQLite DB
    run_record = ModelRun(
        run_id=run_id,
        model_type=request.model_type,
        dataset_version=dataset_ver,
        status="completed",
        mlflow_run_id=mlflow_run_id,
        parameters_json=json.dumps(request.parameters),
        metrics_json=json.dumps(metrics),
        model_path=str(model_dir),
        trained_at=datetime.datetime.utcnow(),
    )
    db.add(run_record)
    db.commit()

    return TrainResult(
        run_id=run_id,
        model_type=request.model_type,
        status="completed",
        dataset_version=dataset_ver,
        split_counts={
            "train": len(train_fids),
            "validation": len(val_fids),
            "test": len(test_fids),
        },
        metrics=metrics,
        mlflow_run_id=mlflow_run_id,
        message=f"Model '{request.model_type}' successfully trained and validated.",
    )


def list_model_runs(db: Session) -> list[ModelRun]:
    """Retrieve all logged model runs ordered by training timestamp."""
    return db.query(ModelRun).order_by(ModelRun.created_at.desc()).all()


def get_model_run(run_id: str, db: Session) -> ModelRun:
    """Retrieve details for a specific model run."""
    run = db.query(ModelRun).filter(ModelRun.run_id == run_id).first()
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model run '{run_id}' not found.",
        )
    return run
