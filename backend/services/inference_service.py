"""Inference service for executing anomaly detection and forecasting on flights."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.db.models import FlightRecord, ModelRun
from backend.schemas.inference import DetectRequest, DetectResult, ForecastRequest, ForecastResult
from backend.services.flight_service import get_flight_record, load_flight_dataframe
from backend.services.training_service import get_model_run
from ml.models.isolation_forest import FlightIsolationForest
from ml.models.lstm_autoencoder import FlightLSTMAutoencoder
from ml.models.predictor import FlightTelemetryPredictor
from ml.preprocess import FlightPreprocessor


def detect_flight_anomalies(
    request: DetectRequest,
    db: Session,
) -> DetectResult:
    """Run anomaly detection inference on an individual flight using a trained model."""
    run = get_model_run(request.run_id, db)
    record = get_flight_record(request.flight_id, db)
    df = load_flight_dataframe(record)

    model_dir = Path(run.model_path)
    preproc_path = model_dir / "preprocessor.joblib"
    if not preproc_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trained preprocessor artifact is missing.",
        )
    preprocessor: FlightPreprocessor = joblib.load(preproc_path)

    windows_out: list[dict[str, Any]] = []
    threshold_used: float = 0.5
    eval_metrics: dict[str, Any] | None = None

    if run.model_type == "isolation_forest":
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model artifact missing.")
        model = FlightIsolationForest.load(model_path)
        threshold_used = request.threshold if request.threshold is not None else model.threshold

        X, meta_list, labels = model.extract_features(df, preprocessor)
        if len(X) > 0:
            scores = model.score(X)
            preds = (scores >= threshold_used).astype(int)

            for i, meta in enumerate(meta_list):
                windows_out.append({
                    "window_index": i,
                    "start_t_s": meta["start_t_s"],
                    "end_t_s": meta["end_t_s"],
                    "phase_mode": meta["phase_mode"],
                    "score": round(float(scores[i]), 4),
                    "threshold": round(float(threshold_used), 4),
                    "is_anomaly": bool(preds[i] == 1),
                    "true_label": labels[i],
                })

            if any(lbl is not None for lbl in labels):
                eval_metrics = model.evaluate(
                    X_test=X,
                    y_test=labels,
                    test_flight_ids=[record.flight_id],
                    test_flight_labels={record.flight_id: "anomalous" if any(lbl == 1 for lbl in labels) else "clean"},
                    threshold=threshold_used,
                )

    elif run.model_type == "lstm_autoencoder":
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Model artifact missing.")
        model = FlightLSTMAutoencoder.load(model_path)
        threshold_used = request.threshold if request.threshold is not None else model.threshold

        X, meta_list, labels = model.extract_sequences(df, preprocessor)
        if len(X) > 0:
            scores = model.reconstruction_errors(X)
            preds = (scores >= threshold_used).astype(int)

            for i, meta in enumerate(meta_list):
                windows_out.append({
                    "window_index": i,
                    "start_t_s": meta["start_t_s"],
                    "end_t_s": meta["end_t_s"],
                    "phase_mode": meta["phase_mode"],
                    "score": round(float(scores[i]), 6),
                    "threshold": round(float(threshold_used), 6),
                    "is_anomaly": bool(preds[i] == 1),
                    "true_label": labels[i],
                })

            if any(lbl is not None for lbl in labels):
                eval_metrics = model.evaluate(
                    X_test=X,
                    y_test=labels,
                    test_flight_ids=[record.flight_id],
                    test_flight_labels={record.flight_id: "anomalous" if any(lbl == 1 for lbl in labels) else "clean"},
                    threshold=threshold_used,
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model type '{run.model_type}' does not support anomaly detection.",
        )

    flagged_count = sum(1 for w in windows_out if w["is_anomaly"])

    return DetectResult(
        run_id=run.run_id,
        model_type=run.model_type,
        flight_id=record.flight_id,
        threshold=threshold_used,
        total_windows=len(windows_out),
        flagged_windows=flagged_count,
        windows=windows_out,
        evaluation_metrics=eval_metrics,
        disclaimer=(
            "Offline post-flight telemetry analysis laboratory. "
            "Detections are algorithmic screening outputs and do not represent FAA/EASA certified safety alerts."
        ),
    )


def forecast_flight_telemetry(
    request: ForecastRequest,
    db: Session,
) -> ForecastResult:
    """Generate predictive baseline telemetry and flag unusual deviations."""
    run = get_model_run(request.run_id, db)
    if run.model_type != "telemetry_predictor":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected model is not a telemetry predictor.",
        )

    record = get_flight_record(request.flight_id, db)
    df = load_flight_dataframe(record)

    model_dir = Path(run.model_path)
    preprocessor: FlightPreprocessor = joblib.load(model_dir / "preprocessor.joblib")
    predictor = FlightTelemetryPredictor.load(model_dir / "model.joblib")

    X, Y, meta_list = predictor.extract_dataset(df, preprocessor, length=30, stride=5)
    forecast_points: list[dict[str, Any]] = []

    if len(X) > 0:
        Y_pred = predictor.predict(X)
        for i, meta in enumerate(meta_list):
            point: dict[str, Any] = {
                "t_s": meta["target_t_s"],
                "window_end_t_s": meta["window_end_t_s"],
            }
            any_flagged = False
            for t_idx, target in enumerate(predictor.targets):
                act_val = float(Y[i, t_idx])
                prd_val = float(Y_pred[i, t_idx])
                err = abs(act_val - prd_val)
                th = predictor.deviation_thresholds_.get(target, 100.0)
                is_dev = err > th

                point[f"actual_{target}"] = round(act_val, 2)
                point[f"pred_{target}"] = round(prd_val, 2)
                point[f"error_{target}"] = round(err, 2)
                point[f"flagged_{target}"] = bool(is_dev)

                if is_dev:
                    any_flagged = True

            point["is_flagged_deviation"] = any_flagged
            forecast_points.append(point)

    metrics = predictor.evaluate(Y, predictor.predict(X)) if len(X) > 0 else {}

    return ForecastResult(
        run_id=run.run_id,
        flight_id=record.flight_id,
        target_columns=predictor.targets,
        forecast_points=forecast_points,
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        disclaimer=(
            "Telemetry deviations highlight mathematical model prediction residual errors. "
            "Never interpret prediction residuals as confirmed aircraft equipment failures."
        ),
    )
