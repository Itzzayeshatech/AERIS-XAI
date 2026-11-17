"""Explainable AI (XAI) service orchestrating SHAP and ablation attributions."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.schemas.explain import ExplainRequest, ExplainResult
from backend.services.flight_service import get_flight_record, load_flight_dataframe
from backend.services.training_service import get_model_run
from ml.explain.sequence_ablation import LSTMAblationExplainer
from ml.explain.tabular_shap import IsolationForestShapExplainer
from ml.models.isolation_forest import FlightIsolationForest
from ml.models.lstm_autoencoder import FlightLSTMAutoencoder
from ml.preprocess import FlightPreprocessor


def explain_anomaly_window(
    request: ExplainRequest,
    db: Session,
) -> ExplainResult:
    """Generate local feature attributions for a flagged window."""
    run = get_model_run(request.run_id, db)
    record = get_flight_record(request.flight_id, db)
    df = load_flight_dataframe(record)

    model_dir = Path(run.model_path)
    preprocessor: FlightPreprocessor = joblib.load(model_dir / "preprocessor.joblib")

    if run.model_type == "isolation_forest":
        model = FlightIsolationForest.load(model_dir / "model.joblib")
        X, meta_list, _ = model.extract_features(df, preprocessor)

        if len(X) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Flight has insufficient length for windows.")

        idx = min(request.window_index, len(X) - 1)
        meta = meta_list[idx]
        score = float(model.score(X[idx : idx + 1])[0])
        is_flagged = bool(score >= model.threshold)

        explainer = IsolationForestShapExplainer(model)
        explanation = explainer.explain_window(X[idx])

        return ExplainResult(
            run_id=run.run_id,
            model_type=run.model_type,
            flight_id=record.flight_id,
            window_index=idx,
            window_start_t_s=meta["start_t_s"],
            window_end_t_s=meta["end_t_s"],
            anomaly_score=round(score, 4),
            is_flagged=is_flagged,
            method=explanation["method"],
            base_value=explanation["base_value"],
            top_features=explanation["top_features"],
            all_attributions=explanation["all_attributions"],
            summary_text=explanation["summary_text"],
        )

    elif run.model_type == "lstm_autoencoder":
        model = FlightLSTMAutoencoder.load(model_dir / "model.pt")
        X, meta_list, _ = model.extract_sequences(df, preprocessor)

        if len(X) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Flight has insufficient length for sequences.")

        idx = min(request.window_index, len(X) - 1)
        meta = meta_list[idx]
        score = float(model.reconstruction_errors(X[idx : idx + 1])[0])
        is_flagged = bool(score >= model.threshold)

        explainer = LSTMAblationExplainer(model)
        explanation = explainer.explain_sequence(X[idx])

        return ExplainResult(
            run_id=run.run_id,
            model_type=run.model_type,
            flight_id=record.flight_id,
            window_index=idx,
            window_start_t_s=meta["start_t_s"],
            window_end_t_s=meta["end_t_s"],
            anomaly_score=round(score, 6),
            is_flagged=is_flagged,
            method=explanation["method"],
            base_value=explanation["base_value"],
            top_features=explanation["top_features"],
            all_attributions=explanation["all_attributions"],
            summary_text=explanation["summary_text"],
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model type '{run.model_type}' does not support explainability.",
        )
