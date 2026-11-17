"""Tests for Predictive Flight Analytics Engine."""
from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from aeris_data.generate import generate_flight
from ml.models.predictor import FlightTelemetryPredictor
from ml.preprocess import FlightPreprocessor


def test_telemetry_predictor_pipeline():
    df_train = pd.concat([generate_flight(f"TRN_P_{i}", n_rows=60, seed=i) for i in range(3)], ignore_index=True)
    df_test = generate_flight("TST_P_0", n_rows=60, seed=15)

    preprocessor = FlightPreprocessor()
    preprocessor.fit(df_train)

    predictor = FlightTelemetryPredictor(random_state=42)
    X_train, Y_train, _ = predictor.extract_dataset(df_train, preprocessor, length=30, stride=5)
    X_test, Y_test, _ = predictor.extract_dataset(df_test, preprocessor, length=30, stride=5)

    assert len(X_train) > 0
    assert Y_train.shape[1] == 3  # altitude_ft, airspeed_kts, n1_pct

    predictor.fit(X_train, Y_train)
    assert predictor.is_fitted is True

    preds = predictor.predict(X_test)
    assert preds.shape == Y_test.shape

    metrics = predictor.evaluate(Y_test, preds)
    assert metrics["status"] == "computed"
    assert "mae_altitude_ft" in metrics
    assert "rmse_altitude_ft" in metrics

    # Save and Load
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "pred_model.joblib"
        predictor.save(model_file)
        loaded = FlightTelemetryPredictor.load(model_file)
        loaded_preds = loaded.predict(X_test)
        np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5)
