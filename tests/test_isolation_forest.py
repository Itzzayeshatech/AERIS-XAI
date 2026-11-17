"""Tests for Isolation Forest Anomaly Detection Engine."""
from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from aeris_data.generate import generate_flight
from ml.models.isolation_forest import FlightIsolationForest
from ml.preprocess import FlightPreprocessor


def test_isolation_forest_pipeline():
    # Generate flights
    df_train = pd.concat([generate_flight(f"TRN_{i}", n_rows=60, seed=i) for i in range(4)], ignore_index=True)
    df_clean = generate_flight("TST_CLN", n_rows=60, seed=10, anomaly_type="none")
    df_anom = generate_flight("TST_ANM", n_rows=60, seed=20, anomaly_type="sensor_spike")
    df_test = pd.concat([df_clean, df_anom], ignore_index=True)

    preprocessor = FlightPreprocessor()
    preprocessor.fit(df_train)

    model = FlightIsolationForest(n_estimators=50, contamination=0.1, random_state=42)
    X_train, _, _ = model.extract_features(df_train, preprocessor)
    X_test, _, y_test = model.extract_features(df_test, preprocessor)

    assert len(X_train) > 0
    assert len(X_test) > 0

    model.fit(X_train)
    scores = model.score(X_test)
    assert len(scores) == len(X_test)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)

    # Predictions
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})

    # Evaluation with labels
    metrics = model.evaluate(
        X_test=X_test,
        y_test=y_test,
        test_flight_ids=["TST_CLN", "TST_ANM"],
        test_flight_labels={"TST_CLN": "clean", "TST_ANM": "anomalous"},
    )
    assert metrics["status"] == "computed"
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics

    # Save and Load roundtrip
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "if_model.joblib"
        model.save(model_file)
        loaded = FlightIsolationForest.load(model_file)
        loaded_scores = loaded.score(X_test)
        np.testing.assert_allclose(scores, loaded_scores, rtol=1e-5)
