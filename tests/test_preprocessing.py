"""Tests for Data Preprocessing and Inspection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aeris_data.generate import generate_flight
from ml.inspection import inspect_frame
from ml.preprocess import FlightPreprocessor


def test_inspection_module():
    df = generate_flight("FL_INSP", n_rows=60, seed=12, anomaly_type="sensor_spike")
    res = inspect_frame(df)

    assert res.n_flights == 1
    assert res.n_rows == 60
    assert res.n_duplicate_timestamps == 0
    assert "altitude_ft" in res.column_profiles
    assert res.column_profiles["altitude_ft"].col_min is not None
    assert res.has_labels is True
    assert res.n_anomalous_rows > 0


def test_flight_preprocessor_fit_transform():
    df_train = generate_flight("FL_TRN", n_rows=80, seed=1)
    df_test = generate_flight("FL_TST", n_rows=80, seed=2)

    preprocessor = FlightPreprocessor()
    preprocessor.fit(df_train)

    assert preprocessor._fitted is True
    assert len(preprocessor.model_columns()) > 10

    # Transform test flight
    transformed = preprocessor.transform(df_test)
    assert len(transformed) == 80
    assert "is_anomaly" not in transformed.columns
    assert "anomaly_type" not in transformed.columns


def test_preprocessor_prevents_eval_leakage():
    df = generate_flight("FL_LEAK", n_rows=50, seed=4, anomaly_type="sensor_spike")
    preprocessor = FlightPreprocessor()
    preprocessor.fit(df)
    transformed = preprocessor.transform(df)

    assert "is_anomaly" not in transformed.columns
    assert "anomaly_type" not in transformed.columns
