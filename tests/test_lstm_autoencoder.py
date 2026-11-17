"""Tests for PyTorch LSTM Autoencoder."""
from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from aeris_data.generate import generate_flight
from ml.models.lstm_autoencoder import FlightLSTMAutoencoder
from ml.preprocess import FlightPreprocessor


def test_lstm_autoencoder_pipeline():
    df_train = pd.concat([generate_flight(f"TRN_L_{i}", n_rows=60, seed=i) for i in range(3)], ignore_index=True)
    df_val = generate_flight("VAL_L_0", n_rows=60, seed=10)
    df_test = generate_flight("TST_L_0", n_rows=60, seed=20, anomaly_type="attitude_excursion")

    preprocessor = FlightPreprocessor()
    preprocessor.fit(df_train)

    model = FlightLSTMAutoencoder(
        seq_len=30,
        hidden_dim=24,
        latent_dim=12,
        epochs=3,  # fast test epochs
        learning_rate=0.01,
        random_seed=42,
    )

    X_train, _, _ = model.extract_sequences(df_train, preprocessor)
    X_val, _, _ = model.extract_sequences(df_val, preprocessor)
    X_test, _, y_test = model.extract_sequences(df_test, preprocessor)

    assert X_train.ndim == 3
    assert X_train.shape[1] == 30  # seq_len

    model.fit(X_train, X_val)
    assert model.is_fitted is True
    assert model.threshold > 0.0

    errors = model.reconstruction_errors(X_test)
    assert len(errors) == len(X_test)
    assert np.all(errors >= 0.0)

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})

    # Save and Load
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file = Path(tmp_dir) / "lstm_model.pt"
        model.save(model_file)
        loaded = FlightLSTMAutoencoder.load(model_file)
        loaded_errors = loaded.reconstruction_errors(X_test)
        np.testing.assert_allclose(errors, loaded_errors, rtol=1e-4)
