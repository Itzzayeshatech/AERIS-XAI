"""Tests for Explainable AI (XAI) modules."""
from __future__ import annotations

import numpy as np
import pandas as pd

from aeris_data.generate import generate_flight
from ml.explain.sequence_ablation import LSTMAblationExplainer
from ml.explain.tabular_shap import IsolationForestShapExplainer
from ml.models.isolation_forest import FlightIsolationForest
from ml.models.lstm_autoencoder import FlightLSTMAutoencoder
from ml.preprocess import FlightPreprocessor


def test_shap_isolation_forest_explanation():
    df_train = pd.concat([generate_flight(f"TRN_X_{i}", n_rows=60, seed=i) for i in range(3)], ignore_index=True)
    preprocessor = FlightPreprocessor()
    preprocessor.fit(df_train)

    model = FlightIsolationForest(n_estimators=30, random_state=42)
    X_train, _, _ = model.extract_features(df_train, preprocessor)
    model.fit(X_train)

    explainer = IsolationForestShapExplainer(model)
    result = explainer.explain_window(X_train[0], top_k=5)

    assert result["method"] == "shap_tree_explainer"
    assert len(result["top_features"]) == 5
    assert len(result["all_attributions"]) == len(model.feature_names)
    assert isinstance(result["summary_text"], str)


def test_lstm_sequence_ablation_explanation():
    df_train = pd.concat([generate_flight(f"TRN_AL_{i}", n_rows=60, seed=i) for i in range(2)], ignore_index=True)
    preprocessor = FlightPreprocessor()
    preprocessor.fit(df_train)

    model = FlightLSTMAutoencoder(seq_len=30, hidden_dim=16, latent_dim=8, epochs=2, random_seed=42)
    X_train, _, _ = model.extract_sequences(df_train, preprocessor)
    model.fit(X_train, X_train)

    explainer = LSTMAblationExplainer(model)
    result = explainer.explain_sequence(X_train[0], top_k=4)

    assert result["method"] == "feature_ablation_perturbation"
    assert len(result["top_features"]) == 4
    assert len(result["all_attributions"]) == len(model.feature_columns)
    assert isinstance(result["summary_text"], str)
