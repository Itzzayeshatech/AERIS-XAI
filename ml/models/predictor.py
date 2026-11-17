"""Predictive Flight Analytics Engine for multi-step telemetry forecasting."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from aeris_data.schema import FORECAST_TARGETS, NUMERIC_FEATURES
from ml.metrics import forecast_metrics
from ml.preprocess import FlightPreprocessor
from ml.windows import aggregate_feature_names, aggregate_window, iter_windows, next_step_targets


class FlightTelemetryPredictor:
    """Predictive baseline model forecasting subsequent flight states from historical windows."""

    def __init__(
        self,
        random_state: int = 42,
        deviation_sigma: float = 3.0,
    ) -> None:
        self.random_state = random_state
        self.deviation_sigma = deviation_sigma
        self.targets: list[str] = list(FORECAST_TARGETS)
        self.feature_names: list[str] = aggregate_feature_names()

        # Multi-output regressor using gradient boosting
        base_estimator = HistGradientBoostingRegressor(random_state=random_state, max_iter=80)
        self.model = MultiOutputRegressor(base_estimator)

        self.residual_means_: np.ndarray | None = None
        self.residual_stds_: np.ndarray | None = None
        self.deviation_thresholds_: dict[str, float] = {}
        self.is_fitted: bool = False

    def extract_dataset(
        self,
        frame: pd.DataFrame,
        preprocessor: FlightPreprocessor,
        length: int = 30,
        stride: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Extract input features from window and next-step targets."""
        X_list: list[np.ndarray] = []
        Y_list: list[np.ndarray] = []
        meta_list: list[dict[str, Any]] = []

        ordered = frame.sort_values(["flight_id", "t_s"])
        for flight_id, group in ordered.groupby("flight_id", sort=False):
            group = group.reset_index(drop=True)
            start = 0
            while start + length < len(group):
                window = group.iloc[start : start + length].copy()
                target_vector = next_step_targets(group, start + length)
                if target_vector is not None:
                    agg_series = aggregate_window(window, preprocessor)
                    feat_vector = agg_series[self.feature_names].to_numpy(dtype=float)
                    X_list.append(feat_vector)
                    Y_list.append(target_vector)

                    next_t = float(group.iloc[start + length]["t_s"])
                    meta_list.append({
                        "flight_id": str(flight_id),
                        "window_end_t_s": float(window["t_s"].iloc[-1]),
                        "target_t_s": next_t,
                    })
                start += stride

        if not X_list:
            return np.empty((0, len(self.feature_names))), np.empty((0, len(self.targets))), []

        return np.vstack(X_list), np.vstack(Y_list), meta_list

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "FlightTelemetryPredictor":
        """Train the multi-target telemetry regression model."""
        if len(X_train) == 0:
            raise ValueError("Training dataset for telemetry predictor cannot be empty.")

        self.model.fit(X_train, Y_train)

        # Estimate residual statistics on training set to set deviation thresholds
        preds = self.model.predict(X_train)
        residuals = np.abs(Y_train - preds)
        self.residual_means_ = np.mean(residuals, axis=0)
        self.residual_stds_ = np.std(residuals, axis=0)

        # Deviation thresholds: mean + k * sigma
        for idx, target in enumerate(self.targets):
            th = float(self.residual_means_[idx] + self.deviation_sigma * max(self.residual_stds_[idx], 1e-4))
            self.deviation_thresholds_[target] = th

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Forecast target values for input windows."""
        if not self.is_fitted:
            raise RuntimeError("Predictor must be fitted before calling predict.")
        if len(X) == 0:
            return np.empty((0, len(self.targets)))
        return self.model.predict(X)

    def evaluate(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, Any]:
        """Compute MAE, RMSE, and standard aviation forecasting metrics."""
        base_metrics = forecast_metrics(Y_true, Y_pred)
        if base_metrics.get("status") == "unavailable":
            return base_metrics

        # Add RMSE per target
        rmses: dict[str, float] = {}
        for idx, target in enumerate(self.targets):
            rmse = float(np.sqrt(np.mean((Y_true[:, idx] - Y_pred[:, idx]) ** 2)))
            rmses[f"rmse_{target}"] = round(rmse, 4)

        base_metrics.update(rmses)
        base_metrics["deviation_thresholds"] = self.deviation_thresholds_
        base_metrics["disclaimer"] = (
            "Aerospace Predictive Analytics Preview. Discrepancies represent mathematical "
            "model deviations, not certified flight safety events or sensor failures."
        )
        return base_metrics

    def save(self, filepath: Path | str) -> None:
        """Save predictor model to disk."""
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "targets": self.targets,
            "residual_means_": self.residual_means_,
            "residual_stds_": self.residual_stds_,
            "deviation_thresholds_": self.deviation_thresholds_,
            "is_fitted": self.is_fitted,
            "params": {
                "random_state": self.random_state,
                "deviation_sigma": self.deviation_sigma,
            },
        }
        joblib.dump(data, filepath)

    @classmethod
    def load(cls, filepath: Path | str) -> "FlightTelemetryPredictor":
        """Load predictor model from disk."""
        data = joblib.load(filepath)
        params = data.get("params", {})
        instance = cls(
            random_state=params.get("random_state", 42),
            deviation_sigma=params.get("deviation_sigma", 3.0),
        )
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.targets = data["targets"]
        instance.residual_means_ = data["residual_means_"]
        instance.residual_stds_ = data["residual_stds_"]
        instance.deviation_thresholds_ = data["deviation_thresholds_"]
        instance.is_fitted = data["is_fitted"]
        return instance
