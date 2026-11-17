"""Isolation Forest anomaly detection model for tabular flight window aggregates."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.metrics import detection_metrics
from ml.preprocess import FlightPreprocessor
from ml.windows import aggregate_feature_names, aggregate_window, iter_windows, window_label


class FlightIsolationForest:
    """Isolation Forest anomaly detector operating on sliding window aggregates."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = 0.05,
        max_samples: float | int = 1.0,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_names: list[str] = aggregate_feature_names()
        self.threshold: float = 0.5
        self.score_min_: float = 0.0
        self.score_max_: float = 1.0
        self.is_fitted: bool = False

    def extract_features(
        self,
        frame: pd.DataFrame,
        preprocessor: FlightPreprocessor,
        length: int = 30,
        stride: int = 10,
    ) -> tuple[np.ndarray, list[dict[str, Any]], list[int | None]]:
        """Extract aggregate window features and window metadata."""
        X_rows: list[np.ndarray] = []
        meta_list: list[dict[str, Any]] = []
        labels: list[int | None] = []

        for flight_id, window in iter_windows(frame, length=length, stride=stride):
            agg_series = aggregate_window(window, preprocessor)
            feat_vector = agg_series[self.feature_names].to_numpy(dtype=float)
            X_rows.append(feat_vector)

            start_t = float(window["t_s"].iloc[0])
            end_t = float(window["t_s"].iloc[-1])
            lbl = window_label(window)

            meta_list.append({
                "flight_id": str(flight_id),
                "start_t_s": start_t,
                "end_t_s": end_t,
                "phase_mode": str(window["phase"].mode(dropna=False).iloc[0]),
            })
            labels.append(lbl)

        if not X_rows:
            return np.empty((0, len(self.feature_names))), [], []

        return np.vstack(X_rows), meta_list, labels

    def fit(self, X_train: np.ndarray) -> "FlightIsolationForest":
        """Fit the Isolation Forest model on training window features."""
        if len(X_train) == 0:
            raise ValueError("Cannot fit Isolation Forest on empty dataset.")

        self.model.fit(X_train)
        # Calibration of anomaly score range on training set
        raw_scores = -self.model.score_samples(X_train)
        self.score_min_ = float(np.min(raw_scores))
        self.score_max_ = float(np.max(raw_scores)) if float(np.max(raw_scores)) > self.score_min_ else self.score_min_ + 1.0
        
        # Determine empirical decision threshold based on contamination
        if isinstance(self.contamination, float) and 0.0 < self.contamination < 0.5:
            self.threshold = float(np.quantile(self._scale_scores(raw_scores), 1.0 - self.contamination))
        else:
            self.threshold = 0.5

        self.is_fitted = True
        return self

    def _scale_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Scale scores to [0, 1] interval where 1.0 indicates maximum anomaly."""
        denom = max(self.score_max_ - self.score_min_, 1e-6)
        scaled = (raw_scores - self.score_min_) / denom
        return np.clip(scaled, 0.0, 1.0)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute continuous anomaly scores in range [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring.")
        if len(X) == 0:
            return np.empty((0,), dtype=float)

        raw_scores = -self.model.score_samples(X)
        return self._scale_scores(raw_scores)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Binary anomaly prediction: 1 for anomaly, 0 for normal."""
        scores = self.score(X)
        th = threshold if threshold is not None else self.threshold
        return (scores >= th).astype(int)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: list[int | None],
        test_flight_ids: list[str],
        test_flight_labels: dict[str, str],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Compute metrics if labels exist, otherwise unsupervised diagnostics."""
        scores = self.score(X_test)
        preds = self.predict(X_test, threshold=threshold)

        has_any_label = any(y is not None for y in y_test)
        if has_any_label:
            # Count clean and anomalous flights
            n_clean = sum(1 for fid, kind in test_flight_labels.items() if kind == "clean")
            n_anom = sum(1 for fid, kind in test_flight_labels.items() if kind == "anomalous")

            valid_indices = [i for i, y in enumerate(y_test) if y is not None]
            y_true_valid = [y_test[i] for i in valid_indices]
            y_score_valid = [float(scores[i]) for i in valid_indices]
            y_pred_valid = [int(preds[i]) for i in valid_indices]

            metrics = detection_metrics(
                y_true=y_true_valid,
                y_score=y_score_valid,
                y_pred=y_pred_valid,
                n_clean_flights=n_clean,
                n_anomalous_flights=n_anom,
            )
            metrics["threshold_applied"] = threshold if threshold is not None else self.threshold
            metrics["total_windows_evaluated"] = len(valid_indices)
            metrics["flagged_windows"] = int(np.sum(preds))
            return metrics

        # Unsupervised diagnostics
        return {
            "evaluation_scope": "unsupervised_diagnostics",
            "status": "unsupervised",
            "reason": "Test dataset contains no ground truth anomaly labels.",
            "threshold_applied": threshold if threshold is not None else self.threshold,
            "total_windows_evaluated": len(X_test),
            "flagged_windows": int(np.sum(preds)),
            "outlier_ratio": float(np.mean(preds)) if len(preds) > 0 else 0.0,
            "score_mean": float(np.mean(scores)) if len(scores) > 0 else 0.0,
            "score_std": float(np.std(scores)) if len(scores) > 0 else 0.0,
            "score_p50": float(np.quantile(scores, 0.50)) if len(scores) > 0 else 0.0,
            "score_p95": float(np.quantile(scores, 0.95)) if len(scores) > 0 else 0.0,
            "score_max": float(np.max(scores)) if len(scores) > 0 else 0.0,
        }

    def save(self, filepath: Path | str) -> None:
        """Serialize model and metadata to disk."""
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "score_min_": self.score_min_,
            "score_max_": self.score_max_,
            "is_fitted": self.is_fitted,
            "params": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "max_samples": self.max_samples,
                "random_state": self.random_state,
            },
        }
        joblib.dump(data, filepath)

    @classmethod
    def load(cls, filepath: Path | str) -> "FlightIsolationForest":
        """Deserialize model from disk."""
        data = joblib.load(filepath)
        params = data.get("params", {})
        instance = cls(
            n_estimators=params.get("n_estimators", 100),
            contamination=params.get("contamination", 0.05),
            max_samples=params.get("max_samples", 1.0),
            random_state=params.get("random_state", 42),
        )
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        instance.threshold = data["threshold"]
        instance.score_min_ = data["score_min_"]
        instance.score_max_ = data["score_max_"]
        instance.is_fitted = data["is_fitted"]
        return instance
