"""Synthetic held-out metrics. Unavailable results stay null, never zero."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, precision_score, recall_score

EVALUATION_SCOPE = "synthetic_held_out_test"
DISCLAIMER = "These figures are not real-world detection or forecast performance."
LABORATORY_LIMITATION = (
    "Research laboratory using synthetic evaluation data; not certified; "
    "not for real-time aircraft control; not validated for operational aviation safety."
)

DETECTION_KEYS = ("precision", "recall", "f1", "average_precision")
FORECAST_KEYS = ("mae_altitude_ft", "mae_airspeed_kts", "mae_n1_pct")


def _base() -> dict:
    return {
        "evaluation_scope": EVALUATION_SCOPE,
        "disclaimer": DISCLAIMER,
    }


def _unavailable(reason: str, keys: tuple[str, ...]) -> dict:
    payload = _base()
    payload["status"] = "unavailable"
    payload["reason"] = reason
    for key in keys:
        payload[key] = None
    return payload


def detection_metrics(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    n_clean_flights: int,
    n_anomalous_flights: int,
) -> dict:
    if n_clean_flights < 1 or n_anomalous_flights < 1:
        return _unavailable(
            "Labeled test data is insufficient: "
            f"labeled clean test flights={n_clean_flights}, "
            f"labeled anomalous test flights={n_anomalous_flights}. "
            "At least one labeled clean flight and one labeled anomalous flight are required.",
            DETECTION_KEYS,
        )
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    preds = np.asarray(y_pred).astype(int)
    if labels.size == 0 or len(set(labels.tolist())) < 2:
        positive = int(labels.sum()) if labels.size else 0
        negative = int(labels.size - positive)
        return _unavailable(
            "Labeled test windows are insufficient: "
            f"positive windows={positive}, negative windows={negative}. "
            "Both classes are required before detection metrics are calculated.",
            DETECTION_KEYS,
        )
    payload = _base()
    payload.update(
        {
            "status": "computed",
            "reason": None,
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "average_precision": float(average_precision_score(labels, scores)),
        }
    )
    return payload


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true is None or len(y_true) < 1:
        rows = 0 if y_true is None else len(y_true)
        return _unavailable(
            f"Held-out test rows={rows}. Forecast MAE was not calculated because no test rows were available.",
            FORECAST_KEYS,
        )
    payload = _base()
    payload["status"] = "computed"
    payload["reason"] = None
    payload["mae_altitude_ft"] = float(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))
    payload["mae_airspeed_kts"] = float(mean_absolute_error(y_true[:, 1], y_pred[:, 1]))
    payload["mae_n1_pct"] = float(mean_absolute_error(y_true[:, 2], y_pred[:, 2]))
    return payload
