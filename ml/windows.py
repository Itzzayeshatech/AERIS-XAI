"""Windows are cut inside a single flight after the split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeris_data.schema import (
    EVAL_COLUMNS,
    FORECAST_TARGETS,
    NUMERIC_FEATURES,
    WINDOW_LENGTH,
    WINDOW_STRIDE,
    phase_feature_names,
)
from ml.preprocess import FlightPreprocessor


def iter_windows(frame: pd.DataFrame, length: int = WINDOW_LENGTH, stride: int = WINDOW_STRIDE):
    ordered = frame.sort_values(["flight_id", "t_s"])
    for flight_id, group in ordered.groupby("flight_id", sort=False):
        group = group.reset_index(drop=True)
        start = 0
        while start + length <= len(group):
            window = group.iloc[start : start + length].copy()
            if window["flight_id"].nunique() != 1:
                raise RuntimeError("A window contains more than one flight_id.")
            yield str(flight_id), window
            start += stride


def assert_no_evaluation_columns(columns) -> None:
    leaked = set(columns).intersection(EVAL_COLUMNS)
    if leaked:
        raise RuntimeError(f"Model matrix contains evaluation columns: {sorted(leaked)}")


def sequence_matrix(window: pd.DataFrame, columns: list[str]) -> np.ndarray:
    assert_no_evaluation_columns(columns)
    assert_no_evaluation_columns(window.columns)
    return window[columns].to_numpy(dtype=float)


def aggregate_window(window: pd.DataFrame, preprocessor: FlightPreprocessor) -> pd.Series:
    """Scaled numeric aggregates plus the saved one-hot encoding of the phase mode."""
    values: dict[str, float] = {}
    for column in NUMERIC_FEATURES:
        series = window[column].astype(float)
        values[f"{column}_mean"] = float(series.mean())
        values[f"{column}_std"] = float(series.std(ddof=0))
        values[f"{column}_min"] = float(series.min())
        values[f"{column}_max"] = float(series.max())
        values[f"{column}_delta"] = float(series.iloc[-1] - series.iloc[0])
    mode = window["phase"].mode(dropna=False).iloc[0]
    encoded = preprocessor.encoder.transform(pd.DataFrame({"phase": [mode]}))
    for index, name in enumerate(phase_feature_names()):
        values[name] = float(encoded[0, index])
    assert_no_evaluation_columns(values.keys())
    return pd.Series(values)


def aggregate_feature_names() -> list[str]:
    names: list[str] = []
    for column in NUMERIC_FEATURES:
        for stat in ("mean", "std", "min", "max", "delta"):
            names.append(f"{column}_{stat}")
    names.extend(phase_feature_names())
    assert_no_evaluation_columns(names)
    return names


def window_label(raw_window: pd.DataFrame) -> int | None:
    if "is_anomaly" not in raw_window.columns:
        return None
    return int(raw_window["is_anomaly"].astype(int).sum() > 0)


def next_step_targets(raw_flight: pd.DataFrame, window_end_position: int) -> np.ndarray | None:
    if window_end_position >= len(raw_flight):
        return None
    row = raw_flight.iloc[window_end_position]
    if row[FORECAST_TARGETS].isna().any():
        return None
    return row[FORECAST_TARGETS].to_numpy(dtype=float)
