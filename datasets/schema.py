"""Closed flight-telemetry schema for AERIS XAI."""

from __future__ import annotations

import pandas as pd

PHASES: list[str] = [
    "taxi",
    "takeoff",
    "climb",
    "cruise",
    "descent",
    "approach",
    "landing",
]

NUMERIC_FEATURES: list[str] = [
    "altitude_ft",
    "airspeed_kts",
    "vertical_speed_fpm",
    "pitch_deg",
    "roll_deg",
    "heading_deg",
    "n1_pct",
    "egt_c",
    "fuel_flow_pph",
]

REPLAY_ONLY: list[str] = ["latitude", "longitude"]

EVAL_COLUMNS: list[str] = ["is_anomaly", "anomaly_type"]

ANOMALY_TYPES: list[str] = [
    "none",
    "sensor_spike",
    "engine_drift",
    "attitude_excursion",
    "airspeed_dropout",
]

IDENTITY: list[str] = ["flight_id", "t_s", "phase"]

REQUIRED_COLUMNS: list[str] = IDENTITY + NUMERIC_FEATURES + REPLAY_ONLY

FORECAST_TARGETS: list[str] = ["altitude_ft", "airspeed_kts", "n1_pct"]

MIN_DISTINCT_FLIGHTS = 5
WINDOW_LENGTH = 30
WINDOW_STRIDE = 10

FORBIDDEN_MODEL_COLUMNS = set(EVAL_COLUMNS)


class SchemaError(ValueError):
    """A telemetry table does not match the AERIS schema."""


def phase_feature_names() -> list[str]:
    return [f"phase_{phase}" for phase in PHASES]


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy or raise SchemaError with a specific reason."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {', '.join(missing)}.")
    present_eval = [column for column in EVAL_COLUMNS if column in frame.columns]
    if present_eval and len(present_eval) != len(EVAL_COLUMNS):
        raise SchemaError("Evaluation columns is_anomaly and anomaly_type must both be present or both be absent.")

    cleaned = frame.copy()
    cleaned["flight_id"] = cleaned["flight_id"].astype(str).str.strip()
    if cleaned["flight_id"].eq("").any() or cleaned["flight_id"].eq("nan").any():
        raise SchemaError("flight_id contains an empty value.")
    if cleaned["flight_id"].nunique() != 1:
        raise SchemaError("Each uploaded file must contain exactly one flight_id.")

    cleaned["t_s"] = pd.to_numeric(cleaned["t_s"], errors="coerce")
    if cleaned["t_s"].isna().any():
        raise SchemaError("t_s must be numeric.")

    unknown_phase = sorted(set(cleaned["phase"].astype(str)) - set(PHASES))
    if unknown_phase:
        raise SchemaError(
            "phase contains values outside the closed vocabulary "
            f"({', '.join(PHASES)}): {', '.join(unknown_phase)}."
        )
    cleaned["phase"] = cleaned["phase"].astype(str)

    for column in NUMERIC_FEATURES + REPLAY_ONLY:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if present_eval:
        cleaned["is_anomaly"] = pd.to_numeric(cleaned["is_anomaly"], errors="coerce")
        if cleaned["is_anomaly"].isna().any() or ~cleaned["is_anomaly"].isin([0, 1]).all():
            raise SchemaError("is_anomaly must be 0 or 1.")
        cleaned["is_anomaly"] = cleaned["is_anomaly"].astype(int)
        cleaned["anomaly_type"] = cleaned["anomaly_type"].astype(str)
        unknown_types = sorted(set(cleaned["anomaly_type"]) - set(ANOMALY_TYPES))
        if unknown_types:
            raise SchemaError(f"anomaly_type contains unknown values: {', '.join(unknown_types)}.")
        mismatch = ((cleaned["is_anomaly"] == 0) & (cleaned["anomaly_type"] != "none")) | (
            (cleaned["is_anomaly"] == 1) & (cleaned["anomaly_type"] == "none")
        )
        if mismatch.any():
            raise SchemaError("is_anomaly and anomaly_type disagree on at least one row.")

    if len(cleaned) < 1:
        raise SchemaError("The file has no data rows.")
    return cleaned.sort_values("t_s").reset_index(drop=True)


def has_evaluation_labels(frame: pd.DataFrame) -> bool:
    return all(column in frame.columns for column in EVAL_COLUMNS)


def label_kind(frame: pd.DataFrame) -> str:
    """Return unlabeled, clean, or anomalous. Labels are not model features."""
    if not has_evaluation_labels(frame):
        return "unlabeled"
    if int(frame["is_anomaly"].sum()) > 0:
        return "anomalous"
    return "clean"
