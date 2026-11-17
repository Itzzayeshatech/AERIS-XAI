"""Data inspection and profiling for flight telemetry DataFrames.

Inspection is strictly read-only: it never modifies the frame, never drops rows,
and reports data quality issues as *warnings* rather than raising exceptions.
Callers decide how to respond.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from aeris_data.schema import EVAL_COLUMNS, NUMERIC_FEATURES, PHASES, REQUIRED_COLUMNS


@dataclass
class ColumnProfile:
    """Summary statistics for a single column."""

    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int
    # Numeric summary — None for non-numeric columns
    col_min: float | None = None
    col_max: float | None = None
    mean: float | None = None
    std: float | None = None
    p01: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p99: float | None = None


@dataclass
class FlightProfile:
    """Per-flight summary."""

    flight_id: str
    n_rows: int
    duration_s: float
    n_duplicate_t_s: int
    phase_counts: dict[str, int]
    n_anomalous_rows: int | None


@dataclass
class InspectionResult:
    """Complete inspection of a (possibly multi-flight) telemetry frame."""

    n_flights: int
    n_rows: int
    n_duplicate_timestamps: int
    phase_distribution: dict[str, int]
    has_labels: bool
    n_anomalous_rows: int | None
    anomaly_type_distribution: dict[str, int] | None
    duration_min_s: float
    duration_max_s: float
    duration_mean_s: float
    column_profiles: dict[str, ColumnProfile]
    per_flight: list[FlightProfile]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return _make_json_safe(asdict(self))


def inspect_frame(frame: pd.DataFrame) -> InspectionResult:
    """Inspect a telemetry DataFrame without modifying it.

    Works on single-flight or multi-flight frames.
    """
    warnings_: list[str] = []

    missing_req = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_req:
        warnings_.append(f"Missing required columns: {', '.join(missing_req)}.")

    n_rows = len(frame)
    n_flights = int(frame["flight_id"].nunique()) if "flight_id" in frame.columns else 0

    # Duplicate timestamps within each flight
    n_duplicate_timestamps = 0
    if "flight_id" in frame.columns and "t_s" in frame.columns:
        n_duplicate_timestamps = int(
            frame.groupby("flight_id", sort=False)["t_s"]
            .apply(lambda s: s.duplicated(keep="first").sum())
            .sum()
        )
        if n_duplicate_timestamps > 0:
            warnings_.append(
                f"{n_duplicate_timestamps} duplicate timestamp(s) found within flights. "
                "The pipeline will remove them, keeping the first occurrence."
            )

    # Phase distribution
    phase_distribution: dict[str, int] = {}
    if "phase" in frame.columns:
        for ph, cnt in frame["phase"].value_counts().items():
            phase_distribution[str(ph)] = int(cnt)
        unknown = sorted(set(phase_distribution) - set(PHASES))
        if unknown:
            warnings_.append(f"Unknown phase values (not in schema): {', '.join(unknown)}.")

    # Evaluation labels
    has_labels = all(c in frame.columns for c in EVAL_COLUMNS)
    n_anomalous_rows: int | None = None
    anomaly_type_distribution: dict[str, int] | None = None
    if has_labels:
        n_anomalous_rows = int(
            pd.to_numeric(frame["is_anomaly"], errors="coerce").fillna(0).sum()
        )
        anomaly_type_distribution = {}
        for at, cnt in frame["anomaly_type"].value_counts().items():
            anomaly_type_distribution[str(at)] = int(cnt)
        if n_anomalous_rows == 0:
            warnings_.append(
                "Evaluation labels are present but all rows are non-anomalous (is_anomaly=0)."
            )

    # Per-flight profiles and duration stats
    per_flight: list[FlightProfile] = []
    durations: list[float] = []
    if "flight_id" in frame.columns:
        for fid, grp in frame.groupby("flight_id", sort=False):
            t = pd.to_numeric(grp["t_s"], errors="coerce")
            dur = float(t.max() - t.min()) if len(t) > 1 else 0.0
            durations.append(dur)
            n_dup = int(t.duplicated(keep="first").sum())
            ph_counts: dict[str, int] = {}
            if "phase" in grp.columns:
                for ph, cnt in grp["phase"].value_counts().items():
                    ph_counts[str(ph)] = int(cnt)
            n_anom: int | None = None
            if has_labels:
                n_anom = int(
                    pd.to_numeric(grp["is_anomaly"], errors="coerce").fillna(0).sum()
                )
            per_flight.append(
                FlightProfile(
                    flight_id=str(fid),
                    n_rows=len(grp),
                    duration_s=dur,
                    n_duplicate_t_s=n_dup,
                    phase_counts=ph_counts,
                    n_anomalous_rows=n_anom,
                )
            )
        per_flight.sort(key=lambda p: p.flight_id)

    duration_min_s = float(min(durations)) if durations else 0.0
    duration_max_s = float(max(durations)) if durations else 0.0
    duration_mean_s = float(np.mean(durations)) if durations else 0.0

    if durations and duration_min_s < 30:
        warnings_.append(
            f"Shortest flight is {duration_min_s:.0f}s. "
            "Flights shorter than 30 s cannot produce any sliding windows (window_length=30)."
        )

    # Per-column profiles
    column_profiles: dict[str, ColumnProfile] = {}
    for col in frame.columns:
        series = frame[col]
        n_missing = int(series.isna().sum())
        pct_missing = round(100.0 * n_missing / max(n_rows, 1), 2)
        n_unique = int(series.nunique(dropna=True))

        num = pd.to_numeric(series, errors="coerce")
        if num.notna().any():
            q = num.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
            profile = ColumnProfile(
                dtype=str(series.dtype),
                n_missing=n_missing,
                pct_missing=pct_missing,
                n_unique=n_unique,
                col_min=float(num.min()),
                col_max=float(num.max()),
                mean=round(float(num.mean()), 6),
                std=round(float(num.std()), 6),
                p01=float(q.iloc[0]),
                p25=float(q.iloc[1]),
                p50=float(q.iloc[2]),
                p75=float(q.iloc[3]),
                p99=float(q.iloc[4]),
            )
        else:
            profile = ColumnProfile(
                dtype=str(series.dtype),
                n_missing=n_missing,
                pct_missing=pct_missing,
                n_unique=n_unique,
            )

        # Warn on unexpected missing values.
        # Suppress warning for airspeed_kts in labelled data where NaN is an
        # expected signal for the airspeed_dropout anomaly type.
        if n_missing > 0:
            is_expected = col == "airspeed_kts" and has_labels
            if not is_expected:
                warnings_.append(
                    f"Column '{col}' has {n_missing} missing value(s) ({pct_missing}%)."
                )

        column_profiles[col] = profile

    return InspectionResult(
        n_flights=n_flights,
        n_rows=n_rows,
        n_duplicate_timestamps=n_duplicate_timestamps,
        phase_distribution=phase_distribution,
        has_labels=has_labels,
        n_anomalous_rows=n_anomalous_rows,
        anomaly_type_distribution=anomaly_type_distribution,
        duration_min_s=duration_min_s,
        duration_max_s=duration_max_s,
        duration_mean_s=duration_mean_s,
        column_profiles=column_profiles,
        per_flight=per_flight,
        warnings=warnings_,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars and arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj
