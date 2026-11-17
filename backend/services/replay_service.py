"""Digital Flight Replay service for synchronized 2D flight path and instrument playback."""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from backend.services.flight_service import get_flight_record, load_flight_dataframe


def get_flight_replay_data(flight_id: str, db: Session, max_points: int = 1500) -> dict[str, Any]:
    """Retrieve trajectory and instrument data structured for 2D map replay."""
    record = get_flight_record(flight_id, db)
    df = load_flight_dataframe(record)

    total_rows = len(df)
    if total_rows > max_points:
        step = int(np.ceil(total_rows / max_points))
        sample_df = df.iloc[::step].copy()
    else:
        sample_df = df.copy()

    # Coordinates
    lat_vals = pd.to_numeric(sample_df["latitude"], errors="coerce").fillna(0.0).tolist()
    lon_vals = pd.to_numeric(sample_df["longitude"], errors="coerce").fillna(0.0).tolist()

    # Telemetry instruments
    timestamps = pd.to_numeric(sample_df["t_s"], errors="coerce").fillna(0.0).tolist()
    phases = sample_df["phase"].astype(str).tolist()
    altitudes = pd.to_numeric(sample_df["altitude_ft"], errors="coerce").fillna(0.0).tolist()
    airspeeds = pd.to_numeric(sample_df["airspeed_kts"], errors="coerce").fillna(0.0).tolist()
    pitches = pd.to_numeric(sample_df["pitch_deg"], errors="coerce").fillna(0.0).tolist()
    rolls = pd.to_numeric(sample_df["roll_deg"], errors="coerce").fillna(0.0).tolist()
    headings = pd.to_numeric(sample_df["heading_deg"], errors="coerce").fillna(0.0).tolist()
    n1_pcts = pd.to_numeric(sample_df["n1_pct"], errors="coerce").fillna(0.0).tolist()
    egt_cs = pd.to_numeric(sample_df["egt_c"], errors="coerce").fillna(0.0).tolist()

    anomaly_flags = [0] * len(sample_df)
    if "is_anomaly" in sample_df.columns:
        anomaly_flags = pd.to_numeric(sample_df["is_anomaly"], errors="coerce").fillna(0).astype(int).tolist()

    frames: list[dict[str, Any]] = []
    for i in range(len(timestamps)):
        frames.append({
            "t_s": timestamps[i],
            "lat": lat_vals[i],
            "lon": lon_vals[i],
            "phase": phases[i],
            "altitude_ft": round(altitudes[i], 1),
            "airspeed_kts": round(airspeeds[i], 1),
            "pitch_deg": round(pitches[i], 2),
            "roll_deg": round(rolls[i], 2),
            "heading_deg": round(headings[i], 1),
            "n1_pct": round(n1_pcts[i], 1),
            "egt_c": round(egt_cs[i], 1),
            "is_anomaly": anomaly_flags[i],
        })

    center_lat = float(np.mean(lat_vals)) if lat_vals else 28.55
    center_lon = float(np.mean(lon_vals)) if lon_vals else 77.15

    return {
        "flight_id": flight_id,
        "source": record.source,
        "duration_s": record.duration_s,
        "total_frames": len(frames),
        "center": {"lat": center_lat, "lon": center_lon},
        "frames": frames,
    }
