"""Deterministic synthetic 1 Hz flight profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aeris_data.schema import ANOMALY_TYPES, PHASES

_PHASE_WEIGHTS = np.array([4, 3, 6, 14, 6, 4, 3], dtype=float)
_PHASE_TARGETS = {
    "taxi": dict(altitude_ft=40, airspeed_kts=18, pitch_deg=1, roll_deg=0, n1_pct=22, egt_c=420, fuel_flow_pph=400),
    "takeoff": dict(altitude_ft=400, airspeed_kts=145, pitch_deg=12, roll_deg=1, n1_pct=96, egt_c=710, fuel_flow_pph=6200),
    "climb": dict(altitude_ft=12000, airspeed_kts=230, pitch_deg=8, roll_deg=1, n1_pct=91, egt_c=660, fuel_flow_pph=4600),
    "cruise": dict(altitude_ft=33000, airspeed_kts=275, pitch_deg=2, roll_deg=0, n1_pct=84, egt_c=590, fuel_flow_pph=3100),
    "descent": dict(altitude_ft=10000, airspeed_kts=250, pitch_deg=-2, roll_deg=2, n1_pct=45, egt_c=490, fuel_flow_pph=1400),
    "approach": dict(altitude_ft=1800, airspeed_kts=145, pitch_deg=1, roll_deg=1, n1_pct=58, egt_c=520, fuel_flow_pph=1800),
    "landing": dict(altitude_ft=50, airspeed_kts=128, pitch_deg=3, roll_deg=0, n1_pct=35, egt_c=450, fuel_flow_pph=800),
}


def phase_counts(n_rows: int) -> list[int]:
    raw = [max(1, int(round(weight / _PHASE_WEIGHTS.sum() * n_rows))) for weight in _PHASE_WEIGHTS]
    while sum(raw) > n_rows:
        index = int(np.argmax(raw))
        if raw[index] > 1:
            raw[index] -= 1
        else:
            break
    while sum(raw) < n_rows:
        raw[PHASES.index("cruise")] += 1
    return raw


def generate_flight(
    flight_id: str,
    n_rows: int = 90,
    seed: int = 7,
    anomaly_type: str = "none",
    anomaly_start: int | None = None,
    anomaly_length: int = 6,
) -> pd.DataFrame:
    if anomaly_type not in ANOMALY_TYPES:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")
    if n_rows < 1:
        raise ValueError("n_rows must be positive.")
    rng = np.random.default_rng(seed)
    counts = phase_counts(n_rows)
    phases: list[str] = []
    knots_t: list[float] = []
    knots_alt: list[float] = []
    knots_ias: list[float] = []
    cursor = 0
    for phase, count in zip(PHASES, counts, strict=True):
        phases.extend([phase] * count)
        knots_t.append(cursor + max(count - 1, 0) / 2)
        knots_alt.append(_PHASE_TARGETS[phase]["altitude_ft"])
        knots_ias.append(_PHASE_TARGETS[phase]["airspeed_kts"])
        cursor += count
    t_s = np.arange(n_rows, dtype=float)
    altitude = np.interp(t_s, knots_t, knots_alt)
    airspeed = np.interp(t_s, knots_t, knots_ias)
    altitude = altitude + rng.normal(0, 8, n_rows)
    airspeed = np.clip(airspeed + rng.normal(0, 1.5, n_rows), 0, None)
    vertical = np.diff(altitude, prepend=altitude[0]) * 60.0
    pitch = np.interp(t_s, knots_t, [ _PHASE_TARGETS[p]["pitch_deg"] for p in PHASES ]) + rng.normal(0, 0.15, n_rows)
    roll = rng.normal(0, 0.4, n_rows)
    heading = (20 + t_s * 0.15 + rng.normal(0, 0.05, n_rows)) % 360
    n1 = np.interp(t_s, knots_t, [_PHASE_TARGETS[p]["n1_pct"] for p in PHASES]) + rng.normal(0, 0.3, n_rows)
    egt = np.interp(t_s, knots_t, [_PHASE_TARGETS[p]["egt_c"] for p in PHASES]) + rng.normal(0, 1.5, n_rows)
    fuel = np.interp(t_s, knots_t, [_PHASE_TARGETS[p]["fuel_flow_pph"] for p in PHASES]) + rng.normal(0, 10, n_rows)
    latitude = 28.55 + t_s * 0.0012
    longitude = 77.15 + t_s * 0.0011
    is_anomaly = np.zeros(n_rows, dtype=int)
    anomaly_labels = np.array(["none"] * n_rows, dtype=object)

    if anomaly_type != "none":
        start = n_rows // 2 if anomaly_start is None else anomaly_start
        start = int(np.clip(start, 0, n_rows - 1))
        stop = min(n_rows, start + anomaly_length)
        is_anomaly[start:stop] = 1
        anomaly_labels[start:stop] = anomaly_type
        if anomaly_type == "sensor_spike":
            altitude[start:stop] += 900
        elif anomaly_type == "engine_drift":
            n1[start:stop] -= 18
            egt[start:stop] += 70
        elif anomaly_type == "attitude_excursion":
            roll[start:stop] += 28
            pitch[start:stop] += 6
        elif anomaly_type == "airspeed_dropout":
            airspeed[start:stop] = np.nan

    return pd.DataFrame(
        {
            "flight_id": flight_id,
            "t_s": t_s,
            "phase": phases,
            "altitude_ft": altitude,
            "airspeed_kts": airspeed,
            "vertical_speed_fpm": vertical,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "heading_deg": heading,
            "n1_pct": n1,
            "egt_c": egt,
            "fuel_flow_pph": fuel,
            "latitude": latitude,
            "longitude": longitude,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_labels,
        }
    )


def anomaly_for_index(index: int) -> str:
    cycle = ["none", "none", "sensor_spike", "none", "engine_drift", "none", "attitude_excursion", "none", "airspeed_dropout"]
    return cycle[index % len(cycle)]
