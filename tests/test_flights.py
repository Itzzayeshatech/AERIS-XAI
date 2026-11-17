"""Unit and integration tests for Flight Data Laboratory and CSV ingestion."""
from __future__ import annotations

import io
import pandas as pd
from fastapi.testclient import TestClient

from aeris_data.generate import generate_flight
from aeris_data.schema import REQUIRED_COLUMNS, SchemaError, validate_frame


def create_sample_flight_csv(
    flight_id: str = "FL_TEST_01",
    n_rows: int = 50,
    with_labels: bool = True,
    corrupt_column: str | None = None,
    corrupt_value: str | None = None,
    duplicate_ts: bool = False,
) -> bytes:
    """Helper creating memory buffer of CSV flight telemetry."""
    df = generate_flight(flight_id=flight_id, n_rows=n_rows, seed=123, anomaly_type="none")
    if not with_labels:
        df = df.drop(columns=["is_anomaly", "anomaly_type"])

    if corrupt_column and corrupt_value is not None:
        df[corrupt_column] = corrupt_value

    if duplicate_ts and len(df) > 5:
        # Duplicate timestamp on row 3
        df.loc[3, "t_s"] = df.loc[2, "t_s"]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_valid_csv_upload_engineer(client: TestClient, engineer_headers: dict[str, str]):
    csv_bytes = create_sample_flight_csv("FL_ENG_01", n_rows=60)
    files = {"file": ("flight_eng_01.csv", csv_bytes, "text/csv")}

    resp = client.post("/api/v1/flights/upload", files=files, headers=engineer_headers)
    assert resp.status_code == 201
    data = resp.json()
    assert data["flight_id"] == "FL_ENG_01"
    assert data["source"] == "uploaded"
    assert data["row_count"] == 60
    assert data["has_labels"] is True


def test_missing_required_columns_rejected(client: TestClient, engineer_headers: dict[str, str]):
    # Create CSV missing 'altitude_ft'
    df = generate_flight(flight_id="FL_MISSING", n_rows=40, seed=1)
    df = df.drop(columns=["altitude_ft"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    files = {"file": ("missing_col.csv", buf.getvalue().encode("utf-8"), "text/csv")}

    resp = client.post("/api/v1/flights/upload", files=files, headers=engineer_headers)
    assert resp.status_code == 422
    assert "Missing required columns" in resp.json()["detail"]


def test_invalid_flight_phase_rejected(client: TestClient, engineer_headers: dict[str, str]):
    csv_bytes = create_sample_flight_csv("FL_BAD_PHASE", corrupt_column="phase", corrupt_value="hyperspace")
    files = {"file": ("bad_phase.csv", csv_bytes, "text/csv")}

    resp = client.post("/api/v1/flights/upload", files=files, headers=engineer_headers)
    assert resp.status_code == 422
    assert "phase contains values outside the closed vocabulary" in resp.json()["detail"]


def test_invalid_numeric_values_rejected(client: TestClient, engineer_headers: dict[str, str]):
    # Corrupt t_s with text
    csv_bytes = create_sample_flight_csv("FL_BAD_NUM", corrupt_column="t_s", corrupt_value="not_a_number")
    files = {"file": ("bad_num.csv", csv_bytes, "text/csv")}

    resp = client.post("/api/v1/flights/upload", files=files, headers=engineer_headers)
    assert resp.status_code == 422
    assert "t_s must be numeric" in resp.json()["detail"]


def test_duplicate_timestamps_rejected(client: TestClient, engineer_headers: dict[str, str]):
    csv_bytes = create_sample_flight_csv("FL_DUP_TS", duplicate_ts=True)
    files = {"file": ("dup_ts.csv", csv_bytes, "text/csv")}

    resp = client.post("/api/v1/flights/upload", files=files, headers=engineer_headers)
    assert resp.status_code == 422
    assert "duplicate timestamps" in resp.json()["detail"]


def test_optional_evaluation_columns_valid(client: TestClient, engineer_headers: dict[str, str]):
    # Without eval labels
    csv_unlabeled = create_sample_flight_csv("FL_UNLABELED", with_labels=False)
    files = {"file": ("unlabeled.csv", csv_unlabeled, "text/csv")}
    resp = client.post("/api/v1/flights/upload", files=files, headers=engineer_headers)
    assert resp.status_code == 201
    assert resp.json()["has_labels"] is False

    # Partial labels (only is_anomaly present, anomaly_type absent) should be rejected
    df = generate_flight("FL_PARTIAL", n_rows=40, seed=5)
    df = df.drop(columns=["anomaly_type"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    files = {"file": ("partial.csv", buf.getvalue().encode("utf-8"), "text/csv")}
    resp = client.post("/api/v1/flights/upload", files=files, headers=engineer_headers)
    assert resp.status_code == 422
    assert "both be present or both be absent" in resp.json()["detail"]


def test_synthetic_generator_reproducibility():
    """Verify that generating with identical seeds produces identical numeric values."""
    df1 = generate_flight(flight_id="SYN_REP_1", n_rows=80, seed=999)
    df2 = generate_flight(flight_id="SYN_REP_1", n_rows=80, seed=999)

    pd.testing.assert_frame_equal(df1, df2)


def test_anomaly_label_consistency():
    """Verify that when anomaly_type != 'none', is_anomaly is strictly 1, and vice versa."""
    df = generate_flight(flight_id="SYN_ANOM", n_rows=100, seed=42, anomaly_type="sensor_spike")
    cleaned = validate_frame(df)

    mismatch = ((cleaned["is_anomaly"] == 0) & (cleaned["anomaly_type"] != "none")) | (
        (cleaned["is_anomaly"] == 1) & (cleaned["anomaly_type"] == "none")
    )
    assert not mismatch.any()
    assert (cleaned["is_anomaly"] == 1).sum() > 0


def test_admin_only_synthetic_generation(client: TestClient, engineer_headers: dict[str, str], admin_headers: dict[str, str]):
    payload = {"flight_count": 2, "duration_rows": 50, "base_seed": 777}

    # Engineer should be forbidden
    resp_eng = client.post("/api/v1/flights/generate", json=payload, headers=engineer_headers)
    assert resp_eng.status_code == 403

    # Admin should succeed
    resp_adm = client.post("/api/v1/flights/generate", json=payload, headers=admin_headers)
    assert resp_adm.status_code == 201
    records = resp_adm.json()
    assert len(records) == 2
    assert records[0]["source"] == "synthetic"


def test_flight_metadata_and_summary_endpoints(client: TestClient, admin_headers: dict[str, str]):
    # First generate a test flight
    payload = {"flight_count": 1, "duration_rows": 60, "base_seed": 555, "prefix": "METATEST"}
    resp_gen = client.post("/api/v1/flights/generate", json=payload, headers=admin_headers)
    assert resp_gen.status_code == 201
    flight_id = resp_gen.json()[0]["flight_id"]

    # Metadata endpoint
    resp_meta = client.get(f"/api/v1/flights/{flight_id}", headers=admin_headers)
    assert resp_meta.status_code == 200
    assert resp_meta.json()["flight_id"] == flight_id

    # Summary endpoint
    resp_sum = client.get(f"/api/v1/flights/{flight_id}/summary", headers=admin_headers)
    assert resp_sum.status_code == 200
    sum_data = resp_sum.json()
    assert "phase_distribution" in sum_data
    assert "telemetry_ranges" in sum_data
    assert "altitude_ft" in sum_data["telemetry_ranges"]

    # Series endpoint
    resp_series = client.get(f"/api/v1/flights/{flight_id}/series", headers=admin_headers)
    assert resp_series.status_code == 200
    series_data = resp_series.json()
    assert len(series_data["series"]) == 60
