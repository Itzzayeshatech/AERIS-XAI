"""End-to-end integration test exercising all laboratory pipelines."""
from __future__ import annotations

import io
from fastapi.testclient import TestClient
from aeris_data.generate import generate_flight


def test_full_laboratory_workflow_end_to_end(
    client: TestClient,
    admin_headers: dict[str, str],
    engineer_headers: dict[str, str],
):
    # 1. Admin generates synthetic telemetry batch (5 flights)
    gen_resp = client.post(
        "/api/v1/flights/generate",
        json={"flight_count": 5, "duration_rows": 60, "base_seed": 100, "inject_anomalies": True},
        headers=admin_headers,
    )
    assert gen_resp.status_code == 201
    created_flights = gen_resp.json()
    assert len(created_flights) == 5

    # 2. Engineer uploads genuine flight CSV
    df_upload = generate_flight("FL_INTEG_ENG", n_rows=60, seed=500)
    buf = io.StringIO()
    df_upload.to_csv(buf, index=False)
    upload_resp = client.post(
        "/api/v1/flights/upload",
        files={"file": ("integ_flight.csv", buf.getvalue().encode("utf-8"), "text/csv")},
        headers=engineer_headers,
    )
    assert upload_resp.status_code == 201
    assert upload_resp.json()["flight_id"] == "FL_INTEG_ENG"

    # 3. Query all flights catalog
    list_resp = client.get("/api/v1/flights", headers=engineer_headers)
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 6

    # 4. Trigger Isolation Forest training
    train_resp = client.post(
        "/api/v1/training/run",
        json={"model_type": "isolation_forest", "seed": 42, "parameters": {"contamination": 0.05, "n_estimators": 50}},
        headers=admin_headers,
    )
    assert train_resp.status_code == 201
    run_data = train_resp.json()
    run_id = run_data["run_id"]
    assert run_data["status"] == "completed"

    # 5. Run anomaly detection on uploaded flight
    detect_resp = client.post(
        "/api/v1/inference/detect",
        json={"run_id": run_id, "flight_id": "FL_INTEG_ENG"},
        headers=engineer_headers,
    )
    assert detect_resp.status_code == 200
    detect_data = detect_resp.json()
    assert "windows" in detect_data
    assert len(detect_data["windows"]) > 0

    # 6. Request explanation for window 0
    explain_resp = client.post(
        "/api/v1/explain",
        json={"run_id": run_id, "flight_id": "FL_INTEG_ENG", "window_index": 0},
        headers=engineer_headers,
    )
    assert explain_resp.status_code == 200
    explain_data = explain_resp.json()
    assert "top_features" in explain_data
    assert len(explain_data["top_features"]) > 0

    # 7. Record investigation feedback
    fb_resp = client.post(
        "/api/v1/investigation/feedback",
        json={
            "flight_id": "FL_INTEG_ENG",
            "timestamp_s": 30.0,
            "engineer_name": "Senior Test Analyst",
            "feedback_type": "confirmed_anomaly",
            "severity": "medium",
            "comments": "Observed localized parameter deviation during approach transition.",
        },
        headers=engineer_headers,
    )
    assert fb_resp.status_code == 201

    # 8. Compile report
    rep_resp = client.post(
        "/api/v1/reports",
        json={
            "title": "Integration Test Laboratory Report",
            "report_type": "flight_summary",
            "format": "html",
            "flight_id": "FL_INTEG_ENG",
            "run_id": run_id,
            "executive_summary": "Autonomous integration verification completed with nominal status.",
        },
        headers=engineer_headers,
    )
    assert rep_resp.status_code == 201
    report_id = rep_resp.json()["report_id"]

    # 9. Download report HTML
    dl_resp = client.get(f"/api/v1/reports/{report_id}/download", headers=engineer_headers)
    assert dl_resp.status_code == 200
    assert "text/html" in dl_resp.headers["content-type"]
    assert b"AERIS XAI ENGINEERING LABORATORY REPORT" in dl_resp.content
