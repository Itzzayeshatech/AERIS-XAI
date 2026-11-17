"""Data Preprocessing & Telemetry Inspection visualizer."""
from __future__ import annotations

from dash import dcc, html
from frontend.app import api_request
from ml.inspection import inspect_frame
from backend.services.flight_service import get_flight_record, load_flight_dataframe
from backend.db.session import SessionLocal


def layout(token: str | None = None):
    # Fetch flights
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []

    flight_options = [{"label": f"{f['flight_id']} ({f['source']})", "value": f["flight_id"]} for f in flights_list]
    initial_flight = flights_list[0]["flight_id"] if flights_list else None

    return html.Div(
        [
            html.H1("DATA PREPROCESSING & TELEMETRY INSPECTION ENGINE", style={"marginBottom": "8px"}),
            html.P("Validation of physical units, duplicate timestamps, missing values, and closed-vocabulary phases.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            html.Div(
                [
                    html.Label("Select Flight to Inspect:", className="form-label"),
                    dcc.Dropdown(
                        id="preproc-flight-selector",
                        options=flight_options,
                        value=initial_flight,
                        clearable=False,
                        className="aeris-select",
                        style={"maxWidth": "400px", "marginBottom": "20px"},
                    ),
                    html.Div(id="preproc-inspection-output"),
                ],
                className="aeris-card",
            ),
        ]
    )
