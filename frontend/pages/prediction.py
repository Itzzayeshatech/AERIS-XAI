"""Predictive Flight Analytics page layout."""
from __future__ import annotations

from dash import dcc, html
from frontend.app import api_request


def layout(token: str | None = None):
    # Fetch flights
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []
    flight_opts = [{"label": f"{f['flight_id']} ({f['source']})", "value": f["flight_id"]} for f in flights_list]

    # Fetch trained predictor models
    status_code, runs = api_request("GET", "/api/v1/training/runs", token=token)
    runs_list = [r for r in (runs if isinstance(runs, list) else []) if r["model_type"] == "telemetry_predictor"]
    model_opts = [{"label": f"{r['run_id']} (Predictor)", "value": r["run_id"]} for r in runs_list]

    return html.Div(
        [
            html.H1("PREDICTIVE FLIGHT ANALYTICS ENGINE", style={"marginBottom": "8px"}),
            html.P("Multi-parameter flight trajectory forecasting and residual deviation monitoring.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            # Selector Card
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Select Flight Record:", className="form-label"),
                                    dcc.Dropdown(
                                        id="predict-flight-select",
                                        options=flight_opts,
                                        value=flight_opts[0]["value"] if flight_opts else None,
                                        clearable=False,
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Select Trained Predictor Model:", className="form-label"),
                                    dcc.Dropdown(
                                        id="predict-model-select",
                                        options=model_opts,
                                        value=model_opts[0]["value"] if model_opts else None,
                                        clearable=False,
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                    ),
                    html.Button("GENERATE TELEMETRY FORECAST", id="btn-run-forecast", className="btn-aeris-primary"),
                ],
                className="aeris-card",
            ),

            # Results Section
            dcc.Loading(
                id="loading-forecast",
                children=html.Div(id="predict-results-container"),
            ),
        ]
    )
