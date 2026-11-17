"""Explainable AI (XAI) dashboard page."""
from __future__ import annotations

from dash import dcc, html
from frontend.app import api_request


def layout(token: str | None = None):
    # Fetch flights
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []
    flight_opts = [{"label": f"{f['flight_id']} ({f['source']})", "value": f["flight_id"]} for f in flights_list]

    # Fetch trained models
    status_code, runs = api_request("GET", "/api/v1/training/runs", token=token)
    runs_list = [r for r in (runs if isinstance(runs, list) else []) if r["model_type"] in ["isolation_forest", "lstm_autoencoder"]]
    model_opts = [{"label": f"{r['run_id']} ({r['model_type']})", "value": r["run_id"]} for r in runs_list]

    return html.Div(
        [
            html.H1("EXPLAINABLE AI (XAI) INVESTIGATION DASHBOARD", style={"marginBottom": "8px"}),
            html.P("Feature attribution and channel sensitivity explaining why specific flight windows were flagged.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            # Selector Card
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Select Flight Record:", className="form-label"),
                                    dcc.Dropdown(
                                        id="explain-flight-select",
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
                                    html.Label("Select Trained Model:", className="form-label"),
                                    dcc.Dropdown(
                                        id="explain-model-select",
                                        options=model_opts,
                                        value=model_opts[0]["value"] if model_opts else None,
                                        clearable=False,
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Window Index to Explain:", className="form-label"),
                                    dcc.Input(id="explain-window-idx", type="number", value=0, min=0, className="aeris-input"),
                                ],
                                style={"flex": "0.6"},
                            ),
                        ],
                        style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                    ),
                    html.Button("GENERATE MODEL EXPLANATION", id="btn-run-explain", className="btn-aeris-primary"),
                ],
                className="aeris-card",
            ),

            # Results Section
            dcc.Loading(
                id="loading-explain",
                children=html.Div(id="explain-results-container"),
            ),
        ]
    )
