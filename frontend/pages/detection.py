"""Anomaly Detection Analysis page layout."""
from __future__ import annotations

from dash import dcc, html
from frontend.app import api_request


def layout(token: str | None = None):
    # Fetch flights
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []
    flight_opts = [{"label": f"{f['flight_id']} ({f['source']})", "value": f["flight_id"]} for f in flights_list]

    # Fetch trained models (IF and LSTM)
    status_code, runs = api_request("GET", "/api/v1/training/runs", token=token)
    runs_list = [r for r in (runs if isinstance(runs, list) else []) if r["model_type"] in ["isolation_forest", "lstm_autoencoder"]]
    model_opts = [{"label": f"{r['run_id']} ({r['model_type']})", "value": r["run_id"]} for r in runs_list]

    return html.Div(
        [
            html.H1("AIRCRAFT ANOMALY DETECTION ENGINE", style={"marginBottom": "8px"}),
            html.P("Execute multivariate anomaly screening using trained Isolation Forest or PyTorch LSTM Autoencoder.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            # Selector Card
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Select Flight Record:", className="form-label"),
                                    dcc.Dropdown(
                                        id="detect-flight-select",
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
                                        id="detect-model-select",
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
                                    html.Label("Detection Threshold Override (Optional):", className="form-label"),
                                    dcc.Input(id="detect-threshold-override", type="number", placeholder="Default model threshold", step=0.01, className="aeris-input"),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                    ),
                    html.Button("RUN ANOMALY SCREENING", id="btn-run-detect", className="btn-aeris-primary"),
                ],
                className="aeris-card",
            ),

            # Results Section
            dcc.Loading(
                id="loading-detection",
                children=html.Div(id="detect-results-container"),
            ),
        ]
    )
