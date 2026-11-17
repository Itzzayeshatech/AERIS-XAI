"""AI Command Centre overview dashboard."""
from __future__ import annotations

from dash import dcc, html
from frontend.app import api_request


def layout(token: str | None = None):
    # Fetch real counts from backend
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []

    status_code, runs = api_request("GET", "/api/v1/training/runs", token=token)
    runs_list = runs if isinstance(runs, list) else []

    status_code, reports = api_request("GET", "/api/v1/reports", token=token)
    reports_list = reports if isinstance(reports, list) else []

    total_flights = len(flights_list)
    synthetic_count = sum(1 for f in flights_list if f.get("source") == "synthetic")
    uploaded_count = sum(1 for f in flights_list if f.get("source") == "uploaded")
    total_models = len(runs_list)
    total_reports = len(reports_list)

    return html.Div(
        [
            html.Div(
                [
                    html.H1("AI COMMAND & SURVEILLANCE CENTRE", style={"color": "#f8fafc", "marginBottom": "6px"}),
                    html.P("Autonomous Explainable Telemetry Analytics, Anomaly Detection & Flight Quality Laboratory.", style={"color": "#94a3b8"}),
                ],
                style={"marginBottom": "24px"},
            ),

            # KPI Grid
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("TOTAL FLIGHTS IN REPOSITORY", className="kpi-label"),
                            html.Div(f"{total_flights}", className="kpi-value"),
                            html.Div(f"{synthetic_count} Synthetic / {uploaded_count} Uploaded", className="kpi-sub"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("TRAINED AI MODELS", className="kpi-label"),
                            html.Div(f"{total_models}", className="kpi-value"),
                            html.Div("IF / LSTM Autoencoder / Predictors", className="kpi-sub"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("COMPILED REPORTS", className="kpi-label"),
                            html.Div(f"{total_reports}", className="kpi-value"),
                            html.Div("Verified Engineering Reports", className="kpi-sub"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("LABORATORY SYSTEM STATUS", className="kpi-label"),
                            html.Div("OPERATIONAL", className="kpi-value", style={"color": "#10b981"}),
                            html.Div("FastAPI + Dash + PyTorch (CPU-Ready)", className="kpi-sub"),
                        ],
                        className="kpi-card",
                    ),
                ],
                className="kpi-grid",
            ),

            # Quick Action Panels
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Flight Data Laboratory", className="card-title"),
                            html.P("Generate reproducible synthetic telemetry batches or upload genuine CSV files adhering to the closed AERIS schema.", style={"fontSize": "13px", "color": "#94a3b8"}),
                            dcc.Link("OPEN FLIGHT LAB →", href="/lab", className="btn-aeris-primary", style={"marginTop": "12px", "display": "inline-block"}),
                        ],
                        className="aeris-card",
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.H3("AI Model Training Centre", className="card-title"),
                            html.P("Train Isolation Forest, PyTorch LSTM Autoencoders, or multi-target telemetry predictors with zero temporal leakage.", style={"fontSize": "13px", "color": "#94a3b8"}),
                            dcc.Link("OPEN TRAINING CENTRE →", href="/training", className="btn-aeris-primary", style={"marginTop": "12px", "display": "inline-block"}),
                        ],
                        className="aeris-card",
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.H3("Explainability & Investigation", className="card-title"),
                            html.P("Inspect SHAP tree attributions and channel ablation sensitivity to pinpoint anomalous flight events.", style={"fontSize": "13px", "color": "#94a3b8"}),
                            dcc.Link("INSPECT EXPLANATIONS →", href="/explain", className="btn-aeris-primary", style={"marginTop": "12px", "display": "inline-block"}),
                        ],
                        className="aeris-card",
                        style={"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
            ),

            # Research Disclaimer Notice
            html.Div(
                [
                    html.Strong("ACADEMIC PROTOTYPE NOTICE: "),
                    "AERIS XAI is an offline post-flight aerospace engineering laboratory developed for research and algorithmic verification. "
                    "Models and predictions do not represent certified flight control computers or live operational aviation advisories.",
                ],
                className="disclaimer-banner",
            ),
        ]
    )
