"""AI Training Centre page layout."""
from __future__ import annotations

from dash import dcc, html
from frontend.app import api_request


def layout(token: str | None = None, role: str | None = None):
    # Fetch flights
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []

    flight_options = [{"label": f"{f['flight_id']} ({f['source']})", "value": f["flight_id"]} for f in flights_list]

    # Fetch trained models
    status_code, runs = api_request("GET", "/api/v1/training/runs", token=token)
    runs_list = runs if isinstance(runs, list) else []

    return html.Div(
        [
            html.H1("AI TRAINING & EVALUATION CENTRE", style={"marginBottom": "8px"}),
            html.P("Model training on strict flight-level splits with zero temporal leakage. Tracked via MLflow.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            # Training Config Card
            html.Div(
                [
                    html.H3("MODEL TRAINING CONFIGURATION", className="card-title"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Select Architecture", className="form-label"),
                                    dcc.Dropdown(
                                        id="train-model-type",
                                        options=[
                                            {"label": "Isolation Forest (Window Aggregates)", "value": "isolation_forest"},
                                            {"label": "PyTorch LSTM Autoencoder (Temporal Sequences)", "value": "lstm_autoencoder"},
                                            {"label": "Telemetry Predictor (Multi-Target Regression)", "value": "telemetry_predictor"},
                                        ],
                                        value="isolation_forest",
                                        clearable=False,
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Random Seed", className="form-label"),
                                    dcc.Input(id="train-seed", type="number", value=42, className="aeris-input"),
                                ],
                                style={"flex": "0.5"},
                            ),
                        ],
                        style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                    ),

                    html.Div(
                        [
                            html.Label("Training Dataset (Select Flights or leave empty for all):", className="form-label"),
                            dcc.Dropdown(
                                id="train-flight-ids",
                                options=flight_options,
                                value=[],
                                multi=True,
                                placeholder="All available flights in repository (min 5 required)",
                                className="aeris-select",
                            ),
                        ],
                        style={"marginBottom": "16px"},
                    ),

                    # Dynamic Hyperparameter Controls
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Hyperparameter 1", id="lbl-param1", className="form-label"),
                                    dcc.Input(id="train-param1", type="number", value=0.05, step=0.01, className="aeris-input"),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Hyperparameter 2", id="lbl-param2", className="form-label"),
                                    dcc.Input(id="train-param2", type="number", value=100, step=1, className="aeris-input"),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        id="hyperparams-container",
                        style={"display": "flex", "gap": "16px", "marginBottom": "20px"},
                    ),

                    html.Button("EXECUTE MODEL TRAINING", id="btn-start-training", className="btn-aeris-primary"),
                    dcc.Loading(id="loading-training", children=html.Div(id="training-result-alert", style={"marginTop": "14px"})),
                ],
                className="aeris-card",
            ),

            # Model Runs Catalog
            html.Div(
                [
                    html.H3(f"TRAINED MODEL RUNS CATALOG ({len(runs_list)})", className="card-title"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([
                                    html.Th("Run ID"),
                                    html.Th("Architecture"),
                                    html.Th("Dataset Version"),
                                    html.Th("Status"),
                                    html.Th("Key Evaluation Metrics"),
                                    html.Th("Trained At"),
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(r["run_id"], style={"fontWeight": "600", "fontFamily": "JetBrains Mono"}),
                                    html.Td(r["model_type"].replace("_", " ").title()),
                                    html.Td(r["dataset_version"], style={"fontFamily": "JetBrains Mono"}),
                                    html.Td(html.Span(r["status"].upper(), style={"color": "#10b981", "fontWeight": "600"})),
                                    html.Td(
                                        f"F1: {r.get('metrics', {}).get('f1', 'N/A')} | "
                                        f"Prec: {r.get('metrics', {}).get('precision', 'N/A')} | "
                                        f"Rec: {r.get('metrics', {}).get('recall', 'N/A')}"
                                        if "f1" in r.get("metrics", {}) else
                                        f"MAE Alt: {r.get('metrics', {}).get('mae_altitude_ft', 'N/A')} ft"
                                    ),
                                    html.Td(r["created_at"][:19].replace("T", " ")),
                                ])
                                for r in runs_list
                            ] if runs_list else [html.Tr([html.Td("No model runs executed yet.", colSpan=6)])]),
                        ],
                        className="aeris-table",
                    ),
                ],
                className="aeris-card",
            ),
        ]
    )
