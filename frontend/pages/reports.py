"""Engineering Reports page layout."""
from __future__ import annotations

from dash import dcc, html
from frontend.app import api_request
from backend.config import settings


def layout(token: str | None = None):
    # Fetch flights
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []
    flight_opts = [{"label": f"{f['flight_id']} ({f['source']})", "value": f["flight_id"]} for f in flights_list]

    # Fetch trained models
    status_code, runs = api_request("GET", "/api/v1/training/runs", token=token)
    runs_list = runs if isinstance(runs, list) else []
    model_opts = [{"label": f"{r['run_id']} ({r['model_type']})", "value": r["run_id"]} for r in runs_list]

    # Fetch generated reports
    status_code, reports = api_request("GET", "/api/v1/reports", token=token)
    reports_list = reports if isinstance(reports, list) else []

    return html.Div(
        [
            html.H1("ENGINEERING REPORTS & VIVA DOCUMENTATION GENERATOR", style={"marginBottom": "8px"}),
            html.P("Compile formal engineering summaries, anomaly dossiers, and benchmark evaluation reports.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            # Generator Card
            html.Div(
                [
                    html.H3("COMPILE NEW TECHNICAL REPORT", className="card-title"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Report Title", className="form-label"),
                                    dcc.Input(id="rep-title", type="text", value="Post-Flight Telemetry & Anomaly Investigation", className="aeris-input"),
                                ],
                                style={"flex": "2"},
                            ),
                            html.Div(
                                [
                                    html.Label("Report Type", className="form-label"),
                                    dcc.Dropdown(
                                        id="rep-type",
                                        options=[
                                            {"label": "Flight Summary Report", "value": "flight_summary"},
                                            {"label": "Anomaly Investigation Dossier", "value": "anomaly_investigation"},
                                            {"label": "Model Evaluation Benchmark", "value": "model_evaluation"},
                                        ],
                                        value="flight_summary",
                                        clearable=False,
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1.5"},
                            ),
                        ],
                        style={"display": "flex", "gap": "16px", "marginBottom": "14px"},
                    ),

                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Associated Flight Record (Optional)", className="form-label"),
                                    dcc.Dropdown(
                                        id="rep-flight-id",
                                        options=flight_opts,
                                        value=flight_opts[0]["value"] if flight_opts else None,
                                        placeholder="Select Flight",
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Associated AI Model Run (Optional)", className="form-label"),
                                    dcc.Dropdown(
                                        id="rep-run-id",
                                        options=model_opts,
                                        value=model_opts[0]["value"] if model_opts else None,
                                        placeholder="Select Model Run",
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "16px", "marginBottom": "14px"},
                    ),

                    html.Div(
                        [
                            html.Label("Executive Summary / Findings Narrative:", className="form-label"),
                            dcc.Textarea(
                                id="rep-summary-text",
                                placeholder="Formal summary of telemetry screening, explainability insights, and engineer feedback...",
                                className="aeris-input",
                                style={"height": "70px"},
                            ),
                        ],
                        style={"marginBottom": "16px"},
                    ),

                    html.Button("GENERATE & COMPILE REPORT", id="btn-create-report", className="btn-aeris-primary"),
                    html.Div(id="report-create-alert", style={"marginTop": "10px"}),
                ],
                className="aeris-card",
            ),

            # Reports List
            html.Div(
                [
                    html.H3(f"GENERATED REPORTS ARCHIVE ({len(reports_list)})", className="card-title"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([
                                    html.Th("Report ID"),
                                    html.Th("Title"),
                                    html.Th("Type"),
                                    html.Th("Format"),
                                    html.Th("Author"),
                                    html.Th("Compiled At"),
                                    html.Th("Actions"),
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(r["report_id"], style={"fontFamily": "JetBrains Mono"}),
                                    html.Td(r["title"], style={"fontWeight": "600"}),
                                    html.Td(r["report_type"].replace("_", " ").title()),
                                    html.Td(html.Span(r["format"].upper(), className="badge-synthetic")),
                                    html.Td(r["created_by"]),
                                    html.Td(r["created_at"][:19].replace("T", " ")),
                                    html.Td(
                                        html.A(
                                            "DOWNLOAD HTML",
                                            href=f"{settings.AERIS_API_BASE}/api/v1/reports/{r['report_id']}/download",
                                            target="_blank",
                                            className="btn-aeris-secondary",
                                            style={"fontSize": "11px", "padding": "4px 8px"},
                                        )
                                    ),
                                ])
                                for r in reports_list
                            ] if reports_list else [html.Tr([html.Td("No reports compiled yet.", colSpan=7)])]),
                        ],
                        className="aeris-table",
                    ),
                ],
                className="aeris-card",
            ),
        ]
    )
