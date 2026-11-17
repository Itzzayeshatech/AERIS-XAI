"""Flight Data Laboratory page layout."""
from __future__ import annotations

from dash import dcc, html
import plotly.graph_objects as go
from frontend.app import api_request


def layout(token: str | None = None, role: str | None = None):
    # Fetch existing flights
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []

    flight_options = [{"label": f"{f['flight_id']} ({f['source']}, {f['row_count']} rows)", "value": f["flight_id"]} for f in flights_list]

    # Admin generate panel
    admin_generate_panel = html.Div()
    if role == "admin":
        admin_generate_panel = html.Div(
            [
                html.H3("SYNTHETIC TELEMETRY GENERATOR (ADMIN ONLY)", className="card-title"),
                html.P("Generate reproducible multi-phase 1 Hz flight profiles with optional anomaly injection.", style={"fontSize": "12px", "color": "#94a3b8"}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Flight Count", className="form-label"),
                                dcc.Input(id="gen-count", type="number", value=5, min=1, max=30, className="aeris-input"),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                html.Label("Duration (Seconds/Rows)", className="form-label"),
                                dcc.Input(id="gen-duration", type="number", value=90, min=30, max=500, className="aeris-input"),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                html.Label("Base Random Seed", className="form-label"),
                                dcc.Input(id="gen-seed", type="number", value=42, className="aeris-input"),
                            ],
                            style={"flex": "1"},
                        ),
                        html.Div(
                            [
                                html.Label("Inject Evaluation Anomalies", className="form-label"),
                                dcc.Dropdown(
                                    id="gen-inject",
                                    options=[{"label": "Yes (Controlled Injection)", "value": "true"}, {"label": "No (Nominal Only)", "value": "false"}],
                                    value="true",
                                    clearable=False,
                                    className="aeris-select",
                                ),
                            ],
                            style={"flex": "1.5"},
                        ),
                    ],
                    style={"display": "flex", "gap": "14px", "marginBottom": "14px"},
                ),
                html.Button("GENERATE SYNTHETIC BATCH", id="btn-generate-flights", className="btn-aeris-primary"),
                html.Div(id="gen-alert", style={"marginTop": "10px"}),
            ],
            className="aeris-card",
        )

    # CSV upload panel
    csv_upload_panel = html.Div(
        [
            html.H3("CSV TELEMETRY INGESTION (ADMIN & ENGINEER)", className="card-title"),
            html.P("Upload flight telemetry adhering to the closed AERIS schema (CSV format, 1 flight_id per file).", style={"fontSize": "12px", "color": "#94a3b8"}),
            dcc.Upload(
                id="upload-data-csv",
                children=html.Div(
                    [
                        html.Span("📂 Drag and Drop or "),
                        html.A("Select Aircraft Telemetry CSV", style={"color": "#00aaff", "fontWeight": "600"}),
                    ]
                ),
                className="aeris-upload-box",
                multiple=False,
            ),
            html.Div(id="upload-alert", style={"marginTop": "10px"}),
        ],
        className="aeris-card",
    )

    # Flight selection & Visualizer section
    selected_flight_id = flights_list[0]["flight_id"] if flights_list else None

    visualizer_section = html.Div(
        [
            html.Div(
                [
                    html.H3("FLIGHT TELEMETRY VISUALIZER", className="card-title"),
                    html.Div(
                        [
                            html.Label("Select Flight Record:", className="form-label"),
                            dcc.Dropdown(
                                id="lab-flight-selector",
                                options=flight_options,
                                value=selected_flight_id,
                                clearable=False,
                                className="aeris-select",
                                style={"minWidth": "320px"},
                            ),
                        ],
                        style={"marginBottom": "20px", "maxWidth": "400px"},
                    ),
                ]
            ),
            # Metadata Summary Cards Container
            html.Div(id="lab-metadata-cards", style={"marginBottom": "20px"}),

            # Charts Grid
            html.Div(
                [
                    html.Div([dcc.Graph(id="chart-altitude-airspeed")], style={"flex": "2", "minWidth": "500px"}),
                    html.Div([dcc.Graph(id="chart-phase-pie")], style={"flex": "1", "minWidth": "300px"}),
                ],
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "20px"},
            ),

            html.Div(
                [
                    dcc.Graph(id="chart-engine-params"),
                ]
            ),
        ],
        className="aeris-card",
    )

    # Flight Library Catalog Table
    table_rows = []
    for f in flights_list:
        badge_cls = "badge-synthetic" if f["source"] == "synthetic" else "badge-uploaded"
        table_rows.append(
            html.Tr([
                html.Td(f["flight_id"], style={"fontWeight": "600", "fontFamily": "JetBrains Mono"}),
                html.Td(html.Span(f["source"].upper(), className=badge_cls)),
                html.Td(f"{f['row_count']}"),
                html.Td(f"{f['duration_s']:.1f} s"),
                html.Td("YES" if f["has_labels"] else "NO", style={"color": "#38bdf8" if f["has_labels"] else "#94a3b8"}),
                html.Td(str(f.get("seed") or "-")),
                html.Td(f["created_at"][:19].replace("T", " ")),
            ])
        )

    table_section = html.Div(
        [
            html.H3(f"FLIGHT REPOSITORY CATALOG ({len(flights_list)} FLIGHTS)", className="card-title"),
            html.Table(
                [
                    html.Thead(
                        html.Tr([
                            html.Th("Flight ID"),
                            html.Th("Dataset Source"),
                            html.Th("Rows"),
                            html.Th("Duration"),
                            html.Th("Evaluation Labels"),
                            html.Th("Seed"),
                            html.Th("Ingested At"),
                        ])
                    ),
                    html.Tbody(table_rows if table_rows else [html.Tr([html.Td("No flight datasets ingested yet.", colSpan=7)])]),
                ],
                className="aeris-table",
            ),
        ],
        className="aeris-card",
    )

    return html.Div(
        [
            html.H1("FLIGHT DATA LABORATORY", style={"marginBottom": "8px"}),
            html.P("Synthetic generation, CSV upload verification, and aerodynamic telemetry profiling.", style={"color": "#94a3b8", "marginBottom": "24px"}),
            admin_generate_panel,
            csv_upload_panel,
            visualizer_section,
            table_section,
        ]
    )
