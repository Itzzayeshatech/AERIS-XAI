"""Interactive Dash callbacks for all AERIS XAI pages."""
from __future__ import annotations

import base64
import io
import json
from dash import Input, Output, State, dcc, html, no_update
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from backend.config import settings
from frontend.app import api_request, app, get_navbar
from frontend.pages import (
    command_centre,
    comparison,
    detection,
    explain,
    investigation,
    lab,
    login,
    prediction,
    preprocessing,
    replay,
    reports,
    training,
    users,
)
from ml.inspection import inspect_frame


# ---------------------------------------------------------------------------
# 1. Routing & Navbar Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    [Output("navbar-container", "children"), Output("page-content", "children")],
    [Input("url", "pathname"), Input("session-store", "data")],
)
def render_page(pathname: str, session: dict | None):
    token = session.get("access_token") if session else None
    role = session.get("role") if session else None
    username = session.get("username") if session else None

    navbar = get_navbar(role=role, username=username)

    if pathname == "/login":
        return navbar, login.layout()
    elif pathname == "/lab":
        return navbar, lab.layout(token=token, role=role)
    elif pathname == "/preprocessing":
        return navbar, preprocessing.layout(token=token)
    elif pathname == "/training":
        return navbar, training.layout(token=token, role=role)
    elif pathname == "/detection":
        return navbar, detection.layout(token=token)
    elif pathname == "/prediction":
        return navbar, prediction.layout(token=token)
    elif pathname == "/explain":
        return navbar, explain.layout(token=token)
    elif pathname == "/comparison":
        return navbar, comparison.layout(token=token)
    elif pathname == "/replay":
        return navbar, replay.layout(token=token)
    elif pathname == "/investigation":
        return navbar, investigation.layout(token=token, username=username)
    elif pathname == "/reports":
        return navbar, reports.layout(token=token)
    elif pathname == "/users":
        return navbar, users.layout(token=token, role=role)
    else:
        return navbar, command_centre.layout(token=token)


# ---------------------------------------------------------------------------
# 2. Authentication Callback
# ---------------------------------------------------------------------------
@app.callback(
    [Output("session-store", "data"), Output("login-alert", "children"), Output("url", "pathname")],
    [Input("login-btn", "n_clicks")],
    [State("login-username", "value"), State("login-password", "value")],
    prevent_initial_call=True,
)
def handle_login(n_clicks: int, username: str | None, password: str | None):
    if not username or not password:
        return no_update, html.Div("Please enter both username and password.", style={"color": "#f43f5e"}), no_update

    import httpx
    try:
        resp = httpx.post(
            f"{settings.AERIS_API_BASE}/api/v1/auth/token",
            data={"username": username, "password": password},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            session = {
                "access_token": data["access_token"],
                "role": data["role"],
                "username": data["username"],
            }
            return session, html.Div("Login successful! Redirecting...", style={"color": "#10b981"}), "/"
        else:
            detail = resp.json().get("detail", "Invalid credentials")
            return no_update, html.Div(f"Authentication failed: {detail}", style={"color": "#f43f5e"}), no_update
    except Exception as e:
        return no_update, html.Div(f"Connection error: {str(e)}", style={"color": "#f43f5e"}), no_update


# ---------------------------------------------------------------------------
# 3. Flight Data Laboratory Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("gen-alert", "children"),
    Input("btn-generate-flights", "n_clicks"),
    [
        State("gen-count", "value"),
        State("gen-duration", "value"),
        State("gen-seed", "value"),
        State("gen-inject", "value"),
        State("session-store", "data"),
    ],
    prevent_initial_call=True,
)
def handle_generate_synthetic(n_clicks: int, count: int, duration: int, seed: int, inject: str, session: dict | None):
    token = session.get("access_token") if session else None
    payload = {
        "flight_count": count or 5,
        "duration_rows": duration or 90,
        "base_seed": seed or 42,
        "inject_anomalies": (inject == "true"),
        "prefix": "SYN",
    }
    status_code, resp = api_request("POST", "/api/v1/flights/generate", token=token, json_data=payload)
    if status_code == 201:
        return html.Div(f"Successfully generated {len(resp)} synthetic flights! Refresh page to view in catalog.", style={"color": "#10b981", "fontWeight": "600"})
    else:
        err = resp.get("detail", "Generation failed") if isinstance(resp, dict) else str(resp)
        return html.Div(f"Error: {err}", style={"color": "#f43f5e"})


@app.callback(
    Output("upload-alert", "children"),
    Input("upload-data-csv", "contents"),
    [State("upload-data-csv", "filename"), State("session-store", "data")],
    prevent_initial_call=True,
)
def handle_csv_upload(contents: str | None, filename: str | None, session: dict | None):
    if not contents or not filename:
        return no_update
    token = session.get("access_token") if session else None

    # Decode base64
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)

    files = {"file": (filename, decoded, "text/csv")}
    status_code, resp = api_request("POST", "/api/v1/flights/upload", token=token, files=files)

    if status_code == 201:
        return html.Div(f"Flight '{resp.get('flight_id')}' uploaded and validated successfully! Rows: {resp.get('row_count')}.", style={"color": "#10b981", "fontWeight": "600"})
    else:
        err = resp.get("detail", "Upload rejected") if isinstance(resp, dict) else str(resp)
        return html.Div(f"Upload rejected: {err}", style={"color": "#f43f5e"})


@app.callback(
    [
        Output("lab-metadata-cards", "children"),
        Output("chart-altitude-airspeed", "figure"),
        Output("chart-phase-pie", "figure"),
        Output("chart-engine-params", "figure"),
    ],
    Input("lab-flight-selector", "value"),
    State("session-store", "data"),
)
def update_lab_visuals(flight_id: str | None, session: dict | None):
    empty_fig = go.Figure().update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    if not flight_id:
        return html.Div("No flight selected."), empty_fig, empty_fig, empty_fig

    token = session.get("access_token") if session else None

    # Fetch summary and series
    _, summary = api_request("GET", f"/api/v1/flights/{flight_id}/summary", token=token)
    _, series_resp = api_request("GET", f"/api/v1/flights/{flight_id}/series", token=token, params={"max_points": 1000})

    if not isinstance(summary, dict) or not isinstance(series_resp, dict):
        return html.Div("Unable to load flight data."), empty_fig, empty_fig, empty_fig

    # 1. Metadata cards
    badge_cls = "badge-synthetic" if summary.get("source") == "synthetic" else "badge-uploaded"
    cards = html.Div(
        [
            html.Div([html.Div("FLIGHT ID", className="kpi-label"), html.Div(summary.get("flight_id"), className="kpi-value", style={"fontSize": "20px"})], className="kpi-card"),
            html.Div([html.Div("SOURCE", className="kpi-label"), html.Div(html.Span(summary.get("source", "").upper(), className=badge_cls), style={"marginTop": "8px"})], className="kpi-card"),
            html.Div([html.Div("DURATION", className="kpi-label"), html.Div(f"{summary.get('duration_s', 0):.1f} s", className="kpi-value", style={"fontSize": "20px"})], className="kpi-card"),
            html.Div([html.Div("RECORDED ROWS", className="kpi-label"), html.Div(f"{summary.get('row_count', 0)}", className="kpi-value", style={"fontSize": "20px"})], className="kpi-card"),
            html.Div([html.Div("EVAL LABELS", className="kpi-label"), html.Div("AVAILABLE" if summary.get("has_labels") else "NONE", className="kpi-value", style={"fontSize": "20px", "color": "#38bdf8" if summary.get("has_labels") else "#94a3b8"})], className="kpi-card"),
        ],
        className="kpi-grid",
    )

    # 2. Altitude & Airspeed chart
    points = series_resp.get("series", [])
    df = pd.DataFrame(points)

    fig_alt = go.Figure()
    if not df.empty and "t_s" in df.columns:
        fig_alt.add_trace(go.Scatter(x=df["t_s"], y=df.get("altitude_ft"), name="Altitude (ft)", line=dict(color="#00aaff", width=2)))
        fig_alt.add_trace(go.Scatter(x=df["t_s"], y=df.get("airspeed_kts"), name="Airspeed (kts)", yaxis="y2", line=dict(color="#34d399", width=2)))

        # Highlight anomalies if present
        if "is_anomaly" in df.columns and df["is_anomaly"].sum() > 0:
            anom_df = df[df["is_anomaly"] == 1]
            fig_alt.add_trace(go.Scatter(x=anom_df["t_s"], y=anom_df.get("altitude_ft"), mode="markers", name="Injected Anomaly", marker=dict(color="#f43f5e", size=6, symbol="x")))

    fig_alt.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title=f"Altitude & Airspeed Telemetry Profile ({flight_id})",
        xaxis=dict(title="Time (seconds)"),
        yaxis=dict(title="Altitude (ft)", title_font=dict(color="#00aaff"), tickfont=dict(color="#00aaff")),
        yaxis2=dict(title="Airspeed (kts)", title_font=dict(color="#34d399"), tickfont=dict(color="#34d399"), overlaying="y", side="right"),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # 3. Phase pie chart
    phase_dist = summary.get("phase_distribution", {})
    fig_phase = px.pie(
        names=list(phase_dist.keys()),
        values=list(phase_dist.values()),
        title="Flight Phase Distribution",
        hole=0.45,
        color_discrete_sequence=px.colors.sequential.Teal,
    ).update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", margin=dict(l=20, r=20, t=50, b=20))

    # 4. Engine parameters chart
    fig_eng = go.Figure()
    if not df.empty and "t_s" in df.columns:
        fig_eng.add_trace(go.Scatter(x=df["t_s"], y=df.get("n1_pct"), name="N1 Engine Fan Speed (%)", line=dict(color="#f59e0b", width=2)))
        fig_eng.add_trace(go.Scatter(x=df["t_s"], y=df.get("egt_c"), name="EGT Exhaust Gas Temp (°C)", yaxis="y2", line=dict(color="#fb7185", width=2)))
        fig_eng.add_trace(go.Scatter(x=df["t_s"], y=df.get("fuel_flow_pph"), name="Fuel Flow (pph)", yaxis="y3", line=dict(color="#a78bfa", width=2)))

    fig_eng.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title=f"Engine Telemetry Profile: N1, EGT & Fuel Flow ({flight_id})",
        xaxis=dict(title="Time (seconds)"),
        yaxis=dict(title="N1 (%)", title_font=dict(color="#f59e0b"), tickfont=dict(color="#f59e0b")),
        yaxis2=dict(title="EGT (°C)", title_font=dict(color="#fb7185"), tickfont=dict(color="#fb7185"), overlaying="y", side="right"),
        yaxis3=dict(title="Fuel (pph)", title_font=dict(color="#a78bfa"), tickfont=dict(color="#a78bfa"), overlaying="y", side="right", position=0.94),
        margin=dict(l=40, r=80, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return cards, fig_alt, fig_phase, fig_eng


# ---------------------------------------------------------------------------
# 4. Preprocessing Inspection Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("preproc-inspection-output", "children"),
    Input("preproc-flight-selector", "value"),
    State("session-store", "data"),
)
def update_preproc_inspection(flight_id: str | None, session: dict | None):
    if not flight_id:
        return html.Div("No flight selected.")
    token = session.get("access_token") if session else None

    _, series_resp = api_request("GET", f"/api/v1/flights/{flight_id}/series", token=token, params={"max_points": 5000})
    if not isinstance(series_resp, dict) or not series_resp.get("series"):
        return html.Div("Unable to load flight telemetry.")

    df = pd.DataFrame(series_resp["series"])
    insp = inspect_frame(df)

    warnings_box = None
    if insp.warnings:
        warnings_box = html.Div(
            [html.Strong("DATA QUALITY & CONSTRAINTS: "), html.Ul([html.Li(w) for w in insp.warnings])],
            className="disclaimer-banner",
        )

    # Column profiles table
    col_rows = []
    for col, prof in insp.column_profiles.items():
        col_rows.append(
            html.Tr([
                html.Td(col, style={"fontFamily": "JetBrains Mono", "fontWeight": "600"}),
                html.Td(prof.dtype),
                html.Td(f"{prof.n_missing} ({prof.pct_missing}%)", style={"color": "#f43f5e" if prof.n_missing > 0 else "#10b981"}),
                html.Td(f"{prof.col_min:.2f}" if prof.col_min is not None else "-"),
                html.Td(f"{prof.col_max:.2f}" if prof.col_max is not None else "-"),
                html.Td(f"{prof.mean:.2f}" if prof.mean is not None else "-"),
                html.Td(f"{prof.std:.2f}" if prof.std is not None else "-"),
            ])
        )

    return html.Div(
        [
            warnings_box,
            html.Table(
                [
                    html.Thead(
                        html.Tr([
                            html.Th("Column Name"),
                            html.Th("Data Type"),
                            html.Th("Missing Values"),
                            html.Th("Min"),
                            html.Th("Max"),
                            html.Th("Mean"),
                            html.Th("Std Dev"),
                        ])
                    ),
                    html.Tbody(col_rows),
                ],
                className="aeris-table",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 5. Training Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    [
        Output("lbl-param1", "children"),
        Output("lbl-param2", "children"),
        Output("train-param1", "value"),
        Output("train-param2", "value"),
    ],
    Input("train-model-type", "value"),
)
def update_hyperparam_labels(model_type: str):
    if model_type == "isolation_forest":
        return "Contamination Factor", "Number of Estimators (Trees)", 0.05, 100
    elif model_type == "lstm_autoencoder":
        return "Training Epochs", "Learning Rate", 20, 0.002
    else:
        return "Deviation Sigma (Threshold)", "Random State", 3.0, 42


@app.callback(
    Output("training-result-alert", "children"),
    Input("btn-start-training", "n_clicks"),
    [
        State("train-model-type", "value"),
        State("train-seed", "value"),
        State("train-flight-ids", "value"),
        State("train-param1", "value"),
        State("train-param2", "value"),
        State("session-store", "data"),
    ],
    prevent_initial_call=True,
)
def execute_model_training(n_clicks: int, model_type: str, seed: int, flight_ids: list[str] | None, p1: float, p2: float, session: dict | None):
    token = session.get("access_token") if session else None

    params: dict = {}
    if model_type == "isolation_forest":
        params = {"contamination": float(p1 or 0.05), "n_estimators": int(p2 or 100)}
    elif model_type == "lstm_autoencoder":
        params = {"epochs": int(p1 or 20), "learning_rate": float(p2 or 0.002)}
    else:
        params = {"deviation_sigma": float(p1 or 3.0)}

    payload = {
        "model_type": model_type,
        "seed": seed or 42,
        "flight_ids": flight_ids if flight_ids else None,
        "parameters": params,
    }

    status_code, resp = api_request("POST", "/api/v1/training/run", token=token, json_data=payload)

    if status_code == 201:
        run_id = resp.get("run_id")
        metrics = resp.get("metrics", {})
        splits = resp.get("split_counts", {})
        return html.Div(
            [
                html.H4(f"Model '{run_id}' Successfully Trained!", style={"color": "#10b981", "marginBottom": "6px"}),
                html.P(f"Splits: {splits.get('train')} Train / {splits.get('validation')} Val / {splits.get('test')} Test. "
                       f"Evaluation metrics: {json.dumps(metrics, indent=2)}", style={"fontSize": "12px", "color": "#e2e8f0"}),
                html.P("Refresh page to update model catalog.", style={"fontSize": "11px", "color": "#94a3b8"}),
            ],
            style={"backgroundColor": "rgba(16, 185, 129, 0.1)", "border": "1px solid #10b981", "padding": "16px", "borderRadius": "6px"}
        )
    else:
        err = resp.get("detail", "Training failed") if isinstance(resp, dict) else str(resp)
        return html.Div(f"Training failed: {err}", style={"color": "#f43f5e", "fontWeight": "600"})


# ---------------------------------------------------------------------------
# 6. Detection Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("detect-results-container", "children"),
    Input("btn-run-detect", "n_clicks"),
    [
        State("detect-flight-select", "value"),
        State("detect-model-select", "value"),
        State("detect-threshold-override", "value"),
        State("session-store", "data"),
    ],
    prevent_initial_call=True,
)
def run_detection_analysis(n_clicks: int, flight_id: str | None, run_id: str | None, th_override: float | None, session: dict | None):
    if not flight_id or not run_id:
        return html.Div("Please select both a flight record and a trained model.", style={"color": "#f59e0b"})

    token = session.get("access_token") if session else None
    payload = {
        "flight_id": flight_id,
        "run_id": run_id,
        "threshold": th_override if th_override else None,
    }

    status_code, resp = api_request("POST", "/api/v1/inference/detect", token=token, json_data=payload)
    if status_code != 200:
        err = resp.get("detail", "Detection inference failed") if isinstance(resp, dict) else str(resp)
        return html.Div(f"Error: {err}", style={"color": "#f43f5e"})

    windows = resp.get("windows", [])
    if not windows:
        return html.Div("No windows generated for this flight.", style={"color": "#94a3b8"})

    w_df = pd.DataFrame(windows)
    threshold = resp.get("threshold", 0.5)

    # Plot score timeline vs threshold
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w_df["end_t_s"], y=w_df["score"], mode="lines+markers", name="Anomaly Score", line=dict(color="#00aaff", width=2)))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#f43f5e", annotation_text=f"Decision Threshold ({threshold:.3f})")

    # Flagged windows markers
    flagged = w_df[w_df["is_anomaly"] == True]
    if not flagged.empty:
        fig.add_trace(go.Scatter(x=flagged["end_t_s"], y=flagged["score"], mode="markers", name="Flagged Anomaly", marker=dict(color="#f43f5e", size=8, symbol="diamond")))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title=f"Multivariate Anomaly Score Timeline: Flight {flight_id} ({resp.get('model_type')})",
        xaxis=dict(title="Window End Timestamp (seconds)"),
        yaxis=dict(title="Anomaly Score"),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    # Flagged table
    table_rows = []
    for w in windows:
        if w["is_anomaly"]:
            table_rows.append(
                html.Tr([
                    html.Td(str(w["window_index"])),
                    html.Td(f"{w['start_t_s']:.1f} s - {w['end_t_s']:.1f} s"),
                    html.Td(w["phase_mode"].title()),
                    html.Td(f"{w['score']:.4f}"),
                    html.Td(html.Span("FLAGGED ANOMALY", className="badge-anomaly")),
                    html.Td(
                        dcc.Link(
                            f"EXPLAIN WINDOW {w['window_index']} →",
                            href="/explain",
                            className="btn-aeris-secondary",
                            style={"fontSize": "11px", "padding": "4px 8px"},
                        )
                    ),
                ])
            )

    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Div("TOTAL WINDOWS", className="kpi-label"), html.Div(f"{resp.get('total_windows')}", className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("FLAGGED WINDOWS", className="kpi-label"), html.Div(f"{resp.get('flagged_windows')}", className="kpi-value", style={"color": "#f43f5e" if resp.get('flagged_windows', 0) > 0 else "#10b981"})], className="kpi-card"),
                    html.Div([html.Div("CALIBRATED THRESHOLD", className="kpi-label"), html.Div(f"{threshold:.4f}", className="kpi-value")], className="kpi-card"),
                ],
                className="kpi-grid",
            ),
            html.Div([dcc.Graph(figure=fig)], className="aeris-card"),
            html.Div(
                [
                    html.H3(f"FLAGGED ANOMALOUS TELEMETRY WINDOWS ({len(table_rows)})", className="card-title"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([
                                    html.Th("Window #"),
                                    html.Th("Time Interval"),
                                    html.Th("Flight Phase"),
                                    html.Th("Anomaly Score"),
                                    html.Th("Status"),
                                    html.Th("Investigate"),
                                ])
                            ),
                            html.Tbody(table_rows if table_rows else [html.Tr([html.Td("No anomalous windows detected. Flight is nominal.", colSpan=6)])]),
                        ],
                        className="aeris-table",
                    ),
                ],
                className="aeris-card",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# 7. Prediction Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("predict-results-container", "children"),
    Input("btn-run-forecast", "n_clicks"),
    [
        State("predict-flight-select", "value"),
        State("predict-model-select", "value"),
        State("session-store", "data"),
    ],
    prevent_initial_call=True,
)
def run_prediction_analysis(n_clicks: int, flight_id: str | None, run_id: str | None, session: dict | None):
    if not flight_id or not run_id:
        return html.Div("Please select both a flight record and a trained predictor model.", style={"color": "#f59e0b"})

    token = session.get("access_token") if session else None
    payload = {"flight_id": flight_id, "run_id": run_id}

    status_code, resp = api_request("POST", "/api/v1/inference/forecast", token=token, json_data=payload)
    if status_code != 200:
        err = resp.get("detail", "Forecasting failed") if isinstance(resp, dict) else str(resp)
        return html.Div(f"Error: {err}", style={"color": "#f43f5e"})

    points = resp.get("forecast_points", [])
    if not points:
        return html.Div("No forecast points available for this flight.", style={"color": "#94a3b8"})

    p_df = pd.DataFrame(points)

    # Plot actual vs predicted altitude
    fig_alt = go.Figure()
    fig_alt.add_trace(go.Scatter(x=p_df["t_s"], y=p_df["actual_altitude_ft"], name="Actual Altitude (ft)", line=dict(color="#00aaff", width=2)))
    fig_alt.add_trace(go.Scatter(x=p_df["t_s"], y=p_df["pred_altitude_ft"], name="Predicted Altitude (ft)", line=dict(color="#38bdf8", width=2, dash="dash")))

    fig_alt.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title=f"Altitude Forecast vs Actual: Flight {flight_id}",
        xaxis=dict(title="Time (seconds)"),
        yaxis=dict(title="Altitude (ft)"),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    # Plot actual vs predicted airspeed
    fig_ias = go.Figure()
    fig_ias.add_trace(go.Scatter(x=p_df["t_s"], y=p_df["actual_airspeed_kts"], name="Actual Airspeed (kts)", line=dict(color="#34d399", width=2)))
    fig_ias.add_trace(go.Scatter(x=p_df["t_s"], y=p_df["pred_airspeed_kts"], name="Predicted Airspeed (kts)", line=dict(color="#10b981", width=2, dash="dash")))

    fig_ias.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title=f"Airspeed Forecast vs Actual: Flight {flight_id}",
        xaxis=dict(title="Time (seconds)"),
        yaxis=dict(title="Airspeed (kts)"),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    metrics = resp.get("metrics", {})

    return html.Div(
        [
            html.Div(
                [
                    html.Div([html.Div("MAE ALTITUDE", className="kpi-label"), html.Div(f"{metrics.get('mae_altitude_ft', 0):.2f} ft", className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("MAE AIRSPEED", className="kpi-label"), html.Div(f"{metrics.get('mae_airspeed_kts', 0):.2f} kts", className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("MAE N1 ENGINE", className="kpi-label"), html.Div(f"{metrics.get('mae_n1_pct', 0):.2f} %", className="kpi-value")], className="kpi-card"),
                ],
                className="kpi-grid",
            ),
            html.Div([dcc.Graph(figure=fig_alt)], className="aeris-card"),
            html.Div([dcc.Graph(figure=fig_ias)], className="aeris-card"),
        ]
    )


# ---------------------------------------------------------------------------
# 8. Explainability Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("explain-results-container", "children"),
    Input("btn-run-explain", "n_clicks"),
    [
        State("explain-flight-select", "value"),
        State("explain-model-select", "value"),
        State("explain-window-idx", "value"),
        State("session-store", "data"),
    ],
    prevent_initial_call=True,
)
def run_xai_explanation(n_clicks: int, flight_id: str | None, run_id: str | None, window_idx: int | None, session: dict | None):
    if not flight_id or not run_id:
        return html.Div("Please select both a flight record and a trained model.", style={"color": "#f59e0b"})

    token = session.get("access_token") if session else None
    payload = {
        "flight_id": flight_id,
        "run_id": run_id,
        "window_index": window_idx or 0,
    }

    status_code, resp = api_request("POST", "/api/v1/explain", token=token, json_data=payload)
    if status_code != 200:
        err = resp.get("detail", "Explainability calculation failed") if isinstance(resp, dict) else str(resp)
        return html.Div(f"Error: {err}", style={"color": "#f43f5e"})

    top_features = resp.get("top_features", [])
    if not top_features:
        return html.Div("No feature attributions generated.", style={"color": "#94a3b8"})

    # Horizontal bar chart of feature contributions
    names = [f["feature_name"] for f in top_features][::-1]
    attrs = [f["attribution"] for f in top_features][::-1]
    colors = ["#f43f5e" if a > 0 else "#00aaff" for a in attrs]

    fig = go.Figure(go.Bar(
        x=attrs,
        y=names,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title=f"Feature Attributions for Window {resp.get('window_index')} ({resp.get('method')})",
        xaxis=dict(title="Attribution Toward Anomaly (Positive = Higher Anomaly Risk)"),
        margin=dict(l=150, r=40, t=50, b=40),
    )

    # Narrative explanation card
    narrative_card = html.Div(
        [
            html.H3("AEROSPACE ENGINEERING EXPLANATION SUMMARY", className="card-title"),
            html.P(resp.get("summary_text"), style={"fontSize": "15px", "color": "#f8fafc", "lineHeight": "1.6"}),
            html.P(f"Observation window: t={resp.get('window_start_t_s'):.1f}s to t={resp.get('window_end_t_s'):.1f}s | Calculated score: {resp.get('anomaly_score')}", style={"fontSize": "12px", "color": "#94a3b8"}),
        ],
        className="aeris-card",
    )

    return html.Div(
        [
            narrative_card,
            html.Div([dcc.Graph(figure=fig)], className="aeris-card"),
        ]
    )


# ---------------------------------------------------------------------------
# 9. Replay Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("replay-container", "children"),
    Input("replay-flight-select", "value"),
    State("session-store", "data"),
)
def render_flight_replay(flight_id: str | None, session: dict | None):
    if not flight_id:
        return html.Div("No flight selected.")

    token = session.get("access_token") if session else None
    status_code, resp = api_request("GET", f"/api/v1/replay/{flight_id}", token=token)
    if status_code != 200 or not isinstance(resp, dict):
        return html.Div("Unable to load replay data.")

    frames = resp.get("frames", [])
    if not frames:
        return html.Div("No trajectory frames recorded.")

    f_df = pd.DataFrame(frames)

    # Trajectory 2D plot using scatter
    fig = px.scatter(
        f_df,
        x="lon",
        y="lat",
        color="altitude_ft",
        hover_data=["t_s", "phase", "altitude_ft", "airspeed_kts", "is_anomaly"],
        color_continuous_scale="Viridis",
        title=f"2D Spatial Flight Track (Latitude vs Longitude): {flight_id}",
    ).update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")

    # Time-series instruments
    fig_inst = go.Figure()
    fig_inst.add_trace(go.Scatter(x=f_df["t_s"], y=f_df["pitch_deg"], name="Pitch (°)", line=dict(color="#00aaff")))
    fig_inst.add_trace(go.Scatter(x=f_df["t_s"], y=f_df["roll_deg"], name="Roll (°)", line=dict(color="#f59e0b")))
    fig_inst.add_trace(go.Scatter(x=f_df["t_s"], y=f_df["heading_deg"], name="Heading (°)", yaxis="y2", line=dict(color="#a78bfa")))

    fig_inst.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title="Cockpit Attitude & Heading Telemetry Timeline",
        xaxis=dict(title="Time (seconds)"),
        yaxis=dict(title="Attitude Angles (deg)"),
        yaxis2=dict(title="Heading (deg)", overlaying="y", side="right"),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return html.Div(
        [
            html.Div([dcc.Graph(figure=fig)], style={"marginBottom": "20px"}),
            html.Div([dcc.Graph(figure=fig_inst)]),
        ]
    )


# ---------------------------------------------------------------------------
# 10. Investigation Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("chart-correlation-heatmap", "figure"),
    Input("inv-flight-select", "value"),
    State("session-store", "data"),
)
def update_correlation_heatmap(flight_id: str | None, session: dict | None):
    empty_fig = go.Figure().update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    if not flight_id:
        return empty_fig

    token = session.get("access_token") if session else None
    status_code, resp = api_request("GET", f"/api/v1/investigation/correlations/{flight_id}", token=token)
    if status_code != 200 or not isinstance(resp, dict):
        return empty_fig

    features = resp.get("features", [])
    matrix = resp.get("matrix", [])

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=features,
        y=features,
        colorscale="Blues",
        zmin=-1.0,
        zmax=1.0,
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        title=f"Flight Parameter Cross-Correlation Matrix ({flight_id})",
        margin=dict(l=120, r=40, t=50, b=120),
    )
    return fig


@app.callback(
    Output("feedback-submit-alert", "children"),
    Input("btn-submit-feedback", "n_clicks"),
    [
        State("inv-flight-select", "value"),
        State("inv-time", "value"),
        State("inv-engineer", "value"),
        State("inv-type", "value"),
        State("inv-severity", "value"),
        State("inv-comments", "value"),
        State("session-store", "data"),
    ],
    prevent_initial_call=True,
)
def handle_feedback_submission(n_clicks: int, flight_id: str | None, t_s: float | None, eng: str | None, ftype: str, sev: str, comm: str | None, session: dict | None):
    if not flight_id or not comm:
        return html.Div("Please select a flight and provide investigation comments.", style={"color": "#f59e0b"})

    token = session.get("access_token") if session else None
    payload = {
        "flight_id": flight_id,
        "timestamp_s": float(t_s or 0.0),
        "engineer_name": eng or "Flight Analyst",
        "feedback_type": ftype,
        "severity": sev,
        "comments": comm,
    }

    status_code, resp = api_request("POST", "/api/v1/investigation/feedback", token=token, json_data=payload)
    if status_code == 201:
        return html.Div("Investigation finding logged successfully! Refresh page to view updated log.", style={"color": "#10b981", "fontWeight": "600"})
    else:
        err = resp.get("detail", "Submission failed") if isinstance(resp, dict) else str(resp)
        return html.Div(f"Error: {err}", style={"color": "#f43f5e"})


# ---------------------------------------------------------------------------
# 11. Reports Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("report-create-alert", "children"),
    Input("btn-create-report", "n_clicks"),
    [
        State("rep-title", "value"),
        State("rep-type", "value"),
        State("rep-flight-id", "value"),
        State("rep-run-id", "value"),
        State("rep-summary-text", "value"),
        State("session-store", "data"),
    ],
    prevent_initial_call=True,
)
def handle_report_creation(n_clicks: int, title: str | None, rtype: str, flight_id: str | None, run_id: str | None, summary_txt: str | None, session: dict | None):
    if not title:
        return html.Div("Please provide a report title.", style={"color": "#f59e0b"})

    token = session.get("access_token") if session else None
    payload = {
        "title": title,
        "report_type": rtype,
        "format": "html",
        "flight_id": flight_id,
        "run_id": run_id,
        "executive_summary": summary_txt,
    }

    status_code, resp = api_request("POST", "/api/v1/reports", token=token, json_data=payload)
    if status_code == 201:
        rep_id = resp.get("report_id")
        return html.Div(f"Report '{rep_id}' compiled successfully! Refresh to download HTML.", style={"color": "#10b981", "fontWeight": "600"})
    else:
        err = resp.get("detail", "Report creation failed") if isinstance(resp, dict) else str(resp)
        return html.Div(f"Error: {err}", style={"color": "#f43f5e"})


# ---------------------------------------------------------------------------
# 12. User Management Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("create-user-alert", "children"),
    Input("btn-create-user", "n_clicks"),
    [
        State("new-username", "value"),
        State("new-password", "value"),
        State("new-role", "value"),
        State("session-store", "data"),
    ],
    prevent_initial_call=True,
)
def handle_create_user(n_clicks: int, uname: str | None, pwd: str | None, role: str, session: dict | None):
    if not uname or not pwd:
        return html.Div("Please provide both username and password.", style={"color": "#f59e0b"})

    token = session.get("access_token") if session else None
    payload = {"username": uname, "password": pwd, "role": role}

    status_code, resp = api_request("POST", "/api/v1/auth/users", token=token, json_data=payload)
    if status_code == 201:
        return html.Div(f"User '{uname}' successfully provisioned as {role.upper()}! Refresh to update table.", style={"color": "#10b981", "fontWeight": "600"})
    else:
        err = resp.get("detail", "Provisioning failed") if isinstance(resp, dict) else str(resp)
        return html.Div(f"Error: {err}", style={"color": "#f43f5e"})
