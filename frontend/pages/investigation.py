"""Engineering Investigation Centre page layout."""
from __future__ import annotations

from dash import dcc, html
from frontend.app import api_request


def layout(token: str | None = None, username: str | None = None):
    # Fetch flights
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []
    flight_opts = [{"label": f"{f['flight_id']} ({f['source']})", "value": f["flight_id"]} for f in flights_list]
    initial_flight = flights_list[0]["flight_id"] if flights_list else None

    # Fetch feedback
    status_code, feedbacks = api_request("GET", "/api/v1/investigation/feedback", token=token)
    fb_list = feedbacks if isinstance(feedbacks, list) else []

    return html.Div(
        [
            html.H1("ENGINEERING INVESTIGATION CENTRE", style={"marginBottom": "8px"}),
            html.P("Parameter cross-correlations, anomaly ground-truth verification, and analyst feedback logging.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            # Selector
            html.Div(
                [
                    html.Label("Select Flight Record:", className="form-label"),
                    dcc.Dropdown(
                        id="inv-flight-select",
                        options=flight_opts,
                        value=initial_flight,
                        clearable=False,
                        className="aeris-select",
                        style={"maxWidth": "400px", "marginBottom": "16px"},
                    ),
                    dcc.Graph(id="chart-correlation-heatmap"),
                ],
                className="aeris-card",
            ),

            # Feedback Submission Form
            html.Div(
                [
                    html.H3("LOG ANALYST INVESTIGATION FINDINGS", className="card-title"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Event Timestamp (t_s)", className="form-label"),
                                    dcc.Input(id="inv-time", type="number", value=45.0, min=0, className="aeris-input"),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Analyst Name", className="form-label"),
                                    dcc.Input(id="inv-engineer", type="text", value=username or "Flight Engineer", className="aeris-input"),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Classification Finding", className="form-label"),
                                    dcc.Dropdown(
                                        id="inv-type",
                                        options=[
                                            {"label": "Confirmed Anomaly", "value": "confirmed_anomaly"},
                                            {"label": "False Positive", "value": "false_positive"},
                                            {"label": "Benign Environmental Variation", "value": "benign_variation"},
                                            {"label": "Sensor Glitch", "value": "sensor_glitch"},
                                        ],
                                        value="confirmed_anomaly",
                                        clearable=False,
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1.5"},
                            ),
                            html.Div(
                                [
                                    html.Label("Severity Level", className="form-label"),
                                    dcc.Dropdown(
                                        id="inv-severity",
                                        options=[
                                            {"label": "Low", "value": "low"},
                                            {"label": "Medium", "value": "medium"},
                                            {"label": "High", "value": "high"},
                                            {"label": "Critical", "value": "critical"},
                                        ],
                                        value="medium",
                                        clearable=False,
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={"display": "flex", "gap": "14px", "marginBottom": "14px"},
                    ),

                    html.Div(
                        [
                            html.Label("Technical Investigation Notes & Aerodynamic Rationalization:", className="form-label"),
                            dcc.Textarea(
                                id="inv-comments",
                                placeholder="Describe physical or sensor mechanism observed (e.g., sudden roll excursion accompanied by aileron trim discrepancy)...",
                                className="aeris-input",
                                style={"height": "80px"},
                            ),
                        ],
                        style={"marginBottom": "16px"},
                    ),

                    html.Button("RECORD INVESTIGATION ENTRY", id="btn-submit-feedback", className="btn-aeris-primary"),
                    html.Div(id="feedback-submit-alert", style={"marginTop": "10px"}),
                ],
                className="aeris-card",
            ),

            # Historical Log Table
            html.Div(
                [
                    html.H3(f"ANALYST INVESTIGATION LOGS ({len(fb_list)})", className="card-title"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([
                                    html.Th("Flight ID"),
                                    html.Th("Timestamp (s)"),
                                    html.Th("Analyst"),
                                    html.Th("Classification"),
                                    html.Th("Severity"),
                                    html.Th("Comments"),
                                    html.Th("Logged At"),
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(fb["flight_id"], style={"fontFamily": "JetBrains Mono"}),
                                    html.Td(f"{fb['timestamp_s']} s"),
                                    html.Td(fb["engineer_name"]),
                                    html.Td(html.Span(fb["feedback_type"].replace("_", " ").title(), className="badge-synthetic")),
                                    html.Td(html.Span(fb["severity"].upper(), className="badge-anomaly" if fb["severity"] in ["high", "critical"] else "badge-uploaded")),
                                    html.Td(fb["comments"]),
                                    html.Td(fb["created_at"][:19].replace("T", " ")),
                                ])
                                for fb in fb_list
                            ] if fb_list else [html.Tr([html.Td("No analyst feedback entries recorded yet.", colSpan=7)])]),
                        ],
                        className="aeris-table",
                    ),
                ],
                className="aeris-card",
            ),
        ]
    )
