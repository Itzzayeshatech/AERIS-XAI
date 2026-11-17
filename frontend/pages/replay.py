"""Digital Flight Replay page layout."""
from __future__ import annotations

from dash import dcc, html
import plotly.graph_objects as go
from frontend.app import api_request


def layout(token: str | None = None):
    # Fetch flights
    status_code, flights = api_request("GET", "/api/v1/flights", token=token)
    flights_list = flights if isinstance(flights, list) else []
    flight_opts = [{"label": f"{f['flight_id']} ({f['source']})", "value": f["flight_id"]} for f in flights_list]
    initial_flight = flights_list[0]["flight_id"] if flights_list else None

    return html.Div(
        [
            html.H1("DIGITAL FLIGHT REPLAY & 2D TRAJECTORY VIEWER", style={"marginBottom": "8px"}),
            html.P("Synchronized spatial flight path reproduction with time-indexed cockpit instrumentation.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            # Selector Card
            html.Div(
                [
                    html.Label("Select Flight Record:", className="form-label"),
                    dcc.Dropdown(
                        id="replay-flight-select",
                        options=flight_opts,
                        value=initial_flight,
                        clearable=False,
                        className="aeris-select",
                        style={"maxWidth": "400px", "marginBottom": "16px"},
                    ),
                    html.Div(id="replay-container"),
                ],
                className="aeris-card",
            ),
        ]
    )
