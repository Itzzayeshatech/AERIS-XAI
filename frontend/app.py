"""Dash Multi-Page Web Application for AERIS XAI Aerospace Laboratory."""
from __future__ import annotations

import os
from typing import Any
import dash
from dash import Input, Output, State, dcc, html
import httpx

from backend.config import settings

# Initialize Dash application
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="AERIS XAI | Flight Telemetry Laboratory",
    update_title=None,
)

# API client helper function
def api_request(
    method: str,
    endpoint: str,
    token: str | None = None,
    json_data: dict | None = None,
    params: dict | None = None,
    files: Any | None = None,
) -> tuple[int, Any]:
    """Execute synchronous HTTP request to FastAPI backend."""
    url = f"{settings.AERIS_API_BASE}{endpoint}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        with httpx.Client(timeout=60.0) as client:
            if method.upper() == "GET":
                resp = client.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                if files:
                    resp = client.post(url, headers=headers, files=files, params=params)
                else:
                    resp = client.post(url, headers=headers, json=json_data, params=params)
            elif method.upper() == "DELETE":
                resp = client.delete(url, headers=headers)
            else:
                return 405, {"detail": "Method not allowed"}

            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, resp.text
    except Exception as e:
        return 503, {"detail": f"Backend connection failed: {str(e)}"}


# Navbar header layout
def get_navbar(role: str | None = None, username: str | None = None):
    links = [
        dcc.Link("Command Centre", href="/", className="nav-link-btn"),
        dcc.Link("Flight Lab", href="/lab", className="nav-link-btn"),
        dcc.Link("Inspection", href="/preprocessing", className="nav-link-btn"),
        dcc.Link("Training Centre", href="/training", className="nav-link-btn"),
        dcc.Link("Anomaly Detection", href="/detection", className="nav-link-btn"),
        dcc.Link("Predictions", href="/prediction", className="nav-link-btn"),
        dcc.Link("Explainability", href="/explain", className="nav-link-btn"),
        dcc.Link("Model Comparison", href="/comparison", className="nav-link-btn"),
        dcc.Link("Digital Replay", href="/replay", className="nav-link-btn"),
        dcc.Link("Investigation", href="/investigation", className="nav-link-btn"),
        dcc.Link("Reports", href="/reports", className="nav-link-btn"),
    ]
    if role == "admin":
        links.append(dcc.Link("Users", href="/users", className="nav-link-btn"))

    user_badge = html.Div(
        [
            html.Span(f"● {username} ({role})" if username else "● Guest", style={
                "color": "#10b981" if username else "#94a3b8",
                "fontSize": "12px",
                "marginRight": "12px",
                "fontWeight": "600",
            }),
            dcc.Link("Login" if not username else "Logout", href="/login", className="btn-aeris-secondary", style={"padding": "4px 10px", "fontSize": "11px"}),
        ],
        style={"display": "flex", "alignItems": "center"}
    )

    return html.Div(
        [
            html.Div(
                [
                    html.Span("AERIS", className="aeris-logo-text"),
                    html.Span("XAI", style={"color": "#f8fafc", "fontFamily": "Rajdhani", "fontWeight": "700", "fontSize": "26px"}),
                    html.Span("AEROSPACE LAB", className="aeris-logo-tag"),
                ],
                className="aeris-logo-brand",
            ),
            html.Div(links, className="nav-links-container"),
            user_badge,
        ],
        className="aeris-nav-header",
    )


# Root layout
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        dcc.Store(id="session-store", storage_type="local"),
        html.Div(id="navbar-container"),
        html.Div(id="page-content", className="main-container"),
    ]
)

# Register callbacks
import frontend.callbacks  # noqa: F401

