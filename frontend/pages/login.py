"""Login page layout and authentication handlers."""
from __future__ import annotations

from dash import html, dcc, Input, Output, State
import httpx
from backend.config import settings


def layout():
    return html.Div(
        [
            html.Div(
                [
                    html.H2("AERIS XAI SYSTEM ACCESS", style={"color": "#00aaff", "textAlign": "center"}),
                    html.P("Enter your engineering or administrative credentials to unlock laboratory pipelines.", style={"color": "#94a3b8", "textAlign": "center", "fontSize": "13px", "marginBottom": "24px"}),

                    html.Div(
                        [
                            html.Label("Username", className="form-label"),
                            dcc.Input(id="login-username", type="text", placeholder="e.g. admin or engineer", className="aeris-input"),
                        ],
                        className="form-group",
                    ),

                    html.Div(
                        [
                            html.Label("Password", className="form-label"),
                            dcc.Input(id="login-password", type="password", placeholder="••••••••", className="aeris-input"),
                        ],
                        className="form-group",
                    ),

                    html.Button("AUTHENTICATE & ENTER", id="login-btn", className="btn-aeris-primary", style={"width": "100%", "justifyContent": "center", "marginTop": "10px"}),

                    html.Div(id="login-alert", style={"marginTop": "16px"}),

                    html.Div(
                        [
                            html.P("Default Laboratory Accounts:", style={"fontWeight": "600", "color": "#64748b", "marginBottom": "4px"}),
                            html.Code("admin / admin123 (Full Control)", style={"color": "#38bdf8", "display": "block", "fontSize": "12px"}),
                            html.Code("engineer / engineer123 (Upload & Analysis)", style={"color": "#34d399", "display": "block", "fontSize": "12px"}),
                        ],
                        style={"marginTop": "24px", "paddingTop": "16px", "borderTop": "1px solid #1e293b", "fontSize": "12px"}
                    ),
                ],
                className="aeris-card",
                style={"maxWidth": "440px", "margin": "60px auto"},
            )
        ]
    )
