"""User Management page layout (Admin Only)."""
from __future__ import annotations

from dash import dcc, html
from frontend.app import api_request


def layout(token: str | None = None, role: str | None = None):
    if role != "admin":
        return html.Div(
            [
                html.H2("ACCESS RESTRICTED", style={"color": "#f43f5e"}),
                html.P("User Management is exclusively available to administrators.", style={"color": "#94a3b8"}),
            ],
            className="aeris-card",
        )

    status_code, users = api_request("GET", "/api/v1/auth/users", token=token)
    user_list = users if isinstance(users, list) else []

    rows = []
    for u in user_list:
        rows.append(
            html.Tr([
                html.Td(str(u["id"])),
                html.Td(u["username"], style={"fontWeight": "600"}),
                html.Td(html.Span(u["role"].upper(), className="badge-synthetic" if u["role"] == "admin" else "badge-uploaded")),
                html.Td("ACTIVE" if u["is_active"] else "INACTIVE", style={"color": "#10b981" if u["is_active"] else "#f43f5e"}),
                html.Td(u["created_at"][:19].replace("T", " ")),
            ])
        )

    return html.Div(
        [
            html.H1("USER & ACCESS CONTROL MANAGEMENT", style={"marginBottom": "8px"}),
            html.P("Provision laboratory analyst accounts and assign Role-Based Access Control (RBAC) permissions.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            # Provision Form
            html.Div(
                [
                    html.H3("PROVISION NEW LABORATORY USER", className="card-title"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Username", className="form-label"),
                                    dcc.Input(id="new-username", type="text", placeholder="e.g. flight_analyst_1", className="aeris-input"),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Password", className="form-label"),
                                    dcc.Input(id="new-password", type="password", placeholder="Minimum 6 characters", className="aeris-input"),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Label("Role Designation", className="form-label"),
                                    dcc.Dropdown(
                                        id="new-role",
                                        options=[
                                            {"label": "Engineer (Upload & Analysis)", "value": "engineer"},
                                            {"label": "Admin (Full Control & Generation)", "value": "admin"},
                                            {"label": "Viewer (Read Only)", "value": "viewer"},
                                        ],
                                        value="engineer",
                                        clearable=False,
                                        className="aeris-select",
                                    ),
                                ],
                                style={"flex": "1.2"},
                            ),
                        ],
                        style={"display": "flex", "gap": "16px", "marginBottom": "16px"},
                    ),
                    html.Button("PROVISION ACCOUNT", id="btn-create-user", className="btn-aeris-primary"),
                    html.Div(id="create-user-alert", style={"marginTop": "10px"}),
                ],
                className="aeris-card",
            ),

            # Users Table
            html.Div(
                [
                    html.H3(f"REGISTERED LABORATORY USERS ({len(user_list)})", className="card-title"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([
                                    html.Th("ID"),
                                    html.Th("Username"),
                                    html.Th("Role"),
                                    html.Th("Account Status"),
                                    html.Th("Created At"),
                                ])
                            ),
                            html.Tbody(rows),
                        ],
                        className="aeris-table",
                    ),
                ],
                className="aeris-card",
            ),
        ]
    )
