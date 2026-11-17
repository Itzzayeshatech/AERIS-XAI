"""Model Comparison dashboard page."""
from __future__ import annotations

from dash import dcc, html
import plotly.graph_objects as go
from frontend.app import api_request


def layout(token: str | None = None):
    status_code, comparisons = api_request("GET", "/api/v1/models/compare", token=token)
    comp_list = comparisons if isinstance(comparisons, list) else []

    # Build comparison table
    rows = []
    model_names = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for c in comp_list:
        m_type = c["model_type"].replace("_", " ").title()
        model_names.append(f"{c['run_id'][:12]} ({m_type})")

        f1 = c.get("f1")
        precision = c.get("precision")
        recall = c.get("recall")
        f1_scores.append(float(f1) if f1 is not None else None)
        precision_scores.append(float(precision) if precision is not None else None)
        recall_scores.append(float(recall) if recall is not None else None)

        rows.append(
            html.Tr([
                html.Td(c["run_id"], style={"fontWeight": "600", "fontFamily": "JetBrains Mono"}),
                html.Td(m_type),
                html.Td(c["dataset_version"], style={"fontFamily": "JetBrains Mono"}),
                html.Td("N/A" if f1 is None else f"{float(f1):.4f}"),
                html.Td("N/A" if precision is None else f"{float(precision):.4f}"),
                html.Td("N/A" if recall is None else f"{float(recall):.4f}"),
                html.Td("N/A" if c.get("average_precision") is None else f"{float(c['average_precision']):.4f}"),
                html.Td("N/A" if c.get("mae_altitude_ft") is None else f"{float(c['mae_altitude_ft']):.4f} ft"),
            ])
        )

    # Bar chart comparing F1, Precision, Recall
    fig = go.Figure()
    if model_names:
        fig.add_trace(go.Bar(name="F1 Score", x=model_names, y=f1_scores, marker_color="#00aaff"))
        fig.add_trace(go.Bar(name="Precision", x=model_names, y=precision_scores, marker_color="#10b981"))
        fig.add_trace(go.Bar(name="Recall", x=model_names, y=recall_scores, marker_color="#f59e0b"))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        barmode="group",
        title="Detection Metrics Comparison Across Architectures",
        yaxis=dict(title="Score", range=[0, 1.05]),
        margin=dict(l=40, r=40, t=50, b=50),
    )

    return html.Div(
        [
            html.H1("AI MODEL BENCHMARK & COMPARISON", style={"marginBottom": "8px"}),
            html.P("Comparative evaluation across Isolation Forest, PyTorch LSTM Autoencoders, and Regressors.", style={"color": "#94a3b8", "marginBottom": "24px"}),

            html.Div(
                [
                    dcc.Graph(figure=fig),
                ],
                className="aeris-card",
            ),

            html.Div(
                [
                    html.H3(f"BENCHMARK METRICS SUMMARY ({len(comp_list)} MODELS)", className="card-title"),
                    html.Table(
                        [
                            html.Thead(
                                html.Tr([
                                    html.Th("Run ID"),
                                    html.Th("Architecture"),
                                    html.Th("Dataset Version"),
                                    html.Th("F1 Score"),
                                    html.Th("Precision"),
                                    html.Th("Recall"),
                                    html.Th("Avg Precision"),
                                    html.Th("Forecast MAE (Alt)"),
                                ])
                            ),
                            html.Tbody(rows if rows else [html.Tr([html.Td("No completed models to compare yet.", colSpan=8)])]),
                        ],
                        className="aeris-table",
                    ),
                ],
                className="aeris-card",
            ),
        ]
    )
