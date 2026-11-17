"""Engineering report generator for flight summaries, anomaly investigations, and model evaluations."""
from __future__ import annotations

import datetime
import json
import uuid
import html
from pathlib import Path
from typing import Any
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from backend.config import settings
from backend.db.models import FlightRecord, ModelRun, ReportRecord
from backend.schemas.reports import ReportGenerateRequest
from backend.services.flight_service import get_flight_record, get_flight_summary
from backend.services.investigation_service import list_investigation_feedback
from backend.services.training_service import get_model_run


def _json_safe(value: Any) -> Any:
    """Convert Pydantic/ORM values into JSON-safe primitives."""
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def generate_engineering_report(
    request: ReportGenerateRequest,
    created_by: str,
    db: Session,
) -> ReportRecord:
    """Generate and store an engineering report in HTML format."""
    settings.ensure_directories()
    report_id = f"rep_{uuid.uuid4().hex[:8]}"

    summary_data: dict[str, Any] = {
        "report_id": report_id,
        "title": request.title,
        "report_type": request.report_type,
        "generated_by": created_by,
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "flight_id": request.flight_id,
        "run_id": request.run_id,
    }

    flight_info = None
    feedback_entries = []
    if request.flight_id:
        flight_sum = get_flight_summary(request.flight_id, db)
        flight_info = flight_sum.model_dump()
        summary_data["flight_summary"] = flight_info
        fbs = list_investigation_feedback(request.flight_id, db)
        feedback_entries = [
            {
                "engineer": fb.engineer_name,
                "type": fb.feedback_type,
                "severity": fb.severity,
                "comments": fb.comments,
                "timestamp_s": fb.timestamp_s,
            }
            for fb in fbs
        ]
        summary_data["feedback_entries"] = feedback_entries

    model_info = None
    if request.run_id:
        run = get_model_run(request.run_id, db)
        model_info = {
            "run_id": run.run_id,
            "model_type": run.model_type,
            "status": run.status,
            "dataset_version": run.dataset_version,
            "metrics": json.loads(run.metrics_json or "{}"),
        }
        summary_data["model_info"] = model_info

    # Build clean, modern HTML report
    safe_title = html.escape(request.title)
    safe_summary = html.escape(request.executive_summary or "This document synthesizes aircraft telemetry observations, algorithmic screening results, and explainability attributions conducted within the AERIS XAI laboratory.")
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AERIS XAI Report - {safe_title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0a0e1a;
            color: #e2e8f0;
            padding: 40px;
            margin: 0;
            line-height: 1.6;
        }}
        .report-header {{
            border-bottom: 2px solid #00aaff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .report-title {{
            color: #00aaff;
            font-size: 28px;
            margin: 0 0 10px 0;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            background-color: #0f172a;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #1e293b;
            margin-bottom: 30px;
        }}
        .meta-item {{ font-size: 14px; color: #94a3b8; }}
        .meta-value {{ font-size: 16px; font-weight: bold; color: #f8fafc; margin-top: 4px; }}
        .card {{
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 25px;
        }}
        h2 {{ color: #38bdf8; font-size: 20px; margin-top: 0; border-bottom: 1px solid #1e293b; padding-bottom: 8px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid #1e293b;
        }}
        th {{ background-color: #1e293b; color: #94a3b8; font-size: 13px; text-transform: uppercase; }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            background-color: #0284c7;
            color: white;
        }}
        .disclaimer-box {{
            background-color: #1e1b4b;
            border-left: 4px solid #818cf8;
            padding: 16px;
            border-radius: 4px;
            font-size: 13px;
            color: #c7d2fe;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="report-header">
        <h1 class="report-title">AERIS XAI ENGINEERING LABORATORY REPORT</h1>
        <p style="margin: 0; color: #94a3b8;">Autonomous Explainable Telemetry Analytics & Anomaly Investigation</p>
    </div>

    <div class="meta-grid">
        <div class="meta-item">Report ID<div class="meta-value">{report_id}</div></div>
        <div class="meta-item">Report Type<div class="meta-value">{request.report_type.replace('_', ' ').title()}</div></div>
        <div class="meta-item">Generated By<div class="meta-value">{created_by}</div></div>
        <div class="meta-item">Timestamp<div class="meta-value">{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</div></div>
    </div>

    <div class="card">
        <h2>Executive Summary</h2>
        <p>{safe_summary}</p>
    </div>
"""

    if flight_info:
        html_content += f"""
    <div class="card">
        <h2>Flight Telemetry Profile: {request.flight_id}</h2>
        <p><strong>Source:</strong> <span class="badge">{flight_info.get('source')}</span> | 
           <strong>Duration:</strong> {flight_info.get('duration_s')} s | 
           <strong>Recorded Rows:</strong> {flight_info.get('row_count')}</p>
        
        <h3>Phase Distribution</h3>
        <table>
            <tr><th>Flight Phase</th><th>Recorded Seconds</th></tr>
            {''.join(f"<tr><td>{ph.title()}</td><td>{cnt}</td></tr>" for ph, cnt in flight_info.get('phase_distribution', {}).items())}
        </table>
    </div>
"""

    if model_info:
        metrics_table = "".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in model_info.get("metrics", {}).items() if v is not None
        )
        html_content += f"""
    <div class="card">
        <h2>AI Model Architecture & Evaluation: {model_info.get('model_type')}</h2>
        <p><strong>Run ID:</strong> {model_info.get('run_id')} | <strong>Status:</strong> {model_info.get('status')}</p>
        <table>
            <tr><th>Metric</th><th>Calculated Value</th></tr>
            {metrics_table}
        </table>
    </div>
"""

    if feedback_entries:
        fb_rows = "".join(
            f"<tr><td>t={fb['timestamp_s']}s</td><td>{fb['engineer']}</td><td>{fb['type']}</td><td>{fb['severity'].upper()}</td><td>{fb['comments']}</td></tr>"
            for fb in feedback_entries
        )
        html_content += f"""
    <div class="card">
        <h2>Investigative Engineer Feedback Log</h2>
        <table>
            <tr><th>Timestamp</th><th>Analyst</th><th>Classification</th><th>Severity</th><th>Findings</th></tr>
            {fb_rows}
        </table>
    </div>
"""

    html_content += """
    <div class="disclaimer-box">
        <strong>AERIS XAI Research Notice:</strong>
        This technical document is generated from an offline post-flight research laboratory.
        Evaluations utilize reproducible telemetry benchmarks and explainability heuristics.
        This system is not certified by FAA or EASA for operational line aviation or live flight deck decision-making.
    </div>
</body>
</html>
"""

    storage_path = settings.DATA_DIR / "reports" / f"{report_id}.html"
    with open(storage_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    report_record = ReportRecord(
        report_id=report_id,
        title=request.title,
        report_type=request.report_type,
        format=request.format,
        storage_path=str(storage_path),
        created_by=created_by,
        summary_json=json.dumps(summary_data, default=_json_safe),
    )
    db.add(report_record)
    db.commit()
    db.refresh(report_record)
    return report_record


def list_reports(db: Session) -> list[ReportRecord]:
    """Retrieve all reports ordered by generation date."""
    return db.query(ReportRecord).order_by(ReportRecord.created_at.desc()).all()


def get_report(report_id: str, db: Session) -> ReportRecord:
    """Retrieve specific report record or raise 404."""
    rep = db.query(ReportRecord).filter(ReportRecord.report_id == report_id).first()
    if not rep:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found.",
        )
    return rep
