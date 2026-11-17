"""Engineering Reports router."""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.db.models import ReportRecord, User
from backend.db.session import get_db
from backend.schemas.reports import ReportGenerateRequest, ReportRecordOut
from backend.security.dependencies import get_current_user, require_role
from backend.services.report_service import (
    generate_engineering_report,
    get_report,
    list_reports,
)

router = APIRouter(prefix="/api/v1/reports", tags=["Engineering Reports"])


@router.post("", response_model=ReportRecordOut, status_code=status.HTTP_201_CREATED)
def create_report(
    request: ReportGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["admin", "engineer"])),
) -> dict:
    """Generate a formal technical report."""
    rec = generate_engineering_report(request, created_by=current_user.username, db=db)
    return {
        "id": rec.id,
        "report_id": rec.report_id,
        "title": rec.title,
        "report_type": rec.report_type,
        "format": rec.format,
        "storage_path": rec.storage_path,
        "created_by": rec.created_by,
        "summary_data": json.loads(rec.summary_json or "{}"),
        "created_at": rec.created_at,
    }


@router.get("", response_model=list[ReportRecordOut])
def list_all_reports(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[dict]:
    """List all available generated reports."""
    records = list_reports(db)
    out = []
    for r in records:
        out.append({
            "id": r.id,
            "report_id": r.report_id,
            "title": r.title,
            "report_type": r.report_type,
            "format": r.format,
            "storage_path": r.storage_path,
            "created_by": r.created_by,
            "summary_data": json.loads(r.summary_json or "{}"),
            "created_at": r.created_at,
        })
    return out


@router.get("/{report_id}/download")
def download_report_file(
    report_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FileResponse:
    """Download the raw HTML/PDF report artifact."""
    rec = get_report(report_id, db)
    path = Path(rec.storage_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file is missing from server storage.",
        )
    return FileResponse(
        path=path,
        filename=path.name,
        media_type="text/html" if rec.format == "html" else "application/pdf",
    )
