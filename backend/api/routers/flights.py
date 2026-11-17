"""Flight Data Laboratory router for synthetic generation, upload, and telemetry querying."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile, status
from sqlalchemy.orm import Session

from backend.db.models import FlightRecord, User
from backend.db.session import get_db
from backend.schemas.flights import (
    FlightGenerateRequest,
    FlightRecordOut,
    FlightSummaryOut,
    SeriesResponse,
)
from backend.security.dependencies import get_current_user, require_role
from backend.services.flight_service import (
    delete_flight_record,
    generate_synthetic_batch,
    get_downsampled_series,
    get_flight_record,
    get_flight_summary,
    list_flights,
    process_uploaded_csv,
)

router = APIRouter(prefix="/api/v1/flights", tags=["Flight Data Laboratory"])


@router.post(
    "/generate",
    response_model=list[FlightRecordOut],
    status_code=status.HTTP_201_CREATED,
)
def generate_synthetic_telemetry(
    request: FlightGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["admin"])),
) -> list[FlightRecord]:
    """Admin-only: Generate reproducible synthetic flight telemetry batch."""
    return generate_synthetic_batch(request, db)


@router.post(
    "/upload",
    response_model=FlightRecordOut,
    status_code=status.HTTP_201_CREATED,
)
async def upload_flight_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["admin", "engineer"])),
) -> FlightRecord:
    """Engineer and Admin: Validate and store an uploaded aircraft telemetry CSV."""
    return await process_uploaded_csv(file, db)


@router.get("", response_model=list[FlightRecordOut])
def get_all_flights(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[FlightRecord]:
    """Retrieve catalog of all registered flight datasets."""
    return list_flights(db)


@router.get("/{flight_id}", response_model=FlightRecordOut)
def get_flight_by_id(
    flight_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FlightRecord:
    """Retrieve metadata for a specific flight."""
    return get_flight_record(flight_id, db)


@router.get("/{flight_id}/summary", response_model=FlightSummaryOut)
def get_flight_summary_statistics(
    flight_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> FlightSummaryOut:
    """Compute flight phase distribution and parameter ranges."""
    return get_flight_summary(flight_id, db)


@router.get("/{flight_id}/series", response_model=SeriesResponse)
def get_flight_telemetry_series(
    flight_id: str,
    max_points: int = Query(1000, ge=50, le=10000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> SeriesResponse:
    """Retrieve downsampled time-series telemetry for visualization."""
    return get_downsampled_series(flight_id, db, max_points=max_points)


@router.delete("/{flight_id}", status_code=status.HTTP_204_NO_CONTENT, response_class=Response)
def delete_flight(
    flight_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(["admin"])),
) -> Response:
    """Admin-only: Remove flight dataset and delete stored file."""
    delete_flight_record(flight_id, db)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
