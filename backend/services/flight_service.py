"""Flight data management: synthetic generation, CSV upload, validation, and series retrieval."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from aeris_data.generate import anomaly_for_index, generate_flight
from aeris_data.schema import (
    EVAL_COLUMNS,
    NUMERIC_FEATURES,
    REQUIRED_COLUMNS,
    SchemaError,
    has_evaluation_labels,
    validate_frame,
)
from backend.config import settings
from backend.db.models import FlightRecord
from backend.schemas.flights import FlightGenerateRequest, FlightSummaryOut, SeriesResponse
from backend.services.storage import generate_storage_path, validate_file_size


def generate_synthetic_batch(
    request: FlightGenerateRequest,
    db: Session,
) -> list[FlightRecord]:
    """Generate a batch of reproducible synthetic flights and persist to disk and DB."""
    settings.ensure_directories()
    created_records: list[FlightRecord] = []

    for i in range(request.flight_count):
        seed = request.base_seed + i
        flight_id = f"{request.prefix}_{seed:04d}"

        # Determine anomaly type if injection is enabled
        anomaly_type = anomaly_for_index(i) if request.inject_anomalies else "none"

        # Generate dataframe
        df = generate_flight(
            flight_id=flight_id,
            n_rows=request.duration_rows,
            seed=seed,
            anomaly_type=anomaly_type,
        )

        # Validate with AERIS schema
        df = validate_frame(df)

        # Generate storage path and save CSV
        filename = f"{flight_id}.csv"
        storage_path = generate_storage_path(source="synthetic", original_filename=filename)
        df.to_csv(storage_path, index=False)

        duration = float(df["t_s"].max() - df["t_s"].min())
        has_labels = has_evaluation_labels(df)

        # Upsert or check existing
        existing = db.query(FlightRecord).filter(FlightRecord.flight_id == flight_id).first()
        if existing:
            existing.filename = filename
            existing.storage_path = str(storage_path)
            existing.row_count = len(df)
            existing.duration_s = duration
            existing.has_labels = has_labels
            existing.seed = seed
            record = existing
        else:
            record = FlightRecord(
                flight_id=flight_id,
                source="synthetic",
                filename=filename,
                storage_path=str(storage_path),
                row_count=len(df),
                duration_s=duration,
                has_labels=has_labels,
                seed=seed,
                notes="Synthetic telemetry generated for laboratory evaluation.",
            )
            db.add(record)

        created_records.append(record)

    db.commit()
    for rec in created_records:
        db.refresh(rec)
    return created_records


async def process_uploaded_csv(
    file: UploadFile,
    db: Session,
) -> FlightRecord:
    """Validate and store an engineer-uploaded aircraft telemetry CSV."""
    settings.ensure_directories()

    # Read and validate file content size
    content = await file.read()
    validate_file_size(content)

    # Parse CSV into DataFrame
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to parse CSV file: {str(e)}",
        )

    # Check maximum allowed rows
    if len(df) > settings.MAX_UPLOAD_ROWS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File contains {len(df)} rows, exceeding limit of {settings.MAX_UPLOAD_ROWS}.",
        )

    # Check for duplicate timestamps before schema validation to provide specific message
    if "flight_id" in df.columns and "t_s" in df.columns:
        if df["t_s"].duplicated().any():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="CSV contains duplicate timestamps (t_s) for the flight.",
            )

    # Validate against closed schema
    try:
        cleaned_df = validate_frame(df)
    except SchemaError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Schema validation error: {str(e)}",
        )

    flight_id = str(cleaned_df["flight_id"].iloc[0])

    # Check if flight_id already exists in database
    existing = db.query(FlightRecord).filter(FlightRecord.flight_id == flight_id).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Flight '{flight_id}' already exists in database.",
        )

    # Save to safe server-side storage path
    original_name = file.filename or f"{flight_id}.csv"
    storage_path = generate_storage_path(source="uploaded", original_filename=original_name)
    cleaned_df.to_csv(storage_path, index=False)

    duration = float(cleaned_df["t_s"].max() - cleaned_df["t_s"].min())
    has_labels = has_evaluation_labels(cleaned_df)

    record = FlightRecord(
        flight_id=flight_id,
        source="uploaded",
        filename=storage_path.name,
        storage_path=str(storage_path),
        row_count=len(cleaned_df),
        duration_s=duration,
        has_labels=has_labels,
        seed=None,
        notes="Uploaded telemetry dataset.",
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def list_flights(db: Session) -> list[FlightRecord]:
    """Retrieve all flight records ordered by creation timestamp."""
    return db.query(FlightRecord).order_by(FlightRecord.created_at.desc()).all()


def get_flight_record(flight_id: str, db: Session) -> FlightRecord:
    """Retrieve flight metadata or raise 404."""
    record = db.query(FlightRecord).filter(FlightRecord.flight_id == flight_id).first()
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flight '{flight_id}' not found.",
        )
    return record


def load_flight_dataframe(record: FlightRecord) -> pd.DataFrame:
    """Load DataFrame from storage path."""
    path = Path(record.storage_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Flight data file '{path.name}' is missing from storage.",
        )
    return pd.read_csv(path)


def get_flight_summary(flight_id: str, db: Session) -> FlightSummaryOut:
    """Calculate summary statistics and phase distribution for a flight."""
    record = get_flight_record(flight_id, db)
    df = load_flight_dataframe(record)

    phase_distribution = df["phase"].value_counts().to_dict()

    anomaly_counts = None
    if record.has_labels and "anomaly_type" in df.columns:
        anomaly_counts = df["anomaly_type"].value_counts().to_dict()

    # Calculate telemetry ranges (min, max, mean) for numeric features
    telemetry_ranges: dict[str, dict[str, float]] = {}
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) > 0:
                telemetry_ranges[col] = {
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "mean": round(float(s.mean()), 2),
                }

    return FlightSummaryOut(
        flight_id=record.flight_id,
        source=record.source,
        row_count=record.row_count,
        duration_s=record.duration_s,
        has_labels=record.has_labels,
        phase_distribution=phase_distribution,
        anomaly_counts=anomaly_counts,
        telemetry_ranges=telemetry_ranges,
        created_at=record.created_at,
    )


def get_downsampled_series(
    flight_id: str,
    db: Session,
    max_points: int = 1000,
) -> SeriesResponse:
    """Downsample telemetry time-series for smooth UI rendering."""
    record = get_flight_record(flight_id, db)
    df = load_flight_dataframe(record)

    total_points = len(df)
    is_downsampled = False

    if total_points > max_points:
        step = int(np.ceil(total_points / max_points))
        sampled_df = df.iloc[::step].copy()
        is_downsampled = True
    else:
        sampled_df = df.copy()

    # Clean NaN values for JSON serialization
    points_dict = sampled_df.replace({np.nan: None}).to_dict(orient="records")

    return SeriesResponse(
        flight_id=record.flight_id,
        total_points=total_points,
        returned_points=len(points_dict),
        is_downsampled=is_downsampled,
        series=points_dict,
    )


def delete_flight_record(flight_id: str, db: Session) -> None:
    """Delete a flight record and its underlying CSV file."""
    record = get_flight_record(flight_id, db)
    path = Path(record.storage_path)
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass
    db.delete(record)
    db.commit()
