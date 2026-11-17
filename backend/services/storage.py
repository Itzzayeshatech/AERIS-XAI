"""File storage management and path sanitization for AERIS XAI."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from fastapi import HTTPException, status

from backend.config import settings


def sanitize_filename(filename: str) -> str:
    """Strip malicious path traversal characters and unsafe characters."""
    base = Path(filename).name
    # Remove all except alphanumeric, underscore, dash, and dot
    clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
    return clean if clean else "flight_data.csv"


def generate_storage_path(source: str, original_filename: str) -> Path:
    """Generate a safe, unique server-side storage path."""
    settings.ensure_directories()
    safe_name = sanitize_filename(original_filename)
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{source}_{unique_id}_{safe_name}"

    sub_dir = settings.DATA_DIR / ("synthetic" if source == "synthetic" else "uploads")
    return sub_dir / filename


def validate_file_size(content: bytes) -> None:
    """Check that file byte size does not exceed configured ceiling."""
    if len(content) > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum permitted size of {max_mb} MB.",
        )
