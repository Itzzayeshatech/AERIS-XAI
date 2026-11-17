"""FastAPI database session dependency."""
from __future__ import annotations

from typing import Generator
from sqlalchemy.orm import Session

from backend.db.base import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a database session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
