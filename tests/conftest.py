"""Shared Pytest fixtures with temporary database and filesystem isolation."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.api.main import app
from backend.config import settings
from backend.db.base import Base
from backend.db.models import User
from backend.db.session import get_db
from backend.security.hashing import hash_password
from backend.security.jwt import create_access_token


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Provide an isolated temporary directory for test datasets and models."""
    tmp = Path(tempfile.mkdtemp(prefix="aeris_test_"))
    old_data_dir = settings.DATA_DIR
    settings.DATA_DIR = tmp
    settings.ensure_directories()
    yield tmp
    settings.DATA_DIR = old_data_dir
    try:
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture(scope="function")
def db_session(temp_dir: Path) -> Generator[Session, None, None]:
    """Provide an isolated SQLite database per test function."""
    db_path = temp_dir / f"test_{tempfile.mktemp(dir='')}.db"
    test_db_url = f"sqlite:///{db_path}"

    engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()

    # Seed default test users
    admin_user = User(
        username="test_admin",
        hashed_password=hash_password("admin_pass123"),
        role="admin",
        is_active=True,
    )
    engineer_user = User(
        username="test_engineer",
        hashed_password=hash_password("engineer_pass123"),
        role="engineer",
        is_active=True,
    )
    session.add(admin_user)
    session.add(engineer_user)
    session.commit()

    yield session

    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with overridden get_db dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def admin_headers() -> dict[str, str]:
    token = create_access_token({"sub": "test_admin", "role": "admin"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def engineer_headers() -> dict[str, str]:
    token = create_access_token({"sub": "test_engineer", "role": "engineer"})
    return {"Authorization": f"Bearer {token}"}
