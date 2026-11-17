"""Main FastAPI application entry point for AERIS XAI."""
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routers import (
    auth,
    explain,
    flights,
    inference,
    investigation,
    models,
    replay,
    reports,
    training,
)
from backend.config import settings
from backend.db.base import Base, SessionLocal, engine
from backend.db.models import User
from backend.security.hashing import hash_password


def init_database_and_seed_defaults() -> None:
    """Create database tables and provision default admin & engineer accounts."""
    settings.ensure_directories()
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Check and seed admin
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin_user = User(
                username="admin",
                hashed_password=hash_password("admin123"),
                role="admin",
                is_active=True,
            )
            db.add(admin_user)

        # Check and seed engineer
        engineer = db.query(User).filter(User.username == "engineer").first()
        if not engineer:
            eng_user = User(
                username="engineer",
                hashed_password=hash_password("engineer123"),
                role="engineer",
                is_active=True,
            )
            db.add(eng_user)

        db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown routines."""
    init_database_and_seed_defaults()
    yield


app = FastAPI(
    title="AERIS XAI Aerospace Laboratory API",
    description="Backend API for synthetic flight telemetry, anomaly detection, forecasting, and explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth.router)
app.include_router(flights.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(explain.router)
app.include_router(models.router)
app.include_router(replay.router)
app.include_router(investigation.router)
app.include_router(reports.router)


@app.get("/api/v1/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "system": "AERIS XAI",
        "version": "1.0.0",
        "environment": "offline_research_laboratory",
    }


@app.get("/", tags=["Root"])
def root_info() -> dict[str, str]:
    """Root metadata."""
    return {
        "project": "AERIS XAI",
        "title": "Explainable AI Aerospace Laboratory",
        "docs_url": "/docs",
        "api_prefix": "/api/v1",
    }
