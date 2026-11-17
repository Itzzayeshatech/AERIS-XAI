"""Application configuration for AERIS XAI backend."""
from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration managed via environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    DATABASE_URL: str = "sqlite:///./aeris.db"
    JWT_SECRET: str = "aeris-xai-super-secret-key-for-local-aerospace-lab-2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    DATA_DIR: Path = Path("./data")
    MLFLOW_TRACKING_URI: str = "sqlite:///./mlflow.db"
    MLFLOW_EXPERIMENT_NAME: str = "aeris_flight_anomaly_detection"

    CORS_ORIGINS: list[str] = [
        "http://127.0.0.1:8050",
        "http://localhost:8050",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]

    AERIS_API_BASE: str = "http://127.0.0.1:8000"
    DASH_HOST: str = "127.0.0.1"
    DASH_PORT: int = 8050

    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
    MAX_UPLOAD_ROWS: int = 150_000

    def ensure_directories(self) -> None:
        """Ensure necessary runtime directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "uploads").mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "synthetic").mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "models").mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "reports").mkdir(parents=True, exist_ok=True)


settings = Settings()
