"""SQLAlchemy database models for AERIS XAI."""
from __future__ import annotations

import datetime
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from backend.db.base import Base


class User(Base):
    """System user with role-based access control."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), default="engineer", nullable=False)  # admin, engineer, viewer
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<User(username={self.username}, role={self.role})>"


class FlightRecord(Base):
    """Metadata catalog for uploaded and synthetically generated flights."""
    __tablename__ = "flight_records"

    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(String(64), unique=True, index=True, nullable=False)
    source = Column(String(32), nullable=False)  # "synthetic" or "uploaded"
    filename = Column(String(256), nullable=False)
    storage_path = Column(String(512), nullable=False)
    row_count = Column(Integer, nullable=False)
    duration_s = Column(Float, nullable=False)
    has_labels = Column(Boolean, default=False, nullable=False)
    seed = Column(Integer, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    feedback_entries = relationship("InvestigationFeedback", back_populates="flight", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<FlightRecord(flight_id={self.flight_id}, source={self.source}, rows={self.row_count})>"


class ModelRun(Base):
    """Tracking record for trained AI models."""
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(64), unique=True, index=True, nullable=False)
    model_type = Column(String(64), nullable=False)  # "isolation_forest", "lstm_autoencoder", "telemetry_predictor"
    dataset_version = Column(String(64), nullable=False)
    status = Column(String(32), default="pending", nullable=False)  # pending, completed, failed
    mlflow_run_id = Column(String(64), nullable=True)
    parameters_json = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)
    model_path = Column(String(512), nullable=True)
    trained_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<ModelRun(run_id={self.run_id}, model_type={self.model_type}, status={self.status})>"


class InvestigationFeedback(Base):
    """Feedback and engineering log entered by flight analysts during investigation."""
    __tablename__ = "investigation_feedback"

    id = Column(Integer, primary_key=True, index=True)
    feedback_id = Column(String(64), unique=True, index=True, nullable=False)
    flight_id = Column(String(64), ForeignKey("flight_records.flight_id", ondelete="CASCADE"), nullable=False)
    timestamp_s = Column(Float, nullable=False)
    engineer_name = Column(String(64), nullable=False)
    feedback_type = Column(String(64), nullable=False)  # "confirmed_anomaly", "false_positive", "benign_variation", "sensor_glitch"
    severity = Column(String(32), default="medium", nullable=False)  # low, medium, high, critical
    comments = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    flight = relationship("FlightRecord", back_populates="feedback_entries")

    def __repr__(self) -> str:
        return f"<InvestigationFeedback(flight_id={self.flight_id}, t={self.timestamp_s}, type={self.feedback_type})>"


class ReportRecord(Base):
    """Generated engineering investigation and flight data reports."""
    __tablename__ = "report_records"

    id = Column(Integer, primary_key=True, index=True)
    report_id = Column(String(64), unique=True, index=True, nullable=False)
    title = Column(String(256), nullable=False)
    report_type = Column(String(64), nullable=False)  # "flight_summary", "anomaly_investigation", "model_evaluation"
    format = Column(String(16), default="html", nullable=False)  # "html" or "pdf"
    storage_path = Column(String(512), nullable=False)
    created_by = Column(String(64), nullable=False)
    summary_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<ReportRecord(report_id={self.report_id}, title={self.title})>"
