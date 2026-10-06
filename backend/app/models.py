from datetime import datetime
from typing import Any
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base

class Timestamped:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Project(Timestamped, Base):
    __tablename__ = "projects"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    framework: Mapped[str] = mapped_column(String(50), default="scikit-learn")
    task_type: Mapped[str] = mapped_column(String(30), default="classification")
    local_model_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="REGISTERED")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    drift_status: Mapped[str] = mapped_column(String(30), default="not_checked")
    check_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_check_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    agent_connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    credentials: Mapped[list["AgentCredential"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    monitoring_windows: Mapped[list["MonitoringWindow"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    events: Mapped[list["ActivityEvent"]] = relationship(back_populates="project", cascade="all, delete-orphan")

class AgentCredential(Timestamped, Base):
    __tablename__ = "agent_credentials"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    token_digest: Mapped[str] = mapped_column(String(64), unique=True)
    label: Mapped[str] = mapped_column(String(120), default="local-agent")
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    project: Mapped[Project] = relationship(back_populates="credentials")

class MonitoringWindow(Timestamped, Base):
    __tablename__ = "monitoring_windows"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    source: Mapped[str] = mapped_column(String(40), default="agent")
    data_quality_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    drift_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    performance_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    severity: Mapped[str] = mapped_column(String(20), default="healthy")
    project: Mapped[Project] = relationship(back_populates="monitoring_windows")

class ActivityEvent(Timestamped, Base):
    __tablename__ = "activity_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    kind: Mapped[str] = mapped_column(String(80))
    message: Mapped[str] = mapped_column(Text)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    project: Mapped[Project] = relationship(back_populates="events")
