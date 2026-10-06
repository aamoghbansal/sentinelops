"""Lightweight SQLite schema upgrades for development (no Alembic)."""

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine


def _has_column(engine: Engine, table: str, column: str) -> bool:
    return column in {c["name"] for c in inspect(engine).get_columns(table)}


def apply_sqlite_migrations(engine: Engine) -> None:
    if not str(engine.url).startswith("sqlite"):
        return
    with engine.begin() as conn:
        if not _has_column(engine, "projects", "model_id"):
            conn.execute(text("ALTER TABLE projects ADD COLUMN model_id VARCHAR(36)"))
        if not _has_column(engine, "projects", "local_model_path"):
            conn.execute(text("ALTER TABLE projects ADD COLUMN local_model_path TEXT"))
        if not _has_column(engine, "projects", "check_requested_at"):
            conn.execute(text("ALTER TABLE projects ADD COLUMN check_requested_at DATETIME"))
        if not _has_column(engine, "projects", "last_check_json"):
            conn.execute(text("ALTER TABLE projects ADD COLUMN last_check_json JSON"))
        if not _has_column(engine, "projects", "agent_connected_at"):
            conn.execute(text("ALTER TABLE projects ADD COLUMN agent_connected_at DATETIME"))
        if not _has_column(engine, "monitoring_windows", "performance_json"):
            conn.execute(text("ALTER TABLE monitoring_windows ADD COLUMN performance_json JSON"))
