import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sklearn.datasets import load_iris
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Base, SessionLocal, engine, get_session
from app.migrate import apply_sqlite_migrations
from app.models import ActivityEvent, AgentCredential, MonitoringWindow, Project
from app.schemas import (
    AgentModelCheckPayload,
    AgentScanPayload,
    AgentWorkResponse,
    CredentialCreate,
    ModelCreate,
    ModelDetail,
    ModelSummary,
    MonitoringPayload,
    ProjectCreate,
)
from app.security import digest_token, mint_agent_token, token_matches
from app.services.drift import compute_drift, drift_detected
from app.services.quality import data_quality_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [backend] %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    apply_sqlite_migrations(engine)
    _backfill_model_ids()
    with SessionLocal() as session:
        seed_demo(session)
    yield


app = FastAPI(title="SentinelOps", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DbSession = Annotated[Session, Depends(get_session)]


def _backfill_model_ids() -> None:
    with SessionLocal() as session:
        rows = session.scalars(select(Project).where(Project.model_id.is_(None))).all()
        for row in rows:
            row.model_id = str(uuid.uuid4())
        if rows:
            session.commit()


def get_project(session: Session, project_id: int) -> Project:
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


def get_project_by_model_id(session: Session, model_id: str) -> Project:
    project = session.scalar(select(Project).where(Project.model_id == model_id))
    if not project:
        raise HTTPException(404, "Model not found")
    return project


def require_agent_for_model(
    model_id: str,
    session: DbSession,
    authorization: Annotated[str | None, Header()] = None,
) -> AgentCredential:
    project = get_project_by_model_id(session, model_id)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bearer credential required")
    token = authorization.removeprefix("Bearer ")
    credential = session.scalar(
        select(AgentCredential).where(
            AgentCredential.project_id == project.id,
            AgentCredential.token_digest == digest_token(token),
        )
    )
    if not credential or credential.revoked or not token_matches(token, credential.token_digest):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or revoked Agent credential")
    if "model:check" not in (credential.scopes or []):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Credential missing model:check scope")
    credential.last_used_at = datetime.now(timezone.utc)
    project.agent_connected_at = datetime.now(timezone.utc)
    if project.status == "REGISTERED":
        project.status = "CONNECTED"
    session.commit()
    return credential


def require_monitor_agent_for_model(
    model_id: str,
    session: DbSession,
    authorization: Annotated[str | None, Header()] = None,
) -> AgentCredential:
    """Authenticate a model Agent with the monitoring write scope."""
    project = get_project_by_model_id(session, model_id)
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bearer credential required")
    token = authorization.removeprefix("Bearer ")
    credential = session.scalar(select(AgentCredential).where(
        AgentCredential.project_id == project.id,
        AgentCredential.token_digest == digest_token(token),
    ))
    if not credential or credential.revoked or not token_matches(token, credential.token_digest):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or revoked Agent credential")
    if "monitor:write" not in (credential.scopes or []):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Credential missing monitor:write scope")
    now = datetime.now(timezone.utc)
    credential.last_used_at = now
    project.agent_connected_at = now
    session.commit()
    return credential


def require_agent(project_id: int, session: DbSession, authorization: Annotated[str | None, Header()] = None) -> AgentCredential:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bearer credential required")
    token = authorization.removeprefix("Bearer ")
    credential = session.scalar(
        select(AgentCredential).where(
            AgentCredential.project_id == project_id,
            AgentCredential.token_digest == digest_token(token),
        )
    )
    if not credential or credential.revoked or not token_matches(token, credential.token_digest):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or revoked Agent credential")
    credential.last_used_at = datetime.now(timezone.utc)
    session.commit()
    return credential


def model_summary(project: Project) -> ModelSummary:
    last = project.last_check_json or {}
    return ModelSummary(
        model_id=project.model_id,
        name=project.name,
        framework=project.framework,
        model_type=project.task_type,
        local_model_path=project.local_model_path,
        status=project.status,
        registered_at=project.created_at,
        agent_connected=project.agent_connected_at is not None,
        last_check_status=last.get("status") if last else None,
    )


def model_detail(project: Project) -> ModelDetail:
    summary = model_summary(project)
    active_credential = next((c for c in project.credentials if not c.revoked), None)
    latest = max(project.monitoring_windows, key=lambda window: window.id, default=None)
    return ModelDetail(
        **summary.model_dump(),
        last_check=project.last_check_json,
        check_pending=project.check_requested_at is not None,
        metadata=project.metadata_json,
        credential_configured=active_credential is not None,
        latest_monitoring=None if latest is None else {
            "id": latest.id,
            "source": latest.source,
            "last_run": latest.created_at,
            "health": latest.severity.upper(),
            "overall_drift_status": latest.drift_json.get("overall_status", "UNKNOWN"),
            "data_quality": latest.data_quality_json,
            "drift": latest.drift_json,
            "performance": latest.performance_json,
        },
    )


def monitoring_window_payload(window: MonitoringWindow) -> dict:
    """Serialize one stored monitoring run for model details and the monitoring UI."""
    quality = window.data_quality_json or {}
    drift = window.drift_json or {}
    performance = window.performance_json or {}
    return {
        "id": window.id,
        "source": window.source,
        "last_run": window.created_at,
        "health": window.severity.upper(),
        "overall_drift_status": drift.get("overall_status", "UNKNOWN"),
        "data_quality": quality,
        "drift": drift,
        "performance": performance,
    }


def latest_monitoring_window(session: Session, project_id: int) -> MonitoringWindow | None:
    return session.scalar(
        select(MonitoringWindow)
        .where(MonitoringWindow.project_id == project_id)
        .order_by(MonitoringWindow.id.desc())
    )


def seed_demo(session: Session) -> None:
    existing = session.scalar(select(Project).where(Project.name == "Iris classification demo"))
    if existing:
        if not existing.local_model_path:
            existing.local_model_path = "../demo_project/model.joblib"
        if existing.status == "connected":
            existing.status = "REGISTERED"
        session.commit()
        return
    project = Project(
        model_id=str(uuid.uuid4()),
        name="Iris classification demo",
        framework="scikit-learn",
        task_type="classification",
        local_model_path="../demo_project/model.joblib",
        status="REGISTERED",
        metadata_json={"dataset": "scikit-learn Iris", "stage": "foundation"},
    )
    session.add(project)
    session.flush()
    session.add(ActivityEvent(project_id=project.id, kind="model.seeded", message="Iris demo model registered."))
    session.commit()


@app.get("/health")
def health():
    return {"status": "ok", "scope": "stage-1 model registration + agent model check"}


# --- Stage 1 model APIs ---


@app.get("/models", response_model=list[ModelSummary])
def list_models(session: DbSession):
    rows = session.scalars(select(Project).order_by(Project.id)).all()
    return [model_summary(p) for p in rows]


@app.post("/models", response_model=ModelDetail, status_code=201)
def register_model(payload: ModelCreate, session: DbSession):
    logger.info("Registering model name=%s framework=%s", payload.name, payload.framework)
    if payload.framework != "scikit-learn":
        raise HTTPException(400, "Stage 1 supports scikit-learn only")
    if payload.model_type != "classification":
        raise HTTPException(400, "Stage 1 supports tabular classification only")
    project = Project(
        model_id=str(uuid.uuid4()),
        name=payload.name,
        framework=payload.framework,
        task_type=payload.model_type,
        local_model_path=payload.local_model_path,
        status="REGISTERED",
        metadata_json={"stage": "foundation"},
    )
    session.add(project)
    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        raise HTTPException(409, "Model name already exists") from exc
    session.refresh(project)
    session.add(
        ActivityEvent(
            project_id=project.id,
            kind="model.registered",
            message=f"Registered model: {project.name} ({project.model_id}).",
        )
    )
    session.commit()
    return model_detail(project)


@app.get("/models/{model_id}", response_model=ModelDetail)
def get_model(model_id: str, session: DbSession):
    return model_detail(get_project_by_model_id(session, model_id))


@app.get("/monitoring")
def list_monitoring(session: DbSession):
    """Latest persisted monitoring result per registered model; no raw datasets exposed."""
    projects = session.scalars(select(Project).order_by(Project.id)).all()
    rows = []
    for project in projects:
        window = latest_monitoring_window(session, project.id)
        rows.append({
            "model_id": project.model_id,
            "name": project.name,
            "framework": project.framework,
            "current_version": project.metadata_json.get("current_version", "—"),
            "agent_connected": project.agent_connected_at is not None,
            "monitoring": monitoring_window_payload(window) if window else None,
        })
    return rows


@app.get("/models/{model_id}/monitoring")
def get_latest_model_monitoring(model_id: str, session: DbSession):
    project = get_project_by_model_id(session, model_id)
    window = latest_monitoring_window(session, project.id)
    return {"model_id": model_id, "monitoring": monitoring_window_payload(window) if window else None}


@app.get("/models/{model_id}/monitoring/history")
def get_model_monitoring_history(model_id: str, session: DbSession):
    project = get_project_by_model_id(session, model_id)
    windows = session.scalars(
        select(MonitoringWindow)
        .where(MonitoringWindow.project_id == project.id)
        .order_by(MonitoringWindow.id.desc())
    ).all()
    return {"model_id": model_id, "runs": [monitoring_window_payload(window) for window in windows]}


@app.post("/models/{model_id}/credentials", status_code=201)
def create_model_credential(
    model_id: str,
    payload: CredentialCreate | None = None,
    session: DbSession = None,
):
    project = get_project_by_model_id(session, model_id)
    cred_payload = payload or CredentialCreate()
    existing = session.scalar(select(AgentCredential).where(
        AgentCredential.project_id == project.id,
        AgentCredential.revoked.is_(False),
    ).order_by(AgentCredential.id.desc()))
    if existing:
        # Secrets are intentionally one-time display values; reopening a model never rotates one.
        return {
            "token": None,
            "credential_configured": True,
            "created": False,
            "scopes": existing.scopes,
            "model_id": project.model_id,
        }
    token = mint_agent_token()
    credential = AgentCredential(
        project_id=project.id,
        token_digest=digest_token(token),
        label=cred_payload.label,
        scopes=cred_payload.scopes,
    )
    session.add(credential)
    session.add(
        ActivityEvent(
            project_id=project.id,
            kind="agent.credential_created",
            message=f"Created scoped Agent credential for model {project.name}.",
        )
    )
    session.commit()
    logger.info("Agent credential created for model_id=%s", model_id)
    return {"token": token, "credential_configured": True, "created": True, "scopes": credential.scopes, "model_id": project.model_id}


def _revoke_active_credentials(session: Session, project: Project) -> int:
    credentials = session.scalars(select(AgentCredential).where(
        AgentCredential.project_id == project.id,
        AgentCredential.revoked.is_(False),
    )).all()
    for credential in credentials:
        credential.revoked = True
    return len(credentials)


@app.post("/models/{model_id}/credentials/rotate", status_code=201)
def rotate_model_credential(
    model_id: str,
    payload: CredentialCreate | None = None,
    session: DbSession = None,
):
    """Explicitly replace the one active Agent credential and show the replacement once."""
    project = get_project_by_model_id(session, model_id)
    revoked_count = _revoke_active_credentials(session, project)
    credential_payload = payload or CredentialCreate()
    token = mint_agent_token()
    credential = AgentCredential(
        project_id=project.id,
        token_digest=digest_token(token),
        label=credential_payload.label,
        scopes=credential_payload.scopes,
    )
    session.add(credential)
    session.add(ActivityEvent(
        project_id=project.id,
        kind="agent.credential_rotated",
        message="Agent credential rotated; previous credential revoked.",
        details_json={"revoked_count": revoked_count},
    ))
    session.commit()
    return {"token": token, "credential_configured": True, "scopes": credential.scopes, "model_id": project.model_id}


@app.post("/models/{model_id}/credentials/disconnect")
def disconnect_model_agent(model_id: str, session: DbSession):
    """Explicitly disconnect the Agent without deleting the registered model or history."""
    project = get_project_by_model_id(session, model_id)
    revoked_count = _revoke_active_credentials(session, project)
    cancelled_check = project.check_requested_at is not None
    project.check_requested_at = None
    project.agent_connected_at = None
    if project.status == "CONNECTED":
        project.status = "REGISTERED"
    session.add(ActivityEvent(
        project_id=project.id,
        kind="agent.disconnected",
        message="Agent disconnected; active credential revoked and pending check cancelled." if cancelled_check else "Agent disconnected; active credential revoked.",
        details_json={"revoked_count": revoked_count, "cancelled_pending_check": cancelled_check},
    ))
    session.commit()
    return {"model_id": model_id, "disconnected": True, "revoked_count": revoked_count, "cancelled_pending_check": cancelled_check}


@app.post("/models/{model_id}/checks/trigger", response_model=ModelDetail)
def trigger_model_check(model_id: str, session: DbSession):
    """Flag a pending check. The local Agent picks it up via /agent/models/{id}/work."""
    project = get_project_by_model_id(session, model_id)
    logger.info("Model check requested for model_id=%s (waiting for local Agent)", model_id)
    project.check_requested_at = datetime.now(timezone.utc)
    project.status = "CONNECTED" if project.agent_connected_at else project.status
    session.add(
        ActivityEvent(
            project_id=project.id,
            kind="model.check_requested",
            message="UI requested a local Agent model check.",
        )
    )
    session.commit()
    session.refresh(project)
    return model_detail(project)


@app.get("/agent/models/{model_id}")
def agent_model_config(model_id: str, session: DbSession, _: AgentCredential = Depends(require_agent_for_model)):
    project = get_project_by_model_id(session, model_id)
    return {
        "model_id": project.model_id,
        "local_model_path": project.local_model_path,
        "framework": project.framework,
        "model_type": project.task_type,
    }


@app.get("/agent/models/{model_id}/work", response_model=AgentWorkResponse)
def agent_work(
    model_id: str,
    response: Response,
    session: DbSession,
    _: AgentCredential = Depends(require_agent_for_model),
):
    # This endpoint must always reflect the current pending work state.
    # Vercel/CDN must never cache an old "action: null" response.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["CDN-Cache-Control"] = "no-store"
    response.headers["Vercel-CDN-Cache-Control"] = "no-store"
    project = get_project_by_model_id(session, model_id)
    if project.check_requested_at:
        return AgentWorkResponse(
            action="model_check",
            model_id=project.model_id,
            local_model_path=project.local_model_path,
            framework=project.framework,
            model_type=project.task_type,
        )
    return AgentWorkResponse(action=None)


@app.post("/agent/models/{model_id}/check")
def agent_model_check(
    model_id: str,
    payload: AgentModelCheckPayload,
    session: DbSession,
    _: AgentCredential = Depends(require_agent_for_model),
):
    project = get_project_by_model_id(session, model_id)
    if payload.model_id != model_id:
        raise HTTPException(400, "model_id mismatch")
    logger.info("Agent model check received model_id=%s status=%s", model_id, payload.status)
    check = payload.model_dump()
    check["received_at"] = datetime.now(timezone.utc).isoformat()
    project.last_check_json = check
    project.check_requested_at = None
    project.status = payload.status if payload.status in {"READY", "ERROR"} else "ERROR"
    session.add(
        ActivityEvent(
            project_id=project.id,
            kind="model.check_completed",
            message=f"Agent model check completed: {project.status}.",
            details_json=check,
        )
    )
    session.commit()
    return {"model_id": model_id, "status": project.status, "check": check}


@app.post("/agent/models/{model_id}/monitoring")
def agent_model_monitoring(
    model_id: str,
    payload: MonitoringPayload,
    session: DbSession,
    _: AgentCredential = Depends(require_monitor_agent_for_model),
):
    """Store results calculated locally by the Agent; raw datasets never leave it."""
    project = get_project_by_model_id(session, model_id)
    detected = payload.overall_drift_status.upper() == "DRIFTED"
    project.drift_status = "detected" if detected else "clear"
    drift = {**payload.drift, "overall_status": payload.overall_drift_status.upper()}
    health = payload.health.upper()
    if health not in {"HEALTHY", "WARNING", "CRITICAL"}:
        raise HTTPException(400, "Invalid monitoring health status")
    window = MonitoringWindow(
        project_id=project.id,
        source="agent",
        data_quality_json=payload.data_quality,
        drift_json=drift,
        performance_json=payload.performance,
        severity=health,
    )
    session.add(window)
    session.add(ActivityEvent(
        project_id=project.id,
        kind="monitoring.completed",
        message=f"Agent monitoring completed: {health} (drift: {payload.overall_drift_status.upper()}).",
        details_json={"health": health, "drift_status": payload.overall_drift_status.upper()},
    ))
    session.commit()
    return {"id": window.id, "health": health, "overall_drift_status": payload.overall_drift_status.upper()}


# --- Legacy project APIs (preserved) ---


@app.get("/projects")
def list_projects(session: DbSession):
    rows = session.scalars(select(Project).order_by(Project.id)).all()
    return [
        {
            "id": p.id,
            "model_id": p.model_id,
            "name": p.name,
            "status": p.status,
            "framework": p.framework,
            "drift_status": p.drift_status,
        }
        for p in rows
    ]


@app.post("/projects", status_code=201)
def create_project(payload: ProjectCreate, session: DbSession):
    project = Project(
        model_id=str(uuid.uuid4()),
        name=payload.name,
        task_type=payload.task_type,
        status="REGISTERED",
        metadata_json={"stage": "foundation"},
    )
    session.add(project)
    try:
        session.commit()
    except Exception as exc:
        session.rollback()
        raise HTTPException(409, "Project name already exists") from exc
    session.refresh(project)
    session.add(ActivityEvent(project_id=project.id, kind="project.registered", message=f"Registered model project: {project.name}."))
    session.commit()
    return {"id": project.id, "model_id": project.model_id, "name": project.name}


@app.get("/projects/{project_id}/dashboard")
def dashboard(project_id: int, session: DbSession):
    project = get_project(session, project_id)
    window = session.scalar(select(MonitoringWindow).where(MonitoringWindow.project_id == project_id).order_by(MonitoringWindow.id.desc()))
    events = session.scalars(select(ActivityEvent).where(ActivityEvent.project_id == project_id).order_by(ActivityEvent.id.desc()).limit(15)).all()
    return {
        "project": {
            "id": project.id,
            "model_id": project.model_id,
            "name": project.name,
            "framework": project.framework,
            "task_type": project.task_type,
            "status": project.status,
            "local_model_path": project.local_model_path,
            "drift_status": project.drift_status,
            "metadata": project.metadata_json,
            "last_check": project.last_check_json,
            "check_pending": project.check_requested_at is not None,
            "agent_connected": project.agent_connected_at is not None,
        },
        "agent_scan": project.metadata_json.get("agent_scan"),
        "latest_window": None
        if not window
        else {
            "id": window.id,
            "severity": window.severity,
            "quality": window.data_quality_json,
            "drift": window.drift_json,
            "source": window.source,
        },
        "events": [{"kind": e.kind, "message": e.message, "created_at": e.created_at} for e in events],
    }


@app.post("/projects/{project_id}/credentials", status_code=201)
def create_credential(project_id: int, payload: CredentialCreate, session: DbSession):
    get_project(session, project_id)
    token = mint_agent_token()
    credential = AgentCredential(project_id=project_id, token_digest=digest_token(token), label=payload.label, scopes=payload.scopes)
    session.add(credential)
    session.add(ActivityEvent(project_id=project_id, kind="agent.credential_created", message="Created scoped Agent credential."))
    session.commit()
    logger.info("Agent credential created for project_id=%s", project_id)
    return {"token": token, "scopes": credential.scopes, "model_id": get_project(session, project_id).model_id}


@app.post("/agent/projects/{project_id}/scan")
def agent_scan(project_id: int, payload: AgentScanPayload, session: DbSession, _: AgentCredential = Depends(require_agent)):
    project = get_project(session, project_id)
    project.metadata_json = {**project.metadata_json, "agent_scan": payload.model_dump()}
    session.add(ActivityEvent(project_id=project_id, kind="agent.scan", message="Agent scanned the local project and uploaded findings.", details_json=payload.findings))
    session.commit()
    return {"status": "recorded", "findings": payload.findings}


@app.post("/agent/projects/{project_id}/monitoring")
def agent_monitoring(project_id: int, payload: MonitoringPayload, session: DbSession, _: AgentCredential = Depends(require_agent)):
    project = get_project(session, project_id)
    detected = drift_detected(payload.drift)
    severity = "critical" if detected else "healthy"
    project.drift_status = "detected" if detected else "clear"
    window = MonitoringWindow(project_id=project_id, source="agent", data_quality_json=payload.data_quality, drift_json=payload.drift, severity=severity)
    session.add(window)
    session.add(ActivityEvent(project_id=project_id, kind="monitoring.completed", message=f"Agent monitoring completed: {severity}.", details_json={"drift_detected": detected}))
    session.commit()
    return {"id": window.id, "severity": severity, "drift_detected": detected}


@app.post("/projects/{project_id}/monitoring/demo")
def demo_monitoring(project_id: int, session: DbSession):
    """Convenience endpoint for presentation if the Agent CLI is not used."""
    get_project(session, project_id)
    data = load_iris(as_frame=True).data
    reference = data.iloc[:75].copy()
    current = data.iloc[75:].copy()
    current["sepal length (cm)"] = current["sepal length (cm)"] + 1.7
    payload = {"data_quality": data_quality_report(current), "drift": compute_drift(reference, current)}
    project = get_project(session, project_id)
    detected = drift_detected(payload["drift"])
    project.drift_status = "detected" if detected else "clear"
    window = MonitoringWindow(project_id=project_id, source="demo", data_quality_json=payload["data_quality"], drift_json=payload["drift"], severity="critical" if detected else "healthy")
    session.add(window)
    session.add(ActivityEvent(project_id=project_id, kind="demo.monitoring", message="Demo monitoring window computed."))
    session.commit()
    return {"id": window.id, "severity": window.severity, "drift_detected": detected}
