from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

class ProjectCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    task_type: str = "classification"

class ModelCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    framework: str = Field(default="scikit-learn", max_length=50)
    model_type: str = Field(default="classification", max_length=30)
    local_model_path: str = Field(min_length=1)

class ModelSummary(BaseModel):
    model_id: str
    name: str
    framework: str
    model_type: str
    local_model_path: str | None
    status: str
    registered_at: datetime
    agent_connected: bool
    last_check_status: str | None

class ModelDetail(ModelSummary):
    last_check: dict[str, Any] | None
    check_pending: bool
    check_requested_at: datetime | None
    metadata: dict[str, Any]
    credential_configured: bool = False
    latest_monitoring: dict[str, Any] | None = None

class CredentialCreate(BaseModel):
    label: str = "local-agent"
    scopes: list[str] = ["model:check", "project:scan", "monitor:write"]

class AgentScanPayload(BaseModel):
    project_root: str
    findings: dict[str, Any]

class MonitoringPayload(BaseModel):
    data_quality: dict[str, Any]
    drift: dict[str, Any]
    performance: dict[str, Any] = Field(default_factory=dict)
    overall_drift_status: str = "UNKNOWN"
    health: str = "UNKNOWN"

class AgentModelCheckPayload(BaseModel):
    model_id: str
    model_loaded: bool
    framework: str
    model_type: str
    model_path: str
    model_file_exists: bool
    framework_supported: bool
    model_type_supported: bool
    prediction_test_status: str
    expected_features: int | None = None
    sample_prediction: Any | None = None
    status: str
    errors: list[str] = Field(default_factory=list)

class AgentWorkResponse(BaseModel):
    action: str | None = None
    model_id: str | None = None
    local_model_path: str | None = None
    framework: str | None = None
    model_type: str | None = None
