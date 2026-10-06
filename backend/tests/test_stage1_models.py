import uuid

import joblib
import pytest
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from app.security import digest_token, mint_agent_token
from app.models import AgentCredential, Project
from app.db import SessionLocal


@pytest.fixture()
def model_file(tmp_path: Path) -> Path:
    model = LogisticRegression(max_iter=200)
    model.fit(load_iris(as_frame=True).data, load_iris().target)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    return path


def test_register_list_and_agent_check_via_api(model_file: Path):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        reg = client.post(
            "/models",
            json={
                "name": f"Stage1-{uuid.uuid4().hex[:8]}",
                "framework": "scikit-learn",
                "model_type": "classification",
                "local_model_path": str(model_file),
            },
        )
        assert reg.status_code == 201, reg.text
        model_id = reg.json()["model_id"]
        listed = client.get("/models")
        assert any(m["model_id"] == model_id for m in listed.json())

        with SessionLocal() as session:
            project = session.query(Project).filter_by(model_id=model_id).one()
            token = mint_agent_token()
            session.add(
                AgentCredential(
                    project_id=project.id,
                    token_digest=digest_token(token),
                    scopes=["model:check"],
                )
            )
            session.commit()

        client.post(f"/models/{model_id}/checks/trigger")
        work = client.get(
            f"/agent/models/{model_id}/work",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert work.status_code == 200
        assert work.json()["action"] == "model_check"

        posted = client.post(
            f"/agent/models/{model_id}/check",
            json={
                "model_id": model_id,
                "model_loaded": True,
                "framework": "scikit-learn",
                "model_type": "classification",
                "model_path": str(model_file),
                "model_file_exists": True,
                "framework_supported": True,
                "model_type_supported": True,
                "prediction_test_status": "passed",
                "status": "READY",
                "errors": [],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert posted.status_code == 200
        detail = client.get(f"/models/{model_id}").json()
        assert detail["status"] == "READY"


def test_invalid_agent_token(model_file: Path):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        reg = client.post(
            "/models",
            json={
                "name": f"Auth-{uuid.uuid4().hex[:8]}",
                "framework": "scikit-learn",
                "model_type": "classification",
                "local_model_path": str(model_file),
            },
        )
        model_id = reg.json()["model_id"]
        res = client.get(
            f"/agent/models/{model_id}/work",
            headers={"Authorization": "Bearer invalid"},
        )
        assert res.status_code == 401


def test_model_credentials_endpoint_and_status_preservation(model_file: Path):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        reg = client.post(
            "/models",
            json={
                "name": f"ModelCred-{uuid.uuid4().hex[:8]}",
                "framework": "scikit-learn",
                "model_type": "classification",
                "local_model_path": str(model_file),
            },
        )
        assert reg.status_code == 201
        model_id = reg.json()["model_id"]

        # Create credential directly via /models/{model_id}/credentials
        cred_res = client.post(f"/models/{model_id}/credentials", json={"label": "test-agent"})
        assert cred_res.status_code == 201
        token = cred_res.json()["token"]
        assert token.startswith("sops_")

        # Initial work check: project status transitions REGISTERED -> CONNECTED
        work = client.get(
            f"/agent/models/{model_id}/work",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert work.status_code == 200
        assert work.json()["action"] is None
        detail = client.get(f"/models/{model_id}").json()
        assert detail["status"] == "CONNECTED"
        assert detail["agent_connected"] is True

        # Trigger check
        client.post(f"/models/{model_id}/checks/trigger")
        work = client.get(
            f"/agent/models/{model_id}/work",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert work.json()["action"] == "model_check"

        # Submit successful check
        client.post(
            f"/agent/models/{model_id}/check",
            json={
                "model_id": model_id,
                "model_loaded": True,
                "framework": "scikit-learn",
                "model_type": "classification",
                "model_path": str(model_file),
                "model_file_exists": True,
                "framework_supported": True,
                "model_type_supported": True,
                "prediction_test_status": "passed",
                "status": "READY",
                "errors": [],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        detail = client.get(f"/models/{model_id}").json()
        assert detail["status"] == "READY"

        # Ensure subsequent agent polls do NOT reset status back to CONNECTED!
        work_after = client.get(
            f"/agent/models/{model_id}/work",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert work_after.status_code == 200
        detail_after = client.get(f"/models/{model_id}").json()
        assert detail_after["status"] == "READY"


def test_agent_check_error_reporting(model_file: Path):
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        reg = client.post(
            "/models",
            json={
                "name": f"ErrModel-{uuid.uuid4().hex[:8]}",
                "framework": "scikit-learn",
                "model_type": "classification",
                "local_model_path": str(model_file),
            },
        )
        model_id = reg.json()["model_id"]
        cred_res = client.post(f"/models/{model_id}/credentials")
        token = cred_res.json()["token"]

        # Submit error check
        client.post(
            f"/agent/models/{model_id}/check",
            json={
                "model_id": model_id,
                "model_loaded": False,
                "framework": "scikit-learn",
                "model_type": "classification",
                "model_path": "nonexistent.joblib",
                "model_file_exists": False,
                "framework_supported": True,
                "model_type_supported": True,
                "prediction_test_status": "not_run",
                "status": "ERROR",
                "errors": ["Model file not found at nonexistent.joblib."],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        detail = client.get(f"/models/{model_id}").json()
        assert detail["status"] == "ERROR"
        assert detail["last_check"]["model_file_exists"] is False
        assert "Model file not found" in detail["last_check"]["errors"][0]

