import uuid

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from sentinelops_agent.monitoring import monitoring_report


def test_credential_is_idempotent_and_monitoring_is_stored():
    reference = pd.DataFrame({"x": range(100), "y": range(100), "target": [0] * 50 + [1] * 50})
    current = pd.DataFrame({"x": [value + 25 for value in range(100)], "y": range(100), "target": [0] * 50 + [1] * 50})
    report = monitoring_report(reference, current, label_column="target")
    assert report["overall_drift_status"] == "DRIFTED"
    assert report["drift"]["features"]["x"]["status"] == "DRIFTED"
    assert report["performance"]["status"] == "UNKNOWN"

    with TestClient(app) as client:
        registration = client.post("/models", json={
            "name": f"Stage2-{uuid.uuid4().hex[:8]}", "framework": "scikit-learn",
            "model_type": "classification", "local_model_path": "local-model.joblib",
        })
        assert registration.status_code == 201
        model_id = registration.json()["model_id"]

        first = client.post(f"/models/{model_id}/credentials", json={"label": "stage2-agent"})
        assert first.status_code == 201
        token = first.json()["token"]
        assert token
        second = client.post(f"/models/{model_id}/credentials", json={"label": "stage2-agent"})
        assert second.status_code == 201
        assert second.json()["created"] is False
        assert second.json()["token"] is None

        rotated = client.post(f"/models/{model_id}/credentials/rotate", json={"label": "stage2-agent"})
        assert rotated.status_code == 201
        replacement_token = rotated.json()["token"]
        assert replacement_token and replacement_token != token
        old_credential = client.get(f"/agent/models/{model_id}", headers={"Authorization": f"Bearer {token}"})
        assert old_credential.status_code == 401

        stored = client.post(
            f"/agent/models/{model_id}/monitoring", json=report,
            headers={"Authorization": f"Bearer {replacement_token}"},
        )
        assert stored.status_code == 200, stored.text
        assert stored.json()["overall_drift_status"] == "DRIFTED"

        detail = client.get(f"/models/{model_id}").json()
        assert detail["credential_configured"] is True
        assert detail["latest_monitoring"]["health"] == "WARNING"
        assert detail["latest_monitoring"]["drift"]["features"]["x"]["psi"] > 0.2

        monitoring_list = client.get("/monitoring")
        assert monitoring_list.status_code == 200
        listed = next(row for row in monitoring_list.json() if row["model_id"] == model_id)
        assert listed["monitoring"]["health"] == "WARNING"
        history = client.get(f"/models/{model_id}/monitoring/history")
        assert history.status_code == 200
        assert len(history.json()["runs"]) == 1

        pending = client.post(f"/models/{model_id}/checks/trigger")
        assert pending.json()["check_pending"] is True
        disconnected = client.post(f"/models/{model_id}/credentials/disconnect")
        assert disconnected.status_code == 200
        assert disconnected.json()["cancelled_pending_check"] is True
        assert client.get(f"/models/{model_id}").json()["check_pending"] is False
        assert client.get(f"/agent/models/{model_id}", headers={"Authorization": f"Bearer {replacement_token}"}).status_code == 401
