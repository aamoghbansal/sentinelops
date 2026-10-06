import subprocess
import time
import requests
import uuid
from pathlib import Path

def test_full_agent_cli_flow(tmp_path: Path):
    from fastapi.testclient import TestClient
    from app.main import app
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression

    # 1. Train real model and save to disk
    model = LogisticRegression(max_iter=200)
    data = load_iris(as_frame=True)
    model.fit(data.data, data.target)
    model_file = tmp_path / "model.joblib"
    joblib.dump(model, model_file)

    sample_csv = tmp_path / "sample.csv"
    data.data.iloc[:1].to_csv(sample_csv, index=False)

    with TestClient(app) as client:
        # 2. Register model in backend
        model_name = f"E2E-Model-{uuid.uuid4().hex[:6]}"
        reg_res = client.post(
            "/models",
            json={
                "name": model_name,
                "framework": "scikit-learn",
                "model_type": "classification",
                "local_model_path": str(model_file),
            },
        )
        assert reg_res.status_code == 201, reg_res.text
        model_id = reg_res.json()["model_id"]
        assert reg_res.json()["status"] == "REGISTERED"

        # 3. Create scoped Agent credential for this model
        cred_res = client.post(f"/models/{model_id}/credentials", json={"label": "e2e-agent"})
        assert cred_res.status_code == 201, cred_res.text
        token = cred_res.json()["token"]
        assert token.startswith("sops_")

        # 4. Agent queries model configuration
        conf_res = client.get(
            f"/agent/models/{model_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert conf_res.status_code == 200
        assert conf_res.json()["model_id"] == model_id
        assert conf_res.json()["framework"] == "scikit-learn"

        # 5. Model check execution directly via agent model_check logic
        from sentinelops_agent.model_check import run_model_check
        check_res = run_model_check(
            model_id=model_id,
            model_path=str(model_file),
            framework="scikit-learn",
            model_type="classification",
            sample_features_path=str(sample_csv),
        )
        assert check_res["status"] == "READY"
        assert check_res["model_loaded"] is True
        assert check_res["prediction_test_status"] == "passed"
        assert check_res["sample_prediction"] is not None
        assert check_res["expected_features"] == 4

        # 6. Submit check result to backend
        sub_res = client.post(
            f"/agent/models/{model_id}/check",
            json=check_res,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert sub_res.status_code == 200

        # 7. Check backend status and details
        detail_res = client.get(f"/models/{model_id}")
        assert detail_res.status_code == 200
        detail = detail_res.json()
        assert detail["status"] == "READY"
        assert detail["agent_connected"] is True
        assert detail["last_check"]["prediction_test_status"] == "passed"
        assert detail["last_check"]["expected_features"] == 4
        assert detail["last_check"]["sample_prediction"] == check_res["sample_prediction"]

        # 8. Trigger another check from UI
        trig_res = client.post(f"/models/{model_id}/checks/trigger")
        assert trig_res.status_code == 200
        assert trig_res.json()["check_pending"] is True

        # 9. Verify agent work polling picks up pending check
        work_res = client.get(
            f"/agent/models/{model_id}/work",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert work_res.status_code == 200
        assert work_res.json()["action"] == "model_check"
