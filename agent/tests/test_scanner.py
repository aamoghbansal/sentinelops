from pathlib import Path
from sentinelops_agent.scanner import scan_project

def test_scan_detects_demo_project(tmp_path: Path):
    (tmp_path / "train.py").write_text("from sklearn.linear_model import LogisticRegression\n", encoding="utf-8")
    (tmp_path / "model.joblib").write_text("x", encoding="utf-8")
    (tmp_path / "data.csv").write_text("x\n1\n", encoding="utf-8")
    result = scan_project(str(tmp_path))
    assert "model.joblib" in result["model_files"]
    assert result["detected_framework"] == "scikit-learn"
