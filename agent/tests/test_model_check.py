from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from sentinelops_agent.model_check import run_model_check


@pytest.fixture()
def iris_model(tmp_path: Path):
    model = LogisticRegression(max_iter=200, random_state=1)
    data = load_iris(as_frame=True)
    model.fit(data.data, data.target)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    data.data.iloc[:1].to_csv(tmp_path / "sample.csv", index=False)
    return path


def test_model_check_success(iris_model: Path):
    result = run_model_check(
        model_id="test-id",
        model_path=str(iris_model),
        framework="scikit-learn",
        model_type="classification",
        sample_features_path=str(iris_model.parent / "sample.csv"),
    )
    assert result["model_loaded"] is True
    assert result["prediction_test_status"] == "passed"
    assert result["status"] == "READY"


def test_model_check_missing_file(tmp_path: Path):
    result = run_model_check(
        model_id="x",
        model_path=str(tmp_path / "missing.joblib"),
        framework="scikit-learn",
        model_type="classification",
    )
    assert result["status"] == "ERROR"
    assert result["model_file_exists"] is False


def test_model_check_invalid_framework(iris_model: Path):
    result = run_model_check(
        model_id="x",
        model_path=str(iris_model),
        framework="pytorch",
        model_type="classification",
    )
    assert result["framework_supported"] is False
    assert result["status"] == "ERROR"


def test_model_check_string_labels(tmp_path: Path):
    model = LogisticRegression(max_iter=200, random_state=1)
    data = load_iris(as_frame=True)
    # Target as string labels
    target_names = [data.target_names[i] for i in data.target]
    model.fit(data.data, target_names)
    path = tmp_path / "string_model.joblib"
    joblib.dump(model, path)
    data.data.iloc[:1].to_csv(tmp_path / "sample.csv", index=False)

    result = run_model_check(
        model_id="string-label-model",
        model_path=str(path),
        framework="scikit-learn",
        model_type="classification",
        sample_features_path=str(tmp_path / "sample.csv"),
    )
    assert result["status"] == "READY"
    assert result["prediction_test_status"] == "passed"
    assert isinstance(result["sample_prediction"], str)


def test_model_check_invalid_object(tmp_path: Path):
    # Save a non-model object into a joblib file
    bad_path = tmp_path / "bad_object.joblib"
    joblib.dump({"key": "not a model"}, bad_path)

    result = run_model_check(
        model_id="bad-obj",
        model_path=str(bad_path),
        framework="scikit-learn",
        model_type="classification",
    )
    assert result["status"] == "ERROR"
    assert result["model_loaded"] is True
    assert any("not a valid scikit-learn predictor" in e for e in result["errors"])



def test_model_check_bad_sample_file_reports_error(tmp_path):
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sentinelops_agent.model_check import run_model_check

    data = load_iris(as_frame=True)
    model = LogisticRegression(max_iter=200).fit(data.data, data.target)
    model_file = tmp_path / "model.joblib"
    joblib.dump(model, model_file)
    sample = tmp_path / "sample.csv"
    sample.write_text("a,b,c,d\nx,y,z,w\n", encoding="utf-8")

    result = run_model_check(
        model_id="m",
        model_path=str(model_file),
        framework="scikit-learn",
        model_type="classification",
        sample_features_path=str(sample),
    )
    assert result["status"] == "ERROR"
    assert result["model_loaded"] is True
    assert any("sample" in e.lower() for e in result["errors"])
