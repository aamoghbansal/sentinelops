"""Local scikit-learn tabular classification model validation (Stage 1)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_FRAMEWORK = "scikit-learn"
SUPPORTED_MODEL_TYPE = "classification"


def _resolve_model_path(model_path: str) -> Path:
    raw = Path(model_path).expanduser()
    if raw.is_file():
        return raw.resolve()
    cwd = Path.cwd()
    if (cwd / raw).is_file():
        return (cwd / raw).resolve()
    cleaned = model_path.lstrip("./\\").replace("../", "").replace("..\\", "")
    if (cwd / cleaned).is_file():
        return (cwd / cleaned).resolve()
    if (cwd.parent / raw).is_file():
        return (cwd.parent / raw).resolve()
    if (cwd.parent / cleaned).is_file():
        return (cwd.parent / cleaned).resolve()
    return raw.resolve()


def _load_sample_features(model_path: Path, sample_path: Path | None) -> list[float] | None:
    if sample_path and sample_path.is_file():
        if sample_path.suffix.lower() == ".json":
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "features" in payload:
                return list(payload["features"])
            if isinstance(payload, list):
                return list(payload)
        if sample_path.suffix.lower() == ".csv":
            row = pd.read_csv(sample_path).iloc[0]
            return row.astype(float).tolist()
    demo_data = model_path.parent / "data" / "reference.csv"
    if demo_data.is_file():
        row = pd.read_csv(demo_data).iloc[0]
        return row.astype(float).tolist()
    return None


def run_model_check(
    *,
    model_id: str,
    model_path: str,
    framework: str,
    model_type: str,
    sample_features_path: str | None = None,
) -> dict[str, Any]:
    path = _resolve_model_path(model_path)
    result: dict[str, Any] = {
        "model_id": model_id,
        "framework": framework,
        "model_type": model_type,
        "model_path": str(path),
        "model_file_exists": path.is_file(),
        "model_loaded": False,
        "framework_supported": framework == SUPPORTED_FRAMEWORK,
        "model_type_supported": model_type == SUPPORTED_MODEL_TYPE,
        "prediction_test_status": "not_run",
        "expected_features": None,
        "sample_prediction": None,
        "status": "ERROR",
        "errors": [],
    }

    if not result["framework_supported"]:
        result["errors"].append(f"Unsupported framework '{framework}' (Stage 1: scikit-learn only).")
        return result
    if not result["model_type_supported"]:
        result["errors"].append(f"Unsupported model type '{model_type}' (Stage 1: classification only).")
        return result
    if not result["model_file_exists"]:
        result["errors"].append(f"Model file not found at {path}.")
        return result

    try:
        model = joblib.load(path)
        result["model_loaded"] = True
    except Exception as exc:  # noqa: BLE001 — return structured agent error
        logger.exception("Failed to load model from %s", path)
        result["errors"].append(f"Model load failed: {exc}")
        return result

    if not hasattr(model, "predict") or not callable(getattr(model, "predict")):
        result["errors"].append(
            "Loaded object is not a valid scikit-learn predictor (missing callable 'predict' method)."
        )
        return result

    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None:
        result["expected_features"] = int(n_features)

    sample_path = Path(sample_features_path).expanduser().resolve() if sample_features_path else None
    try:
        features = _load_sample_features(path, sample_path)
    except Exception as exc:  # noqa: BLE001 — bad sample file must not crash the Agent
        result["errors"].append(f"Could not read sample features: {exc}")
        return result
    if features is None:
        if n_features is not None:
            features = [0.0] * int(n_features)
        else:
            result["errors"].append("No sample features available for a test prediction.")
            return result

    if n_features is not None and len(features) != int(n_features):
        result["errors"].append(
            f"Sample has {len(features)} features but model expects {n_features}."
        )
        return result

    try:
        prediction = model.predict([features])
        pred_val = prediction[0]
        if hasattr(pred_val, "item"):
            pred_val = pred_val.item()
        result["sample_prediction"] = pred_val
        result["prediction_test_status"] = "passed"
        result["status"] = "READY"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Test prediction failed")
        result["prediction_test_status"] = "failed"
        result["errors"].append(f"Test prediction failed: {exc}")

    return result

