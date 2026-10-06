"""Local deterministic monitoring. Only aggregate metrics leave the machine."""
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

EPSILON = 1e-6
PSI_THRESHOLD = 0.20
KS_P_VALUE_THRESHOLD = 0.05

def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)

def _distribution(reference: np.ndarray, current: np.ndarray, bins: int, combined: bool = False):
    source = np.concatenate([reference, current]) if combined else reference
    edges = np.unique(np.quantile(source, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, edges)
    cur_counts, _ = np.histogram(current, edges)
    rp = (ref_counts + EPSILON) / (ref_counts.sum() + len(ref_counts) * EPSILON)
    cp = (cur_counts + EPSILON) / (cur_counts.sum() + len(cur_counts) * EPSILON)
    return rp, cp

def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    ref, cur = _numeric(reference), _numeric(current)
    if len(ref) < 2 or len(cur) < 2: return None
    distribution = _distribution(ref, cur, bins)
    if distribution is None: return 0.0
    rp, cp = distribution
    return float(np.sum((cp - rp) * np.log(cp / rp)))

def kl_divergence(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    """KL(current || reference), smoothed so zero-probability bins are safe."""
    ref, cur = _numeric(reference), _numeric(current)
    if len(ref) < 2 or len(cur) < 2: return None
    distribution = _distribution(ref, cur, bins, combined=True)
    if distribution is None: return 0.0
    rp, cp = distribution
    return float(np.sum(cp * np.log(cp / rp)))

def ks_test(reference: pd.Series, current: pd.Series) -> dict[str, float] | None:
    ref, cur = _numeric(reference), _numeric(current)
    if len(ref) < 2 or len(cur) < 2: return None
    result = ks_2samp(ref, cur)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}

def data_quality(current: pd.DataFrame, reference: pd.DataFrame | None = None) -> dict[str, Any]:
    outliers: dict[str, dict[str, float | int]] = {}
    for column in current.select_dtypes(include="number").columns:
        values = current[column].dropna()
        if len(values) < 4:
            outliers[column] = {"count": 0, "rate": 0.0}; continue
        q1, q3 = values.quantile(.25), values.quantile(.75)
        iqr = q3 - q1
        mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        outliers[column] = {"count": int(mask.sum()), "rate": float(mask.mean())}
    schema: dict[str, Any] = {"missing_columns": [], "added_columns": [], "type_mismatches": []}
    if reference is not None:
        schema["missing_columns"] = sorted(set(reference.columns) - set(current.columns))
        schema["added_columns"] = sorted(set(current.columns) - set(reference.columns))
        schema["type_mismatches"] = sorted(c for c in set(reference.columns) & set(current.columns) if pd.api.types.is_numeric_dtype(reference[c]) != pd.api.types.is_numeric_dtype(current[c]))
    issue = bool(current.isna().any().any() or current.duplicated().any() or any(x["count"] for x in outliers.values()) or any(schema.values()))
    return {"status": "WARNING" if issue else "CLEAR", "rows": len(current), "duplicate_rows": int(current.duplicated().sum()), "null_rates": {c: float(v) for c, v in current.isna().mean().items()}, "iqr_outliers": outliers, "columns": list(current.columns), "schema": schema}

def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    features: dict[str, Any] = {}
    for column in reference.columns:
        if column not in current.columns:
            features[column] = {"psi": None, "ks": None, "kl_divergence": None, "status": "UNKNOWN"}
            continue
        if not pd.api.types.is_numeric_dtype(reference[column]) or not pd.api.types.is_numeric_dtype(current[column]):
            features[column] = {"psi": None, "ks": None, "kl_divergence": None, "status": "UNKNOWN"}
            continue
        feature_psi, feature_ks = psi(reference[column], current[column]), ks_test(reference[column], current[column])
        shifted = (feature_psi is not None and feature_psi > PSI_THRESHOLD) or (feature_ks is not None and feature_ks["p_value"] < KS_P_VALUE_THRESHOLD)
        features[column] = {"psi": feature_psi, "ks": feature_ks, "kl_divergence": kl_divergence(reference[column], current[column]), "status": "DRIFTED" if shifted else "CLEAR"}
    schema = {"missing_columns": sorted(set(reference.columns) - set(current.columns)), "added_columns": sorted(set(current.columns) - set(reference.columns))}
    detected = any(x["status"] == "DRIFTED" for x in features.values()) or bool(schema["missing_columns"])
    return {"features": features, "schema": schema, "overall_status": "DRIFTED" if detected else "CLEAR", "method": "PSI + KS-test + KL-divergence", "thresholds": {"psi": PSI_THRESHOLD, "ks_p_value": KS_P_VALUE_THRESHOLD}}

def performance_report(model: Any, current: pd.DataFrame, label_column: str | None) -> dict[str, Any]:
    if not label_column or label_column not in current.columns:
        return {"status": "UNKNOWN", "reason": "Ground-truth labels are unavailable."}
    try:
        actual, predicted = current[label_column], model.predict(current.drop(columns=[label_column]))
        return {"status": "AVAILABLE", "label_column": label_column, "accuracy": float(accuracy_score(actual, predicted)), "precision": float(precision_score(actual, predicted, average="weighted", zero_division=0)), "recall": float(recall_score(actual, predicted, average="weighted", zero_division=0)), "f1": float(f1_score(actual, predicted, average="weighted", zero_division=0))}
    except Exception as exc:
        return {"status": "UNKNOWN", "reason": f"Performance could not be calculated: {exc}"}

def model_health(quality: dict[str, Any], drift: dict[str, Any], performance: dict[str, Any]) -> str:
    schema = quality.get("schema", {})
    if schema.get("missing_columns") or schema.get("type_mismatches"): return "CRITICAL"
    if performance.get("status") == "AVAILABLE" and performance.get("f1", 1.0) < 0.50: return "CRITICAL"
    if quality.get("status") == "WARNING" or drift.get("overall_status") == "DRIFTED": return "WARNING"
    return "HEALTHY"

def monitoring_report(reference: pd.DataFrame, current: pd.DataFrame, model: Any | None = None, label_column: str | None = None) -> dict[str, Any]:
    quality = data_quality(current, reference)
    drift = drift_report(reference.drop(columns=[label_column], errors="ignore"), current.drop(columns=[label_column], errors="ignore"))
    performance = performance_report(model, current, label_column) if model is not None else {"status": "UNKNOWN", "reason": "No local model supplied."}
    return {"data_quality": quality, "drift": drift, "performance": performance, "overall_drift_status": drift["overall_status"], "health": model_health(quality, drift, performance)}
