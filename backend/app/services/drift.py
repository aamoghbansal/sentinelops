from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPSILON = 1e-6

def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)

def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    ref, cur = _numeric(reference), _numeric(current)
    if len(ref) < 2 or len(cur) < 2:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    rc, _ = np.histogram(ref, edges)
    cc, _ = np.histogram(cur, edges)
    rp = (rc + EPSILON) / (rc.sum() + len(rc) * EPSILON)
    cp = (cc + EPSILON) / (cc.sum() + len(cc) * EPSILON)
    return float(np.sum((cp - rp) * np.log(cp / rp)))

def kl_divergence(reference: pd.Series, current: pd.Series, bins: int = 10) -> float | None:
    ref, cur = _numeric(reference), _numeric(current)
    if len(ref) < 2 or len(cur) < 2:
        return None
    edges = np.unique(np.quantile(np.concatenate([ref, cur]), np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    rc, _ = np.histogram(ref, edges)
    cc, _ = np.histogram(cur, edges)
    rp = (rc + EPSILON) / (rc.sum() + len(rc) * EPSILON)
    cp = (cc + EPSILON) / (cc.sum() + len(cc) * EPSILON)
    return float(np.sum(cp * np.log(cp / rp)))

def ks_test(reference: pd.Series, current: pd.Series) -> dict[str, float] | None:
    ref, cur = _numeric(reference), _numeric(current)
    if len(ref) < 2 or len(cur) < 2:
        return None
    result = ks_2samp(ref, cur)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}

def compute_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    comparable = [
        c for c in reference.columns
        if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])
    ]
    features = {}
    for column in comparable:
        features[column] = {
            "psi": psi(reference[column], current[column]),
            "ks": ks_test(reference[column], current[column]),
            "kl_divergence": kl_divergence(reference[column], current[column]),
        }
        metric = features[column]
        metric["status"] = "DRIFTED" if (
            (metric["psi"] is not None and metric["psi"] > 0.2)
            or ((metric["ks"] or {}).get("p_value") is not None and (metric["ks"] or {})["p_value"] < 0.05)
        ) else "CLEAR"
    return {
        "features": features,
        "schema": {
            "missing_columns": sorted(set(reference.columns) - set(current.columns)),
            "added_columns": sorted(set(current.columns) - set(reference.columns)),
        },
        "method": "PSI + KS-test + KL-divergence",
        "thresholds": {"psi": 0.2, "ks_p_value": 0.05},
    }

def drift_detected(drift: dict[str, Any], psi_threshold: float = 0.2, ks_threshold: float = 0.05) -> bool:
    for detail in drift.get("features", {}).values():
        p = detail.get("psi")
        ks_p = (detail.get("ks") or {}).get("p_value")
        if (p is not None and p > psi_threshold) or (ks_p is not None and ks_p < ks_threshold):
            return True
    return False
