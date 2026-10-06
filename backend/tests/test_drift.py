import pandas as pd
from app.services.drift import compute_drift, drift_detected

def test_drift_detects_shift():
    ref = pd.DataFrame({"x": list(range(100))})
    cur = pd.DataFrame({"x": [v + 20 for v in range(100)]})
    report = compute_drift(ref, cur)
    assert report["features"]["x"]["psi"] is not None
    assert drift_detected(report)
