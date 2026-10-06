import pandas as pd
from sentinelops_agent.monitoring import drift_report, data_quality, monitoring_report
from sklearn.linear_model import LogisticRegression

def test_monitoring_detects_shift():
    ref = pd.DataFrame({"x": range(100)})
    cur = pd.DataFrame({"x": [v + 20 for v in range(100)]})
    report = drift_report(ref, cur)
    assert report["features"]["x"]["psi"] is not None
    assert data_quality(cur)["rows"] == 100


def test_monitoring_calculates_performance_when_labels_exist():
    reference = pd.DataFrame({"x": [0, 1, 2, 3], "target": [0, 0, 1, 1]})
    model = LogisticRegression().fit(reference[["x"]], reference["target"])
    report = monitoring_report(reference, reference.copy(), model, "target")
    assert report["performance"]["status"] == "AVAILABLE"
    assert report["performance"]["accuracy"] == 1.0
    assert report["health"] == "HEALTHY"
