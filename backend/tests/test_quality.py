import pandas as pd
from app.services.quality import data_quality_report

def test_quality_reports_nulls_duplicates_and_outliers():
    frame = pd.DataFrame({"x": [1, 2, 2, 100, None]})
    result = data_quality_report(frame)
    assert result["duplicate_rows"] == 1
    assert result["null_rates"]["x"] > 0
    assert result["iqr_outliers"]["x"]["count"] >= 1
