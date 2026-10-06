from typing import Any
import pandas as pd

def data_quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    null_rates = {c: float(frame[c].isna().mean()) for c in frame.columns}
    outliers: dict[str, dict[str, float | int]] = {}
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna()
        if len(values) < 4:
            outliers[column] = {"count": 0, "rate": 0.0}
            continue
        q1, q3 = values.quantile(.25), values.quantile(.75)
        iqr = q3 - q1
        mask = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
        outliers[column] = {"count": int(mask.sum()), "rate": float(mask.mean())}
    return {
        "rows": len(frame),
        "duplicate_rows": int(frame.duplicated().sum()),
        "null_rates": null_rates,
        "iqr_outliers": outliers,
        "columns": list(frame.columns),
    }
