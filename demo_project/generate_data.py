from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
root = Path(__file__).parent / "data"
root.mkdir(exist_ok=True)
reference = data.data.iloc[:75].copy()
current = data.data.iloc[75:].copy()
current["sepal length (cm)"] = current["sepal length (cm)"] + 1.7
reference.to_csv(root / "reference.csv", index=False)
current.to_csv(root / "current.csv", index=False)

# A labelled pair is provided for a repeatable Stage 2 performance demonstration.
# The files remain local; the Agent uploads only aggregate monitoring metrics.
reference_labeled = reference.copy()
current_labeled = current.copy()
reference_labeled["target"] = data.target.iloc[:75].to_numpy()
current_labeled["target"] = data.target.iloc[75:].to_numpy()
reference_labeled.to_csv(root / "reference_labeled.csv", index=False)
current_labeled.to_csv(root / "current_labeled.csv", index=False)
print("Generated shifted reference/current datasets, including labelled monitoring samples.")
