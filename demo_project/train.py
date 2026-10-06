from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

root = Path(__file__).parent
model = LogisticRegression(max_iter=1000, random_state=17)
data = load_iris(as_frame=True)
model.fit(data.data, data.target)
joblib.dump(model, root / "model.joblib")
print("Saved model.joblib")
