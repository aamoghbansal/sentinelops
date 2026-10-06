from pathlib import Path
import ast
from typing import Any

import pandas as pd

MODEL_EXTENSIONS = {".pkl", ".joblib", ".pt", ".pth", ".keras", ".h5"}
DATA_EXTENSIONS = {".csv", ".parquet"}
TRAINING_NAMES = {"train.py", "training.py", "train_model.py", "retrain.py"}

def _python_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module: imports.append(node.module)
    return sorted(set(imports))

def scan_project(root: str) -> dict:
    base = Path(root).resolve()
    files = [p for p in base.rglob("*") if p.is_file() and ".venv" not in p.parts and "node_modules" not in p.parts]
    models = [str(p.relative_to(base)) for p in files if p.suffix.lower() in MODEL_EXTENSIONS]
    datasets = [str(p.relative_to(base)) for p in files if p.suffix.lower() in DATA_EXTENSIONS]
    training = [str(p.relative_to(base)) for p in files if p.name.lower() in TRAINING_NAMES]
    py_files = [p for p in files if p.suffix == ".py"]
    imports = sorted({item for p in py_files for item in _python_imports(p)})
    framework = "unknown"
    if any(name.startswith("sklearn") for name in imports): framework = "scikit-learn"
    elif any(name.startswith("xgboost") for name in imports): framework = "xgboost"
    elif any(name.startswith("torch") for name in imports): framework = "pytorch"
    elif any(name.startswith("tensorflow") or name.startswith("keras") for name in imports): framework = "tensorflow/keras"
    return {"project_root": str(base), "model_files": models, "dataset_files": datasets, "training_files": training, "python_imports": imports, "detected_framework": framework, "scan_note": "AST/file-signature scan; V1 supports scikit-learn tabular projects."}


LABEL_HINTS = ("target", "label", "class", "outcome", "ground_truth", "ground-truth", "actual", "y")
REFERENCE_HINTS = (("reference", 100), ("baseline", 90), ("train", 60))
CURRENT_HINTS = (("current", 100), ("production", 90), ("live", 80), ("recent", 70), ("serving", 60))

def _data_files(root: Path) -> list[Path]:
    excluded = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or excluded.intersection(path.parts):
            continue
        if path.suffix.lower() == ".csv":
            files.append(path)
    return files

def _filename_score(path: Path, hints: tuple[tuple[str, int], ...]) -> int:
    name = path.stem.lower().replace("-", "_")
    return sum(score for hint, score in hints if hint in name)

def _find_label_column(columns: list[str], model_features: list[str] | None = None) -> str | None:
    by_lower = {str(c).strip().lower(): str(c) for c in columns}
    for hint in LABEL_HINTS:
        if hint in by_lower:
            return by_lower[hint]
    if model_features:
        extras = [c for c in columns if c not in model_features]
        if len(extras) == 1:
            return str(extras[0])
    return None

def discover_monitoring_inputs(model_path: str, model_features: list[str] | None = None) -> dict[str, Any]:
    model_file = Path(model_path).expanduser().resolve()
    root = model_file.parent
    candidates = _data_files(root)
    if len(candidates) < 2:
        raise ValueError(f"Could not auto-detect monitoring data in {root}. Need reference/baseline and current/production CSV files.")
    infos: list[dict[str, Any]] = []
    for path in candidates:
        try:
            sample = pd.read_csv(path, nrows=50)
        except Exception:
            continue
        label = _find_label_column(list(sample.columns), model_features)
        infos.append({"path": path, "columns": list(sample.columns), "label": label, "reference_score": _filename_score(path, REFERENCE_HINTS), "current_score": _filename_score(path, CURRENT_HINTS)})
    refs = [item for item in infos if item["reference_score"] > 0]
    currents = [item for item in infos if item["current_score"] > 0]
    if not refs or not currents:
        raise ValueError("Could not confidently auto-detect reference/current datasets. Use filenames containing reference/baseline/train and current/production/live.")
    best_pair = None
    best_score = -1
    for ref in refs:
        for cur in currents:
            if ref["path"] == cur["path"]:
                continue
            shared = set(ref["columns"]) & set(cur["columns"])
            if not shared:
                continue
            label = ref["label"] if ref["label"] and ref["label"] in cur["columns"] else cur["label"]
            score = ref["reference_score"] + cur["current_score"]
            if label:
                score += 25
            if ref["label"] and ref["label"] == cur["label"]:
                score += 10
            if ref["path"].parent == cur["path"].parent:
                score += 5
            if score > best_score:
                best_score = score
                best_pair = (ref, cur, label)
    if best_pair is None:
        raise ValueError("Could not find compatible reference and current datasets for monitoring.")
    ref, cur, label = best_pair
    return {"reference_path": str(ref["path"]), "current_path": str(cur["path"]), "label_column": label, "project_root": str(root), "confidence": "high" if best_score >= 225 else "medium"}
