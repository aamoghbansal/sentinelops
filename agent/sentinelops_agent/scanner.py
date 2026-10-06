from pathlib import Path
import ast

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
