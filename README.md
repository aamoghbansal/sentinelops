# SentinelOps — Development Build

This repository is a **focused development build** of SentinelOps.

It currently demonstrates the foundation and monitoring layer of the platform:

1. Model/project registration
2. Scoped Agent credential generation
3. Local SentinelOps Agent
4. AST/file-signature project scanning
5. Data-quality monitoring (null rates, duplicates, IQR outliers)
6. Statistical drift detection using PSI, KS-test, and KL divergence
7. React dashboard showing the results

## Planned next components

- Automated retraining
- Model version registry/comparison
- Deployment, canary deployment, rollback
- LLM Copilot
- Additional framework adapters

These are planned parts of the overall SentinelOps project and will be added in later development phases.

## Repository

```text
sentinelops_build/
├── backend/            # FastAPI + SQLite
├── agent/              # Local Python Agent
├── frontend/           # React + Vite dashboard
├── demo_project/       # Small scikit-learn/Iris workload
└── docs/
    └── CURRENT_SCOPE.md
```

## Run backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs`.

## Run frontend

Open another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Prepare demo project

```powershell
cd demo_project
python -m pip install pandas scikit-learn joblib
python train.py
python generate_data.py
```

## Live Agent demo

1. In the dashboard, choose the seeded `Iris classification demo` project.
2. Click **Create token** and copy the token.
3. In a third terminal:

```powershell
cd agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

4. Scan the demo project:

```powershell
sentinelops-agent scan --project-root ..\demo_project --api-url http://localhost:8000 --project-id 1 --token YOUR_TOKEN
```

5. Run monitoring against the reference/current CSV files:

```powershell
sentinelops-agent monitor --project-root ..\demo_project --api-url http://localhost:8000 --project-id 1 --token YOUR_TOKEN --reference ..\demo_project\data\reference.csv --current ..\demo_project\data\current.csv
```

6. Refresh the dashboard. The scan findings, quality results, and drift results will appear.

### Easier presentation shortcut

The dashboard has **Run demo drift check**. It computes the same PSI/KS/KL idea directly on a controlled Iris distribution shift, so the live presentation does not depend on the Agent CLI if time is limited.

## Tests

Backend:

```powershell
cd backend
pytest
```

Agent:

```powershell
cd agent
pytest
```
