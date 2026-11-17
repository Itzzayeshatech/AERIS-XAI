# AERIS XAI

### AI-Driven Aerospace Intelligence and Predictive Flight Behaviour Analysis

AERIS XAI is an offline academic research platform for analysing synthetic aircraft telemetry, detecting unusual flight behaviour, forecasting selected telemetry parameters, explaining model findings, replaying flights, and managing engineering investigations.

> **Academic prototype notice:** AERIS XAI is not certified aviation software. It is not an aircraft flight-control system and must not be used for real-time operational, safety-critical, or certified aviation decisions.

**Repository:** https://github.com/Itzzayeshatech/AERIS-XAI

---

## Overview

AERIS XAI combines FastAPI, Dash, Plotly, SQLAlchemy, scikit-learn, PyTorch, SHAP, and MLflow to provide a local aerospace intelligence laboratory.

### Core capabilities

- Synthetic aircraft telemetry generation with reproducible seeds
- Validated CSV flight-data uploads
- Telemetry visualization and flight metadata
- Flight-level training, validation, and test splitting
- Isolation Forest anomaly detection
- PyTorch LSTM Autoencoder temporal anomaly detection
- Random Forest next-step telemetry prediction
- SHAP and feature-ablation explainability
- MLflow experiment tracking
- Interactive digital flight replay
- Engineering investigation feedback
- HTML investigation report generation
- Admin and Engineer role-based access
- SQLite persistence for local academic use

---

## Technology Stack

| Area | Technology |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Frontend | Dash, Plotly, HTML, CSS |
| Database | SQLite, SQLAlchemy |
| Authentication | JWT, bcrypt |
| Data processing | Pandas, NumPy, Joblib |
| Machine learning | scikit-learn, Isolation Forest, Random Forest |
| Deep learning | PyTorch CPU |
| Explainability | SHAP, feature ablation |
| Experiment tracking | MLflow |
| Testing | pytest, HTTPX, FastAPI TestClient |

---

## System Architecture

```text
AERIS-XAI/
├── backend/
│   ├── api/                 # FastAPI routers
│   ├── db/                  # SQLAlchemy models and sessions
│   ├── schemas/             # Pydantic request and response models
│   ├── security/            # JWT, bcrypt, and RBAC
│   └── services/            # Flight, ML, report, and investigation services
├── frontend/
│   ├── app.py               # Dash application and HTTP API client
│   ├── callbacks.py         # Interactive Dash callbacks
│   ├── pages/               # Application pages
│   └── assets/theme.css     # Aerospace command-centre theme
├── ml/
│   ├── models/              # Isolation Forest, LSTM, and predictor wrappers
│   ├── explain/             # SHAP and feature-ablation modules
│   ├── preprocess.py        # Fitted preprocessing pipeline
│   ├── splits.py            # Flight-level dataset splitting
│   └── windows.py           # Time-window feature engineering
├── datasets/                # Telemetry schema and synthetic generator
├── tests/                   # Unit, API, integration, and ML tests
├── run_backend.py           # FastAPI launcher
├── run_frontend.py          # Dash launcher
└── pyproject.toml           # Dependencies and pytest configuration
```

Dash communicates with FastAPI through authenticated HTTP requests. The frontend does not access SQLite or load trained ML models directly.

---

## Telemetry Schema

Required CSV columns:

- `flight_id`
- `t_s`
- `phase`
- `altitude_ft`
- `airspeed_kts`
- `vertical_speed_fpm`
- `pitch_deg`
- `roll_deg`
- `heading_deg`
- `n1_pct`
- `egt_c`
- `fuel_flow_pph`
- `latitude`
- `longitude`

Optional evaluation columns:

- `is_anomaly`
- `anomaly_type`

Supported flight phases:

```text
Taxi, Takeoff, Climb, Cruise, Descent, Approach, Landing
```

CSV uploads must contain exactly one `flight_id`. Evaluation labels are intended for controlled synthetic testing and must never enter model feature matrices.

---

## Windows Installation

### 1. Install Python

Python 3.11 or later is recommended. Verify the installation:

```powershell
py --version
```

### 2. Create a virtual environment

```powershell
cd C:\path\to\AERIS-XAI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell script execution is blocked:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 4. Configure the environment

Create a `.env` file based on `.env.example`:

```dotenv
DATABASE_URL=sqlite:///./aeris.db
JWT_SECRET=replace-with-a-long-random-local-secret
DATA_DIR=./data
MLFLOW_TRACKING_URI=sqlite:///./mlflow.db
CORS_ORIGINS=["http://127.0.0.1:8050","http://localhost:8050"]
AERIS_API_BASE=http://127.0.0.1:8000
DASH_HOST=127.0.0.1
DASH_PORT=8050
```

Never commit `.env`, database files, model artifacts, credentials, or uploaded telemetry.

---

## Running the Application

### Terminal 1: FastAPI backend

```powershell
cd C:\path\to\AERIS-XAI
.\.venv\Scripts\Activate.ps1
python run_backend.py
```

FastAPI: http://127.0.0.1:8000  
API documentation: http://127.0.0.1:8000/docs

### Terminal 2: Dash frontend

```powershell
cd C:\path\to\AERIS-XAI
.\.venv\Scripts\Activate.ps1
python run_frontend.py
```

Dash interface: http://127.0.0.1:8050

### Health check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/health
```

---

## Demo Accounts

| Role | Username | Password |
|---|---|---|
| Administrator | `admin` | `admin123` |
| Engineer | `engineer` | `engineer123` |

Demo credentials are intended only for the local laboratory. Change or remove them before any shared deployment.

### Role permissions

**Admin**

- User management
- Synthetic dataset generation
- Dataset management
- Model training
- Model comparison and promotion
- System model management

**Engineer**

- Telemetry upload
- Flight review
- Anomaly detection
- Predictive analytics
- Explainability
- Flight replay
- Investigations and reports

---

## Recommended Demonstration Workflow

1. Start the FastAPI backend and Dash frontend.
2. Log in as `admin`.
3. Open the Flight Laboratory.
4. Generate at least five synthetic flights.
5. Open the Training Centre.
6. Train an Isolation Forest model.
7. Open Anomaly Detection and select a synthetic flight.
8. Review anomaly windows and anomaly scores.
9. Open Explainability to view feature attributions.
10. Compare model runs.
11. Replay a selected flight.
12. Create an engineering investigation report.
13. Log out and repeat key read-only workflows as `engineer`.

---

## Machine Learning Design

### Dataset splitting

- 60% training
- 20% validation
- 20% held-out testing
- Splitting is performed by `flight_id`, not individual telemetry rows
- Preprocessing transformations are fitted using training flights only
- Evaluation labels are excluded from feature matrices

### Isolation Forest

The Isolation Forest operates on sliding-window telemetry aggregates. Scores are normalized to a 0–1 range and thresholded for window-level anomaly classification.

### LSTM Autoencoder

The LSTM Autoencoder learns normal temporal sequences, reconstructs input windows, and uses reconstruction error for temporal anomaly scoring.

### Predictive analytics

Random Forest models predict:

- Next-step altitude
- Next-step airspeed
- Next-step engine N1

MAE and RMSE are reported only when held-out test targets are available.

### Explainability

- Isolation Forest explanations use SHAP TreeExplainer
- LSTM explanations use feature-ablation reconstruction-error analysis

Every explanation should be interpreted together with the flight ID, time window, model run, score, and explanation method.

---

## Testing

Run the complete test suite:

```powershell
python -m pytest -q
```

Run with a coverage report:

```powershell
python -m pytest --cov=backend --cov=ml --cov=datasets --cov=frontend
```

Tests cover authentication, permissions, API integration, synthetic generation, uploads, preprocessing, splitting, Isolation Forest, LSTM, predictors, and explainability.

---

## Security

AERIS XAI includes:

- bcrypt password hashing
- Signed JWT authentication
- Protected API endpoints
- Backend role authorization
- Upload size limits
- CSV schema validation
- Sanitized uploaded filenames
- Restricted CORS configuration
- Environment-backed application secrets

The application is a local research prototype. It should not be publicly deployed without HTTPS, a production secret manager, secure cookies or a hardened token strategy, database migrations, rate limiting, audit logging, and user-management hardening.

---

## Limitations

- Synthetic telemetry is not real aircraft measurement data.
- Metrics generated from synthetic flights must not be presented as real operational performance.
- Predictions are next-step research estimates, not future flight guarantees.
- SHAP and feature-ablation outputs indicate model contributions, not certified causal relationships.
- The system is unsuitable for real-time cockpit, operational, safety-critical, or airworthiness use.
- The SQLite database and local filesystem are intended for development and academic demonstrations.

---

## Project Status

AERIS XAI provides implemented backend, frontend, synthetic telemetry, authentication, preprocessing, model training, inference, comparison, replay, investigation, and reporting workflows. Final readiness should be assessed using executed test results and live verification rather than by source-code presence alone.

---

## Author

**Ayesha Siddiqa**  
GitHub: https://github.com/Itzzayeshatech

## License

Add the project license before public academic or commercial reuse. Until a license is added, copyright remains with the repository owner.
