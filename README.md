# ML Infrastructure Portfolio: Market Value Prediction Service

A production-grade MLOps demonstration showcasing end-to-end machine learning infrastructure: data pipeline, model training with systematic evaluation, REST API serving, containerized deployment, and performance benchmarking.

## Overview

This project demonstrates core MLOps competencies through a real-world regression problem: predicting football player market values from career trajectory data.

**Technical Stack:**
- Python 3.11+ / scikit-learn / pandas / NumPy
- FastAPI / Uvicorn (async REST API)
- Docker (multi-stage production builds)
- Custom load testing framework (asyncio/aiohttp)

**MLOps Capabilities Demonstrated:**

| Capability | Implementation |
|------------|----------------|
| Data Pipeline | Reproducible preprocessing with null handling, IQR outlier removal, StandardScaler, OneHotEncoder |
| Model Training | RandomForest with 5-fold cross-validation, hyperparameter sweep (n_estimators: 50-1000) |
| Experiment Tracking | Append-only CSV logging (R2, RMSE, MAE, training time, CV statistics) |
| Model Serialization | Bundled model + preprocessor artifact (joblib) |
| API Serving | FastAPI with health checks, structured request/response schemas, latency tracking |
| Web Interface | Interactive prediction UI with real-time API status |
| Containerization | Multi-stage Docker build with non-root user, health checks |
| Performance Testing | Concurrent load testing with P50/P95/P99 latency percentiles |

## Problem Domain

**Task**: Regression model to predict football player market values (EUR) from career trajectory features.

**Dataset**: Transfermarkt.com player valuations
- 508 players, 36 raw features
- Feature categories: demographics, career metrics (CAGR, multipliers), trajectory indicators
- Target: `current_value_eur` (continuous)

**Model**: RandomForestRegressor
- 30+ engineered features after preprocessing
- Best configuration: n_estimators=200, Test R2=0.9997

## Key MLOps Concepts Demonstrated

| Concept | Implementation Detail |
|---------|----------------------|
| Reproducibility | Fixed random seeds, deterministic preprocessing, version-pinned dependencies |
| Data Quality | Null handling (median/mode imputation), outlier removal (IQR method) |
| Feature Engineering | StandardScaler for numerics, OneHotEncoder for categoricals |
| Model Validation | Train/test split (80/20), 5-fold cross-validation |
| Hyperparameter Search | Systematic sweep over n_estimators with full metrics logging |
| Artifact Management | Single joblib file bundles model + preprocessor + metadata |
| API Design | RESTful endpoints, Pydantic schemas, structured error handling |
| Observability | Per-request latency, health endpoints, experiment logs |
| Production Deployment | Multi-stage Docker, security hardening (non-root), health checks |

## Features

- **train.py** -> Loads football data, preprocesses, trains RandomForest with cross-validation, logs metrics to training_logs.csv, saves model.joblib
- **preprocessing.py** -> Reusable pipeline for data loading, null handling, outlier removal, scaling, encoding
- **app.py** -> FastAPI service with:
  - GET /: Web UI redirect (interactive prediction interface)
  - GET /health: Basic health check
  - GET /health/detailed: Model info (n_estimators, R2, feature counts)
  - POST /predict: Regression endpoint (input features -> predicted value EUR + confidence interval + latency)
- **static/index.html** -> Interactive web UI for making predictions with:
  - Player feature input form
  - Quick presets (Rising Star, Elite Player, Veteran, Random)
  - Real-time prediction display with confidence intervals
  - API health status indicator
  - Latency metrics display
- **Dockerfile** -> Multi-stage container image for optimized deployment
- **load_test.py** -> Concurrent load tester measuring P50/P95/P99 latencies and throughput
- **requirements.txt** -> All dependencies (FastAPI, scikit-learn, pandas, joblib, aiohttp)
- **training_logs.csv** -> Append-only metrics: timestamp, n_estimators, train/test R2/RMSE/MAE, CV stats, timing

## Prerequisites

- Linux-like environment (WSL2 on Windows, macOS, or native Linux)
- Python 3.11+
- Docker (Docker Desktop recommended on Windows/macOS)
- git
- (optional) VS Code + Remote - WSL for smooth edit/run experience

## Getting started

```bash
# Setup
cd ~/ml-infrastart
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train model (tests n_estimators: 50, 100, 200, 500, 1000)
python train.py
# Outputs: model.joblib, training_logs.csv

# Run the API locally
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Open the web UI in your browser
# http://localhost:8000

# In another terminal: Test prediction via curl
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 25.0,
      "career_span_years": 5.0,
      "years_to_peak": 3.5,
      "value_cagr": 0.35,
      "value_to_peak_cagr": 0.45,
      "value_multiplier_x": 10.0,
      "post_peak_decline_pct": 0.0,
      "value_volatility": 0.4,
      "mean_yoy_growth_rate": 0.2,
      "num_valuation_points": 15,
      "num_clubs_career": 2,
      "position_group": "Forward",
      "league_name": "Premier League",
      "position": "Centre-Forward",
      "trajectory": "growing"
    }
  }' | jq

# Check health
curl -s http://localhost:8000/health | jq

# Run load tests
python load_test.py
```

## Docker

```bash
docker build -t ml-infra-start:football .
docker run --rm -p 8000:8000 ml-infra-start:football

# In another terminal
python load_test.py
```

## Data pipeline overview

**Preprocessing steps** (see preprocessing.py):

1. **Load**: Extract footballTransfer.zip and read transfermarkt_player_values.csv
2. **Handle nulls**: Drop rows with >50% missing, fill numeric with median, categorical with mode
3. **Validate types**: Ensure numeric columns are float/int, remove unconvertible rows
4. **Remove outliers**: IQR method on market value, age, career span (1.5x multiplier)
5. **Select features**: ~30 features, exclude IDs/names/dates, separate numeric/categorical
6. **Encode**: One-hot encode positions and league names
7. **Scale**: StandardScaler on numeric features (mean=0, std=1)
8. **Split**: 80% train / 20% test (random_state=42)

**Result**: Features ready for RandomForest. All preprocessing bundled into model.joblib for inference.

## Training results tracking

Each run of `python train.py` appends to training_logs.csv:

```
timestamp, n_estimators, training_time_sec, train_r2, test_r2, 
train_rmse, test_rmse, train_mae, test_mae, pred_time_ms,
cv_train_r2_mean, cv_train_r2_std, cv_test_r2_mean, cv_test_r2_std,
cv_train_rmse_mean, cv_test_rmse_mean, cv_train_mae_mean, cv_test_mae_mean
```

Compare performance:
```bash
# View training history
head -20 training_logs.csv

# Find best n_estimators (sort by test_r2)
tail -5 training_logs.csv | sort -t',' -k5 -nr
```

## Expected files

- requirements.txt -> Python dependencies (FastAPI, scikit-learn, pandas, aiohttp, etc.)
- train.py -> Training orchestration + hyperparameter sweep
- preprocessing.py -> Reusable data pipeline
- app.py -> FastAPI inference service + static file serving
- static/index.html -> Interactive web UI for predictions
- Dockerfile -> Multi-stage container definition for production
- load_test.py -> Concurrency-based latency/throughput tester
- .gitignore -> Ignore .venv, __pycache__, IDE files, OS files
- README.md (this file)

## How to measure and record

For each experiment (local/Docker/optimized):

1. **Environment**: OS, CPU cores, RAM, Python version
2. **Test parameters**: Number of requests (n), concurrency level
3. **Latency metrics**: P50, P95, P99 (milliseconds)
4. **Throughput**: Requests per second
5. **Model metrics**: Test R2, RMSE, MAE (from training_logs.csv)
6. **Note**: Why this change matters (baseline -> new approach -> improvement)

### Example metrics table

| Environment | n_estimators | Test R2 | Test RMSE | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (req/s) | Notes |
|---|---|---|---|---|---|---|---|---|
| Local (uvicorn) | 100 | 0.78 | 8.2M | 4.2 | 8.5 | 15.3 | 240 | Single worker |
| Local (uvicorn) | 500 | 0.82 | 6.9M | 6.8 | 14.2 | 28.6 | 145 | Better accuracy, slower |
| Docker | 500 | 0.82 | 6.9M | 7.1 | 15.0 | 30.1 | 138 | Minor overhead vs local |

## Project structure

```
ml-infrastart/
├── .gitignore                  # Git ignore patterns
├── requirements.txt            # Python dependencies
├── train.py                    # Hyperparameter sweep + logging
├── preprocessing.py            # Data pipeline (reusable)
├── app.py                      # FastAPI service + static file serving
├── load_test.py                # Latency/throughput tester
├── Dockerfile                  # Multi-stage container definition
├── static/
│   └── index.html              # Interactive web UI
├── footballTransfer.zip        # Source dataset
├── training_logs.csv           # Metrics history (auto-generated)
├── model.joblib                # Serialized model + preprocessor (auto-generated)
└── README.md
```

## Web Interface

Minimalistic prediction interface available at `http://localhost:8000`.

**Features:**
- Player feature input form (15 fields)
- Quick presets: Rising Star, Elite Player, Veteran, Random
- Real-time prediction display with confidence intervals
- Live API health status indicator
- Per-request latency metrics

```
+------------------------------------------------------------------+
|  Football Player Market Value Predictor                          |
|  ML-powered market value estimation                              |
|                         [API Online / Model Loaded]              |
+-------------------------------+----------------------------------+
|  PLAYER FEATURES              |  PREDICTION RESULT               |
|                               |                                  |
|  Quick Presets:               |  +----------------------------+  |
|  [Rising Star] [Elite] ...    |  |  PREDICTED MARKET VALUE    |  |
|                               |  |       EUR 45.50M           |  |
|  Age: [25.0    ]              |  +----------------------------+  |
|  Career Span: [5.0  ]         |                                  |
|  Position: [Forward    v]     |  Lower        Upper              |
|  League: [Premier Lg   v]     |  EUR 38.68M - EUR 52.33M         |
|  ...                          |                                  |
|                               |  Latency: 4.23 ms                |
|  [  PREDICT MARKET VALUE  ]   |                                  |
+-------------------------------+----------------------------------+
```

## Future Improvements (MLOps Roadmap)

| Category | Enhancement | MLOps Value |
|----------|-------------|-------------|
| **Monitoring** | Prometheus metrics + Grafana dashboards | Production observability |
| **Model Registry** | MLflow integration for artifact versioning | Experiment reproducibility |
| **CI/CD** | GitHub Actions for automated testing + deployment | Continuous delivery |
| **Orchestration** | Kubernetes deployment with HPA | Auto-scaling inference |
| **A/B Testing** | Traffic splitting between model versions | Safe model rollouts |
| **Feature Store** | Centralized feature management | Data consistency |
| **Model Validation** | Automated drift detection | Production reliability |
| **Cost Optimization** | Spot instances, right-sizing | Infrastructure efficiency |

## Resume / Portfolio Presentation

**One-liner:**
> Designed and deployed production ML infrastructure: data pipeline, model training with cross-validation, FastAPI serving, Docker containerization, and latency benchmarking (P50/P95/P99).

**Technical Summary:**
> Built end-to-end MLOps pipeline for market value regression. Implemented reproducible data preprocessing (IQR outlier removal, feature scaling/encoding), systematic hyperparameter optimization with 5-fold CV, experiment tracking to CSV, and model artifact serialization. Deployed as containerized FastAPI service with interactive web UI. Benchmarked inference performance under concurrent load, achieving P50 latency <50ms at 100+ req/s throughput.

**Skills Demonstrated:**
- Machine Learning: scikit-learn, cross-validation, hyperparameter tuning, regression metrics
- Data Engineering: pandas, data cleaning, feature engineering, preprocessing pipelines
- Backend Development: FastAPI, REST API design, Pydantic schemas, async Python
- DevOps/MLOps: Docker multi-stage builds, health checks, load testing, performance profiling
- Software Engineering: Modular code design, error handling, documentation

## Contributing

Contributions welcome. Please include:
- Focused, single-purpose changes
- Before/after metrics (training_logs.csv, load test results)
- Clear rationale for the improvement

## License

MIT License

---

**Author**: MLOps Portfolio Project  
**Last Updated**: February 2026


