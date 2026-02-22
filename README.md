# ml-infra-start: Football Transfer Market Value Predictor

Minimal ML inference infra: RandomForest regressor -> FastAPI prediction service -> Dockerized deployment -> latency benchmarking (P50/P95/P99, throughput).

## Project summary

This repository demonstrates a complete ML infrastructure pipeline for predicting football player market values:

- **Train**: RandomForestRegressor on comprehensive player trajectory data (~500 players, 30+ features)
- **Preprocess**: Data cleaning (null handling), outlier removal (IQR method), feature engineering, scaling, and categorical encoding
- **Evaluate**: Cross-validation (5-fold) and hyperparameter testing (n_estimators: 50, 100, 200, 500, 1000)
- **Serve**: Lightweight FastAPI service with regression predictions + confidence intervals
- **Deploy**: Dockerized container for reproducible deployment
- **Measure**: Repeatable load tests with latency/throughput metrics (P50/P95/P99)

This repo exemplifies ML-Infra / MLOps capabilities: problem definition -> data pipeline -> baseline -> metric -> optimization -> measurable results.

## The task: Predict player market value

**Goal**: Given a player's characteristics (age, position, league, career trajectory metrics), predict their current market value in EUR.

**Dataset**: Football Transfer Market Value Trajectories from Transfermarkt.com
- 508 professional football players
- 36 features including:
  - Demographics: age, position, league, nationality
  - Career metrics: CAGR, value multiplier, career span, peak date
  - Trajectory indicators: rising_star, growing, stable, declining, falling_sharply
  - Value thresholds: ever_100m, ever_50m, ever_10m
- Target: current_value_eur (market value in EUR)

**Baseline model**: RandomForestRegressor
- 30+ input features after preprocessing
- Predicts continuous player market value
- Includes per-request latency measurement

## Why this project

- **End-to-end pipeline**: Data loading -> preprocessing -> training -> serialization -> serving
- **Production-ready patterns**: Cross-validation, hyperparameter sweep, metrics logging, error handling
- **Real dataset**: Rich football transfer data with engineering-friendly features (CAGR, multipliers)
- **Measurable improvements**: Compare performance across n_estimators values via training_logs.csv
- **MLOps foundations**: Reproducible training, containerization, latency/throughput tracking
- **Interview-ready**: Clear story: dataset -> preprocessing decisions -> model selection -> measurements

## Features

- **train.py** -> Loads football data, preprocesses, trains RandomForest with cross-validation, logs metrics to training_logs.csv, saves model.joblib
- **preprocessing.py** -> Reusable pipeline for data loading, null handling, outlier removal, scaling, encoding
- **app.py** -> FastAPI service with:
  - GET /health: Basic health check
  - GET /health/detailed: Model info (n_estimators, R2, feature counts)
  - POST /predict: Regression endpoint (input features -> predicted value EUR + confidence interval + latency)
- **Dockerfile** -> Container image for deployment
- **load_test.py** -> Concurrent load tester measuring P50/P95/P99 latencies and throughput
- **requirements.txt** -> All dependencies (FastAPI, scikit-learn, pandas, joblib)
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

# In another terminal: Test prediction
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

- requirements.txt -> Python dependencies
- train.py -> Training orchestration + hyperparameter sweep
- preprocessing.py -> Reusable data pipeline
- app.py -> FastAPI inference service
- Dockerfile -> Container definition
- load_test.py -> Concurrency-based latency/throughput tester
- .gitignore -> Ignore .venv, __pycache__, *.joblib, *.csv
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
├── .gitignore
├── requirements.txt
├── train.py                    # Hyperparameter sweep + logging
├── preprocessing.py            # Data pipeline (reusable)
├── app.py                      # FastAPI service
├── load_test.py                # Latency/throughput tester
├── Dockerfile                  # Container definition
├── footballTransfer.zip        # Source dataset
├── training_logs.csv           # Metrics history (auto-generated)
├── model.joblib                # Serialized model + preprocessor (auto-generated)
└── README.md
```

## Next experiments (suggested roadmap)

- **Feature importance analysis**: Which features drive predictions most? (analyze model.feature_importance)
- **Hyperparameter tuning**: Grid search over max_depth, min_samples_split
- **Different models**: Gradient Boosting (XGBoost), neural network (verify RF is best)
- **Batching**: Implement dynamic request batching to improve P95 latency at high concurrency
- **Caching**: Cache predictions for recent players (few unique queries likely repeated)
- **Metrics collection**: Add Prometheus instrumentation (request count, latency histogram, model accuracy)
- **Kubernetes**: Deploy to minikube/kind with HPA (autoscale based on throughput)
- **Cost engineering**: Run on spot instances, measure cost per 1000 predictions
- **A/B testing**: Compare RF vs XGBoost with live traffic; measure user-facing latency impact

Document each: What change -> How measured -> Baseline metrics -> New metrics -> Why.

## How to present this on your resume / portfolio

**Single-line bullet**:
Built end-to-end ML infrastructure (data preprocessing -> RandomForest regression training -> FastAPI inference -> Docker deployment) on real football transfer data with systematic hyperparameter sweep and latency benchmarking (P50/P95/P99).

**Paragraph**:
Implemented a complete ML inference pipeline predicting football player market values. Designed a production-grade data preprocessing pipeline (null handling, outlier removal via IQR, feature scaling/encoding). Trained a RandomForest regressor with 5-fold cross-validation across 5 n_estimators configurations, logging metrics (R2, RMSE, MAE) to enable data-driven model selection. Deployed as a FastAPI service returning confidence intervals and per-request latency. Containerized with Docker and benchmarked with reproducible load tests, capturing P50/P95/P99 latencies. Demonstrated production ML infrastructure skills: data rigor, hyperparameter methodology, serialization, serving, and measurable performance assessment.

## Contributing

Fork, create feature branch, open PR with:
- One focused change
- Before/after metrics (training_logs.csv and load test output)
- Clear explanation of why the change matters

## License

MIT License (see LICENSE file or add to your repo)


