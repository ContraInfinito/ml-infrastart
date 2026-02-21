# ml-infra-start

Minimal ML inference infra: sklearn model -> FastAPI inference service -> Dockerized deployment -> latency benchmarking (P50/P95/P99, throughput).

## Project summary
This repository implements a minimal ML inference pipeline designed for people who want to learn and demonstrate ML infrastructure skills:

- Train a small model (scikit-learn) and serialize it.
- Serve predictions with a lightweight FastAPI service.
- Package the service in a Docker container.
- Run repeatable latency/throughput tests and record P50/P95/P99 numbers.
- Provide a clean baseline that you can iterate on (TorchScript, ONNX, Triton, batching, autoscaling, etc.).

This repo is explicitly geared toward ML-Infra / MLOps interview stories: problem -> baseline -> metric -> optimization -> result.

## Why this project
- Teaches the full loop (train -> serialize -> serve -> measure).
- Forces you to confront real infra problems: serialization formats, web serving, containers, concurrency, and measurement.
- Produces concise, measurable artifacts (latency tables, before/after experiments) that interviewers and hiring managers value.
- Minimal, reproducible, and easy to extend into more advanced infra experiments.

## Features
- train.py -> trains and saves a scikit-learn model (model.joblib).
- app.py -> FastAPI app exposing:
  - GET /health endpoint
  - POST /predict inference endpoint (returns class probabilities + per-request latency)
- Dockerfile -> container image for the service.
- load_test.py -> simple Python-based load tester (no external tools required).
- .gitignore and an optional CI workflow example.
- README with commands and a place to record metrics.

## Prerequisites
- Linux-like environment (WSL2 on Windows, macOS, or native Linux)
- Python 3.11+
- Docker (Docker Desktop recommended on Windows/macOS)
- git
- (optional) VS Code + Remote - WSL for a smooth edit/run experience

## Getting started
Run these commands from a terminal inside the project folder.

```bash
# create project and virtualenv
mkdir ~/ml-infra-start && cd ~/ml-infra-start
python3 -m venv .venv
source .venv/bin/activate

# initialize git (optional if you already have a repo)
git init

# install dependencies
pip install -r requirements.txt

# train model
python train.py

# run the API locally
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# in another terminal: quick curl test
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}' | jq

# run the load tester
python load_test.py
```

## Docker
Build and run the container:

```bash
docker build -t ml-infra-start:baseline .
docker run --rm -p 8000:8000 ml-infra-start:baseline

# re-run load tests against the running container
python load_test.py
```

If using WSL2 on Windows, run Docker Desktop with WSL integration enabled.

## Expected files
- requirements.txt -> Python deps
- train.py -> trains and saves model.joblib
- app.py -> FastAPI inference service
- Dockerfile -> image definition
- load_test.py -> basic concurrency-based load tester
- .gitignore -> ignores venv, pycache, and model binary
- README.md

## How to measure and what to record
For each run (local and Docker) record:

- Environment: OS / CPU / RAM / Python version
- Test parameters: n (requests), concurrency
- P50, P95, P99 latencies (ms)
- Mean latency (ms)
- Approximate throughput (req/s)
- Short note explaining why a change helped or regressed performance

### Example metrics
Replace with your measured values.

- Environment: WSL2 (Ubuntu) -> 4 vCPU, 8 GB RAM
- Baseline (uvicorn, local)
  - P50: 8 ms
  - P95: 20 ms
  - P99: 45 ms
  - Throughput: ~200 req/s
- Container (Docker)
  - P50: 9 ms
  - P95: 22 ms
  - P99: 50 ms
  - Throughput: ~180 req/s
- Notes: Docker overhead minor; single-worker Uvicorn OK for low concurrency but worker count/batching needed for production throughput.

## Project structure
```
ml-infra-start/
├- .gitignore
├- requirements.txt
├- train.py
├- app.py
├- load_test.py
├- Dockerfile
└- README.md
```

## Next experiments (suggested roadmap)
- PyTorch baseline -> replace RandomForest with a tiny PyTorch model and compare latency.
- TorchScript / ONNX -> export and run ONNX Runtime; measure CPU and latency gains.
- Serving runtimes -> integrate NVIDIA Triton or Ray Serve and compare throughput/cost.
- Dynamic batching -> implement request coalescing and measure P95 at high concurrency.
- Prometheus + Grafana -> add metrics and visualize latency/throughput trends.
- Kubernetes -> move to minikube or kind, and experiment with autoscaling.
- Cost engineering -> run experiments on spot/preemptible instances and compare $/1k inferences.

Document every change as: What -> How measured -> Baseline metrics -> New metrics -> Why.

## How to present this on your resume / portfolio
Example single-line resume bullet:

Built an end-to-end ML inference baseline (training -> FastAPI serving -> Docker) and performed systematic latency benchmarking (P50/P95/P99), demonstrating production-focused optimization and measurable improvements.

Longer paragraph for portfolio:

Implemented an end-to-end ML inference pipeline with a serialized model, a FastAPI inference service, and Dockerized deployment. Created reproducible load tests and documented latency and throughput (P50/P95/P99) before and after optimizations, demonstrating practical ML infra skills and an ability to measure and improve production performance.

## Contributing
Fork the repo, create a feature branch, open a PR with a clear description and metrics for any performance experiments.

Keep experiments small and measurable: each PR should include one change + the before/after metrics.

## License
Add a license you prefer (MIT recommended for small demos). Example: MIT License.
