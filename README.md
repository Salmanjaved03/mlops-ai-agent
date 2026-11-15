# MLOps AI Agent

[![MLOps Agent CI](https://github.com/Salmanjaved03/mlops-ai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Salmanjaved03/mlops-ai-agent/actions/workflows/ci.yml)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Client-326ce5)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-2.x%20%2F%203.x-017CEE)
![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6)

---

## 1. Executive Summary & Architecture

The **MLOps AI Agent** is a Python prototype for querying MLOps services through a command-line agent. Its integrations inspect DVC/Git metadata, MLflow experiments, Airflow DAGs, and Kubernetes resources. The Airflow training DAG and FastAPI prediction endpoint currently use simulated data/model behavior; they are examples, not production model-serving or training implementations.

### High-Level Architecture Overview

For full system design and sequence flows, see [architecture.md](architecture.md).

```
                      +---------------------------------------+
                      |         MLOps AI Agent REPL / CLI    |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       MLOpsAgent Orchestrator         |
                      |  - Intent Classifier                  |
                      |  - Tool Planner                       |
                      |  - Safety & Confirmation Gate         |
                      |  - Evidence Synthesizer               |
                      +---------------------------------------+
                         |           |           |          |
        +----------------+     +-----+           +----+     +----------------+
        |                      |                      |                      |
        v                      v                      v                      v
+---------------+      +---------------+      +---------------+      +---------------+
|   DVC Agent   |      |  MLflow Agent |      | Airflow Agent |      |   K8s Agent   |
|   (Subprocess)|      |  (SDK Client) |      |  (REST API)   |      | (K8s Client)  |
+---------------+      +---------------+      +---------------+      +---------------+
        |                      |                      |                      |
        v                      v                      v                      v
 [Local/Git DVC]       [MLflow Server]       [Airflow API]          [K8s Cluster]
```

---

## 2. Directory Structure

```text
mlops-ai-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py                  # Agent CLI entry point
│   ├── orchestrator.py          # Central agent orchestrator & intent planner
│   ├── schemas.py               # Strongly-typed Pydantic schemas
│   └── tools/
│       ├── __init__.py
│       ├── airflow_tools.py     # Airflow REST API integration
│       ├── dvc_tools.py         # DVC & Git CLI integration
│       ├── kubernetes_tools.py  # Kubernetes CoreV1 & AppsV1 integration
│       └── mlflow_tools.py      # MLflow tracking client integration
├── api/
│   ├── __init__.py
│   └── main.py                  # Production FastAPI inference service
├── dags/
│   └── training_pipeline.py     # 5-stage Airflow training DAG with Quality Gate
├── data/
│   └── .gitkeep
├── kubernetes/
│   ├── configmap.yaml           # Non-sensitive inference configurations
│   ├── deployment.yaml          # 2 replicas, rolling updates, health probes
│   └── service.yaml             # ClusterIP routing
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared mocks & pytest fixtures
│   ├── test_airflow_tools.py    # Airflow tool tests
│   ├── test_dvc_tools.py        # DVC tool tests
│   ├── test_kubernetes_tools.py # Kubernetes tool & namespace tests
│   ├── test_mlflow_tools.py     # MLflow tracking tests
│   ├── test_orchestrator.py     # Orchestrator & multi-tool tests
│   └── test_safety_and_failures.py # Safety, timeout, non-fabrication & failure tests
├── training/
│   ├── __init__.py
│   ├── prepare_data.py          # Raw to processed data split script
│   └── train.py                 # Training script logging to MLflow
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI workflow (lint + tests)
├── architecture.md              # Detailed architectural design & sequence diagrams
├── Dockerfile                   # Container build for inference service
├── dvc.yaml                     # DVC multi-stage pipeline configuration
├── pyproject.toml               # Python package specification & dependencies
└── README.md                    # Project documentation
```

---

## 3. Installation & Quickstart

### 3.1. Prerequisites

- Python 3.10+
- Git & DVC
- Docker (optional for local deployment)
- Access to MLflow, Airflow, or Kubernetes only for the corresponding live integrations

### 3.2. Setup Environment

```bash
# Clone the repository
git clone https://github.com/Salmanjaved03/mlops-ai-agent.git
cd mlops-ai-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with development extras
pip install -e ".[dev]"
```

### 3.3. Configure Integrations

The tools read configuration from process environment variables. Set only the values for services you use; the defaults target local services and should be changed outside development.

```dotenv
MLFLOW_TRACKING_URI=http://localhost:5000
AIRFLOW_BASE_URL=http://localhost:8080
AIRFLOW_USERNAME=airflow
AIRFLOW_PASSWORD=airflow
K8S_NAMESPACE=mlops
MODEL_NAME=iris_classifier
MODEL_VERSION=latest
```

The Python application does not automatically load `.env` files. Set these variables in your shell, IDE launch configuration, or container environment. Do not commit credentials.

---

## 4. Launching the Services

### 4.1. Run the FastAPI Inference API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Health Endpoint: `http://localhost:8000/health`  
Interactive Swagger Docs: `http://localhost:8000/docs`

The API also exposes `POST /predict` and `GET /model/info`. Predictions currently come from a simple Iris feature heuristic; the endpoint does not load a trained model from MLflow. The `/health` endpoint reports the simulated model state.

### 4.2. Run the Airflow DAG

Deploy `dags/training_pipeline.py` to your Airflow DAGs folder. The DAG's data validation, preprocessing, training, and evaluation tasks currently log simulated values rather than operating on real dataset/model artifacts. Its quality gate checks minimum accuracy of `0.90` and F1 score of `0.85`:

- **Logical Stages**: `validate_data` → `preprocess_data` → `train_model` → `evaluate_model` → `quality_gate`.
- **Quality Gate**: Raises an exception when accuracy is below `0.90` or F1 score is below `0.85`.

### 4.3. Deploy Kubernetes Manifests

```bash
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

---

## 5. Interacting with the MLOps AI Agent

### 5.1. Interactive REPL Mode

```bash
python -m agent.main --mode interactive
```

### 5.2. Single Query Mode

```bash
python -m agent.main --mode single --query "Find the best model from our recent experiments"
```

---

## 6. Tool Definitions & Permissions Matrix

| Domain         | Tool Name                  | Callable                     | Permissions / Safety                    | Description                                                   |
| -------------- | -------------------------- | ---------------------------- | --------------------------------------- | ------------------------------------------------------------- |
| **DVC**        | `dvc_list_tracked`         | `list_tracked_files()`       | Read-only                               | Lists all datasets and files tracked under DVC.               |
| **DVC**        | `dvc_get_version`          | `get_dataset_version()`      | Read-only                               | Retrieves current Git commit and dataset lockfile hashes.     |
| **DVC**        | `dvc_compare_versions`     | `compare_dataset_versions()` | Read-only                               | Computes diff between dataset versions across commits.        |
| **DVC**        | `dvc_check_status`         | `check_data_status()`        | Read-only                               | Verifies if local data has changed vs DVC tracking.           |
| **DVC**        | `dvc_explain_reproduce`    | `explain_reproduce_steps()`  | Read-only                               | Outputs exact reproduction checkout commands.                 |
| **MLflow**     | `mlflow_list_experiments`  | `list_experiments()`         | Read-only                               | Lists all active MLflow experiments.                          |
| **MLflow**     | `mlflow_list_runs`         | `list_runs()`                | Read-only                               | Enumerates run details and hyperparameter records.            |
| **MLflow**     | `mlflow_get_best_run`      | `get_best_run()`             | Read-only                               | Identifies top run by metric with production readiness notes. |
| **MLflow**     | `mlflow_compare_runs`      | `compare_runs()`             | Read-only                               | Side-by-side metric comparison across run IDs.                |
| **Airflow**    | `airflow_list_dags`        | `list_dags()`                | Read-only                               | Lists DAG metadata, paused states, and schedules.             |
| **Airflow**    | `airflow_trigger_dag`      | `trigger_dag()`              | **Destructive** (requires confirmation) | Triggers approved training DAGs only.                         |
| **Airflow**    | `airflow_get_failed_tasks` | `get_failed_tasks()`         | Read-only                               | Retrieves failed task instances and execution details.        |
| **Airflow**    | `airflow_get_task_logs`    | `get_task_logs()`            | Read-only                               | Fetches diagnostic execution logs for a task.                 |
| **Kubernetes** | `k8s_list_pods`            | `list_pods()`                | Read-only (`mlops`, `default`)          | Lists running pods and readiness indicators.                  |
| **Kubernetes** | `k8s_list_deployments`     | `list_deployments()`         | Read-only (`mlops`, `default`)          | Reports deployment status and replica availability.           |
| **Kubernetes** | `k8s_get_unhealthy_pods`   | `get_unhealthy_pods()`       | Read-only (`mlops`, `default`)          | Flags pods with `restart_count > 3` or unready containers.    |
| **Kubernetes** | `k8s_get_pod_events`       | `get_pod_events()`           | Read-only (`mlops`, `default`)          | Fetches Kubernetes cluster events for a pod.                  |

---

## 7. Illustrative Agent Conversations

The scenarios below are illustrative examples only. Their IDs, metrics, cluster states, and responses are sample data, not results from a live infrastructure run or a claim that those values exist in this repository.

### Scenario 1: Dataset Provenance

**User Query**: _"Which dataset version was used for experiment 25?"_
**Agent Action**: Executes `dvc_get_dataset_for_experiment(experiment_ref='exp-25')`.  
**Actual Tool Evidence**:

```json
{
  "experiment_ref": "exp-25",
  "commit": "a4c92e1deadbeef8921",
  "dvc_files": [
    "data/processed/X_train.npy.dvc",
    "data/processed/y_train.npy.dvc"
  ],
  "timestamp": "2026-10-15T12:00:00Z"
}
```

**Agent Response**:

> "Experiment 25 used dataset version associated with Git commit `a4c92e1`. The tracked data files were `data/processed/X_train.npy` and `data/processed/y_train.npy`. The data was retrieved from the configured DVC storage remote."

---

### Scenario 2: Model Selection

**User Query**: _"Compare the recent training runs and recommend the best candidate based on F1 score."_
**Agent Action**: Executes `mlflow_get_best_run(metric='f1_score')` and `mlflow_compare_runs()`.  
**Actual Tool Evidence**:

```json
{
  "best_run": {
    "run_id": "8f2a1b9c4d",
    "metrics": { "f1_score": 0.982, "accuracy": 0.985, "precision": 0.98 },
    "params": { "n_estimators": 100, "max_depth": 5 },
    "note": "Highest evaluation metric does not automatically guarantee production readiness."
  }
}
```

**Agent Response**:

> "The best candidate run based on F1 score is `8f2a1b9c4d` with an F1 score of **0.982** and accuracy of **0.985** (hyperparameters: `n_estimators=100`, `max_depth=5`).  
> _Engineering Note_: While this candidate achieved the highest test F1 score, evaluate inference latency and test set distribution before promoting to Production stage."

---

### Scenario 3: Pipeline Execution

**User Query**: _"Trigger the model training pipeline and report whether execution succeeded."_
**Agent Action**: Blocks unconfirmed run, alerts operator. When confirmed: executes `airflow_trigger_dag(dag_id='training_pipeline')` and monitors `airflow_get_dag_run_status()`.  
**Actual Tool Evidence**:

```json
{
  "dag_id": "training_pipeline",
  "dag_run_id": "manual__2026-11-15T12:30:00",
  "state": "success",
  "tasks": [
    { "task_id": "validate_data", "state": "success" },
    { "task_id": "preprocess_data", "state": "success" },
    { "task_id": "train_model", "state": "success" },
    { "task_id": "evaluate_model", "state": "success" },
    { "task_id": "quality_gate", "state": "success" }
  ]
}
```

**Agent Response**:

> "Pipeline execution triggered and completed successfully. All 5 pipeline stages (`validate_data` → `preprocess_data` → `train_model` → `evaluate_model` → `quality_gate`) finished in `success` state."

---

### Scenario 4: Deployment Diagnosis

**User Query**: _"Why is my prediction API unavailable?"_
**Agent Action**: Executes `k8s_get_unhealthy_pods(namespace='mlops')` and `k8s_get_pod_events()`.  
**Actual Tool Evidence**:

```json
{
  "pod_name": "ml-inference-api-6b78d-x9qwp",
  "status": "Running",
  "ready": false,
  "restart_count": 8,
  "events": [
    {
      "reason": "BackOff",
      "message": "Back-off restarting failed container",
      "type": "Warning"
    }
  ],
  "logs": "HTTPException: Model file 'models/model.pkl' not found on startup."
}
```

**Agent Response**:

> "The prediction API is unavailable because pod `ml-inference-api-6b78d-x9qwp` is trapped in a `CrashLoopBackOff` state with 8 restarts. Container logs reveal the root cause: `models/model.pkl` is missing from the container image. Remount the model artifact volume or download the model from MLflow during container initialization."

---

### Scenario 5: Mandatory Multi-Tool Reasoning

**User Query**: _"Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."_
**Agent Action**: Coordinates `mlflow_get_best_run` + `dvc_get_version` + `k8s_list_deployments` + `k8s_get_unhealthy_pods` + `airflow_list_dags`.  
**Actual Tool Evidence**:

- MLflow: Best run `8f2a1b9c4d`, F1 score: `0.982`.
- DVC: Git Commit `a4c92e1`, tracking `data/processed/`.
- Kubernetes: Deployment `ml-inference-api` has 2/2 ready replicas, 0 unhealthy pods.
- Airflow: `training_pipeline` active and succeeded.  
  **Agent Response**:
  > "### MLOps Operational Report
  >
  > #### 1. Factual Evidence Across Systems:
  >
  > - **Best Model**: Run `8f2a1b9c4d` (MLflow) with F1 score `0.982` and accuracy `0.985`.
  > - **Dataset Provenance**: Associated with Git Commit `a4c92e1` (DVC-tracked).
  > - **Deployment Health**: Kubernetes deployment `ml-inference-api` is operating normally in namespace `mlops` with 2/2 ready replicas and 0 restart events.
  > - **Pipeline Status**: Airflow DAG `training_pipeline` is active and healthy.
  >
  > #### 2. Retraining Assessment & Recommendation:
  >
  > **Retraining is NOT recommended at this time.**
  > _Justification_: The deployed model meets accuracy SLA (> 95%), the serving infrastructure is healthy, and DVC reports no data drift or unstaged data updates. Automatic retraining should only be initiated if model degradation or data changes are detected."

---

## 8. Illustrative Failure Investigations

These failure narratives are examples of the intended diagnostic workflow. They are not reports of observed incidents in a live deployment.

### Investigation 1: Kubernetes CrashLoopBackOff (Container Startup Crash)

- **Observed Symptom**: Inference service returns `503 Service Unavailable` or connection timeouts.
- **Agent Diagnosis**:
  1. Ran `k8s_get_unhealthy_pods(namespace='mlops')` which detected `restart_count = 14`.
  2. Queried `k8s_get_pod_events` which revealed repeated `BackOff` events.
  3. Inspected `k8s_get_pod_logs` which caught `ModuleNotFoundError: No module named 'scipy'`.
- **Root Cause**: A newly added preprocessing dependency was omitted from the production `Dockerfile`.
- **Resolution**: Updated `pyproject.toml` with `scipy` and rebuilt the container image.

### Investigation 2: Airflow Training Pipeline Task Failure (Quality Gate Rejection)

- **Observed Symptom**: Training DAG ended with status `failed`.
- **Agent Diagnosis**:
  1. Ran `airflow_get_failed_tasks(dag_id='training_pipeline')` pinpointing task `quality_gate`.
  2. Extracted task execution logs via `airflow_get_task_logs`.
  3. Discovered that evaluated test accuracy was `0.784`, below the mandatory threshold `0.850`.
- **Root Cause**: Experimental hyperparameter tuning (`max_depth=1`) caused severe underfitting.
- **Resolution**: Reverted hyperparameters and re-executed the pipeline, successfully clearing the quality gate.

---

## 9. Automated Testing & Verification

Run the entire automated test suite:

```bash
pytest tests/ -v --tb=short
```

The test suite covers:

- **Mocked External Tools**: Unit tests with mocked responses for MLflow, Airflow, Kubernetes, and DVC.
- **Safety Enforcement**: Proves that requests to delete/trigger operations require explicit confirmation.
- **Namespace Guardrails**: Proves that access to unauthorized namespaces (`kube-system`, etc.) is blocked.
- **Non-Fabrication**: Validates that when services are offline, the agent reports operational uncertainty and never invents fake metrics or running pods.
- **Injected Failure Diagnostic Tests**: Tests simulating pod crash loops and failed DAG stages.

---

## 10. Known Limitations & Future Improvements

1. **Airflow Authentication**: Currently uses HTTP Basic Auth; upgrading to OAuth2 / Keycloak tokens is recommended for enterprise deployments.
2. **Dynamic Kubernetes Scaling**: Scale operations currently require operator CLI intervention; introducing canary deployment automation is a viable future enhancement.
3. **Automated Drift Detection**: Integrating EvidentlyAI with DVC tools would enable automated data drift alerts to trigger informed retraining pipelines.
