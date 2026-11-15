"""Model training script with MLflow tracking."""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "iris-classifier")
MODEL_DIR = Path("models")
METRICS_FILE = Path("metrics.json")


def load_data():
    """Load and split the iris dataset."""
    logger.info("Loading iris dataset...")
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, data.target_names


def train_model(X_train, y_train, n_estimators=100, max_depth=5, random_state=42):
    """Train a Random Forest classifier."""
    logger.info(f"Training RandomForest (n_estimators={n_estimators}, max_depth={max_depth})...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "f1_score": round(f1_score(y_test, predictions, average="weighted"), 4),
        "precision": round(precision_score(y_test, predictions, average="weighted"), 4),
        "recall": round(recall_score(y_test, predictions, average="weighted"), 4),
    }
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def main():
    """Run the full training pipeline with MLflow tracking."""
    # Hyperparameters
    n_estimators = int(os.getenv("N_ESTIMATORS", "100"))
    max_depth = int(os.getenv("MAX_DEPTH", "5"))
    random_state = int(os.getenv("RANDOM_STATE", "42"))

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("dataset", "iris")

        # Load data
        X_train, X_test, y_train, y_test, target_names = load_data()
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        # Train
        model = train_model(X_train, y_train, n_estimators, max_depth, random_state)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODEL_DIR / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {model_path}")

        # Save metrics
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {METRICS_FILE}")

        mlflow.log_artifact(str(METRICS_FILE))

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run completed: {run_id}")
        logger.info(f"Metrics: {metrics}")

    return metrics


if __name__ == "__main__":
    main()
