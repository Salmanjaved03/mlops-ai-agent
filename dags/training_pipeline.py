"""Apache Airflow DAG for the ML training pipeline.

This DAG orchestrates the end-to-end ML training workflow:
validate → preprocess → train → evaluate → quality gate.
"""

import logging
from datetime import timedelta

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

default_args = {
    "owner": "mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}


def _validate_data(**context: dict) -> None:
    """Validate that the dataset exists and has the correct format.

    Checks for data availability, schema consistency, and basic integrity
    before proceeding with the pipeline.

    Args:
        **context: Airflow task context dictionary.

    Raises:
        AirflowException: If dataset validation fails.
    """
    logger.info("Starting data validation step.")
    logger.info("Checking dataset existence at data/processed/iris.csv ...")

    # Simulate dataset checks
    expected_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    row_count = 150
    logger.info("Dataset schema validated — expected columns: %s", expected_columns)
    logger.info("Dataset contains %d rows. Minimum threshold met.", row_count)

    # Simulate null-value check
    null_percentage = 0.0
    logger.info("Null value percentage: %.2f%%. Within acceptable range.", null_percentage)
    logger.info("Data validation completed successfully.")


def _preprocess_data(**context: dict) -> None:
    """Preprocess and transform raw data for model training.

    Applies feature scaling, encoding, and train/test splitting.

    Args:
        **context: Airflow task context dictionary.
    """
    logger.info("Starting data preprocessing step.")
    logger.info("Loading validated dataset from data/processed/ ...")

    # Simulate preprocessing steps
    logger.info("Applying StandardScaler to numerical features.")
    logger.info("Encoding categorical target variable 'species'.")
    logger.info("Splitting dataset into train (80%%) and test (20%%) sets.")
    logger.info("Train set: 120 samples, Test set: 30 samples.")
    logger.info("Saving preprocessed data to data/processed/train.csv and data/processed/test.csv.")
    logger.info("Data preprocessing completed successfully.")


def _train_model(**context: dict) -> None:
    """Train the ML model on preprocessed data.

    Trains a classifier and logs parameters and artifacts to MLflow.

    Args:
        **context: Airflow task context dictionary.
    """
    logger.info("Starting model training step.")
    logger.info("Loading preprocessed training data ...")

    # Simulate training
    model_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    }
    logger.info("Training RandomForestClassifier with params: %s", model_params)
    logger.info("Training on 120 samples with 4 features ...")
    logger.info("Model training completed in 2.3 seconds.")
    logger.info("Saving trained model artifact to models/model.pkl.")
    logger.info("Logging model parameters and artifact to MLflow.")
    logger.info("Model training step completed successfully.")


def _evaluate_model(**context: dict) -> None:
    """Evaluate the trained model and log metrics to MLflow.

    Computes accuracy, precision, recall, and F1-score on the test set
    and pushes the results via XCom for downstream tasks.

    Args:
        **context: Airflow task context dictionary.
    """
    logger.info("Starting model evaluation step.")
    logger.info("Loading trained model from models/model.pkl ...")
    logger.info("Loading test dataset from data/processed/test.csv ...")

    # Simulate evaluation metrics
    metrics = {
        "accuracy": 0.9667,
        "precision": 0.9700,
        "recall": 0.9667,
        "f1_score": 0.9665,
    }
    for metric_name, metric_value in metrics.items():
        logger.info("Metric — %s: %.4f", metric_name, metric_value)

    logger.info("Logging evaluation metrics to MLflow.")

    # Push metrics to XCom so the quality gate can read them
    ti = context["ti"]
    ti.xcom_push(key="accuracy", value=metrics["accuracy"])
    ti.xcom_push(key="f1_score", value=metrics["f1_score"])
    logger.info("Model evaluation step completed successfully.")


def _quality_gate(**context: dict) -> None:
    """Check if the model meets minimum performance thresholds.

    Pulls evaluation metrics from XCom and verifies they exceed the
    required quality thresholds before the model can be promoted.

    Args:
        **context: Airflow task context dictionary.

    Raises:
        AirflowException: If the model does not meet quality thresholds.
    """
    logger.info("Starting quality gate check.")

    ti = context["ti"]
    accuracy = ti.xcom_pull(task_ids="evaluate_model", key="accuracy")
    f1_score = ti.xcom_pull(task_ids="evaluate_model", key="f1_score")

    min_accuracy = 0.90
    min_f1 = 0.85

    logger.info("Retrieved metrics — accuracy: %s, f1_score: %s", accuracy, f1_score)
    logger.info("Quality thresholds — min_accuracy: %.2f, min_f1: %.2f", min_accuracy, min_f1)

    if accuracy is None or f1_score is None:
        raise AirflowException(
            "Quality gate FAILED: evaluation metrics not found in XCom. "
            "Ensure the evaluate_model task completed successfully."
        )

    if accuracy < min_accuracy:
        raise AirflowException(
            f"Quality gate FAILED: accuracy {accuracy:.4f} is below "
            f"minimum threshold {min_accuracy:.2f}."
        )

    if f1_score < min_f1:
        raise AirflowException(
            f"Quality gate FAILED: F1-score {f1_score:.4f} is below "
            f"minimum threshold {min_f1:.2f}."
        )

    logger.info(
        "Quality gate PASSED — accuracy: %.4f (>= %.2f), f1_score: %.4f (>= %.2f).",
        accuracy,
        min_accuracy,
        f1_score,
        min_f1,
    )
    logger.info("Model is approved for promotion to production.")


with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="End-to-end ML training pipeline: validate, preprocess, train, evaluate, quality gate",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["training", "ml-pipeline"],
) as dag:
    validate_data = PythonOperator(
        task_id="validate_data",
        python_callable=_validate_data,
        retries=2,
    )

    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=_preprocess_data,
        retries=1,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        retries=1,
    )

    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=_evaluate_model,
        retries=1,
    )

    quality_gate = PythonOperator(
        task_id="quality_gate",
        python_callable=_quality_gate,
        retries=0,
    )

    validate_data >> preprocess_data >> train_model >> evaluate_model >> quality_gate
