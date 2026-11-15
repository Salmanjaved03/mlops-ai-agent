"""Tests for MLflow tools module."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
try:
    from mlflow.exceptions import MlflowException
except ImportError:
    MlflowException = Exception  # type: ignore

from agent.tools.mlflow_tools import (
    list_experiments,
    list_runs,
    get_run_details,
    get_best_run,
    compare_runs,
    get_model_versions,
    get_model_info,
    TOOLS,
)


class TestMlflowExperiments:
    def test_list_experiments_success(self, mock_mlflow_client):
        """Test listing experiments successfully."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "0"
        mock_exp.name = "iris-classifier"
        mock_exp.artifact_location = "mlflow-artifacts:/0"
        mock_exp.lifecycle_stage = "active"
        mock_exp.tags = {"env": "prod"}
        mock_mlflow_client.search_experiments.return_value = [mock_exp]

        result = list_experiments()
        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["name"] == "iris-classifier"

    def test_list_experiments_connection_error(self, mock_mlflow_client):
        """Test handling connection failure to MLflow server."""
        mock_mlflow_client.search_experiments.side_effect = Exception("Connection refused to MLflow")
        result = list_experiments()
        assert result["success"] is False
        assert "Connection error" in result["error"] or "refused" in result["error"]


class TestMlflowBestRun:
    def test_get_best_run_success(self, mock_mlflow_client):
        """Test finding the best run and verifying advisory note."""
        mock_run = MagicMock()
        mock_run.info.run_id = "run-abc-123"
        mock_run.info.experiment_id = "0"
        mock_run.info.status = "FINISHED"
        mock_run.info.start_time = 1690000000000
        mock_run.info.end_time = 1690000005000
        mock_run.info.artifact_uri = "s3://mlflow/run-abc"
        mock_run.data.metrics = {"f1_score": 0.965, "accuracy": 0.97}
        mock_run.data.params = {"n_estimators": "100", "max_depth": "5"}
        mock_run.data.tags = {"git_commit": "a4c92e1"}

        mock_exp = MagicMock()
        mock_exp.experiment_id = "0"
        mock_exp.name = "iris-classifier"
        mock_mlflow_client.get_experiment_by_name.return_value = mock_exp
        mock_mlflow_client.search_runs.return_value = [mock_run]

        result = get_best_run(metric="f1_score")
        assert result["success"] is True
        assert result["data"]["run_id"] == "run-abc-123"
        assert result["data"]["metrics"]["f1_score"] == 0.965
        # Must contain advisory note
        assert "note" in result["data"]
        assert "production" in result["data"]["note"].lower()

    def test_get_best_run_no_runs_found(self, mock_mlflow_client):
        """Test behavior when no runs match."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "0"
        mock_exp.name = "iris-classifier"
        mock_mlflow_client.get_experiment_by_name.return_value = mock_exp
        mock_mlflow_client.search_runs.return_value = []
        result = get_best_run(metric="f1_score")
        assert result["success"] is False
        assert "No runs found" in result["error"]


class TestMlflowCompareRuns:
    def test_compare_runs_success(self, mock_mlflow_client):
        """Test comparing two MLflow runs."""
        r1 = MagicMock()
        r1.info.run_id = "run-1"
        r1.info.status = "FINISHED"
        r1.info.start_time = 1690000000000
        r1.info.end_time = 1690000005000
        r1.data.metrics = {"f1_score": 0.92}
        r1.data.params = {"n_estimators": "50"}

        r2 = MagicMock()
        r2.info.run_id = "run-2"
        r2.info.status = "FINISHED"
        r2.info.start_time = 1690000000000
        r2.info.end_time = 1690000005000
        r2.data.metrics = {"f1_score": 0.96}
        r2.data.params = {"n_estimators": "100"}

        mock_mlflow_client.get_run.side_effect = [r1, r2]

        result = compare_runs(["run-1", "run-2"])
        assert result["success"] is True
        assert len(result["data"]) == 2


class TestMlflowToolRegistry:
    def test_all_tools_registered(self):
        expected = [
            "mlflow_list_experiments",
            "mlflow_list_runs",
            "mlflow_get_run",
            "mlflow_get_best_run",
            "mlflow_compare_runs",
            "mlflow_get_model_versions",
            "mlflow_get_model_info",
        ]
        for name in expected:
            assert name in TOOLS
            assert callable(TOOLS[name])
