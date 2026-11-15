"""Tests for Airflow tools module."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
import requests

from agent.tools.airflow_tools import (
    list_dags,
    get_dag_runs,
    trigger_dag,
    get_dag_run_status,
    get_task_instances,
    get_failed_tasks,
    get_task_logs,
    TOOLS,
    APPROVED_DAGS,
)


class TestAirflowDAGs:
    def test_list_dags_success(self, mock_requests_session):
        """Test listing DAGs via Airflow API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "dags": [
                {
                    "dag_id": "training_pipeline",
                    "description": "ML Training Pipeline",
                    "is_paused": False,
                    "schedule_interval": None,
                    "tags": [{"name": "training"}],
                }
            ],
            "total_entries": 1,
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests_session.get.return_value = mock_resp

        result = list_dags()
        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["dag_id"] == "training_pipeline"

    def test_list_dags_connection_error(self, mock_requests_session):
        """Test handling Airflow unreachable error."""
        mock_requests_session.get.side_effect = requests.exceptions.ConnectionError("Airflow down")
        result = list_dags()
        assert result["success"] is False
        assert "Connection error" in result["error"]


class TestAirflowTriggerSafety:
    def test_trigger_unapproved_dag_rejected(self, mock_requests_session):
        """Test that unapproved DAG triggering is blocked by safety guard."""
        result = trigger_dag("drop_production_database")
        assert result["success"] is False
        assert "not approved" in result["error"].lower()
        # Ensure HTTP POST was NEVER called
        mock_requests_session.post.assert_not_called()

    def test_trigger_approved_dag_success(self, mock_requests_session):
        """Test triggering an approved training DAG."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "dag_id": "training_pipeline",
            "dag_run_id": "manual__2026-11-15T12:00:00",
            "state": "queued",
            "execution_date": "2026-11-15T12:00:00+00:00",
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests_session.post.return_value = mock_resp

        result = trigger_dag("training_pipeline")
        assert result["success"] is True
        assert result["data"]["state"] == "queued"


class TestAirflowFailedTasks:
    def test_get_failed_tasks(self, mock_requests_session):
        """Test identifying failed tasks in a DAG run."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "task_instances": [
                {"task_id": "validate_data", "state": "success", "try_number": 1},
                {"task_id": "preprocess_data", "state": "success", "try_number": 1},
                {"task_id": "train_model", "state": "failed", "try_number": 2},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests_session.get.return_value = mock_resp

        result = get_failed_tasks("training_pipeline", "manual__1")
        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["task_id"] == "train_model"
        assert result["data"][0]["state"] == "failed"


class TestAirflowToolRegistry:
    def test_all_tools_registered(self):
        expected = [
            "airflow_list_dags",
            "airflow_get_dag_runs",
            "airflow_trigger_dag",
            "airflow_get_dag_run_status",
            "airflow_get_task_instances",
            "airflow_get_failed_tasks",
            "airflow_get_task_logs",
        ]
        for name in expected:
            assert name in TOOLS
            assert callable(TOOLS[name])
