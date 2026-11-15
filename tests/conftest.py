"""Shared pytest fixtures for MLOps AI Agent test suite."""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch

from agent.schemas import AgentRequest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("AIRFLOW_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("K8S_NAMESPACE", "mlops")


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run for DVC tool tests."""
    with patch("subprocess.run") as mock:
        yield mock


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for experiment tracking tests."""
    with patch("agent.tools.mlflow_tools._get_client") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_requests_session():
    """Mock requests.Session for Airflow tool tests."""
    with patch("agent.tools.airflow_tools._get_session") as mock:
        session = MagicMock()
        mock.return_value = session
        yield session


@pytest.fixture
def mock_k8s_clients():
    """Mock Kubernetes API clients."""
    with patch("agent.tools.kubernetes_tools._get_clients") as mock:
        core_v1 = MagicMock()
        apps_v1 = MagicMock()
        mock.return_value = (core_v1, apps_v1)
        yield core_v1, apps_v1


@pytest.fixture
def sample_agent_request():
    """Create a sample AgentRequest."""
    return AgentRequest(user_input="Find the best model from our recent experiments")
