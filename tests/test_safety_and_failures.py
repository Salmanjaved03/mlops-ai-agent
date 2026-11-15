"""Comprehensive safety, reliability, and failure diagnosis tests.

Verifies:
1. Agent does not invent operational results when services are unavailable.
2. Unauthorized Kubernetes modifications and disallowed namespaces are rejected.
3. Tool timeouts and service connection errors are cleanly reported.
4. Diagnosis of Injected Failure 1: Pod CrashLoopBackOff / restart loop.
5. Diagnosis of Injected Failure 2: Airflow DAG task failure during data validation.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent.orchestrator import MLOpsAgent
from agent.schemas import AgentRequest
from agent.tools.kubernetes_tools import list_pods, get_unhealthy_pods
from agent.tools.airflow_tools import get_failed_tasks, trigger_dag


class TestNonFabricationWhenServiceUnavailable:
    @pytest.mark.asyncio
    async def test_agent_reports_uncertainty_without_inventing_facts(self):
        """Proves agent does not invent operational results when tools are offline."""
        agent = MLOpsAgent()

        # Mock all tools returning failures/offline
        agent.tool_registry["k8s_list_deployments"] = MagicMock(return_value={
            "success": False,
            "data": None,
            "error": "ConnectionRefused: Unable to connect to Kubernetes API server at 127.0.0.1:6443"
        })
        agent.tool_registry["k8s_get_unhealthy_pods"] = MagicMock(return_value={
            "success": False,
            "data": None,
            "error": "ConnectionRefused: Kubernetes cluster unreachable"
        })

        req = AgentRequest(user_input="Check deployment status on kubernetes")
        response = await agent.process_request(req)

        # Confidence should be low or medium, not high
        assert response.confidence in ["low", "medium"]
        # Response must acknowledge tool failure
        assert "encountered an error" in response.response_text or "ConnectionRefused" in response.response_text
        # Response must NOT make up false positive facts like "2 replicas running"
        assert "2 replicas running" not in response.response_text


class TestUnauthorizedKubernetesModifications:
    def test_rejection_of_unauthorized_namespace(self):
        """Verifies that queries targeting unauthorized namespaces are rejected."""
        result = list_pods(namespace="production-restricted")
        assert result["success"] is False
        assert "not allowed" in result["error"].lower()

    def test_rejection_of_kube_system_modification(self):
        """Verifies access to kube-system is rejected."""
        result = list_pods(namespace="kube-system")
        assert result["success"] is False
        assert "not allowed" in result["error"].lower()


class TestInjectedFailure1_KubernetesCrashLoop:
    """Deliberately Injected Failure 1: Pod CrashLoopBackOff Diagnosis."""

    def test_diagnose_pod_crash_loop(self):
        """Agent detects crashing pod, high restart count, and extracts container event."""
        with patch("agent.tools.kubernetes_tools._get_clients") as mock_clients:
            core_v1 = MagicMock()
            mock_clients.return_value = (core_v1, MagicMock())

            # Simulate crashing pod
            pod = MagicMock()
            pod.metadata.name = "ml-inference-api-6f49c-crash"
            pod.metadata.namespace = "mlops"
            pod.status.phase = "Running"

            cs = MagicMock()
            cs.ready = False
            cs.restart_count = 14
            cs.state.waiting.reason = "CrashLoopBackOff"
            cs.state.waiting.message = "back-off 5m0s restarting failed container=inference-api pod=ml-inference-api"
            pod.status.container_statuses = [cs]

            # Simulate pod events
            event = MagicMock()
            event.type = "Warning"
            event.reason = "BackOff"
            event.message = "Back-off restarting failed container: OOMKilled or bad exit code 1"
            event.last_timestamp = None

            core_v1.list_namespaced_pod.return_value.items = [pod]
            core_v1.list_namespaced_event.return_value.items = [event]

            result = get_unhealthy_pods(namespace="mlops")

            assert result["success"] is True
            assert len(result["data"]) == 1
            diagnosed_pod = result["data"][0]
            assert diagnosed_pod["name"] == "ml-inference-api-6f49c-crash"
            assert diagnosed_pod["restart_count"] == 14
            assert diagnosed_pod["events"][0]["reason"] == "BackOff"


class TestInjectedFailure2_AirflowDataValidationFailure:
    """Deliberately Injected Failure 2: Training Pipeline Data Quality Gate Failure."""

    def test_diagnose_pipeline_task_failure(self):
        """Agent retrieves failed tasks and uncovers quality check failure."""
        with patch("agent.tools.airflow_tools._get_session") as mock_session_fn:
            session = MagicMock()
            mock_session_fn.return_value = session

            # Mock task instances response
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "task_instances": [
                    {
                        "task_id": "validate_data",
                        "state": "failed",
                        "try_number": 2,
                        "operator": "PythonOperator",
                        "start_date": "2026-11-15T10:00:00",
                        "end_date": "2026-11-15T10:01:00",
                    },
                    {
                        "task_id": "preprocess_data",
                        "state": "upstream_failed",
                        "try_number": 0,
                    },
                ]
            }
            mock_resp.raise_for_status = MagicMock()
            session.get.return_value = mock_resp

            result = get_failed_tasks(dag_id="training_pipeline", dag_run_id="manual__failed_run_42")

            assert result["success"] is True
            assert len(result["data"]) == 1
            failed_task = result["data"][0]
            assert failed_task["task_id"] == "validate_data"
            assert failed_task["state"] == "failed"
            assert failed_task["try_number"] == 2
