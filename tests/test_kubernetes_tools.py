"""Tests for Kubernetes tools module."""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from kubernetes.client.exceptions import ApiException

from agent.tools.kubernetes_tools import (
    list_pods,
    get_pod_logs,
    list_deployments,
    list_services,
    get_pod_status,
    get_unhealthy_pods,
    get_deployment_version,
    get_pod_events,
    TOOLS,
    ALLOWED_NAMESPACES,
)


class TestK8sNamespaceSafety:
    def test_unauthorized_namespace_rejected(self, mock_k8s_clients):
        """Test that operations targeting disallowed namespaces are rejected."""
        result = list_pods(namespace="kube-system")
        assert result["success"] is False
        assert "not allowed" in result["error"].lower()

    def test_allowed_namespace_accepted(self, mock_k8s_clients):
        """Test that operations within allowed namespaces succeed."""
        core_v1, _ = mock_k8s_clients
        mock_pod = MagicMock()
        mock_pod.metadata.name = "ml-inference-api-7b89d"
        mock_pod.metadata.namespace = "mlops"
        mock_pod.metadata.labels = {"app": "ml-inference-api"}
        mock_pod.metadata.creation_timestamp = None
        mock_pod.status.phase = "Running"
        mock_pod.status.pod_ip = "10.244.0.5"
        mock_pod.spec.node_name = "node-1"

        cs = MagicMock()
        cs.ready = True
        cs.restart_count = 0
        mock_pod.status.container_statuses = [cs]

        core_v1.list_namespaced_pod.return_value.items = [mock_pod]

        result = list_pods(namespace="mlops")
        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["name"] == "ml-inference-api-7b89d"


class TestK8sDeployments:
    def test_list_deployments_success(self, mock_k8s_clients):
        """Test listing deployments in namespace."""
        _, apps_v1 = mock_k8s_clients
        dep = MagicMock()
        dep.metadata.name = "ml-inference-api"
        dep.metadata.namespace = "mlops"
        dep.metadata.labels = {"app": "ml-inference-api"}
        dep.spec.replicas = 2
        dep.status.ready_replicas = 2
        dep.status.available_replicas = 2
        dep.spec.strategy.type = "RollingUpdate"

        container = MagicMock()
        container.image = "ml-inference-api:v1.2.0"
        dep.spec.template.spec.containers = [container]

        apps_v1.list_namespaced_deployment.return_value.items = [dep]

        result = list_deployments(namespace="mlops")
        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["ready_replicas"] == 2
        assert "ml-inference-api:v1.2.0" in result["data"][0]["container_images"]


class TestK8sDiagnostics:
    def test_get_unhealthy_pods_detects_crash(self, mock_k8s_clients):
        """Test detection and reporting of unhealthy pod with restart events."""
        core_v1, _ = mock_k8s_clients

        pod = MagicMock()
        pod.metadata.name = "ml-inference-api-crashing"
        pod.metadata.namespace = "mlops"
        pod.status.phase = "Running"

        cs = MagicMock()
        cs.ready = False
        cs.restart_count = 5
        pod.status.container_statuses = [cs]

        event = MagicMock()
        event.type = "Warning"
        event.reason = "BackOff"
        event.message = "Back-off restarting failed container"
        event.last_timestamp = None

        core_v1.list_namespaced_pod.return_value.items = [pod]
        core_v1.list_namespaced_event.return_value.items = [event]

        result = get_unhealthy_pods(namespace="mlops")
        assert result["success"] is True
        assert len(result["data"]) == 1
        assert result["data"][0]["name"] == "ml-inference-api-crashing"
        assert result["data"][0]["restart_count"] == 5


class TestK8sApiExceptionHandling:
    def test_api_exception_handling(self, mock_k8s_clients):
        """Test handling Kubernetes API errors gracefully."""
        core_v1, _ = mock_k8s_clients
        core_v1.list_namespaced_pod.side_effect = ApiException(status=500, reason="Internal Server Error")

        result = list_pods(namespace="mlops")
        assert result["success"] is False
        assert "500" in result["error"] or "Internal Server Error" in result["error"]


class TestK8sToolRegistry:
    def test_all_tools_registered(self):
        expected = [
            "k8s_list_pods",
            "k8s_get_pod_logs",
            "k8s_list_deployments",
            "k8s_list_services",
            "k8s_get_pod_status",
            "k8s_get_unhealthy_pods",
            "k8s_get_deployment_version",
            "k8s_get_pod_events",
        ]
        for name in expected:
            assert name in TOOLS
            assert callable(TOOLS[name])
