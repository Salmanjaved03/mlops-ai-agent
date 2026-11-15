"""Kubernetes tools for the AI agent.

Provides read-only tools for inspecting Kubernetes cluster resources
such as pods, deployments, services, and events.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from kubernetes import client, config
from kubernetes.client.exceptions import ApiException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_NAMESPACE: str = os.environ.get("K8S_NAMESPACE", "mlops")
ALLOWED_NAMESPACES: list[str] = ["mlops", "default"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_clients() -> tuple[client.CoreV1Api, client.AppsV1Api]:
    """Load cluster config and return (CoreV1Api, AppsV1Api).

    Attempts in-cluster config first, then falls back to the local
    kubeconfig file.

    Raises:
        config.ConfigException: If no valid configuration is found.
    """
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    return client.CoreV1Api(), client.AppsV1Api()


def _validate_namespace(namespace: str | None) -> str:
    """Return a validated namespace string.

    Parameters
    ----------
    namespace:
        The namespace to validate.  Falls back to *DEFAULT_NAMESPACE* when
        ``None``.

    Returns
    -------
    str
        The validated namespace.

    Raises
    ------
    ValueError
        If *namespace* is not in :pydata:`ALLOWED_NAMESPACES`.
    """
    ns = namespace if namespace is not None else DEFAULT_NAMESPACE
    if ns not in ALLOWED_NAMESPACES:
        raise ValueError(
            f"Namespace '{ns}' is not allowed. "
            f"Allowed namespaces: {ALLOWED_NAMESPACES}"
        )
    return ns


def _isoformat(dt: datetime | None) -> str | None:
    """Convert a datetime to ISO-8601 string, or return ``None``."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _error_response(error: Exception) -> dict[str, Any]:
    """Build a standardised error response dict."""
    return {"success": False, "data": None, "error": str(error)}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def list_pods(namespace: str | None = None) -> dict[str, Any]:
    """List all pods in the given namespace.

    Parameters
    ----------
    namespace:
        Kubernetes namespace.  Defaults to *DEFAULT_NAMESPACE*.

    Returns
    -------
    dict
        Structured response with ``success``, ``data``, and ``error`` keys.
        ``data`` contains a list of pod summary dicts.
    """
    try:
        ns = _validate_namespace(namespace)
        core_v1, _ = _get_clients()

        pods = core_v1.list_namespaced_pod(namespace=ns)
        pod_list: list[dict[str, Any]] = []

        for pod in pods.items:
            ready_containers = 0
            total_containers = 0
            restart_count = 0

            if pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    total_containers += 1
                    if cs.ready:
                        ready_containers += 1
                    restart_count += cs.restart_count

            pod_list.append(
                {
                    "name": pod.metadata.name,
                    "status": pod.status.phase,
                    "ready_containers": f"{ready_containers}/{total_containers}",
                    "restart_count": restart_count,
                    "node": pod.spec.node_name,
                    "ip": pod.status.pod_ip,
                    "labels": pod.metadata.labels or {},
                    "creation_timestamp": _isoformat(
                        pod.metadata.creation_timestamp
                    ),
                }
            )

        logger.info("Listed %d pods in namespace '%s'", len(pod_list), ns)
        return {"success": True, "data": pod_list, "error": None}

    except ValueError as exc:
        logger.warning("Namespace validation failed: %s", exc)
        return _error_response(exc)
    except ApiException as exc:
        logger.error("Kubernetes API error listing pods: %s", exc.reason)
        return _error_response(exc)
    except config.ConfigException as exc:
        logger.error("Kubernetes config error: %s", exc)
        return _error_response(exc)


def get_pod_logs(
    pod_name: str,
    namespace: str | None = None,
    tail_lines: int = 100,
) -> dict[str, Any]:
    """Retrieve logs from a specific pod.

    Parameters
    ----------
    pod_name:
        Name of the pod.
    namespace:
        Kubernetes namespace.  Defaults to *DEFAULT_NAMESPACE*.
    tail_lines:
        Number of most-recent log lines to return.

    Returns
    -------
    dict
        Structured response with the log text in ``data``.
    """
    try:
        ns = _validate_namespace(namespace)
        core_v1, _ = _get_clients()

        log_text: str = core_v1.read_namespaced_pod_log(
            name=pod_name,
            namespace=ns,
            tail_lines=tail_lines,
        )

        logger.info(
            "Retrieved logs for pod '%s' in namespace '%s'", pod_name, ns
        )
        return {
            "success": True,
            "data": {"pod_name": pod_name, "logs": log_text},
            "error": None,
        }

    except ValueError as exc:
        logger.warning("Namespace validation failed: %s", exc)
        return _error_response(exc)
    except ApiException as exc:
        logger.error(
            "Kubernetes API error getting logs for pod '%s': %s",
            pod_name,
            exc.reason,
        )
        return _error_response(exc)
    except config.ConfigException as exc:
        logger.error("Kubernetes config error: %s", exc)
        return _error_response(exc)


def list_deployments(namespace: str | None = None) -> dict[str, Any]:
    """List all deployments in the given namespace.

    Parameters
    ----------
    namespace:
        Kubernetes namespace.  Defaults to *DEFAULT_NAMESPACE*.

    Returns
    -------
    dict
        Structured response with a list of deployment summary dicts.
    """
    try:
        ns = _validate_namespace(namespace)
        _, apps_v1 = _get_clients()

        deployments = apps_v1.list_namespaced_deployment(namespace=ns)
        dep_list: list[dict[str, Any]] = []

        for dep in deployments.items:
            container_images = [
                c.image for c in dep.spec.template.spec.containers
            ]
            dep_list.append(
                {
                    "name": dep.metadata.name,
                    "replicas": dep.spec.replicas,
                    "ready_replicas": dep.status.ready_replicas or 0,
                    "available_replicas": dep.status.available_replicas or 0,
                    "strategy": dep.spec.strategy.type if dep.spec.strategy else None,
                    "labels": dep.metadata.labels or {},
                    "container_images": container_images,
                }
            )

        logger.info(
            "Listed %d deployments in namespace '%s'", len(dep_list), ns
        )
        return {"success": True, "data": dep_list, "error": None}

    except ValueError as exc:
        logger.warning("Namespace validation failed: %s", exc)
        return _error_response(exc)
    except ApiException as exc:
        logger.error(
            "Kubernetes API error listing deployments: %s", exc.reason
        )
        return _error_response(exc)
    except config.ConfigException as exc:
        logger.error("Kubernetes config error: %s", exc)
        return _error_response(exc)


def list_services(namespace: str | None = None) -> dict[str, Any]:
    """List all services in the given namespace.

    Parameters
    ----------
    namespace:
        Kubernetes namespace.  Defaults to *DEFAULT_NAMESPACE*.

    Returns
    -------
    dict
        Structured response with a list of service summary dicts.
    """
    try:
        ns = _validate_namespace(namespace)
        core_v1, _ = _get_clients()

        services = core_v1.list_namespaced_service(namespace=ns)
        svc_list: list[dict[str, Any]] = []

        for svc in services.items:
            ports = []
            if svc.spec.ports:
                for p in svc.spec.ports:
                    ports.append(
                        {
                            "name": p.name,
                            "port": p.port,
                            "target_port": (
                                str(p.target_port) if p.target_port else None
                            ),
                            "protocol": p.protocol,
                            "node_port": p.node_port,
                        }
                    )

            svc_list.append(
                {
                    "name": svc.metadata.name,
                    "type": svc.spec.type,
                    "cluster_ip": svc.spec.cluster_ip,
                    "ports": ports,
                    "selector": svc.spec.selector or {},
                }
            )

        logger.info(
            "Listed %d services in namespace '%s'", len(svc_list), ns
        )
        return {"success": True, "data": svc_list, "error": None}

    except ValueError as exc:
        logger.warning("Namespace validation failed: %s", exc)
        return _error_response(exc)
    except ApiException as exc:
        logger.error(
            "Kubernetes API error listing services: %s", exc.reason
        )
        return _error_response(exc)
    except config.ConfigException as exc:
        logger.error("Kubernetes config error: %s", exc)
        return _error_response(exc)


def get_pod_status(
    pod_name: str, namespace: str | None = None
) -> dict[str, Any]:
    """Get detailed status for a specific pod.

    Includes pod conditions, container statuses, and recent events.

    Parameters
    ----------
    pod_name:
        Name of the pod.
    namespace:
        Kubernetes namespace.  Defaults to *DEFAULT_NAMESPACE*.

    Returns
    -------
    dict
        Structured response with detailed pod status information.
    """
    try:
        ns = _validate_namespace(namespace)
        core_v1, _ = _get_clients()

        pod = core_v1.read_namespaced_pod(name=pod_name, namespace=ns)

        # --- conditions ---
        conditions: list[dict[str, Any]] = []
        if pod.status.conditions:
            for cond in pod.status.conditions:
                conditions.append(
                    {
                        "type": cond.type,
                        "status": cond.status,
                        "reason": cond.reason,
                        "message": cond.message,
                        "last_transition_time": _isoformat(
                            cond.last_transition_time
                        ),
                    }
                )

        # --- container statuses ---
        container_statuses: list[dict[str, Any]] = []
        if pod.status.container_statuses:
            for cs in pod.status.container_statuses:
                state_info: dict[str, Any] = {}
                if cs.state:
                    if cs.state.running:
                        state_info = {
                            "state": "running",
                            "started_at": _isoformat(
                                cs.state.running.started_at
                            ),
                        }
                    elif cs.state.waiting:
                        state_info = {
                            "state": "waiting",
                            "reason": cs.state.waiting.reason,
                            "message": cs.state.waiting.message,
                        }
                    elif cs.state.terminated:
                        state_info = {
                            "state": "terminated",
                            "reason": cs.state.terminated.reason,
                            "exit_code": cs.state.terminated.exit_code,
                            "finished_at": _isoformat(
                                cs.state.terminated.finished_at
                            ),
                        }

                container_statuses.append(
                    {
                        "name": cs.name,
                        "ready": cs.ready,
                        "restart_count": cs.restart_count,
                        "image": cs.image,
                        **state_info,
                    }
                )

        # --- events ---
        events = core_v1.list_namespaced_event(
            namespace=ns,
            field_selector=f"involvedObject.name={pod_name}",
        )
        event_list: list[dict[str, Any]] = []
        for evt in events.items:
            event_list.append(
                {
                    "type": evt.type,
                    "reason": evt.reason,
                    "message": evt.message,
                    "timestamp": _isoformat(evt.last_timestamp),
                }
            )

        data = {
            "pod_name": pod_name,
            "namespace": ns,
            "phase": pod.status.phase,
            "conditions": conditions,
            "container_statuses": container_statuses,
            "events": event_list,
        }

        logger.info(
            "Retrieved detailed status for pod '%s' in namespace '%s'",
            pod_name,
            ns,
        )
        return {"success": True, "data": data, "error": None}

    except ValueError as exc:
        logger.warning("Namespace validation failed: %s", exc)
        return _error_response(exc)
    except ApiException as exc:
        logger.error(
            "Kubernetes API error getting status for pod '%s': %s",
            pod_name,
            exc.reason,
        )
        return _error_response(exc)
    except config.ConfigException as exc:
        logger.error("Kubernetes config error: %s", exc)
        return _error_response(exc)


def get_unhealthy_pods(namespace: str | None = None) -> dict[str, Any]:
    """Find pods that are not Ready or have high restart counts (>3).

    Parameters
    ----------
    namespace:
        Kubernetes namespace.  Defaults to *DEFAULT_NAMESPACE*.

    Returns
    -------
    dict
        Structured response with a list of unhealthy pod details and
        their recent events.
    """
    try:
        ns = _validate_namespace(namespace)
        core_v1, _ = _get_clients()

        pods = core_v1.list_namespaced_pod(namespace=ns)
        unhealthy: list[dict[str, Any]] = []

        for pod in pods.items:
            is_unhealthy = False
            total_restarts = 0
            ready_containers = 0
            total_containers = 0

            if pod.status.container_statuses:
                for cs in pod.status.container_statuses:
                    total_containers += 1
                    total_restarts += cs.restart_count
                    if cs.ready:
                        ready_containers += 1
                    else:
                        is_unhealthy = True

            if total_restarts > 3:
                is_unhealthy = True

            if pod.status.phase not in ("Running", "Succeeded"):
                is_unhealthy = True

            if not is_unhealthy:
                continue

            # Fetch events for this unhealthy pod
            events = core_v1.list_namespaced_event(
                namespace=ns,
                field_selector=f"involvedObject.name={pod.metadata.name}",
            )
            event_list: list[dict[str, Any]] = []
            for evt in events.items:
                event_list.append(
                    {
                        "type": evt.type,
                        "reason": evt.reason,
                        "message": evt.message,
                        "timestamp": _isoformat(evt.last_timestamp),
                    }
                )

            unhealthy.append(
                {
                    "name": pod.metadata.name,
                    "status": pod.status.phase,
                    "ready_containers": f"{ready_containers}/{total_containers}",
                    "restart_count": total_restarts,
                    "node": pod.spec.node_name,
                    "ip": pod.status.pod_ip,
                    "labels": pod.metadata.labels or {},
                    "creation_timestamp": _isoformat(
                        pod.metadata.creation_timestamp
                    ),
                    "events": event_list,
                }
            )

        logger.info(
            "Found %d unhealthy pods in namespace '%s'", len(unhealthy), ns
        )
        return {"success": True, "data": unhealthy, "error": None}

    except ValueError as exc:
        logger.warning("Namespace validation failed: %s", exc)
        return _error_response(exc)
    except ApiException as exc:
        logger.error(
            "Kubernetes API error finding unhealthy pods: %s", exc.reason
        )
        return _error_response(exc)
    except config.ConfigException as exc:
        logger.error("Kubernetes config error: %s", exc)
        return _error_response(exc)


def get_deployment_version(
    deployment_name: str, namespace: str | None = None
) -> dict[str, Any]:
    """Get the container image version/tag for a deployment.

    Parameters
    ----------
    deployment_name:
        Name of the deployment.
    namespace:
        Kubernetes namespace.  Defaults to *DEFAULT_NAMESPACE*.

    Returns
    -------
    dict
        Structured response with image version information.
    """
    try:
        ns = _validate_namespace(namespace)
        _, apps_v1 = _get_clients()

        dep = apps_v1.read_namespaced_deployment(
            name=deployment_name, namespace=ns
        )

        containers: list[dict[str, str | None]] = []
        for c in dep.spec.template.spec.containers:
            image = c.image or ""
            # Extract tag after the last ':'
            if ":" in image:
                tag = image.rsplit(":", 1)[1]
            else:
                tag = "latest"

            containers.append(
                {
                    "container_name": c.name,
                    "image": image,
                    "tag": tag,
                }
            )

        data = {
            "deployment_name": deployment_name,
            "namespace": ns,
            "containers": containers,
        }

        logger.info(
            "Retrieved version info for deployment '%s' in namespace '%s'",
            deployment_name,
            ns,
        )
        return {"success": True, "data": data, "error": None}

    except ValueError as exc:
        logger.warning("Namespace validation failed: %s", exc)
        return _error_response(exc)
    except ApiException as exc:
        logger.error(
            "Kubernetes API error getting deployment version for '%s': %s",
            deployment_name,
            exc.reason,
        )
        return _error_response(exc)
    except config.ConfigException as exc:
        logger.error("Kubernetes config error: %s", exc)
        return _error_response(exc)


def get_pod_events(
    pod_name: str, namespace: str | None = None
) -> dict[str, Any]:
    """Get events for a specific pod.

    Parameters
    ----------
    pod_name:
        Name of the pod.
    namespace:
        Kubernetes namespace.  Defaults to *DEFAULT_NAMESPACE*.

    Returns
    -------
    dict
        Structured response with a list of event dicts.
    """
    try:
        ns = _validate_namespace(namespace)
        core_v1, _ = _get_clients()

        events = core_v1.list_namespaced_event(
            namespace=ns,
            field_selector=f"involvedObject.name={pod_name}",
        )

        event_list: list[dict[str, Any]] = []
        for evt in events.items:
            event_list.append(
                {
                    "type": evt.type,
                    "reason": evt.reason,
                    "message": evt.message,
                    "timestamp": _isoformat(evt.last_timestamp),
                }
            )

        logger.info(
            "Retrieved %d events for pod '%s' in namespace '%s'",
            len(event_list),
            pod_name,
            ns,
        )
        return {
            "success": True,
            "data": {"pod_name": pod_name, "events": event_list},
            "error": None,
        }

    except ValueError as exc:
        logger.warning("Namespace validation failed: %s", exc)
        return _error_response(exc)
    except ApiException as exc:
        logger.error(
            "Kubernetes API error getting events for pod '%s': %s",
            pod_name,
            exc.reason,
        )
        return _error_response(exc)
    except config.ConfigException as exc:
        logger.error("Kubernetes config error: %s", exc)
        return _error_response(exc)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, Any] = {
    "k8s_list_pods": list_pods,
    "k8s_get_pod_logs": get_pod_logs,
    "k8s_list_deployments": list_deployments,
    "k8s_list_services": list_services,
    "k8s_get_pod_status": get_pod_status,
    "k8s_get_unhealthy_pods": get_unhealthy_pods,
    "k8s_get_deployment_version": get_deployment_version,
    "k8s_get_pod_events": get_pod_events,
}

DESTRUCTIVE_TOOLS: list[str] = []  # No destructive k8s tools - read only operations
