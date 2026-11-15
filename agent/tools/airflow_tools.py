"""Airflow REST API tools for the AI agent.

Provides functions to interact with Apache Airflow via its REST API,
enabling the agent to list DAGs, trigger runs, monitor status, and
diagnose failures.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------
AIRFLOW_BASE_URL: str = os.getenv("AIRFLOW_BASE_URL", "http://localhost:8080")
AIRFLOW_USERNAME: str = os.getenv("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD: str = os.getenv("AIRFLOW_PASSWORD", "airflow")

# ---------------------------------------------------------------------------
# Safety guardrails – only these DAGs may be triggered by the agent.
# ---------------------------------------------------------------------------
APPROVED_DAGS: list[str] = [
    "training_pipeline",
    "training_retrain",
    "training_evaluation",
]

_REQUEST_TIMEOUT: int = int(os.getenv("AIRFLOW_TIMEOUT", "3"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_session() -> requests.Session:
    """Return a :class:`requests.Session` configured with Basic-Auth."""
    session = requests.Session()
    session.auth = HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD)
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    return session


def _api_url(endpoint: str) -> str:
    """Build a full Airflow REST API URL for *endpoint*.

    Parameters
    ----------
    endpoint:
        The API path **without** the leading ``/api/v1`` prefix,
        e.g. ``"/dags"`` or ``"/dags/{dag_id}/dagRuns"``.

    Returns
    -------
    str
        Fully-qualified URL.
    """
    base = AIRFLOW_BASE_URL.rstrip("/")
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{base}/api/v1{endpoint}"


def _handle_error(exc: Exception, context: str) -> dict[str, Any]:
    """Return a standardised error response dict and log the exception."""
    logger.error("%s: %s", context, exc)
    return {"success": False, "data": None, "error": f"{context}: {exc}"}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def list_dags() -> dict[str, Any]:
    """List all DAGs registered in Airflow.

    Returns
    -------
    dict
        ``success`` – bool, ``data`` – list of DAG summaries,
        ``error`` – error message or ``None``.
    """
    try:
        session = _get_session()
        response = session.get(_api_url("/dags"), timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        raw_dags = response.json().get("dags", [])
        dags = [
            {
                "dag_id": d.get("dag_id"),
                "description": d.get("description"),
                "is_paused": d.get("is_paused"),
                "schedule_interval": d.get("schedule_interval"),
                "tags": [t.get("name") for t in d.get("tags", [])],
            }
            for d in raw_dags
        ]
        logger.info("Listed %d DAGs from Airflow.", len(dags))
        return {"success": True, "data": dags, "error": None}

    except requests.exceptions.ConnectionError as exc:
        return _handle_error(exc, "Connection error while listing DAGs")
    except requests.exceptions.Timeout as exc:
        return _handle_error(exc, "Timeout while listing DAGs")
    except requests.exceptions.HTTPError as exc:
        return _handle_error(exc, "HTTP error while listing DAGs")
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc, "Unexpected error while listing DAGs")


def get_dag_runs(dag_id: str, limit: int = 10) -> dict[str, Any]:
    """Fetch recent DAG runs for *dag_id*.

    Parameters
    ----------
    dag_id:
        Identifier of the DAG.
    limit:
        Maximum number of runs to return (default ``10``).

    Returns
    -------
    dict
        ``success`` – bool, ``data`` – list of run summaries,
        ``error`` – error message or ``None``.
    """
    try:
        session = _get_session()
        url = _api_url(f"/dags/{dag_id}/dagRuns")
        response = session.get(
            url,
            params={"limit": limit, "order_by": "-execution_date"},
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        raw_runs = response.json().get("dag_runs", [])
        runs = [
            {
                "dag_run_id": r.get("dag_run_id"),
                "state": r.get("state"),
                "execution_date": r.get("execution_date"),
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
            }
            for r in raw_runs
        ]
        logger.info("Fetched %d runs for DAG '%s'.", len(runs), dag_id)
        return {"success": True, "data": runs, "error": None}

    except requests.exceptions.ConnectionError as exc:
        return _handle_error(exc, f"Connection error fetching runs for DAG '{dag_id}'")
    except requests.exceptions.Timeout as exc:
        return _handle_error(exc, f"Timeout fetching runs for DAG '{dag_id}'")
    except requests.exceptions.HTTPError as exc:
        return _handle_error(exc, f"HTTP error fetching runs for DAG '{dag_id}'")
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc, f"Unexpected error fetching runs for DAG '{dag_id}'")


def trigger_dag(dag_id: str, conf: dict[str, Any] | None = None) -> dict[str, Any]:
    """Trigger a new DAG run for *dag_id*.

    Only DAGs whose id starts with ``"training_"`` **or** that appear in
    :data:`APPROVED_DAGS` are allowed to be triggered.

    Parameters
    ----------
    dag_id:
        Identifier of the DAG to trigger.
    conf:
        Optional configuration dict passed to the DAG run.

    Returns
    -------
    dict
        ``success`` – bool, ``data`` – triggered run info,
        ``error`` – error message or ``None``.
    """
    # --- Safety check ---
    if not dag_id.startswith("training_") and dag_id not in APPROVED_DAGS:
        msg = (
            f"DAG '{dag_id}' is not approved for triggering. "
            f"Only DAGs starting with 'training_' or in {APPROVED_DAGS} are allowed."
        )
        logger.warning(msg)
        return {"success": False, "data": None, "error": msg}

    try:
        session = _get_session()
        url = _api_url(f"/dags/{dag_id}/dagRuns")
        payload: dict[str, Any] = {"conf": conf or {}}
        response = session.post(url, json=payload, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        run = response.json()
        data = {
            "dag_run_id": run.get("dag_run_id"),
            "state": run.get("state"),
            "execution_date": run.get("execution_date"),
            "start_date": run.get("start_date"),
            "end_date": run.get("end_date"),
            "conf": run.get("conf"),
        }
        logger.info("Triggered DAG '%s' – run id: %s", dag_id, data["dag_run_id"])
        return {"success": True, "data": data, "error": None}

    except requests.exceptions.ConnectionError as exc:
        return _handle_error(exc, f"Connection error triggering DAG '{dag_id}'")
    except requests.exceptions.Timeout as exc:
        return _handle_error(exc, f"Timeout triggering DAG '{dag_id}'")
    except requests.exceptions.HTTPError as exc:
        return _handle_error(exc, f"HTTP error triggering DAG '{dag_id}'")
    except Exception as exc:  # noqa: BLE001
        return _handle_error(exc, f"Unexpected error triggering DAG '{dag_id}'")


def get_dag_run_status(dag_id: str, dag_run_id: str) -> dict[str, Any]:
    """Get detailed status of a specific DAG run.

    Parameters
    ----------
    dag_id:
        Identifier of the DAG.
    dag_run_id:
        Identifier of the DAG run.

    Returns
    -------
    dict
        ``success`` – bool, ``data`` – run status details,
        ``error`` – error message or ``None``.
    """
    try:
        session = _get_session()
        url = _api_url(f"/dags/{dag_id}/dagRuns/{dag_run_id}")
        response = session.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        run = response.json()
        data = {
            "dag_id": run.get("dag_id"),
            "dag_run_id": run.get("dag_run_id"),
            "state": run.get("state"),
            "execution_date": run.get("execution_date"),
            "start_date": run.get("start_date"),
            "end_date": run.get("end_date"),
            "external_trigger": run.get("external_trigger"),
            "conf": run.get("conf"),
        }
        logger.info(
            "DAG '%s' run '%s' status: %s", dag_id, dag_run_id, data["state"],
        )
        return {"success": True, "data": data, "error": None}

    except requests.exceptions.ConnectionError as exc:
        return _handle_error(
            exc, f"Connection error fetching status for DAG '{dag_id}' run '{dag_run_id}'",
        )
    except requests.exceptions.Timeout as exc:
        return _handle_error(
            exc, f"Timeout fetching status for DAG '{dag_id}' run '{dag_run_id}'",
        )
    except requests.exceptions.HTTPError as exc:
        return _handle_error(
            exc, f"HTTP error fetching status for DAG '{dag_id}' run '{dag_run_id}'",
        )
    except Exception as exc:  # noqa: BLE001
        return _handle_error(
            exc, f"Unexpected error fetching status for DAG '{dag_id}' run '{dag_run_id}'",
        )


def get_task_instances(dag_id: str, dag_run_id: str) -> dict[str, Any]:
    """Retrieve all task instances for a specific DAG run.

    Parameters
    ----------
    dag_id:
        Identifier of the DAG.
    dag_run_id:
        Identifier of the DAG run.

    Returns
    -------
    dict
        ``success`` – bool, ``data`` – list of task instance summaries,
        ``error`` – error message or ``None``.
    """
    try:
        session = _get_session()
        url = _api_url(f"/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances")
        response = session.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        raw_tasks = response.json().get("task_instances", [])
        tasks = [
            {
                "task_id": t.get("task_id"),
                "state": t.get("state"),
                "start_date": t.get("start_date"),
                "end_date": t.get("end_date"),
                "duration": t.get("duration"),
                "try_number": t.get("try_number"),
                "operator": t.get("operator"),
            }
            for t in raw_tasks
        ]
        logger.info(
            "Fetched %d task instances for DAG '%s' run '%s'.",
            len(tasks), dag_id, dag_run_id,
        )
        return {"success": True, "data": tasks, "error": None}

    except requests.exceptions.ConnectionError as exc:
        return _handle_error(
            exc,
            f"Connection error fetching tasks for DAG '{dag_id}' run '{dag_run_id}'",
        )
    except requests.exceptions.Timeout as exc:
        return _handle_error(
            exc, f"Timeout fetching tasks for DAG '{dag_id}' run '{dag_run_id}'",
        )
    except requests.exceptions.HTTPError as exc:
        return _handle_error(
            exc, f"HTTP error fetching tasks for DAG '{dag_id}' run '{dag_run_id}'",
        )
    except Exception as exc:  # noqa: BLE001
        return _handle_error(
            exc,
            f"Unexpected error fetching tasks for DAG '{dag_id}' run '{dag_run_id}'",
        )


def get_failed_tasks(dag_id: str, dag_run_id: str) -> dict[str, Any]:
    """Return only the **failed** task instances for a DAG run.

    Internally calls :func:`get_task_instances` and filters for tasks
    whose ``state`` is ``"failed"``.

    Parameters
    ----------
    dag_id:
        Identifier of the DAG.
    dag_run_id:
        Identifier of the DAG run.

    Returns
    -------
    dict
        ``success`` – bool, ``data`` – list of failed task details,
        ``error`` – error message or ``None``.
    """
    result = get_task_instances(dag_id, dag_run_id)
    if not result["success"]:
        return result

    failed = [
        {
            "task_id": t["task_id"],
            "state": t["state"],
            "try_number": t["try_number"],
            "start_date": t.get("start_date"),
            "end_date": t.get("end_date"),
            "duration": t.get("duration"),
            "operator": t.get("operator"),
        }
        for t in result["data"]
        if t.get("state") == "failed"
    ]
    logger.info(
        "Found %d failed task(s) in DAG '%s' run '%s'.",
        len(failed), dag_id, dag_run_id,
    )
    return {"success": True, "data": failed, "error": None}


def get_task_logs(
    dag_id: str,
    dag_run_id: str,
    task_id: str,
    try_number: int = 1,
) -> dict[str, Any]:
    """Fetch execution logs for a specific task instance.

    Parameters
    ----------
    dag_id:
        Identifier of the DAG.
    dag_run_id:
        Identifier of the DAG run.
    task_id:
        Identifier of the task.
    try_number:
        The attempt number whose logs to retrieve (default ``1``).

    Returns
    -------
    dict
        ``success`` – bool, ``data`` – log content string,
        ``error`` – error message or ``None``.
    """
    try:
        session = _get_session()
        url = _api_url(
            f"/dags/{dag_id}/dagRuns/{dag_run_id}"
            f"/taskInstances/{task_id}/logs/{try_number}",
        )
        # Logs endpoint may return plain text.
        session.headers.update({"Accept": "application/json"})
        response = session.get(url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        # The Airflow logs endpoint can return JSON or plain text depending
        # on the Airflow version / config.  Handle both gracefully.
        try:
            body = response.json()
            log_content = body.get("content", response.text)
        except ValueError:
            log_content = response.text

        logger.info(
            "Retrieved logs for task '%s' (try %d) in DAG '%s' run '%s'.",
            task_id, try_number, dag_id, dag_run_id,
        )
        return {"success": True, "data": log_content, "error": None}

    except requests.exceptions.ConnectionError as exc:
        return _handle_error(
            exc,
            f"Connection error fetching logs for task '{task_id}' "
            f"in DAG '{dag_id}' run '{dag_run_id}'",
        )
    except requests.exceptions.Timeout as exc:
        return _handle_error(
            exc,
            f"Timeout fetching logs for task '{task_id}' "
            f"in DAG '{dag_id}' run '{dag_run_id}'",
        )
    except requests.exceptions.HTTPError as exc:
        return _handle_error(
            exc,
            f"HTTP error fetching logs for task '{task_id}' "
            f"in DAG '{dag_id}' run '{dag_run_id}'",
        )
    except Exception as exc:  # noqa: BLE001
        return _handle_error(
            exc,
            f"Unexpected error fetching logs for task '{task_id}' "
            f"in DAG '{dag_id}' run '{dag_run_id}'",
        )


# ---------------------------------------------------------------------------
# Tool registry – maps human-readable names to callable functions.
# ---------------------------------------------------------------------------
TOOLS: dict[str, Any] = {
    "airflow_list_dags": list_dags,
    "airflow_get_dag_runs": get_dag_runs,
    "airflow_trigger_dag": trigger_dag,
    "airflow_get_dag_run_status": get_dag_run_status,
    "airflow_get_task_instances": get_task_instances,
    "airflow_get_failed_tasks": get_failed_tasks,
    "airflow_get_task_logs": get_task_logs,
}
