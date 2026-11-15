"""MLflow tools module for the AI agent.

Provides tool functions that the agent can invoke to interact with MLflow
for experiment tracking, run inspection, and model registry queries.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

try:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None  # type: ignore
    MlflowException = Exception  # type: ignore
    MlflowClient = None  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client helper
# ---------------------------------------------------------------------------

_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def _get_client() -> MlflowClient:
    """Return an :class:`MlflowClient` configured with the tracking URI."""
    if mlflow is None:
        raise RuntimeError("MLflow is not installed or available.")
    mlflow.set_tracking_uri(_TRACKING_URI)
    return MlflowClient(tracking_uri=_TRACKING_URI)


def _ok(data: Any) -> dict:
    """Wrap *data* in a success response envelope."""
    return {"success": True, "data": data, "error": None}


def _err(message: str) -> dict:
    """Wrap an error *message* in a failure response envelope."""
    return {"success": False, "data": None, "error": message}


def _ts_to_iso(ts_ms: int | None) -> str | None:
    """Convert a millisecond-epoch timestamp to an ISO-8601 string."""
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def list_experiments() -> dict:
    """List all MLflow experiments.

    Returns:
        A dict with ``success``, ``data`` (list of experiment info dicts),
        and ``error`` keys.
    """
    try:
        client = _get_client()
        experiments = client.search_experiments()
        data = [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
            }
            for exp in experiments
        ]
        logger.info("Listed %d experiments.", len(data))
        return _ok(data)
    except MlflowException as exc:
        logger.error("MLflow error listing experiments: %s", exc)
        return _err(f"MLflow error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Connection/unexpected error listing experiments: %s", exc)
        return _err(f"Connection error: {exc}")


def list_runs(
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    max_results: int = 10,
) -> dict:
    """Search runs in an experiment.

    Provide either *experiment_name* or *experiment_id*.  If both are given
    *experiment_id* takes precedence.

    Args:
        experiment_name: Human-readable experiment name.
        experiment_id: MLflow experiment ID.
        max_results: Maximum number of runs to return.

    Returns:
        A dict with ``success``, ``data`` (list of run summary dicts),
        and ``error`` keys.
    """
    try:
        client = _get_client()

        # Resolve experiment_id from name when necessary.
        if experiment_id is None and experiment_name is not None:
            exp = client.get_experiment_by_name(experiment_name)
            if exp is None:
                return _err(f"Experiment '{experiment_name}' not found.")
            experiment_id = exp.experiment_id
        elif experiment_id is None and experiment_name is None:
            return _err(
                "Either 'experiment_name' or 'experiment_id' must be provided."
            )

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results,
        )

        data = [
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": _ts_to_iso(run.info.start_time),
                "end_time": _ts_to_iso(run.info.end_time),
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
            }
            for run in runs
        ]
        logger.info(
            "Listed %d runs for experiment '%s'.",
            len(data),
            experiment_id,
        )
        return _ok(data)
    except MlflowException as exc:
        logger.error("MLflow error listing runs: %s", exc)
        return _err(f"MLflow error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Connection/unexpected error listing runs: %s", exc)
        return _err(f"Connection error: {exc}")


def get_run_details(run_id: str) -> dict:
    """Get full details of a specific run.

    Args:
        run_id: The MLflow run ID.

    Returns:
        A dict with ``success``, ``data`` (run detail dict), and ``error``
        keys.
    """
    try:
        client = _get_client()
        run = client.get_run(run_id)

        # Collect artifact list (top-level only).
        try:
            artifacts = [
                {"path": a.path, "is_dir": a.is_dir, "file_size": a.file_size}
                for a in client.list_artifacts(run_id)
            ]
        except Exception:  # noqa: BLE001
            artifacts = []

        data = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": _ts_to_iso(run.info.start_time),
            "end_time": _ts_to_iso(run.info.end_time),
            "artifact_uri": run.info.artifact_uri,
            "lifecycle_stage": run.info.lifecycle_stage,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
            "artifacts": artifacts,
        }
        logger.info("Retrieved details for run '%s'.", run_id)
        return _ok(data)
    except MlflowException as exc:
        logger.error("MLflow error getting run details: %s", exc)
        return _err(f"MLflow error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Connection/unexpected error getting run details: %s", exc)
        return _err(f"Connection error: {exc}")


def get_best_run(
    experiment_name: str | None = None,
    metric: str = "f1_score",
    ascending: bool = False,
    max_results: int = 5,
) -> dict:
    """Find the best run in an experiment ordered by a given metric.

    Args:
        experiment_name: Name of the experiment to search.
        metric: Metric key to sort by (default ``"f1_score"``).
        ascending: If ``True`` sort ascending (lower is better); otherwise
            sort descending (higher is better).
        max_results: Number of top runs to consider.

    Returns:
        A dict with ``success``, ``data`` (best run details and advisory
        note), and ``error`` keys.
    """
    try:
        client = _get_client()

        if experiment_name is None:
            default_exp = os.environ.get("MLFLOW_EXPERIMENT_NAME", "iris-classifier")
            exp = client.get_experiment_by_name(default_exp)
            if exp is None:
                exps = client.search_experiments()
                exp = exps[0] if exps else None
            if exp is not None:
                exp_ids = [exp.experiment_id]
                experiment_name = exp.name
            else:
                exp_ids = []
        else:
            exp = client.get_experiment_by_name(experiment_name)
            if exp is None:
                return _err(f"Experiment '{experiment_name}' not found.")
            exp_ids = [exp.experiment_id]

        order = "ASC" if ascending else "DESC"
        runs = client.search_runs(
            experiment_ids=exp_ids if exp_ids else None,
            order_by=[f"metrics.{metric} {order}"],
            max_results=max_results,
        )

        if not runs:
            return _err(
                f"No runs found in experiment '{experiment_name}'."
            )

        best = runs[0]
        data = {
            "best_run": {
                "run_id": best.info.run_id,
                "status": best.info.status,
                "start_time": _ts_to_iso(best.info.start_time),
                "end_time": _ts_to_iso(best.info.end_time),
                "metrics": dict(best.data.metrics),
                "params": dict(best.data.params),
            },
            "run_id": best.info.run_id,
            "metrics": dict(best.data.metrics),
            "params": dict(best.data.params),
            "metric_used": metric,
            "order": order,
            "note": (
                "The highest metric value alone does not guarantee "
                "production readiness. Consider factors such as "
                "overfitting, data leakage, latency, fairness, and "
                "reproducibility before promoting a model."
            ),
        }
        logger.info(
            "Best run for metric '%s' in '%s': %s",
            metric,
            experiment_name,
            best.info.run_id,
        )
        return _ok(data)
    except MlflowException as exc:
        logger.error("MLflow error getting best run: %s", exc)
        return _err(f"MLflow error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Connection/unexpected error getting best run: %s", exc)
        return _err(f"Connection error: {exc}")


def compare_runs(run_ids: list[str]) -> dict:
    """Compare multiple runs side by side.

    Args:
        run_ids: List of MLflow run IDs to compare.

    Returns:
        A dict with ``success``, ``data`` (mapping of run_id → metrics/params),
        and ``error`` keys.
    """
    try:
        if not run_ids:
            return _err("'run_ids' must be a non-empty list.")

        client = _get_client()
        comparison: dict[str, dict] = {}

        for rid in run_ids:
            run = client.get_run(rid)
            comparison[rid] = {
                "status": run.info.status,
                "start_time": _ts_to_iso(run.info.start_time),
                "end_time": _ts_to_iso(run.info.end_time),
                "metrics": dict(run.data.metrics),
                "params": dict(run.data.params),
            }

        logger.info("Compared %d runs.", len(comparison))
        return _ok(comparison)
    except MlflowException as exc:
        logger.error("MLflow error comparing runs: %s", exc)
        return _err(f"MLflow error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Connection/unexpected error comparing runs: %s", exc)
        return _err(f"Connection error: {exc}")


def get_model_versions(model_name: str) -> dict:
    """Get all registered versions of a model.

    Args:
        model_name: The registered model name.

    Returns:
        A dict with ``success``, ``data`` (list of version info dicts),
        and ``error`` keys.
    """
    try:
        client = _get_client()
        versions = client.search_model_versions(f"name='{model_name}'")

        data = [
            {
                "name": v.name,
                "version": v.version,
                "current_stage": v.current_stage,
                "status": v.status,
                "run_id": v.run_id,
                "source": v.source,
                "creation_timestamp": _ts_to_iso(v.creation_timestamp),
                "last_updated_timestamp": _ts_to_iso(v.last_updated_timestamp),
                "description": v.description,
            }
            for v in versions
        ]
        logger.info(
            "Found %d versions for model '%s'.", len(data), model_name
        )
        return _ok(data)
    except MlflowException as exc:
        logger.error("MLflow error getting model versions: %s", exc)
        return _err(f"MLflow error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Connection/unexpected error getting model versions: %s", exc
        )
        return _err(f"Connection error: {exc}")


def get_model_info(model_name: str) -> dict:
    """Get registered model details.

    Includes the model description, tags, and latest version metadata.

    Args:
        model_name: The registered model name.

    Returns:
        A dict with ``success``, ``data`` (model info dict), and ``error``
        keys.
    """
    try:
        client = _get_client()
        model = client.get_registered_model(model_name)

        latest_versions = [
            {
                "version": v.version,
                "current_stage": v.current_stage,
                "status": v.status,
                "run_id": v.run_id,
                "source": v.source,
                "creation_timestamp": _ts_to_iso(v.creation_timestamp),
                "last_updated_timestamp": _ts_to_iso(v.last_updated_timestamp),
                "description": v.description,
            }
            for v in (model.latest_versions or [])
        ]

        data = {
            "name": model.name,
            "description": model.description,
            "creation_timestamp": _ts_to_iso(model.creation_timestamp),
            "last_updated_timestamp": _ts_to_iso(model.last_updated_timestamp),
            "tags": {t.key: t.value for t in (model.tags or {}).values()}
            if isinstance(model.tags, dict)
            else dict(model.tags) if model.tags else {},
            "latest_versions": latest_versions,
        }
        logger.info("Retrieved info for model '%s'.", model_name)
        return _ok(data)
    except MlflowException as exc:
        logger.error("MLflow error getting model info: %s", exc)
        return _err(f"MLflow error: {exc}")
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Connection/unexpected error getting model info: %s", exc
        )
        return _err(f"Connection error: {exc}")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, Any] = {
    "mlflow_list_experiments": list_experiments,
    "mlflow_list_runs": list_runs,
    "mlflow_get_run": get_run_details,
    "mlflow_get_best_run": get_best_run,
    "mlflow_compare_runs": compare_runs,
    "mlflow_get_model_versions": get_model_versions,
    "mlflow_get_model_info": get_model_info,
}
