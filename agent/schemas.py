"""Pydantic models for the MLOps AI Agent system.

Defines request/response schemas and data models for DVC, MLflow,
Airflow, and Kubernetes tool integrations.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core agent schemas
# ---------------------------------------------------------------------------

class ToolCall(BaseModel):
    """Represents a single tool invocation request."""

    tool_name: str = Field(..., description="Name of the tool to invoke")
    arguments: dict = Field(default_factory=dict, description="Arguments passed to the tool")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the call was created",
    )


class ToolResult(BaseModel):
    """Represents the result returned by a tool execution."""

    tool_name: str = Field(..., description="Name of the tool that produced this result")
    success: bool = Field(..., description="Whether the tool execution succeeded")
    data: Any = Field(default=None, description="Payload returned by the tool")
    error: Optional[str] = Field(default=None, description="Error message if the tool failed")
    duration_seconds: float = Field(..., description="Wall-clock seconds the tool took")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this result",
    )


class AgentRequest(BaseModel):
    """Incoming request from a user to the agent."""

    user_input: str = Field(..., description="Raw natural-language query from the user")
    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this request",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the request was created",
    )


class AgentResponse(BaseModel):
    """Response produced by the agent after processing a request."""

    request_id: str = Field(..., description="Identifier of the originating request")
    response_text: str = Field(..., description="Human-readable response text")
    tool_results: list[ToolResult] = Field(
        default_factory=list,
        description="Results from every tool that was executed",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Agent confidence in the response",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings or caveats about the response",
    )


# ---------------------------------------------------------------------------
# DVC schemas
# ---------------------------------------------------------------------------

class DVCFileInfo(BaseModel):
    """Metadata about a file tracked by DVC."""

    path: str = Field(..., description="Relative path to the DVC-tracked file")
    md5: Optional[str] = Field(default=None, description="MD5 hash of the file content")
    size: Optional[int] = Field(default=None, description="File size in bytes")
    tracked: bool = Field(..., description="Whether the file is currently tracked by DVC")


class DVCDatasetVersion(BaseModel):
    """A snapshot of a dataset version managed by DVC + Git."""

    git_commit: str = Field(..., description="Git commit SHA associated with this version")
    commit_message: str = Field(..., description="Git commit message")
    timestamp: str = Field(..., description="ISO-8601 timestamp of the commit")
    dvc_files: list[str] = Field(
        default_factory=list,
        description="List of DVC file paths included in this version",
    )


# ---------------------------------------------------------------------------
# MLflow schemas
# ---------------------------------------------------------------------------

class MLflowRunInfo(BaseModel):
    """Summary of a single MLflow experiment run."""

    run_id: str = Field(..., description="Unique MLflow run identifier")
    experiment_id: str = Field(..., description="Experiment this run belongs to")
    experiment_name: str = Field(..., description="Human-readable experiment name")
    status: str = Field(..., description="Run status (RUNNING, FINISHED, FAILED, …)")
    metrics: dict = Field(default_factory=dict, description="Logged metrics")
    params: dict = Field(default_factory=dict, description="Logged parameters")
    tags: dict = Field(default_factory=dict, description="Run tags")
    start_time: Optional[str] = Field(default=None, description="ISO-8601 start time")
    end_time: Optional[str] = Field(default=None, description="ISO-8601 end time")
    artifact_uri: Optional[str] = Field(default=None, description="URI to the artifact store")


class MLflowModelVersion(BaseModel):
    """A registered model version in MLflow Model Registry."""

    name: str = Field(..., description="Registered model name")
    version: str = Field(..., description="Model version string")
    stage: str = Field(..., description="Current stage (None, Staging, Production, Archived)")
    run_id: str = Field(..., description="Run that produced this model version")
    description: Optional[str] = Field(default=None, description="Version description")


# ---------------------------------------------------------------------------
# Airflow schemas
# ---------------------------------------------------------------------------

class AirflowDAGInfo(BaseModel):
    """Metadata about an Airflow DAG."""

    dag_id: str = Field(..., description="Unique DAG identifier")
    description: Optional[str] = Field(default=None, description="DAG description")
    is_paused: bool = Field(..., description="Whether the DAG is currently paused")
    schedule_interval: Optional[str] = Field(
        default=None, description="Cron or preset schedule interval"
    )
    tags: list[str] = Field(default_factory=list, description="Tags associated with the DAG")


class AirflowDAGRun(BaseModel):
    """A single execution (run) of an Airflow DAG."""

    dag_id: str = Field(..., description="DAG this run belongs to")
    dag_run_id: str = Field(..., description="Unique run identifier")
    state: str = Field(..., description="Run state (queued, running, success, failed)")
    execution_date: str = Field(..., description="Logical execution date (ISO-8601)")
    start_date: Optional[str] = Field(default=None, description="Actual start time")
    end_date: Optional[str] = Field(default=None, description="Actual end time")


class AirflowTaskInstance(BaseModel):
    """A single task instance within an Airflow DAG run."""

    task_id: str = Field(..., description="Task identifier within the DAG")
    dag_id: str = Field(..., description="Parent DAG identifier")
    state: str = Field(..., description="Task state (queued, running, success, failed, …)")
    start_date: Optional[str] = Field(default=None, description="Task start time")
    end_date: Optional[str] = Field(default=None, description="Task end time")
    try_number: int = Field(..., description="Current attempt number")
    log_url: Optional[str] = Field(default=None, description="URL to the task log")


# ---------------------------------------------------------------------------
# Kubernetes schemas
# ---------------------------------------------------------------------------

class K8sPodInfo(BaseModel):
    """Metadata about a Kubernetes pod."""

    name: str = Field(..., description="Pod name")
    namespace: str = Field(..., description="Kubernetes namespace")
    status: str = Field(..., description="Pod phase (Pending, Running, Succeeded, Failed, …)")
    ready: bool = Field(..., description="Whether all containers are ready")
    restart_count: int = Field(..., description="Total container restart count")
    node: Optional[str] = Field(default=None, description="Node the pod is scheduled on")
    ip: Optional[str] = Field(default=None, description="Pod IP address")
    labels: dict = Field(default_factory=dict, description="Pod labels")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp (ISO-8601)")


class K8sDeploymentInfo(BaseModel):
    """Metadata about a Kubernetes Deployment."""

    name: str = Field(..., description="Deployment name")
    namespace: str = Field(..., description="Kubernetes namespace")
    replicas: int = Field(..., description="Desired replica count")
    ready_replicas: int = Field(..., description="Number of ready replicas")
    available_replicas: int = Field(..., description="Number of available replicas")
    strategy: str = Field(..., description="Deployment strategy (RollingUpdate, Recreate)")
    labels: dict = Field(default_factory=dict, description="Deployment labels")
    image: Optional[str] = Field(default=None, description="Container image")


class K8sServiceInfo(BaseModel):
    """Metadata about a Kubernetes Service."""

    name: str = Field(..., description="Service name")
    namespace: str = Field(..., description="Kubernetes namespace")
    type: str = Field(..., description="Service type (ClusterIP, NodePort, LoadBalancer, …)")
    cluster_ip: Optional[str] = Field(default=None, description="Cluster-internal IP")
    ports: list[dict] = Field(default_factory=list, description="Port mappings")
    selector: dict = Field(default_factory=dict, description="Label selector")
