"""DVC Tools – AI-agent callable functions for Data Version Control.

This module exposes a collection of *tool* functions that an AI agent can
invoke to inspect, compare, and reason about DVC-tracked datasets and their
relationship to Git history.

Every public function follows the **tool contract**:
  • Returns ``dict`` with at least ``{"success": bool, "data": …, "error": …}``.
  • Logs the subprocess command being executed (secrets are never logged).
  • Handles ``subprocess.CalledProcessError`` and ``subprocess.TimeoutExpired``.
  • Uses a default subprocess timeout of **30 s**.
"""

from __future__ import annotations

import configparser
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT: int = 30


def _run_command(
    cmd: list[str],
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute *cmd* as a subprocess and return the ``CompletedProcess``.

    Parameters
    ----------
    cmd:
        The command and its arguments.
    timeout:
        Maximum wall-clock seconds the process is allowed to run.
    cwd:
        Working directory for the subprocess.  Defaults to the current
        working directory.

    Returns
    -------
    subprocess.CompletedProcess[str]

    Raises
    ------
    subprocess.CalledProcessError
        If the process exits with a non-zero return code.
    subprocess.TimeoutExpired
        If the process does not finish within *timeout* seconds.
    """
    logger.info("Running command: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        check=True,
    )


def _error_response(error: Exception, context: str = "") -> dict[str, Any]:
    """Build a standardised error-response dict."""
    msg = f"{context}: {error}" if context else str(error)
    logger.error(msg)
    return {"success": False, "data": None, "error": msg}


# ---------------------------------------------------------------------------
# 1. list_tracked_files
# ---------------------------------------------------------------------------


def list_tracked_files() -> list[dict] | dict[str, Any]:
    """List all DVC-tracked files in the current repository.

    The function first attempts ``dvc list . --dvc-only --json``.  If that
    fails (e.g. DVC is not installed or the project is not initialised) it
    falls back to scanning the working tree for ``*.dvc`` files and parsing
    their YAML content.

    Returns
    -------
    dict
        ``{"success": True, "data": [{"path": …, "md5": …, "size": …}, …]}``
        on success, or ``{"success": False, "data": None, "error": …}`` on
        failure.
    """
    logger.info("Listing DVC-tracked files")

    # --- Strategy 1: ``dvc list`` -----------------------------------------
    try:
        result = _run_command(["dvc", "list", ".", "--dvc-only", "--json"])
        files: list[dict[str, Any]] = json.loads(result.stdout)
        logger.info("dvc list returned %d entries", len(files))
        return {"success": True, "data": files, "error": None}
    except FileNotFoundError:
        logger.warning("DVC CLI not found on PATH – falling back to .dvc file parsing")
    except subprocess.CalledProcessError as exc:
        logger.warning("dvc list failed (rc=%d) – falling back to .dvc file parsing", exc.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("dvc list timed out – falling back to .dvc file parsing")
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse dvc list JSON output: %s", exc)

    # --- Strategy 2: parse *.dvc files ------------------------------------
    try:
        import yaml  # optional dependency – only needed for fallback
    except ImportError:
        return _error_response(
            RuntimeError("Neither DVC CLI nor PyYAML are available"),
            "list_tracked_files",
        )

    try:
        tracked: list[dict[str, Any]] = []
        for dvc_file in Path(".").rglob("*.dvc"):
            logger.debug("Parsing %s", dvc_file)
            with open(dvc_file, encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            if not doc:
                continue
            for out in doc.get("outs", []):
                tracked.append(
                    {
                        "path": out.get("path", str(dvc_file)),
                        "md5": out.get("md5"),
                        "size": out.get("size"),
                    }
                )
        logger.info("Parsed %d DVC-tracked entries from .dvc files", len(tracked))
        return {"success": True, "data": tracked, "error": None}
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, "list_tracked_files fallback")


# ---------------------------------------------------------------------------
# 2. get_dataset_version
# ---------------------------------------------------------------------------


def get_dataset_version() -> dict[str, Any]:
    """Return the current dataset version based on Git + DVC metadata.

    Collects:
    * ``git_commit`` – the current ``HEAD`` SHA.
    * ``commit_message`` – the one-line subject of that commit.
    * ``dvc_files`` – a mapping of each ``.dvc`` / ``dvc.lock`` file to its
      content hash (via ``git hash-object``).
    * ``timestamp`` – ISO-8601 author date of the current commit.

    Returns
    -------
    dict
        Standardised tool response.
    """
    logger.info("Getting current dataset version info")
    try:
        # Git commit SHA
        sha_result = _run_command(["git", "rev-parse", "HEAD"])
        git_commit = sha_result.stdout.strip()

        # Commit message (subject line)
        msg_result = _run_command(["git", "log", "-1", "--format=%s"])
        commit_message = msg_result.stdout.strip()

        # Author date (ISO-8601)
        date_result = _run_command(["git", "log", "-1", "--format=%aI"])
        timestamp = date_result.stdout.strip()

        # Enumerate .dvc / dvc.lock files tracked by Git
        ls_result = _run_command(["git", "ls-files", "*.dvc", "dvc.lock"])
        dvc_paths = [p for p in ls_result.stdout.strip().splitlines() if p]

        dvc_files: dict[str, str | None] = {}
        for path in dvc_paths:
            try:
                hash_result = _run_command(["git", "hash-object", path])
                dvc_files[path] = hash_result.stdout.strip()
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                dvc_files[path] = None

        data = {
            "git_commit": git_commit,
            "commit_message": commit_message,
            "dvc_files": dvc_files,
            "timestamp": timestamp,
        }
        logger.info("Dataset version resolved at commit %s", git_commit[:8])
        return {"success": True, "data": data, "error": None}

    except subprocess.CalledProcessError as exc:
        return _error_response(exc, "get_dataset_version")
    except subprocess.TimeoutExpired as exc:
        return _error_response(exc, "get_dataset_version (timeout)")


# ---------------------------------------------------------------------------
# 3. compare_dataset_versions
# ---------------------------------------------------------------------------


def compare_dataset_versions(commit_a: str, commit_b: str) -> dict[str, Any]:
    """Compare DVC-tracked file metadata between two Git commits.

    Uses ``git diff <commit_a> <commit_b> -- '*.dvc' '*.dvc.lock'`` to
    identify which tracked datasets changed, were added, or were removed.

    Parameters
    ----------
    commit_a:
        The *from* Git ref (SHA, tag, branch …).
    commit_b:
        The *to* Git ref.

    Returns
    -------
    dict
        ``{"success": True, "data": {"changes": […], "added": […],
        "removed": […], "commit_a": …, "commit_b": …}}``
    """
    logger.info("Comparing dataset versions: %s .. %s", commit_a, commit_b)
    try:
        diff_result = _run_command(
            [
                "git", "diff", "--name-status",
                commit_a, commit_b,
                "--", "*.dvc", "*.dvc.lock",
            ]
        )

        changes: list[dict[str, str]] = []
        added: list[str] = []
        removed: list[str] = []

        for line in diff_result.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("\t")
            status = parts[0]
            file_path = parts[1] if len(parts) > 1 else ""

            if status.startswith("A"):
                added.append(file_path)
            elif status.startswith("D"):
                removed.append(file_path)
            elif status.startswith("M") or status.startswith("R"):
                old_path = file_path
                new_path = parts[2] if len(parts) > 2 else file_path
                changes.append(
                    {"status": status, "old_path": old_path, "new_path": new_path}
                )
            else:
                changes.append({"status": status, "path": file_path})

        data = {
            "commit_a": commit_a,
            "commit_b": commit_b,
            "changes": changes,
            "added": added,
            "removed": removed,
            "total_affected": len(changes) + len(added) + len(removed),
        }
        logger.info(
            "Diff complete: %d changed, %d added, %d removed",
            len(changes), len(added), len(removed),
        )
        return {"success": True, "data": data, "error": None}

    except subprocess.CalledProcessError as exc:
        return _error_response(exc, "compare_dataset_versions")
    except subprocess.TimeoutExpired as exc:
        return _error_response(exc, "compare_dataset_versions (timeout)")


# ---------------------------------------------------------------------------
# 4. check_data_status
# ---------------------------------------------------------------------------


def check_data_status() -> dict[str, Any]:
    """Run ``dvc status`` and report which tracked data has changed.

    Returns
    -------
    dict
        ``{"success": True, "data": {"changed_files": {…}, "is_clean": bool}}``
        where ``changed_files`` is the parsed JSON output of ``dvc status``,
        or the raw text if JSON parsing fails.
    """
    logger.info("Checking DVC data status")
    try:
        # Try JSON output first for structured data
        try:
            result = _run_command(["dvc", "status", "--json"])
            status_data = json.loads(result.stdout)
            is_clean = len(status_data) == 0
            logger.info("DVC status: %s", "clean" if is_clean else "changes detected")
            return {
                "success": True,
                "data": {"changed_files": status_data, "is_clean": is_clean},
                "error": None,
            }
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            # Fall back to plain text
            result = _run_command(["dvc", "status"])
            raw = result.stdout.strip()
            is_clean = not raw or "Data and pipelines are up to date" in raw
            logger.info("DVC status (text): %s", "clean" if is_clean else "changes detected")
            return {
                "success": True,
                "data": {"raw_output": raw, "is_clean": is_clean},
                "error": None,
            }

    except FileNotFoundError:
        return _error_response(
            FileNotFoundError("DVC CLI not found on PATH"),
            "check_data_status",
        )
    except subprocess.CalledProcessError as exc:
        return _error_response(exc, "check_data_status")
    except subprocess.TimeoutExpired as exc:
        return _error_response(exc, "check_data_status (timeout)")


# ---------------------------------------------------------------------------
# 5. get_dataset_for_experiment
# ---------------------------------------------------------------------------


def get_dataset_for_experiment(experiment_ref: str) -> dict[str, Any]:
    """Retrieve the dataset version associated with a given experiment.

    Given a Git ref (tag, branch, commit SHA) that represents an experiment,
    this function inspects the DVC lock / ``.dvc`` files at that point in
    history to determine which dataset version was used.

    Parameters
    ----------
    experiment_ref:
        A Git reference (commit SHA, tag, or branch name).

    Returns
    -------
    dict
        ``{"success": True, "data": {"experiment_ref": …,
        "resolved_commit": …, "dvc_metadata": {…}}}``
    """
    logger.info("Getting dataset info for experiment ref: %s", experiment_ref)
    try:
        # Resolve the reference to a full SHA
        rev_result = _run_command(["git", "rev-parse", experiment_ref])
        resolved_commit = rev_result.stdout.strip()

        # Commit message for context
        msg_result = _run_command(
            ["git", "log", "-1", "--format=%s", resolved_commit]
        )
        commit_message = msg_result.stdout.strip()

        # List DVC-relevant files at that commit
        ls_result = _run_command(
            ["git", "ls-tree", "-r", "--name-only", resolved_commit]
        )
        all_files = ls_result.stdout.strip().splitlines()
        dvc_files = [
            f for f in all_files
            if f.endswith(".dvc") or f.endswith("dvc.lock")
        ]

        # Read the content of each DVC metadata file at that commit
        dvc_metadata: dict[str, str | None] = {}
        for dvc_path in dvc_files:
            try:
                show_result = _run_command(
                    ["git", "show", f"{resolved_commit}:{dvc_path}"]
                )
                dvc_metadata[dvc_path] = show_result.stdout
            except subprocess.CalledProcessError:
                logger.warning(
                    "Could not read %s at %s", dvc_path, resolved_commit[:8]
                )
                dvc_metadata[dvc_path] = None

        data = {
            "experiment_ref": experiment_ref,
            "resolved_commit": resolved_commit,
            "commit_message": commit_message,
            "dvc_files": dvc_files,
            "dvc_metadata": dvc_metadata,
        }
        logger.info(
            "Found %d DVC file(s) at experiment ref %s (%s)",
            len(dvc_files), experiment_ref, resolved_commit[:8],
        )
        return {"success": True, "data": data, "error": None}

    except subprocess.CalledProcessError as exc:
        return _error_response(exc, "get_dataset_for_experiment")
    except subprocess.TimeoutExpired as exc:
        return _error_response(exc, "get_dataset_for_experiment (timeout)")


# ---------------------------------------------------------------------------
# 6. explain_reproduce_steps
# ---------------------------------------------------------------------------


def explain_reproduce_steps(target_commit: str) -> dict[str, Any]:
    """Generate step-by-step instructions to reproduce a dataset version.

    Given a *target_commit* (or any valid Git ref), this function verifies
    the ref exists and returns the exact CLI commands needed to restore both
    the code **and** the DVC-tracked data to that point in time.

    Parameters
    ----------
    target_commit:
        The Git commit / tag / branch to reproduce.

    Returns
    -------
    dict
        ``{"success": True, "data": {"target_commit": …,
        "resolved_sha": …, "steps": […]}}``
    """
    logger.info("Generating reproduce steps for: %s", target_commit)
    try:
        # Validate the ref
        rev_result = _run_command(["git", "rev-parse", target_commit])
        resolved_sha = rev_result.stdout.strip()

        # Commit metadata
        meta_result = _run_command(
            ["git", "log", "-1", "--format=%H|%s|%aI", resolved_sha]
        )
        parts = meta_result.stdout.strip().split("|", 2)
        commit_sha = parts[0] if len(parts) > 0 else resolved_sha
        commit_msg = parts[1] if len(parts) > 1 else ""
        commit_date = parts[2] if len(parts) > 2 else ""

        steps = [
            {
                "step": 1,
                "description": "Ensure you have a clean working tree (commit or stash changes).",
                "command": "git stash  # or: git commit -am 'WIP'",
            },
            {
                "step": 2,
                "description": f"Check out the target commit ({commit_sha[:8]}).",
                "command": f"git checkout {resolved_sha}",
            },
            {
                "step": 3,
                "description": "Synchronise DVC-tracked files to match the checked-out metadata.",
                "command": "dvc checkout",
            },
            {
                "step": 4,
                "description": (
                    "Pull data from the remote if it is not already cached locally."
                ),
                "command": "dvc pull",
            },
            {
                "step": 5,
                "description": "Verify that all data is up to date.",
                "command": "dvc status",
            },
            {
                "step": 6,
                "description": (
                    "(Optional) Return to your original branch when done."
                ),
                "command": "git checkout -  # switches back to previous branch",
            },
        ]

        data = {
            "target_commit": target_commit,
            "resolved_sha": resolved_sha,
            "commit_message": commit_msg,
            "commit_date": commit_date,
            "steps": steps,
        }
        logger.info("Reproduce steps generated for %s", resolved_sha[:8])
        return {"success": True, "data": data, "error": None}

    except subprocess.CalledProcessError as exc:
        return _error_response(exc, "explain_reproduce_steps")
    except subprocess.TimeoutExpired as exc:
        return _error_response(exc, "explain_reproduce_steps (timeout)")


# ---------------------------------------------------------------------------
# 7. get_dvc_remote_info
# ---------------------------------------------------------------------------

_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "access_key_id",
    "secret_access_key",
    "session_token",
    "password",
    "token",
    "credentialpath",
    "key",
    "keyfile",
    "account_key",
    "sas_token",
    "connection_string",
    "client_secret",
})


def get_dvc_remote_info() -> dict[str, Any]:
    """Parse ``.dvc/config`` (and local overrides) for remote storage info.

    Secrets (access keys, tokens, passwords …) are **redacted** before the
    result is returned.

    Returns
    -------
    dict
        ``{"success": True, "data": {"remotes": {…}, "core": {…}}}``
    """
    logger.info("Reading DVC remote configuration")
    try:
        config = configparser.ConfigParser()
        config_paths: list[Path] = [
            Path(".dvc") / "config",
            Path(".dvc") / "config.local",
        ]

        read_files = config.read([str(p) for p in config_paths], encoding="utf-8")
        if not read_files:
            logger.warning("No DVC config files found at %s", config_paths)
            return {
                "success": True,
                "data": {"remotes": {}, "core": {}, "config_files_read": []},
                "error": None,
            }

        remotes: dict[str, dict[str, str]] = {}
        core: dict[str, str] = {}

        for section in config.sections():
            items: dict[str, str] = {}
            for key, value in config.items(section):
                # Redact sensitive values
                if key.lower() in _SENSITIVE_KEYS:
                    items[key] = "***REDACTED***"
                else:
                    items[key] = value

            if section.startswith("'remote"):
                # Section looks like: 'remote "myremote"'
                remote_name = section.split('"')[1] if '"' in section else section
                remotes[remote_name] = items
            elif section == "core":
                core = items
            else:
                # Preserve other sections (e.g., cache) under their name
                remotes[section] = items

        data = {
            "remotes": remotes,
            "core": core,
            "config_files_read": [str(p) for p in read_files],
        }
        logger.info("Parsed DVC remote config: %d remote(s) found", len(remotes))
        return {"success": True, "data": data, "error": None}

    except configparser.Error as exc:
        return _error_response(exc, "get_dvc_remote_info (config parse)")
    except OSError as exc:
        return _error_response(exc, "get_dvc_remote_info (file I/O)")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOLS: dict[str, Any] = {
    "dvc_list_tracked": list_tracked_files,
    "dvc_get_version": get_dataset_version,
    "dvc_compare_versions": compare_dataset_versions,
    "dvc_check_status": check_data_status,
    "dvc_get_dataset_for_experiment": get_dataset_for_experiment,
    "dvc_explain_reproduce": explain_reproduce_steps,
    "dvc_remote_info": get_dvc_remote_info,
}
