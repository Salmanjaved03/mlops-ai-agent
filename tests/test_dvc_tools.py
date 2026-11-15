"""Tests for DVC tools module."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch
import pytest

from agent.tools.dvc_tools import (
    list_tracked_files,
    get_dataset_version,
    check_data_status,
    compare_dataset_versions,
    get_dataset_for_experiment,
    explain_reproduce_steps,
    get_dvc_remote_info,
    TOOLS,
)


class TestDvcListTracked:
    def test_successful_list_via_dvc(self, mock_subprocess):
        """Test listing DVC-tracked files successfully."""
        mock_subprocess.return_value = MagicMock(
            stdout='[{"path": "data/processed/X_train.npy", "is_dvc": true}]',
            stderr="",
            returncode=0,
        )
        result = list_tracked_files()
        assert result["success"] is True
        assert len(result["data"]) == 1

    def test_subprocess_called_process_error(self, mock_subprocess):
        """Test handling of subprocess failure when fallback also fails."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["dvc", "list"], stderr="DVC repo not found"
        )
        with patch("pathlib.Path.rglob", side_effect=OSError("Disk read failure")):
            result = list_tracked_files()
            assert result["success"] is False
            assert "error" in result

    def test_subprocess_timeout(self, mock_subprocess):
        """Test handling of subprocess timeout on commands like diff."""
        mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd=["git", "diff"], timeout=30)
        result = compare_dataset_versions("HEAD~1", "HEAD")
        assert result["success"] is False
        assert "timed out" in result["error"].lower()


class TestDvcGetVersion:
    def test_get_dataset_version_success(self, mock_subprocess):
        """Test retrieving current commit and DVC metadata."""
        def run_side_effect(cmd, *args, **kwargs):
            if "rev-parse" in cmd:
                return MagicMock(stdout="a4c92e1deadbeef\n", stderr="", returncode=0)
            if "log" in cmd:
                return MagicMock(stdout="train: update iris dataset\n2026-10-15T12:00:00Z\n", stderr="", returncode=0)
            return MagicMock(stdout="data/processed.dvc\n", stderr="", returncode=0)

        mock_subprocess.side_effect = run_side_effect
        result = get_dataset_version()
        assert result["success"] is True
        assert "git_commit" in result["data"]
        assert result["data"]["git_commit"].startswith("a4c92e1")


class TestDvcDataStatus:
    def test_check_data_status_clean(self, mock_subprocess):
        """Test checking data status when everything is up to date."""
        mock_subprocess.return_value = MagicMock(stdout="{}", stderr="", returncode=0)
        result = check_data_status()
        assert result["success"] is True
        assert "is_clean" in result["data"]

    def test_check_data_status_changed(self, mock_subprocess):
        """Test checking data status when files have modified."""
        mock_subprocess.return_value = MagicMock(
            stdout='{"data/raw/features.npy": "modified"}', stderr="", returncode=0
        )
        result = check_data_status()
        assert result["success"] is True


class TestDvcReproduceAndExperiment:
    def test_explain_reproduce_steps(self, mock_subprocess):
        """Test explanation of steps required to reproduce previous dataset version."""
        mock_subprocess.return_value = MagicMock(stdout="a4c92e1", stderr="", returncode=0)
        result = explain_reproduce_steps("a4c92e1")
        assert result["success"] is True
        assert "steps" in result["data"]
        assert len(result["data"]["steps"]) > 0

    def test_get_dataset_for_experiment(self, mock_subprocess):
        """Test retrieving dataset version for a specific experiment commit."""
        def run_side_effect(cmd, *args, **kwargs):
            if "rev-parse" in cmd:
                return MagicMock(stdout="a4c92e1\n", stderr="", returncode=0)
            if "show" in cmd:
                return MagicMock(stdout="outs:\n- md5: 12345\n  path: data.csv\n", stderr="", returncode=0)
            return MagicMock(stdout="data.dvc\n", stderr="", returncode=0)

        mock_subprocess.side_effect = run_side_effect
        result = get_dataset_for_experiment("exp-25")
        assert result["success"] is True


class TestDvcToolRegistry:
    def test_dvc_tools_registered(self):
        """Test all 7 required DVC tools are registered."""
        expected = [
            "dvc_list_tracked",
            "dvc_get_version",
            "dvc_compare_versions",
            "dvc_check_status",
            "dvc_get_dataset_for_experiment",
            "dvc_explain_reproduce",
            "dvc_remote_info",
        ]
        for name in expected:
            assert name in TOOLS
            assert callable(TOOLS[name])
