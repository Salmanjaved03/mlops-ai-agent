"""Tests for MLOps Agent orchestrator, multi-tool workflows, and synthesis."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from agent.orchestrator import MLOpsAgent
from agent.schemas import AgentRequest, AgentResponse, ToolResult


class TestIntentClassification:
    def setup_method(self):
        with patch.object(MLOpsAgent, "_build_tool_registry", return_value={}):
            self.agent = MLOpsAgent()

    def test_dvc_intent(self):
        intents = self.agent._classify_intent("What dataset version was used for experiment 25?")
        assert "dvc" in intents

    def test_mlflow_intent(self):
        intents = self.agent._classify_intent("Compare the last 5 runs and show best candidate on f1")
        assert "mlflow" in intents

    def test_airflow_intent(self):
        intents = self.agent._classify_intent("Trigger the training pipeline and check dag run status")
        assert "airflow" in intents

    def test_kubernetes_intent(self):
        intents = self.agent._classify_intent("Why is my prediction API pod crashing or unhealthy?")
        assert "kubernetes" in intents

    def test_multi_tool_intent(self):
        intents = self.agent._classify_intent(
            "Find the best recent model, identify the dataset version used to train it, "
            "check the deployment status, and explain whether retraining is necessary."
        )
        assert len(intents) >= 3


class TestAgentExecutionAndSafety:
    def setup_method(self):
        self.agent = MLOpsAgent()

    @pytest.mark.asyncio
    async def test_destructive_tool_blocked_without_confirmation(self):
        """Test destructive tool requires explicit confirmation flag."""
        req = AgentRequest(user_input="trigger training pipeline")
        
        # When confirmed=False, trigger_dag must be blocked
        response = await self.agent.process_request(req, confirmed=False)
        assert any("confirmation" in w.lower() for w in response.warnings)
        failed_tools = [r for r in response.tool_results if not r.success]
        assert len(failed_tools) > 0
        assert "requires explicit confirmation" in failed_tools[0].error.lower()

    @pytest.mark.asyncio
    async def test_tool_execution_logs_and_timing(self):
        """Test tool execution captures execution duration and returns ToolResult."""
        mock_tool = MagicMock(return_value={"success": True, "data": {"status": "ok"}})
        self.agent.tool_registry["test_dummy_tool"] = mock_tool

        result = self.agent._execute_tool("test_dummy_tool")
        assert result.success is True
        assert result.duration_seconds >= 0.0
        assert result.data == {"status": "ok"}
        assert result.tool_name == "test_dummy_tool"


class TestMandatoryIntegrationScenario:
    @pytest.mark.asyncio
    async def test_multi_tool_orchestration_flow(self):
        """Verify the mandatory integration scenario correlates all 4 tools and does not trigger retraining."""
        agent = MLOpsAgent()

        # Mock tools
        agent.tool_registry["mlflow_get_best_run"] = MagicMock(return_value={
            "success": True,
            "data": {
                "run_id": "exp-run-99",
                "metrics": {"f1_score": 0.98, "accuracy": 0.985},
                "params": {"n_estimators": 100},
                "note": "Candidate metrics high but evaluate thoroughly."
            }
        })
        agent.tool_registry["dvc_get_version"] = MagicMock(return_value={
            "success": True,
            "data": {
                "git_commit": "a4c92e1",
                "dvc_files": ["data/processed/X_train.npy.dvc"]
            }
        })
        agent.tool_registry["k8s_list_deployments"] = MagicMock(return_value={
            "success": True,
            "data": [{"name": "ml-inference-api", "ready_replicas": 2, "replicas": 2}]
        })
        agent.tool_registry["k8s_get_unhealthy_pods"] = MagicMock(return_value={
            "success": True,
            "data": []
        })
        agent.tool_registry["airflow_list_dags"] = MagicMock(return_value={
            "success": True,
            "data": [{"dag_id": "training_pipeline", "is_paused": False}]
        })

        req = AgentRequest(
            user_input="Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
        )

        response = await agent.process_request(req)
        assert response.confidence == "high"
        # Must contain MLflow evidence
        assert "exp-run-99" in response.response_text
        assert "0.98" in response.response_text
        # Must contain DVC commit
        assert "a4c92e1" in response.response_text
        # Must contain K8s deployment info
        assert "ml-inference-api" in response.response_text
        # Must explicitly justify why retraining is NOT automatically triggered
        assert "retraining is not" in response.response_text.lower() or "not automatically triggered" in response.response_text.lower()
