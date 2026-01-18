"""Basic tests for the Agent Orchestration System."""

import pytest
import asyncio
import os
import tempfile

from counsel import (
    Config,
    TaskGraph,
    Task,
    TaskStatus,
    Workspace,
    Shell,
    ShellResult,
)


class TestTaskGraph:
    """Tests for TaskGraph."""
    
    def test_create_empty_graph(self):
        graph = TaskGraph()
        assert len(graph) == 0
        assert graph.is_complete()
    
    def test_add_task(self):
        graph = TaskGraph()
        task = graph.add_task("task_1", "Test task")
        
        assert len(graph) == 1
        assert "task_1" in graph
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
    
    def test_task_dependencies(self):
        graph = TaskGraph()
        graph.add_task("task_1", "First task")
        graph.add_task("task_2", "Second task", dependencies=["task_1"])
        graph.finalize()
        
        ready = graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "task_1"
    
    def test_execution_levels(self):
        graph = TaskGraph()
        graph.add_task("task_1", "First")
        graph.add_task("task_2", "Second", dependencies=["task_1"])
        graph.add_task("task_3", "Third", dependencies=["task_1"])
        graph.add_task("task_4", "Fourth", dependencies=["task_2", "task_3"])
        graph.finalize()
        
        levels = graph.get_execution_levels()
        assert len(levels) == 3
        assert levels[0] == ["task_1"]
        assert set(levels[1]) == {"task_2", "task_3"}
        assert levels[2] == ["task_4"]
    
    def test_cycle_detection(self):
        graph = TaskGraph()
        graph.add_task("task_1", "First", dependencies=["task_2"])
        graph.add_task("task_2", "Second", dependencies=["task_1"])
        
        with pytest.raises(ValueError, match="cycles"):
            graph.finalize()
    
    def test_mark_completed(self):
        graph = TaskGraph()
        graph.add_task("task_1", "First")
        graph.add_task("task_2", "Second", dependencies=["task_1"])
        graph.finalize()
        
        graph.mark_running("task_1")
        assert graph.get_task("task_1").status == TaskStatus.RUNNING
        
        graph.mark_completed("task_1", "result")
        assert graph.get_task("task_1").status == TaskStatus.COMPLETED
        assert graph.get_task("task_2").status == TaskStatus.READY
    
    def test_failure_propagation(self):
        graph = TaskGraph()
        graph.add_task("task_1", "First")
        graph.add_task("task_2", "Second", dependencies=["task_1"])
        graph.add_task("task_3", "Third", dependencies=["task_2"])
        graph.finalize()
        
        graph.mark_failed("task_1", "Error!")
        
        assert graph.get_task("task_1").status == TaskStatus.FAILED
        assert graph.get_task("task_2").status == TaskStatus.BLOCKED
        assert graph.get_task("task_3").status == TaskStatus.BLOCKED


class TestShell:
    """Tests for Shell."""
    
    @pytest.fixture
    def shell(self):
        return Shell()
    
    def test_pwd(self, shell):
        assert shell.pwd() == os.getcwd()
    
    @pytest.mark.asyncio
    async def test_run_command(self, shell):
        result = await shell.run("echo hello")
        assert result.success
        assert "hello" in result.stdout
    
    @pytest.mark.asyncio
    async def test_run_failing_command(self, shell):
        result = await shell.run("exit 1")
        assert not result.success
        assert result.return_code == 1
    
    @pytest.mark.asyncio
    async def test_blocked_command(self, shell):
        result = await shell.run("rm -rf /")
        assert result.blocked
        assert not result.success
    
    def test_cd(self, shell):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = shell.cd(tmpdir)
            assert result.success
            assert shell.cwd == tmpdir
    
    def test_cd_nonexistent(self, shell):
        result = shell.cd("/nonexistent/path/12345")
        assert not result.success
        assert "not found" in result.error.lower()


class TestWorkspace:
    """Tests for Workspace."""
    
    @pytest.fixture
    def workspace(self):
        Workspace.reset()
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = Workspace(tmpdir)
            yield ws
            Workspace.reset()
    
    def test_create_workspace(self, workspace):
        assert workspace.root_dir is not None
        assert workspace.cwd == workspace.root_dir
    
    def test_register_file(self, workspace):
        workspace.register_file("test.txt", agent_id="agent_1")
        files = workspace.get_files()
        assert "test.txt" in files
    
    def test_register_directory(self, workspace):
        workspace.register_directory("subdir", agent_id="agent_1")
        dirs = workspace.get_directories()
        assert "subdir" in dirs
    
    def test_log_activity(self, workspace):
        workspace.log_activity("agent_1", "task_1", "test", "Test activity")
        activities = workspace.get_recent_activities()
        assert len(activities) == 1
        assert activities[0].agent_id == "agent_1"
    
    def test_shared_variables(self, workspace):
        workspace.set_variable("key", "value")
        assert workspace.get_variable("key") == "value"
        assert workspace.get_variable("nonexistent", "default") == "default"
    
    def test_agent_tracking(self, workspace):
        workspace.set_agent_active("agent_1", "Working on task")
        active = workspace.get_active_agents()
        assert "agent_1" in active
        
        workspace.set_agent_inactive("agent_1")
        active = workspace.get_active_agents()
        assert "agent_1" not in active


class TestConfig:
    """Tests for Config."""
    
    def test_default_config(self):
        config = Config()
        assert config.llm.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.execution.max_parallel_agents == 3
    
    def test_testing_config(self):
        config = Config.for_testing()
        assert config.llm.load_in_4bit == False
        assert config.execution.persist_state == False
    
    def test_env_config(self, monkeypatch):
        monkeypatch.setenv("AGENT_LLM_MODEL", "test-model")
        monkeypatch.setenv("AGENT_MAX_PARALLEL", "5")
        
        config = Config.from_env()
        assert config.llm.model_name == "test-model"
        assert config.execution.max_parallel_agents == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
