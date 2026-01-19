"""
Counsel of Agents - Multi-agent orchestration system.

A system that breaks down complex tasks into a dependency graph
and executes them in parallel using general-purpose agents with shell access.
"""

from counsel.config import Config, get_config, set_config
from counsel.orchestrator import Orchestrator, ExecutionResult, run_task
from counsel.agent import Agent, AgentPool, AgentResult
from counsel.task_graph import TaskGraph, Task, TaskStatus
from counsel.workspace import Workspace, get_workspace
from counsel.llm import LLM, get_llm
from counsel.shell import Shell, ShellResult, get_shell
from counsel.jobs import Job, JobStatus, JobManager, JobMessage, get_job_manager

__version__ = "0.3.0"
__all__ = [
    "Config",
    "get_config", 
    "set_config",
    "Orchestrator",
    "ExecutionResult",
    "run_task",
    "Agent",
    "AgentPool",
    "AgentResult",
    "TaskGraph",
    "Task",
    "TaskStatus",
    "Workspace",
    "get_workspace",
    "LLM",
    "get_llm",
    "Shell",
    "ShellResult",
    "get_shell",
    "Job",
    "JobStatus",
    "JobManager",
    "JobMessage",
    "get_job_manager",
]
