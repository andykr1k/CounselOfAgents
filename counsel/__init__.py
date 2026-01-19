"""
Counsel Of Agents Orchestration Platform

Enterprise-grade multi-agent orchestration system for automated task execution.
Breaks down complex tasks into dependency graphs and executes them in parallel
using intelligent agents with shell access, verification, and self-correction.

Key Features:
    - LLM-powered task planning and decomposition
    - Parallel execution with dependency management
    - Task verification and automatic remediation
    - Stuck detection with supervisor guidance
    - Professional logging and metrics
    - Job persistence and recovery

Example:
    from counsel import Orchestrator, Config
    
    config = Config.from_env()
    orchestrator = Orchestrator(config=config)
    
    result = await orchestrator.run(
        "Create a REST API with user authentication",
        verify_tasks=True
    )
"""

from counsel.config import Config, SupervisorConfig, get_config, set_config
from counsel.orchestrator import Orchestrator, ExecutionResult, run_task
from counsel.agent import Agent, AgentPool, AgentResult
from counsel.task_graph import TaskGraph, Task, TaskStatus
from counsel.workspace import Workspace, get_workspace
from counsel.llm import LLM, get_llm
from counsel.shell import Shell, ShellResult, get_shell
from counsel.jobs import Job, JobStatus, JobManager, JobMessage, get_job_manager
from counsel.verification import (
    TaskVerifier,
    VerificationManager,
    VerificationResult,
    VerificationStatus,
    VerificationIssue,
    get_verifier,
    get_verification_manager
)
from counsel.logging import (
    CounselLogger,
    get_logger,
    configure_logging,
    LogLevel
)

__version__ = "1.3.0"
__author__ = "Counsel Of Agents"
__license__ = "MIT"

__all__ = [
    # Configuration
    "Config",
    "SupervisorConfig",
    "get_config", 
    "set_config",
    
    # Orchestration
    "Orchestrator",
    "ExecutionResult",
    "run_task",
    
    # Agents
    "Agent",
    "AgentPool",
    "AgentResult",
    
    # Task Graph
    "TaskGraph",
    "Task",
    "TaskStatus",
    
    # Workspace
    "Workspace",
    "get_workspace",
    
    # LLM
    "LLM",
    "get_llm",
    
    # Shell
    "Shell",
    "ShellResult",
    "get_shell",
    
    # Jobs
    "Job",
    "JobStatus",
    "JobManager",
    "JobMessage",
    "get_job_manager",
    
    # Verification
    "TaskVerifier",
    "VerificationManager",
    "VerificationResult",
    "VerificationStatus",
    "VerificationIssue",
    "get_verifier",
    "get_verification_manager",
    
    # Logging
    "CounselLogger",
    "get_logger",
    "configure_logging",
    "LogLevel",
]
