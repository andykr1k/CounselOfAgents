"""
Orchestrator Agent for the Counsel Of Agents Orchestration Platform.

This module provides the central coordination layer for multi-agent task execution.
It handles task planning, dependency management, parallel execution, and verification.

Key Components:
    - Orchestrator: Main coordinator class
    - ExecutionResult: Comprehensive result container
    - Planning system prompt: LLM-based task decomposition
"""

import asyncio
import json
import re
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from counsel.config import Config, get_config
from counsel.llm import LLM, Message, get_llm
from counsel.task_graph import TaskGraph, Task, TaskStatus
from counsel.agent import AgentPool, AgentResult, DebugCallback
from counsel.workspace import Workspace, get_workspace
from counsel.verification import (
    TaskVerifier, VerificationManager, VerificationResult, VerificationStatus,
    get_verification_manager
)


PLANNING_SYSTEM_PROMPT = """You are an expert task planner for a multi-agent system. Your job is to break down complex tasks into smaller, executable subtasks with proper dependencies.

## Your Role

Given a high-level task, you must:
1. Analyze what needs to be done
2. Break it down into atomic subtasks that can be executed independently
3. Identify dependencies between subtasks
4. Output a structured task graph

## Workspace Context

{workspace_context}

## Agent Capabilities

Each agent has access to:
1. **File Operations** (preferred for file work):
   - `<read_file>` - Read file contents with line numbers
   - `<write_file>` - Create or overwrite files
   - `<edit_file>` - Edit specific parts of files
   - `<list_dir>` - List directory contents

2. **Shell Commands** (for running programs):
   - `<shell>` - Run commands like `python script.py`, `pip install`, `mkdir dirname`, etc.

3. **Help System** (when stuck):
   - `<help>` - Request guidance from a supervisor

## CRITICAL: Agent Environment Limitations

Each agent runs shell commands in **separate subprocesses**. This means:
- Environment variables DO NOT persist between commands
- Virtual environment activation DOES NOT persist
- Shell state (except `cd`) is NOT shared between commands

**DO NOT create separate tasks for:**
- "Activate virtual environment" (won't persist anyway)
- "Set environment variables" (won't persist)
- "Change shell settings"

**Instead, bundle related work together:**
- "Create venv AND install dependencies" (one task using ./venv/bin/pip)
- "Write code AND run tests" (can be one task)

**For Python projects, agents should use venv binaries directly:**
- `./venv/bin/python`, `./venv/bin/pip`, etc.

## Output Format

You MUST respond with a valid JSON object in this exact format:

```json
{{
  "analysis": "Brief analysis of the task",
  "tasks": [
    {{
      "id": "task_1",
      "description": "Clear description of what this subtask should accomplish",
      "dependencies": []
    }},
    {{
      "id": "task_2", 
      "description": "Another subtask description",
      "dependencies": ["task_1"]
    }}
  ]
}}
```

## Rules for Task Decomposition

1. **Atomic Tasks**: Each task should be completable by a single agent with shell access
2. **Clear Descriptions**: Each description should be specific and actionable
3. **Proper Dependencies**: A task should only depend on tasks whose output it needs
4. **Parallel When Possible**: Independent tasks should NOT have false dependencies
5. **Use Existing Files**: Reference existing files/directories from workspace context
6. **Don't over-decompose**: Simple projects (few files) can be 2-4 tasks, not 10+
7. **Bundle environment setup**: Venv creation + pip install should be ONE task

## Examples

Task: "Create a Python calculator CLI"
```json
{{
  "analysis": "Need project structure, calculator code, and entry point",
  "tasks": [
    {{
      "id": "task_1",
      "description": "Create project directory 'calculator' with a venv and install any needed packages using ./venv/bin/pip",
      "dependencies": []
    }},
    {{
      "id": "task_2",
      "description": "Create calculator.py with add, subtract, multiply, divide functions",
      "dependencies": ["task_1"]
    }},
    {{
      "id": "task_3",
      "description": "Create main.py CLI entry point that imports calculator and provides interactive menu",
      "dependencies": ["task_2"]
    }},
    {{
      "id": "task_4",
      "description": "Test the calculator by running ./venv/bin/python main.py with sample inputs",
      "dependencies": ["task_3"]
    }}
  ]
}}
```

Now analyze and plan the following task:"""


@dataclass
class ExecutionResult:
    """
    Comprehensive result of executing a task graph.
    
    Contains:
        - Overall success/failure status
        - Task graph with final states
        - Individual agent results
        - Verification results (if enabled)
        - Performance metrics
        - Error information
    """
    success: bool
    task_graph: TaskGraph
    results: Dict[str, AgentResult] = field(default_factory=dict)
    verification_results: Dict[str, Dict] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    # Performance metrics
    total_duration_ms: int = 0
    planning_duration_ms: int = 0
    execution_duration_ms: int = 0
    verification_duration_ms: int = 0
    
    # Retry tracking
    tasks_retried: List[str] = field(default_factory=list)
    retry_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "task_graph": self.task_graph.to_dict(),
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "verification_results": self.verification_results,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms,
            "planning_duration_ms": self.planning_duration_ms,
            "execution_duration_ms": self.execution_duration_ms,
            "verification_duration_ms": self.verification_duration_ms,
            "tasks_retried": self.tasks_retried,
            "retry_count": self.retry_count
        }
    
    def get_files_created(self) -> List[str]:
        """Get all files created across all tasks."""
        files = []
        for result in self.results.values():
            files.extend(result.files_created)
        return files
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """Get a summary of verification results."""
        if not self.verification_results:
            return {"enabled": False}
        
        passed = sum(1 for v in self.verification_results.values() if v.get("status") == "passed")
        failed = sum(1 for v in self.verification_results.values() if v.get("status") == "failed")
        partial = sum(1 for v in self.verification_results.values() if v.get("status") == "partial")
        
        return {
            "enabled": True,
            "passed": passed,
            "failed": failed,
            "partial": partial,
            "total": len(self.verification_results),
            "pass_rate": passed / len(self.verification_results) if self.verification_results else 0
        }


class Orchestrator:
    """
    Main orchestrator that plans and coordinates task execution.
    
    The Orchestrator is the central coordinator for the Counsel platform.
    It provides:
        - LLM-based task planning and decomposition
        - Parallel task execution with dependency management
        - Optional task verification with retry logic
        - Progress reporting and debugging
        - Performance metrics collection
    
    Usage:
        orchestrator = Orchestrator(config=config)
        graph = await orchestrator.plan("Create a Python web app")
        result = await orchestrator.execute()
    
    With verification:
        result = await orchestrator.execute(verify_tasks=True)
        if not result.success:
            # Check verification results
            for task_id, verification in result.verification_results.items():
                if verification['status'] == 'failed':
                    print(f"Task {task_id} failed verification: {verification['summary']}")
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        llm: Optional[LLM] = None,
        workspace: Optional[Workspace] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        debug_callback: Optional[DebugCallback] = None,
        verification_manager: Optional[VerificationManager] = None
    ):
        self.config = config or get_config()
        self.llm = llm or get_llm(self.config.llm)
        self.workspace = workspace or get_workspace()
        self.progress_callback = progress_callback
        self.debug_callback = debug_callback
        self.verification_manager = verification_manager
        self._current_graph: Optional[TaskGraph] = None
    
    def _update_progress(self, status: str, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(status, message)
    
    def _parse_task_graph(self, response: str) -> TaskGraph:
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        
        if "tasks" not in data:
            raise ValueError("Response missing 'tasks' field")
        
        graph = TaskGraph()
        
        for task_data in data["tasks"]:
            if not isinstance(task_data, dict):
                continue
            
            task_id = task_data.get("id", f"task_{len(graph) + 1}")
            description = task_data.get("description", "")
            dependencies = task_data.get("dependencies", [])
            
            if not description:
                continue
            
            graph.add_task(
                task_id=task_id,
                description=description,
                dependencies=dependencies,
                metadata={"raw": task_data}
            )
        
        return graph
    
    async def plan(self, user_request: str) -> TaskGraph:
        self._update_progress("planning", "Analyzing task and creating execution plan...")
        
        workspace_context = self.workspace.get_context_for_agent()
        system_prompt = PLANNING_SYSTEM_PROMPT.format(workspace_context=workspace_context)
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Task: {user_request}")
        ]
        
        response = await self.llm.chat(messages)
        
        try:
            graph = self._parse_task_graph(response)
        except ValueError:
            self._update_progress("planning", "Refining plan...")
            
            messages.append(Message(role="assistant", content=response))
            messages.append(Message(
                role="user",
                content="Please provide the task breakdown as valid JSON with 'tasks' array."
            ))
            
            response = await self.llm.chat(messages)
            graph = self._parse_task_graph(response)
        
        graph.finalize()
        self._current_graph = graph
        self._update_progress("planned", f"Created {len(graph)} tasks")
        
        self.workspace.log_activity("orchestrator", "", "planned", f"Created {len(graph)} tasks")
        
        return graph
    
    async def execute(
        self,
        graph: Optional[TaskGraph] = None,
        continue_on_failure: Optional[bool] = None,
        verify_tasks: Optional[bool] = None,
        max_retries: Optional[int] = None
    ) -> ExecutionResult:
        """
        Execute the task graph with optional verification and retry logic.
        
        Args:
            graph: Task graph to execute (uses current if not provided)
            continue_on_failure: Continue executing other tasks if one fails
            verify_tasks: Enable task verification (uses config.verification.enabled if not specified)
            max_retries: Max retries for failed verifications (uses config.verification.max_retries if not specified)
            
        Returns:
            ExecutionResult with comprehensive status and metrics
        """
        execution_start = datetime.now()
        
        graph = graph or self._current_graph
        if not graph:
            raise ValueError("No task graph to execute")
        
        # Use config defaults if not specified
        continue_on_failure = (
            continue_on_failure 
            if continue_on_failure is not None 
            else self.config.execution.continue_on_failure
        )
        verify_tasks = (
            verify_tasks 
            if verify_tasks is not None 
            else self.config.verification.enabled
        )
        max_retries = (
            max_retries
            if max_retries is not None
            else self.config.verification.max_retries
        )
        
        result = ExecutionResult(success=False, task_graph=graph)
        
        # Initialize verification manager if needed
        if verify_tasks and not self.verification_manager:
            self.verification_manager = get_verification_manager()
        
        # Track when we last saved to debounce disk writes
        last_save_time = datetime.now()
        save_interval_seconds = 5  # Only save every 5 seconds at most
        
        pool = AgentPool(
            max_agents=self.config.execution.max_parallel_agents,
            llm=self.llm,
            supervisor_config=self.config.supervisor,
            workspace=self.workspace,
            debug_callback=self.debug_callback
        )
        
        self._update_progress("executing", "Starting task execution...")
        
        completed_results: Dict[str, Any] = {}
        running_tasks: Dict[str, asyncio.Task] = {}  # task_id -> asyncio.Task
        task_retry_counts: Dict[str, int] = {}  # Track retries per task
        failed = False
        
        async def run_single_task(task: Task, retry_context: Optional[str] = None) -> tuple[str, AgentResult]:
            """Run a single task and return its result."""
            context = {}
            for dep_id in task.dependencies:
                if dep_id in completed_results:
                    # Include both the result text and structured info
                    dep_result = result.results.get(dep_id)
                    if dep_result:
                        context[dep_id] = {
                            'result': completed_results[dep_id],
                            'files_created': dep_result.files_created,
                            'files_modified': dep_result.files_modified,
                            'success': dep_result.success
                        }
                    else:
                        context[dep_id] = {'result': completed_results[dep_id]}
            
            # If this is a retry, inject remediation context
            if retry_context:
                context['_remediation'] = retry_context
            
            agent_result = await pool.execute_task(task, context)
            return task.id, agent_result
        
        async def verify_and_maybe_retry(task_id: str, agent_result: AgentResult) -> Tuple[bool, Optional[str]]:
            """
            Verify a completed task and determine if retry is needed.
            
            Returns:
                (should_retry, remediation_instructions)
            """
            if not verify_tasks or not self.verification_manager:
                return False, None
            
            task = graph.get_task(task_id)
            if not task:
                return False, None
            
            self._update_progress("verifying", f"🔍 Verifying {task_id}...")
            
            verification = await self.verification_manager.verify_task(
                task=task,
                agent_result=agent_result.result,
                shell_history=agent_result.shell_history,
                files_created=agent_result.files_created,
                files_modified=agent_result.files_modified
            )
            
            # Store verification result
            result.verification_results[task_id] = verification.to_dict()
            
            if verification.passed:
                self._update_progress("verified", f"✓ {task_id} verified ({verification.score:.0%})")
                return False, None
            
            # Check if we should retry
            retry_count = task_retry_counts.get(task_id, 0)
            should_retry = self.verification_manager.should_retry(verification, retry_count)
            
            if should_retry:
                remediation = self.verification_manager.get_remediation_prompt(verification)
                self._update_progress(
                    "retry_needed",
                    f"⚠ {task_id} needs remediation (attempt {retry_count + 1}/{max_retries})"
                )
                return True, remediation
            
            self._update_progress(
                "verification_failed",
                f"✗ {task_id} failed verification: {verification.summary}"
            )
            return False, None
        
        # Main execution loop - starts tasks as soon as they're ready
        while not graph.is_complete() and not failed:
            # Start any newly ready tasks
            ready_tasks = graph.get_ready_tasks()
            for task in ready_tasks:
                if task.id not in running_tasks:
                    graph.mark_running(task.id)
                    total_running = sum(1 for t in graph.tasks.values() if t.status == TaskStatus.RUNNING)
                    self._update_progress("task_started", f"▶ Starting {task.id} ({total_running} running)")
                    # Create task and start it immediately
                    running_tasks[task.id] = asyncio.create_task(run_single_task(task))
            
            if not running_tasks:
                # No tasks running and none ready - we're done or stuck
                break
            
            # Wait for ANY task to complete (not all of them)
            done, pending = await asyncio.wait(
                running_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for completed_task in done:
                task_id, agent_result = await completed_task
                
                # Remove from running
                del running_tasks[task_id]
                
                result.results[task_id] = agent_result
                
                if agent_result.success:
                    # Verify the task if verification is enabled
                    should_retry, remediation = await verify_and_maybe_retry(task_id, agent_result)
                    
                    if should_retry and remediation:
                        # Reset task for retry
                        retry_count = task_retry_counts.get(task_id, 0) + 1
                        task_retry_counts[task_id] = retry_count
                        result.tasks_retried.append(task_id)
                        result.retry_count += 1
                        
                        # Re-queue the task with remediation context
                        task = graph.get_task(task_id)
                        if task:
                            task.status = TaskStatus.READY
                            # Inject remediation into task description
                            original_desc = task.metadata.get('original_description', task.description)
                            task.metadata['original_description'] = original_desc
                            task.description = f"{original_desc}\n\n{remediation}"
                        continue
                    
                    # Task passed (or verification not enabled)
                    graph.mark_completed(task_id, agent_result.result)
                    completed_results[task_id] = agent_result.result
                    self._update_progress("task_done", f"✓ {task_id} completed")
                    
                    # Store comprehensive result in workspace for other agents
                    self.workspace.set_variable(f"result_{task_id}", agent_result.result)
                    self.workspace.set_variable(f"files_created_{task_id}", agent_result.files_created)
                    self.workspace.set_variable(f"files_modified_{task_id}", agent_result.files_modified)
                    
                    # Log what was accomplished
                    self.workspace.log_activity(
                        "orchestrator", task_id, "task_completed",
                        f"Completed {task_id}: {len(agent_result.files_created)} files created, {len(agent_result.files_modified)} modified"
                    )
                else:
                    graph.mark_failed(task_id, agent_result.error or "Unknown error")
                    self._update_progress("task_failed", f"✗ {task_id} failed: {agent_result.error}")
                    
                    if not continue_on_failure:
                        result.error = f"Task {task_id} failed: {agent_result.error}"
                        failed = True
                        # Cancel remaining tasks
                        for remaining in running_tasks.values():
                            remaining.cancel()
                        break
            
            # Debounced save - only write to disk every few seconds
            if self.config.execution.persist_state and not failed:
                now = datetime.now()
                if (now - last_save_time).total_seconds() >= save_interval_seconds:
                    graph.save(self.config.execution.state_file)
                    last_save_time = now
        
        # Wait for any remaining tasks if we didn't fail
        if running_tasks and not failed:
            remaining_results = await asyncio.gather(*running_tasks.values(), return_exceptions=True)
            for res in remaining_results:
                if isinstance(res, tuple):
                    task_id, agent_result = res
                    result.results[task_id] = agent_result
                    if agent_result.success:
                        graph.mark_completed(task_id, agent_result.result)
                        self._update_progress("task_done", f"✓ {task_id} completed")
                    else:
                        graph.mark_failed(task_id, agent_result.error or "Unknown error")
        
        result.success = graph.is_successful()
        result.completed_at = datetime.now().isoformat()
        
        # Calculate execution duration
        result.execution_duration_ms = int(
            (datetime.now() - execution_start).total_seconds() * 1000
        )
        
        # Final save to ensure last state is persisted
        if self.config.execution.persist_state:
            graph.save(self.config.execution.state_file)
        
        if result.success:
            verification_summary = result.get_verification_summary()
            if verification_summary.get("enabled"):
                self._update_progress(
                    "complete",
                    f"All tasks completed and verified ({verification_summary['pass_rate']:.0%} pass rate)"
                )
            else:
                self._update_progress("complete", "All tasks completed successfully")
        else:
            summary = graph.get_summary()
            self._update_progress("complete", f"Finished with {summary['failed']} failed, {summary['blocked']} blocked")
        
        return result
    
    async def run(
        self,
        user_request: str,
        verify_tasks: Optional[bool] = None,
        max_retries: Optional[int] = None
    ) -> ExecutionResult:
        """
        Plan and execute a user request in a single call.
        
        Args:
            user_request: Natural language description of what to do
            verify_tasks: Enable task verification (uses config.verification.enabled if not specified)
            max_retries: Max retries for verifications (uses config.verification.max_retries if not specified)
            
        Returns:
            ExecutionResult with comprehensive status and metrics
        """
        total_start = datetime.now()
        
        # Planning phase
        plan_start = datetime.now()
        graph = await self.plan(user_request)
        planning_duration = int((datetime.now() - plan_start).total_seconds() * 1000)
        
        self._update_progress("graph", graph.visualize())
        
        # Execution phase
        result = await self.execute(graph, verify_tasks=verify_tasks, max_retries=max_retries)
        
        # Update timing metrics
        result.planning_duration_ms = planning_duration
        result.total_duration_ms = int((datetime.now() - total_start).total_seconds() * 1000)
        
        return result
    
    def get_current_graph(self) -> Optional[TaskGraph]:
        return self._current_graph
    
    def get_workspace(self) -> Workspace:
        return self.workspace
    
    def visualize(self) -> str:
        if self._current_graph:
            return self._current_graph.visualize()
        return "No task graph"


async def run_task(user_request: str, config: Optional[Config] = None) -> ExecutionResult:
    """Run a task from start to finish."""
    orchestrator = Orchestrator(config=config)
    return await orchestrator.run(user_request)
