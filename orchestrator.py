"""Orchestrator agent for planning and coordinating task execution."""

import asyncio
import json
import re
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from config import Config, ExecutionConfig, get_config
from llm import LLM, Message, get_llm
from task_graph import TaskGraph, Task, TaskStatus
from agent import AgentPool, AgentResult
from workspace import Workspace, get_workspace


PLANNING_SYSTEM_PROMPT = """You are an expert task planner for a multi-agent system. Your job is to break down complex tasks into smaller, executable subtasks with proper dependencies.

## Your Role

Given a high-level task, you must:
1. Analyze what needs to be done
2. Break it down into atomic subtasks that can be executed independently
3. Identify dependencies between subtasks
4. Output a structured task graph

## Workspace Context

{workspace_context}

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
6. **Coordinate Work**: If multiple tasks work in same area, set up proper dependencies

## Examples

Task: "Create a Python project with a hello world script"
```json
{{
  "analysis": "Need to create directory structure and a simple Python file",
  "tasks": [
    {{
      "id": "task_1",
      "description": "Create a new directory called 'hello_project' for the Python project",
      "dependencies": []
    }},
    {{
      "id": "task_2",
      "description": "Create a file called 'hello.py' inside hello_project with a hello world program that prints 'Hello, World!'",
      "dependencies": ["task_1"]
    }}
  ]
}}
```

Task: "Set up a basic Express.js server with a health endpoint"
```json
{{
  "analysis": "Need to initialize npm, install express, and create server file",
  "tasks": [
    {{
      "id": "task_1",
      "description": "Create a new directory called 'express-server' and initialize it with npm init -y",
      "dependencies": []
    }},
    {{
      "id": "task_2",
      "description": "Install express package using npm install express",
      "dependencies": ["task_1"]
    }},
    {{
      "id": "task_3",
      "description": "Create index.js with an Express server that listens on port 3000 and has a GET /health endpoint returning {{status: 'ok'}}",
      "dependencies": ["task_2"]
    }}
  ]
}}
```

Now analyze and plan the following task:"""


@dataclass
class ExecutionResult:
    """Result of executing a task graph."""
    success: bool
    task_graph: TaskGraph
    results: Dict[str, AgentResult] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "task_graph": self.task_graph.to_dict(),
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }
    
    def get_summary(self) -> str:
        """Get a human-readable summary."""
        graph_summary = self.task_graph.get_summary()
        
        lines = [
            "=" * 50,
            "Execution Summary",
            "=" * 50,
            f"Status: {'Success' if self.success else 'Failed'}",
            f"Total tasks: {len(self.task_graph)}",
            f"  Completed: {graph_summary['completed']}",
            f"  Failed: {graph_summary['failed']}",
            f"  Blocked: {graph_summary['blocked']}",
        ]
        
        if self.error:
            lines.append(f"Error: {self.error}")
        
        lines.append("\nTask Results:")
        for task_id, result in self.results.items():
            status = "✓" if result.success else "✗"
            lines.append(f"  {status} {task_id}: {result.result or result.error}")
        
        return "\n".join(lines)
    
    def get_files_created(self) -> list[str]:
        """Get all files created during execution."""
        files = []
        for result in self.results.values():
            files.extend(result.files_created)
        return files


class Orchestrator:
    """
    Main orchestrator that plans and coordinates task execution.
    
    The orchestrator:
    1. Takes a high-level user request
    2. Uses LLM to decompose it into a task graph
    3. Manages a shared workspace for agent coordination
    4. Executes tasks respecting dependencies
    5. Enables parallel execution where possible
    6. Aggregates and returns results
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        llm: Optional[LLM] = None,
        workspace: Optional[Workspace] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Configuration object
            llm: LLM instance (will create one if not provided)
            workspace: Shared workspace (will create one if not provided)
            progress_callback: Optional callback(status, message) for progress updates
        """
        self.config = config or get_config()
        self.llm = llm or get_llm(self.config.llm)
        self.workspace = workspace or get_workspace()
        self.progress_callback = progress_callback
        self._current_graph: Optional[TaskGraph] = None
    
    def _update_progress(self, status: str, message: str) -> None:
        """Send a progress update if callback is registered."""
        if self.progress_callback:
            self.progress_callback(status, message)
    
    def _parse_task_graph(self, response: str) -> TaskGraph:
        """Parse LLM response into a TaskGraph."""
        # Try to extract JSON from response
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
        """
        Plan the execution by decomposing the user request into tasks.
        
        Args:
            user_request: The high-level task from the user
            
        Returns:
            TaskGraph with the planned tasks
        """
        self._update_progress("planning", "Analyzing task and creating execution plan...")
        
        # Get current workspace context
        workspace_context = self.workspace.get_context_for_agent()
        
        # Build planning prompt
        system_prompt = PLANNING_SYSTEM_PROMPT.format(workspace_context=workspace_context)
        
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Task: {user_request}")
        ]
        
        # Get planning response
        response = await self.llm.chat(messages)
        
        # Parse into task graph
        try:
            graph = self._parse_task_graph(response)
        except ValueError as e:
            self._update_progress("planning", "Refining plan...")
            
            messages.append(Message(role="assistant", content=response))
            messages.append(Message(
                role="user",
                content="Please provide the task breakdown as valid JSON with 'tasks' array. Each task needs 'id', 'description', and 'dependencies' fields."
            ))
            
            response = await self.llm.chat(messages)
            graph = self._parse_task_graph(response)
        
        # Finalize the graph
        graph.finalize()
        
        self._current_graph = graph
        self._update_progress("planned", f"Created {len(graph)} tasks")
        
        # Log planning activity
        self.workspace.log_activity(
            "orchestrator", "", "planned",
            f"Created {len(graph)} tasks for: {user_request[:50]}"
        )
        
        return graph
    
    async def execute(
        self,
        graph: Optional[TaskGraph] = None,
        continue_on_failure: Optional[bool] = None
    ) -> ExecutionResult:
        """
        Execute a task graph with parallel agent coordination.
        
        Args:
            graph: TaskGraph to execute (uses current if not provided)
            continue_on_failure: Whether to continue if a task fails
            
        Returns:
            ExecutionResult with all task outcomes
        """
        graph = graph or self._current_graph
        if not graph:
            raise ValueError("No task graph to execute")
        
        continue_on_failure = (
            continue_on_failure 
            if continue_on_failure is not None 
            else self.config.execution.continue_on_failure
        )
        
        result = ExecutionResult(success=False, task_graph=graph)
        
        # Create agent pool with shared workspace
        pool = AgentPool(
            max_agents=self.config.execution.max_parallel_agents,
            llm=self.llm,
            workspace=self.workspace
        )
        
        self._update_progress("executing", "Starting task execution...")
        
        # Store results for context passing
        completed_results: Dict[str, Any] = {}
        
        # Execute until complete
        while not graph.is_complete():
            ready_tasks = graph.get_ready_tasks()
            
            if not ready_tasks:
                if not any(t.status == TaskStatus.RUNNING for t in graph.tasks.values()):
                    break
                await asyncio.sleep(0.1)
                continue
            
            self._update_progress(
                "executing",
                f"Running {len(ready_tasks)} task(s) in parallel..."
            )
            
            # Mark tasks as running
            for task in ready_tasks:
                graph.mark_running(task.id)
            
            # Build context for each task from dependencies
            contexts: Dict[str, Dict[str, Any]] = {}
            for task in ready_tasks:
                task_context = {}
                for dep_id in task.dependencies:
                    if dep_id in completed_results:
                        task_context[dep_id] = completed_results[dep_id]
                contexts[task.id] = task_context
            
            # Execute tasks in parallel
            task_results = await pool.execute_tasks(ready_tasks, contexts)
            
            # Process results
            for task_id, agent_result in task_results.items():
                result.results[task_id] = agent_result
                
                if agent_result.success:
                    graph.mark_completed(task_id, agent_result.result)
                    completed_results[task_id] = agent_result.result
                    self._update_progress("task_done", f"✓ {task_id} completed")
                    
                    # Store result in workspace for other tasks
                    self.workspace.set_variable(f"result_{task_id}", agent_result.result)
                else:
                    graph.mark_failed(task_id, agent_result.error or "Unknown error")
                    self._update_progress("task_failed", f"✗ {task_id} failed: {agent_result.error}")
                    
                    if not continue_on_failure:
                        result.error = f"Task {task_id} failed: {agent_result.error}"
                        result.completed_at = datetime.now().isoformat()
                        return result
            
            # Save state if persistence enabled
            if self.config.execution.persist_state:
                graph.save(self.config.execution.state_file)
        
        # Final status
        result.success = graph.is_successful()
        result.completed_at = datetime.now().isoformat()
        
        if result.success:
            self._update_progress("complete", "All tasks completed successfully")
        else:
            failed = graph.get_summary()["failed"]
            blocked = graph.get_summary()["blocked"]
            self._update_progress("complete", f"Finished with {failed} failed, {blocked} blocked")
        
        return result
    
    async def run(self, user_request: str) -> ExecutionResult:
        """
        Plan and execute a user request in one call.
        
        Args:
            user_request: The high-level task from the user
            
        Returns:
            ExecutionResult with all outcomes
        """
        graph = await self.plan(user_request)
        self._update_progress("graph", graph.visualize())
        return await self.execute(graph)
    
    def get_current_graph(self) -> Optional[TaskGraph]:
        """Get the current task graph."""
        return self._current_graph
    
    def get_workspace(self) -> Workspace:
        """Get the shared workspace."""
        return self.workspace
    
    def visualize(self) -> str:
        """Get visualization of current graph."""
        if self._current_graph:
            return self._current_graph.visualize()
        return "No task graph"


# Convenience function
async def run_task(user_request: str, config: Optional[Config] = None) -> ExecutionResult:
    """Run a task from start to finish."""
    orchestrator = Orchestrator(config=config)
    return await orchestrator.run(user_request)
