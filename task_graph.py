"""Task graph (DAG) for managing task dependencies and parallel execution."""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from datetime import datetime


class TaskStatus(Enum):
    """Status of a task in the graph."""
    PENDING = "pending"        # Not yet ready (dependencies not met)
    READY = "ready"            # Ready to execute (all deps complete)
    RUNNING = "running"        # Currently being executed
    COMPLETED = "completed"    # Successfully completed
    FAILED = "failed"          # Failed during execution
    BLOCKED = "blocked"        # Blocked due to failed dependency


@dataclass
class Task:
    """A task in the execution graph."""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Task":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            dependencies=data.get("dependencies", []),
            status=TaskStatus(data.get("status", "pending")),
            result=data.get("result"),
            error=data.get("error"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metadata=data.get("metadata", {})
        )


class TaskGraph:
    """
    Directed Acyclic Graph (DAG) for task management.
    
    Handles:
    - Task dependencies
    - Execution order (topological sort)
    - Parallel execution levels
    - Status tracking
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self._dependents: Dict[str, Set[str]] = {}  # task_id -> tasks that depend on it
    
    def add_task(
        self,
        task_id: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Task:
        """
        Add a task to the graph.
        
        Args:
            task_id: Unique identifier for the task
            description: What the task should accomplish
            dependencies: List of task IDs this task depends on
            metadata: Additional task metadata
            
        Returns:
            The created Task
        """
        if task_id in self.tasks:
            raise ValueError(f"Task {task_id} already exists")
        
        deps = dependencies or []
        
        # Validate dependencies exist (or will be added)
        task = Task(
            id=task_id,
            description=description,
            dependencies=deps,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        
        # Track reverse dependencies (who depends on this task)
        for dep_id in deps:
            if dep_id not in self._dependents:
                self._dependents[dep_id] = set()
            # Don't add yet - wait until all tasks are added
        
        # Initialize dependents for this task
        if task_id not in self._dependents:
            self._dependents[task_id] = set()
        
        return task
    
    def finalize(self) -> None:
        """
        Finalize the graph after all tasks are added.
        
        This builds the reverse dependency map and validates the graph.
        """
        # Build reverse dependencies
        self._dependents = {task_id: set() for task_id in self.tasks}
        
        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id in self.tasks:
                    self._dependents[dep_id].add(task_id)
        
        # Check for cycles
        if self.has_cycle():
            raise ValueError("Task graph contains cycles")
        
        # Update initial status
        self._update_ready_tasks()
    
    def _update_ready_tasks(self) -> None:
        """Update status of tasks that are ready to run."""
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                if self._are_dependencies_met(task.id):
                    task.status = TaskStatus.READY
    
    def _are_dependencies_met(self, task_id: str) -> bool:
        """Check if all dependencies of a task are completed."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def _has_failed_dependency(self, task_id: str) -> bool:
        """Check if any dependency has failed."""
        task = self.tasks.get(task_id)
        if not task:
            return False
        
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.status in (TaskStatus.FAILED, TaskStatus.BLOCKED):
                return True
        
        return False
    
    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute."""
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.READY
        ]
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def mark_running(self, task_id: str) -> None:
        """Mark a task as running."""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
    
    def mark_completed(self, task_id: str, result: Any = None) -> None:
        """Mark a task as completed and update dependent tasks."""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now().isoformat()
            
            # Update tasks that depend on this one
            for dependent_id in self._dependents.get(task_id, []):
                dependent = self.tasks.get(dependent_id)
                if dependent and dependent.status == TaskStatus.PENDING:
                    if self._are_dependencies_met(dependent_id):
                        dependent.status = TaskStatus.READY
    
    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task as failed and block dependent tasks."""
        task = self.tasks.get(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.now().isoformat()
            
            # Block all tasks that depend on this one
            self._propagate_failure(task_id)
    
    def _propagate_failure(self, failed_task_id: str) -> None:
        """Propagate failure to all dependent tasks."""
        visited = set()
        queue = deque(self._dependents.get(failed_task_id, []))
        
        while queue:
            task_id = queue.popleft()
            if task_id in visited:
                continue
            visited.add(task_id)
            
            task = self.tasks.get(task_id)
            if task and task.status in (TaskStatus.PENDING, TaskStatus.READY):
                task.status = TaskStatus.BLOCKED
                task.error = f"Blocked due to failed dependency: {failed_task_id}"
                
                # Add this task's dependents to the queue
                queue.extend(self._dependents.get(task_id, []))
    
    def has_cycle(self) -> bool:
        """Check if the graph has cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {task_id: WHITE for task_id in self.tasks}
        
        def dfs(task_id: str) -> bool:
            color[task_id] = GRAY
            
            task = self.tasks[task_id]
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    continue
                if color[dep_id] == GRAY:
                    return True  # Back edge = cycle
                if color[dep_id] == WHITE and dfs(dep_id):
                    return True
            
            color[task_id] = BLACK
            return False
        
        for task_id in self.tasks:
            if color[task_id] == WHITE:
                if dfs(task_id):
                    return True
        
        return False
    
    def get_execution_levels(self) -> List[List[str]]:
        """
        Get tasks grouped by execution level.
        
        Tasks in the same level can be executed in parallel.
        
        Returns:
            List of lists of task IDs, where each inner list is a parallel level
        """
        if not self.tasks:
            return []
        
        levels = []
        remaining = set(self.tasks.keys())
        completed = set()
        
        while remaining:
            # Find tasks with all dependencies in completed set
            current_level = []
            for task_id in list(remaining):
                task = self.tasks[task_id]
                deps_met = all(
                    dep_id in completed or dep_id not in self.tasks
                    for dep_id in task.dependencies
                )
                if deps_met:
                    current_level.append(task_id)
            
            if not current_level:
                # Deadlock - remaining tasks have unmet dependencies
                break
            
            levels.append(current_level)
            completed.update(current_level)
            remaining -= set(current_level)
        
        return levels
    
    def is_complete(self) -> bool:
        """Check if all tasks are in a terminal state."""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.BLOCKED)
            for task in self.tasks.values()
        )
    
    def is_successful(self) -> bool:
        """Check if all tasks completed successfully."""
        return all(
            task.status == TaskStatus.COMPLETED
            for task in self.tasks.values()
        )
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of task statuses."""
        summary = {status.value: 0 for status in TaskStatus}
        for task in self.tasks.values():
            summary[task.status.value] += 1
        return summary
    
    def to_dict(self) -> Dict:
        """Serialize the graph to a dictionary."""
        return {
            "tasks": [task.to_dict() for task in self.tasks.values()],
            "created_at": datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "TaskGraph":
        """Deserialize a graph from a dictionary."""
        graph = cls()
        for task_data in data.get("tasks", []):
            task = Task.from_dict(task_data)
            graph.tasks[task.id] = task
        
        # Rebuild dependents map
        graph._dependents = {task_id: set() for task_id in graph.tasks}
        for task_id, task in graph.tasks.items():
            for dep_id in task.dependencies:
                if dep_id in graph.tasks:
                    graph._dependents[dep_id].add(task_id)
        
        return graph
    
    def save(self, filepath: str) -> None:
        """Save the graph to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "TaskGraph":
        """Load a graph from a JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))
    
    def visualize(self) -> str:
        """
        Create a text visualization of the task graph.
        
        Returns:
            ASCII representation of the graph
        """
        if not self.tasks:
            return "Empty task graph"
        
        status_icons = {
            TaskStatus.PENDING: "○",
            TaskStatus.READY: "◐",
            TaskStatus.RUNNING: "◑",
            TaskStatus.COMPLETED: "●",
            TaskStatus.FAILED: "✗",
            TaskStatus.BLOCKED: "◌"
        }
        
        lines = ["Task Graph", "=" * 50]
        
        # Group by execution level
        levels = self.get_execution_levels()
        
        for level_idx, level in enumerate(levels):
            lines.append(f"\nLevel {level_idx + 1}:")
            for task_id in level:
                task = self.tasks[task_id]
                icon = status_icons.get(task.status, "?")
                desc = task.description[:50] + "..." if len(task.description) > 50 else task.description
                deps = f" (deps: {', '.join(task.dependencies)})" if task.dependencies else ""
                lines.append(f"  {icon} [{task_id}] {desc}{deps}")
        
        # Summary
        summary = self.get_summary()
        lines.append("\n" + "-" * 50)
        lines.append(f"Total: {len(self.tasks)} tasks")
        lines.append(f"  ● Completed: {summary['completed']}")
        lines.append(f"  ◑ Running: {summary['running']}")
        lines.append(f"  ◐ Ready: {summary['ready']}")
        lines.append(f"  ○ Pending: {summary['pending']}")
        lines.append(f"  ✗ Failed: {summary['failed']}")
        lines.append(f"  ◌ Blocked: {summary['blocked']}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __contains__(self, task_id: str) -> bool:
        return task_id in self.tasks
