"""Task graph for managing task dependencies and parallel execution."""

from typing import Dict, List, Set, Optional
from collections import deque
from agent import Task


class TaskGraph:
    """Manages task dependencies and execution order."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.dependencies: Dict[str, Set[str]] = {}  # task_id -> set of dependency IDs
        self.reverse_dependencies: Dict[str, Set[str]] = {}  # task_id -> set of tasks that depend on it
    
    def add_task(self, task: Task) -> None:
        """Add a task to the graph."""
        self.tasks[task.id] = task
        self.dependencies[task.id] = set(task.dependencies)
        
        # Update reverse dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.reverse_dependencies:
                self.reverse_dependencies[dep_id] = set()
            self.reverse_dependencies[dep_id].add(task.id)
        
        # Initialize reverse dependencies for this task if not exists
        if task.id not in self.reverse_dependencies:
            self.reverse_dependencies[task.id] = set()
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute (all dependencies completed)."""
        ready = []
        for task in self.tasks.values():
            if task.status == "pending":
                # Check if all dependencies are completed
                if all(
                    self.tasks[dep_id].status == "completed"
                    for dep_id in self.dependencies[task.id]
                    if dep_id in self.tasks
                ):
                    ready.append(task)
        return ready
    
    def mark_completed(self, task_id: str) -> None:
        """Mark a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
    
    def mark_in_progress(self, task_id: str) -> None:
        """Mark a task as in progress."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "in_progress"
    
    def mark_failed(self, task_id: str) -> None:
        """Mark a task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
    
    def get_all_tasks(self) -> List[Task]:
        """Get all tasks in the graph."""
        return list(self.tasks.values())
    
    def has_cycles(self) -> bool:
        """Check if the graph has cycles using DFS."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # Check dependencies
            for dep_id in self.dependencies.get(node_id, set()):
                if dep_id not in self.tasks:
                    continue
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for task_id in self.tasks:
            if task_id not in visited:
                if has_cycle(task_id):
                    return True
        return False
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get tasks grouped by execution level (tasks in same level can run in parallel).
        Returns a list of lists, where each inner list contains task IDs that can run in parallel.
        """
        if self.has_cycles():
            raise ValueError("Cannot determine execution order: graph has cycles")
        
        levels = []
        remaining = set(self.tasks.keys())
        completed = set()
        
        while remaining:
            # Find all tasks with no incomplete dependencies
            current_level = []
            for task_id in remaining:
                deps = self.dependencies.get(task_id, set())
                if all(dep_id in completed or dep_id not in self.tasks for dep_id in deps):
                    current_level.append(task_id)
            
            if not current_level:
                # This shouldn't happen if graph is acyclic, but handle it
                break
            
            levels.append(current_level)
            completed.update(current_level)
            remaining -= set(current_level)
        
        return levels
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status == "completed" for task in self.tasks.values())
    
    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks."""
        return [task for task in self.tasks.values() if task.status == "failed"]
