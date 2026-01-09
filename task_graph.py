"""Task graph for managing task dependencies and parallel execution."""

from typing import Dict, List, Set, Optional
from collections import deque
from agent import Task
try:
    from rich.tree import Tree
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


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
    
    def get_visualization(self) -> Optional[str]:
        """
        Get a visual representation of the task graph using Rich Tree.
        Returns a string representation if Rich is not available.
        
        Returns:
            Rich Tree object if Rich is available, otherwise formatted string
        """
        if not RICH_AVAILABLE:
            # Fallback to text representation
            lines = ["Task Graph:"]
            for task in self.tasks.values():
                status_color = {
                    "completed": "✓",
                    "pending": "○",
                    "in_progress": "→",
                    "failed": "✗"
                }
                icon = status_color.get(task.status, "•")
                deps_str = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
                lines.append(f"  {icon} {task.id}: {task.description[:50]}{deps_str}")
            return "\n".join(lines)
        
        # Status color mapping
        status_styles = {
            "completed": "[green]✓[/green]",
            "pending": "[yellow]○[/yellow]",
            "in_progress": "[blue]→[/blue]",
            "failed": "[red]✗[/red]"
        }
        
        # Find root tasks (tasks with no dependencies)
        root_tasks = [
            task_id for task_id, task in self.tasks.items()
            if not self.dependencies.get(task_id, set())
        ]
        
        # If no root tasks, show all tasks at top level
        if not root_tasks:
            root_tasks = list(self.tasks.keys())
        
        # Build tree structure
        tree = Tree("📋 Task Graph", guide_style="dim")
        added = set()
        
        def add_task_node(task_id: str, parent_node):
            """Recursively add task nodes to the tree."""
            if task_id in added:
                return
            
            task = self.tasks.get(task_id)
            if not task:
                return
            
            added.add(task_id)
            
            # Get status icon
            status_icon = status_styles.get(task.status, "•")
            
            # Truncate description if too long
            desc = task.description[:60] + "..." if len(task.description) > 60 else task.description
            
            # Create node label
            label = f"{status_icon} [bold]{task_id}[/bold]: {desc}"
            
            # Add node
            if parent_node is None:
                node = tree.add(label)
            else:
                node = parent_node.add(label)
            
            # Add child tasks (tasks that depend on this one)
            children = self.reverse_dependencies.get(task_id, set())
            for child_id in sorted(children):
                if child_id in self.tasks:
                    add_task_node(child_id, node)
        
        # Start from root tasks
        for root_id in sorted(root_tasks):
            add_task_node(root_id, None)
        
        # Add any remaining tasks that weren't connected (shouldn't happen in DAG, but handle it)
        for task_id in sorted(self.tasks.keys()):
            if task_id not in added:
                task = self.tasks[task_id]
                status_icon = status_styles.get(task.status, "•")
                desc = task.description[:60] + "..." if len(task.description) > 60 else task.description
                label = f"{status_icon} [bold]{task_id}[/bold]: {desc}"
                tree.add(label)
        
        return tree