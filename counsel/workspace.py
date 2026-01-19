"""Shared workspace for agent coordination and state tracking."""

import os
import json
import threading
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class FileInfo:
    """Information about a file in the workspace."""
    path: str
    created_by: Optional[str] = None
    modified_by: Optional[str] = None
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "path": self.path,
            "created_by": self.created_by,
            "modified_by": self.modified_by,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "description": self.description
        }


@dataclass
class AgentActivity:
    """Record of an agent's activity."""
    agent_id: str
    task_id: str
    action: str
    details: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "action": self.action,
            "details": self.details,
            "timestamp": self.timestamp
        }


class Workspace:
    """
    Shared workspace that tracks state across all agents.
    
    Provides:
    - Current working directory tracking
    - File registry (what files were created/modified)
    - Activity log (what agents have done)
    - Shared variables (key-value store for coordination)
    - Project structure awareness
    
    Thread-safe for concurrent agent access.
    """
    
    _instance: Optional["Workspace"] = None
    _lock = threading.Lock()
    
    def __new__(cls, root_dir: Optional[str] = None):
        """Singleton pattern - one workspace per session."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
    
    def __init__(self, root_dir: Optional[str] = None):
        if self._initialized:
            return
        
        self._root_dir = root_dir or os.getcwd()
        self._cwd = self._root_dir
        self._files: Dict[str, FileInfo] = {}
        self._directories: Set[str] = {self._root_dir}
        self._activities: List[AgentActivity] = []
        self._variables: Dict[str, Any] = {}
        self._active_agents: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._initialized = True
        
        self._scan_directory(self._root_dir)
    
    def _scan_directory(self, path: str, max_depth: int = 3) -> None:
        """Scan a directory and register existing files."""
        try:
            for item in os.listdir(path):
                full_path = os.path.join(path, item)
                rel_path = os.path.relpath(full_path, self._root_dir)
                
                if item.startswith('.') or item in ('node_modules', '__pycache__', 'venv', '.git'):
                    continue
                
                if os.path.isfile(full_path):
                    self._files[rel_path] = FileInfo(
                        path=rel_path,
                        description="Pre-existing file"
                    )
                elif os.path.isdir(full_path) and max_depth > 0:
                    self._directories.add(rel_path)
                    self._scan_directory(full_path, max_depth - 1)
        except PermissionError:
            pass
    
    @property
    def root_dir(self) -> str:
        return self._root_dir
    
    @property
    def cwd(self) -> str:
        with self._lock:
            return self._cwd
    
    def set_cwd(self, path: str) -> None:
        with self._lock:
            if os.path.isabs(path):
                self._cwd = path
            else:
                self._cwd = os.path.normpath(os.path.join(self._cwd, path))
            self._directories.add(self._cwd)
    
    def register_file(
        self,
        path: str,
        agent_id: Optional[str] = None,
        description: Optional[str] = None,
        is_modification: bool = False
    ) -> None:
        with self._lock:
            if os.path.isabs(path):
                rel_path = os.path.relpath(path, self._root_dir)
            else:
                rel_path = path
            
            now = datetime.now().isoformat()
            
            if rel_path in self._files and is_modification:
                self._files[rel_path].modified_by = agent_id
                self._files[rel_path].modified_at = now
            else:
                self._files[rel_path] = FileInfo(
                    path=rel_path,
                    created_by=agent_id,
                    created_at=now,
                    description=description
                )
            
            parent = os.path.dirname(rel_path)
            if parent:
                self._directories.add(parent)
    
    def register_directory(self, path: str, agent_id: Optional[str] = None) -> None:
        with self._lock:
            if os.path.isabs(path):
                rel_path = os.path.relpath(path, self._root_dir)
            else:
                rel_path = path
            
            self._directories.add(rel_path)
            
            if agent_id:
                self.log_activity(agent_id, "", "created_directory", f"Created: {rel_path}")
    
    def log_activity(self, agent_id: str, task_id: str, action: str, details: str) -> None:
        with self._lock:
            activity = AgentActivity(
                agent_id=agent_id,
                task_id=task_id,
                action=action,
                details=details
            )
            self._activities.append(activity)
            
            if len(self._activities) > 100:
                self._activities = self._activities[-100:]
    
    def set_agent_active(self, agent_id: str, task_description: str) -> None:
        with self._lock:
            self._active_agents[agent_id] = task_description
    
    def set_agent_inactive(self, agent_id: str) -> None:
        with self._lock:
            self._active_agents.pop(agent_id, None)
    
    def get_active_agents(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._active_agents)
    
    def set_variable(self, key: str, value: Any) -> None:
        with self._lock:
            self._variables[key] = value
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._variables.get(key, default)
    
    def get_files(self) -> List[str]:
        with self._lock:
            return list(self._files.keys())
    
    def get_directories(self) -> List[str]:
        with self._lock:
            return sorted(self._directories)
    
    def get_recent_activities(self, limit: int = 20) -> List[AgentActivity]:
        with self._lock:
            return self._activities[-limit:]
    
    def _generate_file_tree(self, root: str, max_depth: int = 4, prefix: str = "") -> List[str]:
        """Generate a visual file tree representation."""
        lines = []
        
        try:
            items = sorted(os.listdir(root))
        except (PermissionError, FileNotFoundError):
            return lines
        
        # Filter out common non-essential items
        skip_items = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 
                      '.pytest_cache', '.mypy_cache', '.tox', 'dist', 'build',
                      '.egg-info', '*.pyc', '.DS_Store', '.env'}
        
        items = [i for i in items if i not in skip_items and not i.endswith('.pyc')]
        
        # Separate dirs and files
        dirs = []
        files = []
        for item in items:
            full_path = os.path.join(root, item)
            if os.path.isdir(full_path):
                dirs.append(item)
            else:
                files.append(item)
        
        all_items = dirs + files
        
        for i, item in enumerate(all_items):
            is_last = (i == len(all_items) - 1)
            connector = "└── " if is_last else "├── "
            full_path = os.path.join(root, item)
            
            if os.path.isdir(full_path):
                lines.append(f"{prefix}{connector}{item}/")
                if max_depth > 1:
                    extension = "    " if is_last else "│   "
                    lines.extend(self._generate_file_tree(full_path, max_depth - 1, prefix + extension))
            else:
                # Mark files created by agents
                rel_path = os.path.relpath(full_path, self._root_dir)
                marker = ""
                if rel_path in self._files:
                    info = self._files[rel_path]
                    if info.created_by:
                        marker = f" ← created by {info.created_by}"
                lines.append(f"{prefix}{connector}{item}{marker}")
        
        return lines
    
    def refresh_file_tree(self) -> None:
        """Rescan the filesystem to update the internal file tree.
        
        Call this when you want to ensure the file tree reflects recent changes.
        """
        # Clear existing scanned files (but keep agent-registered files)
        agent_files = {path: info for path, info in self._files.items() 
                       if info.created_by or info.modified_by}
        self._files = agent_files
        self._directories = {self._root_dir}
        
        # Rescan
        self._scan_directory(self._root_dir)
    
    def get_file_tree(self, max_depth: int = 4, refresh: bool = False) -> str:
        """Get a visual file tree of the workspace.
        
        Args:
            max_depth: Maximum depth to traverse
            refresh: If True, rescan filesystem first
        """
        if refresh:
            self.refresh_file_tree()
        
        project_name = os.path.basename(self._root_dir)
        lines = [f"{project_name}/"]
        lines.extend(self._generate_file_tree(self._root_dir, max_depth))
        return "\n".join(lines)
    
    def get_context_for_agent(self, task_id: str = "") -> str:
        """Get a context summary for an agent.
        
        This method refreshes the file tree to ensure agents see the latest state.
        """
        with self._lock:
            parts = []
            
            # Clear header with location
            project_name = os.path.basename(self._root_dir)
            parts.append("=" * 50)
            parts.append(f"PROJECT: {project_name}")
            parts.append(f"ROOT PATH: {self._root_dir}")
            parts.append(f"CURRENT DIRECTORY: {self._cwd}")
            parts.append("=" * 50)
            parts.append("")
            
            # Refresh and show file tree - ensures agents see latest files
            parts.append("## File Tree (actual filesystem - refreshed)")
            parts.append("```")
            parts.append(self.get_file_tree(max_depth=3, refresh=True))
            parts.append("```")
            parts.append("")
            
            # Files created/modified by agents (important for coordination)
            agent_files = [(path, info) for path, info in self._files.items() 
                          if info.created_by or info.modified_by]
            if agent_files:
                parts.append("## Files Created/Modified by Agents:")
                for path, info in sorted(agent_files, key=lambda x: x[1].modified_at or x[1].created_at or "", reverse=True)[:10]:
                    if info.created_by:
                        parts.append(f"  📝 {path} (created by {info.created_by})")
                    elif info.modified_by:
                        parts.append(f"  ✏️ {path} (modified by {info.modified_by})")
                parts.append("")
            
            # Active agents
            if self._active_agents:
                parts.append("## Other Agents Working Now:")
                for agent_id, task_desc in self._active_agents.items():
                    parts.append(f"  • {agent_id}: {task_desc[:60]}...")
                parts.append("")
            
            # Recent activities - shorter list
            recent = self._activities[-5:]
            if recent:
                parts.append("## Recent Activities:")
                for activity in recent:
                    parts.append(f"  • [{activity.agent_id}] {activity.action}: {activity.details[:40]}")
                parts.append("")
            
            # Shared results from other tasks with file info
            task_results = {k: v for k, v in self._variables.items() if k.startswith("result_")}
            if task_results:
                parts.append("## Results from Completed Tasks:")
                for key, value in task_results.items():
                    task_name = key.replace("result_", "")
                    val_str = str(value)[:80]
                    
                    # Get files created/modified for this task
                    files_created = self._variables.get(f"files_created_{task_name}", [])
                    files_modified = self._variables.get(f"files_modified_{task_name}", [])
                    
                    parts.append(f"  • **{task_name}**: {val_str}")
                    if files_created:
                        parts.append(f"    └─ Created: {', '.join(files_created[:5])}")
                        if len(files_created) > 5:
                            parts.append(f"       ... and {len(files_created) - 5} more")
                    if files_modified:
                        parts.append(f"    └─ Modified: {', '.join(files_modified[:5])}")
                parts.append("")
            
            return "\n".join(parts)
    
    def to_dict(self) -> Dict:
        with self._lock:
            return {
                "root_dir": self._root_dir,
                "cwd": self._cwd,
                "files": {k: v.to_dict() for k, v in self._files.items()},
                "directories": list(self._directories),
                "variables": self._variables,
                "activities": [a.to_dict() for a in self._activities[-50:]]
            }
    
    def save(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "Workspace":
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        workspace = cls(data["root_dir"])
        workspace._cwd = data.get("cwd", workspace._root_dir)
        workspace._directories = set(data.get("directories", []))
        workspace._variables = data.get("variables", {})
        
        for path, info_data in data.get("files", {}).items():
            workspace._files[path] = FileInfo(**info_data)
        
        return workspace
    
    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None


def get_workspace(root_dir: Optional[str] = None) -> Workspace:
    """Get the singleton workspace instance."""
    return Workspace(root_dir)
