"""Job persistence and management for the Agent Orchestration System."""

import os
import json
import uuid
import shutil
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


class JobStatus(Enum):
    """Status of a job."""
    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobMessage:
    """A message sent to or from a job."""
    id: str
    direction: str  # "user" or "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processed: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "JobMessage":
        return cls(**data)


@dataclass
class Job:
    """A job represents a complete task execution session."""
    id: str
    name: str  # Short name derived from request
    request: str  # Original user request
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    # Task data
    task_graph: Optional[Dict] = None
    results: Dict[str, Dict] = field(default_factory=dict)
    
    # Execution metadata
    model_used: Optional[str] = None
    total_iterations: int = 0
    files_created: List[str] = field(default_factory=list)
    
    # Messages (for user interaction during execution)
    messages: List[JobMessage] = field(default_factory=list)
    pending_messages: List[JobMessage] = field(default_factory=list)
    
    # Error info
    error: Optional[str] = None
    
    # Workspace state
    workspace_cwd: Optional[str] = None
    
    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "name": self.name,
            "request": self.request,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "task_graph": self.task_graph,
            "results": self.results,
            "model_used": self.model_used,
            "total_iterations": self.total_iterations,
            "files_created": self.files_created,
            "messages": [m.to_dict() for m in self.messages],
            "pending_messages": [m.to_dict() for m in self.pending_messages],
            "error": self.error,
            "workspace_cwd": self.workspace_cwd,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        return cls(
            id=data["id"],
            name=data["name"],
            request=data["request"],
            status=JobStatus(data.get("status", "pending")),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
            task_graph=data.get("task_graph"),
            results=data.get("results", {}),
            model_used=data.get("model_used"),
            total_iterations=data.get("total_iterations", 0),
            files_created=data.get("files_created", []),
            messages=[JobMessage.from_dict(m) for m in data.get("messages", [])],
            pending_messages=[JobMessage.from_dict(m) for m in data.get("pending_messages", [])],
            error=data.get("error"),
            workspace_cwd=data.get("workspace_cwd"),
        )
    
    def add_user_message(self, content: str) -> JobMessage:
        """Add a message from the user to the job."""
        msg = JobMessage(
            id=str(uuid.uuid4())[:8],
            direction="user",
            content=content,
        )
        self.pending_messages.append(msg)
        self.updated_at = datetime.now().isoformat()
        return msg
    
    def add_system_message(self, content: str) -> JobMessage:
        """Add a system message to the job."""
        msg = JobMessage(
            id=str(uuid.uuid4())[:8],
            direction="system",
            content=content,
        )
        self.messages.append(msg)
        self.updated_at = datetime.now().isoformat()
        return msg
    
    def get_pending_messages(self) -> List[JobMessage]:
        """Get unprocessed user messages."""
        pending = [m for m in self.pending_messages if not m.processed]
        return pending
    
    def mark_message_processed(self, msg_id: str) -> None:
        """Mark a message as processed."""
        for msg in self.pending_messages:
            if msg.id == msg_id:
                msg.processed = True
                self.messages.append(msg)
        self.pending_messages = [m for m in self.pending_messages if not m.processed]
    
    @property
    def short_id(self) -> str:
        """Get a short version of the job ID."""
        return self.id[:8]
    
    @property
    def display_name(self) -> str:
        """Get display name for the job."""
        return f"{self.short_id}: {self.name}"


class JobManager:
    """Manages job persistence and retrieval."""
    
    DEFAULT_DIR = os.path.expanduser("~/.counsel")
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or self.DEFAULT_DIR)
        self.jobs_dir = self.base_dir / "jobs"
        self.current_job_file = self.base_dir / "current_job.txt"
        
        # Ensure directories exist
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self._jobs_cache: Dict[str, Job] = {}
        self._current_job_id: Optional[str] = None
        
        # Load current job ID
        self._load_current_job_id()
    
    def _load_current_job_id(self) -> None:
        """Load the current job ID from file."""
        try:
            if self.current_job_file.exists():
                self._current_job_id = self.current_job_file.read_text().strip()
        except:
            self._current_job_id = None
    
    def _save_current_job_id(self) -> None:
        """Save the current job ID to file."""
        try:
            if self._current_job_id:
                self.current_job_file.write_text(self._current_job_id)
            elif self.current_job_file.exists():
                self.current_job_file.unlink()
        except:
            pass
    
    def _job_file(self, job_id: str) -> Path:
        """Get the file path for a job."""
        return self.jobs_dir / f"{job_id}.json"
    
    def create_job(self, request: str, workspace_cwd: Optional[str] = None) -> Job:
        """Create a new job."""
        job_id = str(uuid.uuid4())
        
        # Create a short name from the request
        name = request[:50].strip()
        if len(request) > 50:
            name += "..."
        
        job = Job(
            id=job_id,
            name=name,
            request=request,
            workspace_cwd=workspace_cwd,
        )
        
        self._jobs_cache[job_id] = job
        self.save_job(job)
        
        return job
    
    def save_job(self, job: Job) -> None:
        """Save a job to disk."""
        job.updated_at = datetime.now().isoformat()
        job_file = self._job_file(job.id)
        
        try:
            with open(job_file, 'w') as f:
                json.dump(job.to_dict(), f, indent=2)
            self._jobs_cache[job.id] = job
        except Exception as e:
            raise RuntimeError(f"Failed to save job: {e}")
    
    def load_job(self, job_id: str) -> Optional[Job]:
        """Load a job from disk."""
        # Check cache first
        if job_id in self._jobs_cache:
            return self._jobs_cache[job_id]
        
        # Try to load from file
        job_file = self._job_file(job_id)
        if not job_file.exists():
            # Try short ID match
            for f in self.jobs_dir.glob("*.json"):
                if f.stem.startswith(job_id):
                    job_file = f
                    job_id = f.stem
                    break
            else:
                return None
        
        try:
            with open(job_file, 'r') as f:
                data = json.load(f)
            job = Job.from_dict(data)
            self._jobs_cache[job_id] = job
            return job
        except Exception:
            return None
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a job."""
        job_file = self._job_file(job_id)
        
        # Remove from cache
        self._jobs_cache.pop(job_id, None)
        
        # Remove file
        try:
            if job_file.exists():
                job_file.unlink()
            return True
        except:
            return False
    
    def list_jobs(self, limit: int = 50, status: Optional[JobStatus] = None) -> List[Job]:
        """List all jobs, most recent first."""
        jobs = []
        
        for job_file in self.jobs_dir.glob("*.json"):
            job = self.load_job(job_file.stem)
            if job:
                if status is None or job.status == status:
                    jobs.append(job)
        
        # Sort by updated_at descending
        jobs.sort(key=lambda j: j.updated_at, reverse=True)
        
        return jobs[:limit]
    
    def get_running_jobs(self) -> List[Job]:
        """Get all currently running jobs."""
        return self.list_jobs(status=JobStatus.RUNNING)
    
    def set_current_job(self, job_id: Optional[str]) -> None:
        """Set the current active job."""
        self._current_job_id = job_id
        self._save_current_job_id()
    
    def get_current_job(self) -> Optional[Job]:
        """Get the current active job."""
        if self._current_job_id:
            return self.load_job(self._current_job_id)
        return None
    
    def get_job_by_short_id(self, short_id: str) -> Optional[Job]:
        """Find a job by its short ID prefix."""
        for job_file in self.jobs_dir.glob("*.json"):
            if job_file.stem.startswith(short_id):
                return self.load_job(job_file.stem)
        return None
    
    def cleanup_old_jobs(self, days: int = 30) -> int:
        """Delete jobs older than specified days."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted = 0
        
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                mtime = job_file.stat().st_mtime
                if mtime < cutoff:
                    job_file.unlink()
                    deleted += 1
            except:
                pass
        
        return deleted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about jobs."""
        jobs = self.list_jobs(limit=1000)
        
        stats = {
            "total_jobs": len(jobs),
            "by_status": {},
            "total_files_created": 0,
            "total_iterations": 0,
        }
        
        for job in jobs:
            status = job.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
            stats["total_files_created"] += len(job.files_created)
            stats["total_iterations"] += job.total_iterations
        
        return stats


# Global job manager instance
_JOB_MANAGER: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        _JOB_MANAGER = JobManager()
    return _JOB_MANAGER
