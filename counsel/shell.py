"""Safe shell execution wrapper for agents."""

import asyncio
import subprocess
import os
import shlex
from typing import Optional, Tuple, List
from dataclasses import dataclass

from counsel.config import ShellConfig, get_config


@dataclass
class ShellResult:
    """Result of a shell command execution."""
    command: str
    return_code: int
    stdout: str
    stderr: str
    working_directory: str
    timed_out: bool = False
    blocked: bool = False
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.return_code == 0 and not self.timed_out and not self.blocked
    
    @property
    def output(self) -> str:
        """Get combined output."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr] {self.stderr}")
        if self.error:
            parts.append(f"[error] {self.error}")
        return "\n".join(parts) if parts else "(no output)"
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status}] {self.command}\n{self.output}"


class Shell:
    """
    Safe shell execution environment for agents.
    
    Features:
    - Command blocking for dangerous operations
    - Timeout handling
    - Working directory management
    - Output truncation
    """
    
    def __init__(self, config: Optional[ShellConfig] = None):
        """Initialize the shell."""
        self.config = config or get_config().shell
        self._cwd = self.config.working_directory or os.getcwd()
    
    @property
    def cwd(self) -> str:
        """Get current working directory."""
        return self._cwd
    
    def cd(self, path: str) -> ShellResult:
        """Change working directory."""
        try:
            if not os.path.isabs(path):
                new_path = os.path.normpath(os.path.join(self._cwd, path))
            else:
                new_path = os.path.normpath(path)
            
            if not os.path.exists(new_path):
                return ShellResult(
                    command=f"cd {path}",
                    return_code=1,
                    stdout="",
                    stderr=f"Directory not found: {new_path}",
                    working_directory=self._cwd,
                    error="Directory not found"
                )
            
            if not os.path.isdir(new_path):
                return ShellResult(
                    command=f"cd {path}",
                    return_code=1,
                    stdout="",
                    stderr=f"Not a directory: {new_path}",
                    working_directory=self._cwd,
                    error="Not a directory"
                )
            
            self._cwd = new_path
            return ShellResult(
                command=f"cd {path}",
                return_code=0,
                stdout=f"Changed to: {new_path}",
                stderr="",
                working_directory=new_path
            )
        except Exception as e:
            return ShellResult(
                command=f"cd {path}",
                return_code=1,
                stdout="",
                stderr=str(e),
                working_directory=self._cwd,
                error=str(e)
            )
    
    def _is_blocked(self, command: str) -> Tuple[bool, Optional[str]]:
        """Check if a command should be blocked."""
        command_lower = command.lower()
        
        for pattern in self.config.blocked_patterns:
            if pattern.lower() in command_lower:
                return True, f"Blocked pattern: {pattern}"
        
        if not self.config.allow_sudo:
            if command_lower.startswith("sudo ") or " sudo " in command_lower:
                return True, "sudo commands are not allowed"
        
        return False, None
    
    def _truncate_output(self, output: str) -> str:
        """Truncate output if too long."""
        if len(output) > self.config.max_output_size:
            half = self.config.max_output_size // 2
            return (
                output[:half] +
                f"\n\n... [truncated {len(output) - self.config.max_output_size} characters] ...\n\n" +
                output[-half:]
            )
        return output
    
    async def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None
    ) -> ShellResult:
        """Run a shell command asynchronously."""
        # Check if blocked
        is_blocked, reason = self._is_blocked(command)
        if is_blocked:
            return ShellResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command blocked: {reason}",
                working_directory=self._cwd,
                blocked=True,
                error=reason
            )
        
        # Handle cd specially
        stripped = command.strip()
        if stripped.startswith("cd "):
            path = stripped[3:].strip()
            if (path.startswith('"') and path.endswith('"')) or \
               (path.startswith("'") and path.endswith("'")):
                path = path[1:-1]
            return self.cd(path)
        elif stripped == "cd":
            return self.cd(os.path.expanduser("~"))
        
        work_dir = cwd or self._cwd
        cmd_timeout = timeout or self.config.default_timeout
        
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env={**os.environ, "TERM": "dumb"}
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=cmd_timeout
                )
                
                stdout_str = self._truncate_output(stdout.decode("utf-8", errors="replace"))
                stderr_str = self._truncate_output(stderr.decode("utf-8", errors="replace"))
                
                return ShellResult(
                    command=command,
                    return_code=process.returncode,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    working_directory=work_dir
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                return ShellResult(
                    command=command,
                    return_code=-1,
                    stdout="",
                    stderr=f"Command timed out after {cmd_timeout} seconds",
                    working_directory=work_dir,
                    timed_out=True,
                    error="Timeout"
                )
                
        except Exception as e:
            return ShellResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=str(e),
                working_directory=work_dir,
                error=str(e)
            )
    
    def run_sync(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None
    ) -> ShellResult:
        """Run a shell command synchronously."""
        is_blocked, reason = self._is_blocked(command)
        if is_blocked:
            return ShellResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command blocked: {reason}",
                working_directory=self._cwd,
                blocked=True,
                error=reason
            )
        
        stripped = command.strip()
        if stripped.startswith("cd "):
            path = stripped[3:].strip()
            if (path.startswith('"') and path.endswith('"')) or \
               (path.startswith("'") and path.endswith("'")):
                path = path[1:-1]
            return self.cd(path)
        elif stripped == "cd":
            return self.cd(os.path.expanduser("~"))
        
        work_dir = cwd or self._cwd
        cmd_timeout = timeout or self.config.default_timeout
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=cmd_timeout,
                cwd=work_dir,
                env={**os.environ, "TERM": "dumb"}
            )
            
            return ShellResult(
                command=command,
                return_code=result.returncode,
                stdout=self._truncate_output(result.stdout),
                stderr=self._truncate_output(result.stderr),
                working_directory=work_dir
            )
            
        except subprocess.TimeoutExpired:
            return ShellResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command timed out after {cmd_timeout} seconds",
                working_directory=work_dir,
                timed_out=True,
                error="Timeout"
            )
            
        except Exception as e:
            return ShellResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=str(e),
                working_directory=work_dir,
                error=str(e)
            )
    
    async def run_many(
        self,
        commands: List[str],
        stop_on_error: bool = True
    ) -> List[ShellResult]:
        """Run multiple commands sequentially."""
        results = []
        for cmd in commands:
            result = await self.run(cmd)
            results.append(result)
            if not result.success and stop_on_error:
                break
        return results
    
    def pwd(self) -> str:
        """Get current working directory."""
        return self._cwd


def get_shell(config: Optional[ShellConfig] = None) -> Shell:
    """Get a new shell instance."""
    return Shell(config)
