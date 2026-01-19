"""General-purpose worker agent with shell access, LLM reasoning, and workspace awareness."""

import asyncio
import re
import os
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from counsel.config import AgentConfig, SupervisorConfig, get_config
from counsel.llm import LLM, Message, get_llm
from counsel.shell import Shell, ShellResult, get_shell
from counsel.workspace import Workspace, get_workspace
from counsel.task_graph import Task

# Debug callback type: (agent_id, event_type, content)
DebugCallback = Callable[[str, str, str], None]


@dataclass
class AgentResult:
    """Result from an agent's task execution."""
    task_id: str
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    shell_history: List[Dict] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    iterations: int = 0
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "shell_history": self.shell_history,
            "files_created": self.files_created,
            "files_modified": self.files_modified,
            "iterations": self.iterations,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


AGENT_SYSTEM_PROMPT = """You are a general-purpose AI agent that executes tasks using file operations and shell commands.

You have access to powerful file tools and a Linux/Unix shell to complete your task.

## Available Actions

Use XML tags to take actions:

### File Operations (PREFERRED for file work)

1. **Read a file** - Get full file contents with line numbers:
   <read_file>path/to/file.py</read_file>

2. **List directory** - See all files in a directory:
   <list_dir>path/to/directory</list_dir>

3. **Write/Create a file** - Create or overwrite entire file:
   <write_file path="path/to/file.py">
file contents here
multiple lines supported
</write_file>

4. **Edit a file** - Replace specific text in a file:
   <edit_file path="path/to/file.py">
OLD_TEXT_TO_FIND
|||
NEW_TEXT_TO_REPLACE_WITH
</edit_file>
   NOTE: OLD_TEXT must match exactly (including whitespace). Include enough context to be unique.

### Shell Commands (for running programs, installing packages, etc.)

5. **Run a shell command**:
   <shell>command here</shell>

### Control Actions

6. **Think/reason about the task**:
   <think>your reasoning here</think>

7. **Ask for help when stuck** ⭐ IMPORTANT:
   <help>description of what you're trying to do and what's not working</help>
   
8. **Mark task complete with result**:
   <done>final result or summary</done>

9. **Report an error if task cannot be completed**:
   <error>description of what went wrong</error>

## Workspace Context

{workspace_context}

## CRITICAL: Ask for Help When Stuck! ⭐

**After 2 failed attempts at the same thing, USE <help>!**

Don't keep trying the same failing approach. When something isn't working:
1. Try ONE alternative approach
2. If that also fails, IMMEDIATELY use <help> to get guidance
3. Describe what you're trying to do and what errors you're seeing

Example:
<help>I'm trying to install numpy but pip keeps failing with "Permission denied". I tried using --user flag but it still fails. What should I do?</help>

A supervisor will provide specific guidance to help you succeed.

## IMPORTANT: When to Use File Operations vs Shell

**USE FILE OPERATIONS FOR:**
- Reading file contents → <read_file>
- Viewing directory structure → <list_dir>
- Creating new files → <write_file>
- Editing existing files → <edit_file> or <write_file>

**USE SHELL COMMANDS FOR:**
- Running programs: `python script.py`, `npm start`
- Installing packages: `pip install`, `npm install`
- Git operations: `git add`, `git commit`
- System commands: `mkdir -p`, `chmod`, `mv`, `cp`
- Running tests: `pytest`, `npm test`

## CRITICAL: Shell Environment

**Each shell command runs in a NEW subprocess.** This means:
- Environment changes DO NOT persist between commands
- `cd` is handled specially and DOES persist
- `export VAR=value` will NOT persist to the next command
- Virtual environment activation will NOT persist

**For Python virtual environments:**
- DON'T try to `source venv/bin/activate` - it won't persist!
- Instead, use the venv binaries directly:
  - `./venv/bin/python script.py`
  - `./venv/bin/pip install package`
  - `venv/bin/python -m pytest`

## Rules

1. **ALWAYS read files before editing** - Use <read_file> to see current contents
2. Use <think> to plan before taking action
3. Run ONE action at a time and observe the result
4. **ASK FOR HELP after 2 failures** - Don't spin on the same error!
5. Be aware of files/directories created by other agents (see workspace context)
6. Use <done> when complete, or <error> if impossible
7. DON'T give up - ask for <help> instead!

## File Editing Best Practices

When editing files:
1. First use <read_file> to see the current contents
2. Use <edit_file> for small changes (provide exact text to replace)
3. Use <write_file> for creating new files or complete rewrites
4. After editing, optionally <read_file> again to verify changes

## Coordination with Other Agents ⭐ IMPORTANT

You may be part of a multi-agent workflow where previous tasks have been completed.

**Check the Dependency Context below for:**
- Results from tasks that ran before yours
- Files that were created by previous agents
- Files that were modified by previous agents

**Use this information to:**
- Find files created by previous tasks (don't recreate them!)
- Build on work already done
- Import/use modules or code created earlier
- Know where to find project structure set up by others

**Check the Workspace Context for:**
- Current file tree (what exists now)
- Recent activity from all agents
- Files created/modified across all tasks

## Current Task

{task_description}

## Context from Dependencies (What previous tasks did)

{dependency_context}

**IMPORTANT**: If dependencies created files, those files exist now. Use <read_file> or <list_dir> to examine them before starting your work.

Begin working. Start with <think> to plan your approach, reviewing what dependencies have done."""


class Agent:
    """General-purpose worker agent with workspace awareness and supervisor support."""
    
    def __init__(
        self,
        agent_id: str,
        llm: Optional[LLM] = None,
        config: Optional[AgentConfig] = None,
        supervisor_config: Optional[SupervisorConfig] = None,
        workspace: Optional[Workspace] = None,
        debug_callback: Optional[DebugCallback] = None
    ):
        self.agent_id = agent_id
        self.llm = llm or get_llm()
        self.config = config or get_config().agent
        self.supervisor_config = supervisor_config or get_config().supervisor
        self.workspace = workspace or get_workspace()
        self.shell = get_shell()
        self._conversation: List[Message] = []
        self.debug_callback = debug_callback
        
        # Action tracking for better supervisor context
        self._action_history: List[Dict[str, Any]] = []
        self._consecutive_failures: int = 0
        self._help_requests: int = 0
        
        self.shell._cwd = self.workspace.cwd
    
    def _debug(self, event: str, content: str) -> None:
        """Emit debug event if callback is set."""
        if self.debug_callback:
            self.debug_callback(self.agent_id, event, content)
    
    def _truncate_conversation(self, max_messages: int = 20) -> None:
        """Truncate conversation history to prevent unbounded growth.
        
        Keeps system prompt and last N messages to stay within context limits.
        """
        if len(self._conversation) <= max_messages + 1:  # +1 for system prompt
            return
        
        # Keep system prompt (first message) and last max_messages
        system_prompt = self._conversation[0]
        recent = self._conversation[-(max_messages):]
        
        # Add a summary message
        truncated_count = len(self._conversation) - max_messages - 1
        summary = Message(
            role="user",
            content=f"[Note: {truncated_count} earlier messages were truncated to save context. Continue with the task.]"
        )
        
        self._conversation = [system_prompt, summary] + recent
        self._debug("truncate", f"Truncated {truncated_count} old messages from conversation")
    
    def _track_action(self, action_type: str, content: str, success: bool, result: str = "") -> None:
        """Track an action for supervisor context."""
        self._action_history.append({
            "type": action_type,
            "content": content[:200],
            "success": success,
            "result": result[:300],
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 20 actions
        if len(self._action_history) > 20:
            self._action_history = self._action_history[-20:]
        
        # Track consecutive failures
        if success:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1
    
    def _should_suggest_help(self) -> bool:
        """Check if we should suggest asking for help based on consecutive failures."""
        if not self.supervisor_config.enabled:
            return False
        return self._consecutive_failures >= self.supervisor_config.failures_before_intervention
    
    def _detect_stuck(self, shell_history: List[Dict], recent_errors: List[str]) -> Optional[str]:
        """Detect if the agent is stuck in a loop.
        
        Returns a description of the stuck pattern if detected, None otherwise.
        """
        # Check consecutive failures from action tracking
        if self._consecutive_failures >= self.supervisor_config.failures_before_intervention:
            return f"Multiple consecutive failures ({self._consecutive_failures} in a row)"
        
        if len(shell_history) < 2:
            return None
        
        # Check for repeated commands (same command 2+ times in last 4)
        recent_cmds = [h['command'].strip() for h in shell_history[-4:]]
        for cmd in set(recent_cmds):
            if recent_cmds.count(cmd) >= 2:
                return f"Repeating the same command: '{cmd[:50]}...'"
        
        # Check for repeated errors
        if len(recent_errors) >= 2:
            # Check if same error type keeps appearing
            error_patterns = {}
            for err in recent_errors[-4:]:
                # Extract key part of error
                key = err[:100] if err else "unknown"
                error_patterns[key] = error_patterns.get(key, 0) + 1
            
            for pattern, count in error_patterns.items():
                if count >= 2:
                    return f"Repeated error pattern: '{pattern[:80]}...'"
        
        # Check for failed commands loop
        recent_results = [h.get('success', True) for h in shell_history[-4:]]
        if recent_results.count(False) >= 3:
            return "Most recent commands are failing"
        
        return None
    
    async def _get_supervisor_guidance(self, task_description: str, stuck_reason: str, 
                                        shell_history: List[Dict], recent_errors: List[str],
                                        help_request: str = "") -> str:
        """Get guidance from a supervisor perspective when stuck."""
        
        # Build context from action history (more comprehensive than shell_history)
        recent_actions = []
        for action in self._action_history[-8:]:
            status = "✓" if action['success'] else "✗"
            action_type = action['type']
            content = action['content'][:80]
            result = action['result'][:150] if action['result'] else ""
            recent_actions.append(f"  {status} [{action_type}] {content}")
            if result:
                recent_actions.append(f"      → {result}")
        
        actions_str = "\n".join(recent_actions) if recent_actions else "No actions yet"
        
        # Also include shell history for additional context
        shell_attempts = []
        for h in shell_history[-5:]:
            cmd = h['command'][:100]
            success = "✓" if h.get('success') else "✗"
            output = (h.get('output', '') or '')[:200]
            shell_attempts.append(f"  {success} `{cmd}`\n    → {output}")
        
        shell_str = "\n".join(shell_attempts) if shell_attempts else "No shell commands yet"
        errors_str = "\n".join(f"  - {e[:150]}" for e in recent_errors[-3:]) if recent_errors else "None"
        
        # If agent explicitly asked for help, include their description
        help_context = ""
        if help_request:
            help_context = f"""
## Agent's Help Request
The agent is asking: {help_request}
"""
        
        supervisor_prompt = f"""You are a senior engineer helping a junior developer who is stuck.

## The Task
{task_description}
{help_context}
## Problem
{stuck_reason}

## Recent Actions (all types)
{actions_str}

## Recent Shell Commands
{shell_str}

## Recent Errors
{errors_str}

## Your Job
Analyze what's going wrong and provide SPECIFIC, ACTIONABLE guidance.

Think about:
1. Is there a fundamental misunderstanding of the task?
2. Are they missing a prerequisite step?
3. Should they try a completely different approach?
4. Is there an environment issue to address first?
5. Are they using the right tools? (file operations vs shell)

**Provide step-by-step guidance** (2-5 specific steps) that will help them succeed.
Be concrete - give exact commands or file operations they should try.
Start your response directly with the guidance."""

        try:
            guidance = await self.llm.chat([
                Message(role="system", content="You are a helpful senior engineer. Be direct, specific, and provide exact commands or steps."),
                Message(role="user", content=supervisor_prompt)
            ])
            return guidance.strip()
        except Exception as e:
            return f"Try a different approach. The current method isn't working: {stuck_reason}"
    
    def _parse_action(self, response: str) -> tuple[str, str]:
        # File operation patterns (check first - more specific)
        # read_file: <read_file>path</read_file>
        read_match = re.search(r"<read_file>(.*?)</read_file>", response, re.DOTALL | re.IGNORECASE)
        if read_match:
            return "read_file", read_match.group(1).strip()
        
        # list_dir: <list_dir>path</list_dir>
        list_match = re.search(r"<list_dir>(.*?)</list_dir>", response, re.DOTALL | re.IGNORECASE)
        if list_match:
            return "list_dir", list_match.group(1).strip()
        
        # write_file: <write_file path="...">content</write_file>
        write_match = re.search(
            r'<write_file\s+path=["\']([^"\']+)["\']>(.*?)</write_file>', 
            response, re.DOTALL | re.IGNORECASE
        )
        if write_match:
            return "write_file", f"{write_match.group(1)}|||{write_match.group(2)}"
        
        # edit_file: <edit_file path="...">old|||new</edit_file>
        edit_match = re.search(
            r'<edit_file\s+path=["\']([^"\']+)["\']>(.*?)</edit_file>', 
            response, re.DOTALL | re.IGNORECASE
        )
        if edit_match:
            return "edit_file", f"{edit_match.group(1)}|||{edit_match.group(2)}"
        
        # Standard action patterns
        patterns = [
            (r"<shell>(.*?)</shell>", "shell"),
            (r"<think>(.*?)</think>", "think"),
            (r"<help>(.*?)</help>", "help"),
            (r"<done>(.*?)</done>", "done"),
            (r"<error>(.*?)</error>", "error"),
        ]
        
        for pattern, action_type in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return action_type, match.group(1).strip()
        
        response_lower = response.lower()
        if "```bash" in response_lower or "```sh" in response_lower:
            match = re.search(r"```(?:bash|sh)\n(.*?)```", response, re.DOTALL)
            if match:
                return "shell", match.group(1).strip()
        
        return "none", response
    
    def _format_shell_result(self, result: ShellResult) -> str:
        lines = [f"Command: {result.command}"]
        lines.append(f"Exit code: {result.return_code}")
        lines.append(f"Working directory: {result.working_directory}")
        
        if result.stdout:
            lines.append(f"Output:\n{result.stdout}")
        if result.stderr:
            lines.append(f"Stderr:\n{result.stderr}")
        if result.blocked:
            lines.append(f"⚠️ Command was blocked: {result.error}")
        if result.timed_out:
            lines.append("⚠️ Command timed out")
        
        return "\n".join(lines)
    
    def _resolve_path(self, path: str) -> str:
        """Resolve a path relative to shell's current working directory."""
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.shell.cwd, path))
    
    def _read_file(self, path: str, max_lines: int = 2000) -> tuple[bool, str]:
        """Read a file and return contents with line numbers.
        
        Returns (success, content_or_error)
        """
        full_path = self._resolve_path(path)
        
        if not os.path.exists(full_path):
            return False, f"File not found: {path}"
        
        if not os.path.isfile(full_path):
            return False, f"Not a file: {path}"
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            
            # Truncate if too long
            if total_lines > max_lines:
                # Show first and last portions
                half = max_lines // 2
                shown_lines = lines[:half] + [f"\n... ({total_lines - max_lines} lines omitted) ...\n\n"] + lines[-half:]
                lines = shown_lines
            
            # Add line numbers
            numbered_lines = []
            line_num = 1
            for line in lines:
                if "lines omitted" in line:
                    numbered_lines.append(line)
                    line_num = total_lines - (max_lines // 2) + 1
                else:
                    # Right-align line numbers for readability
                    numbered_lines.append(f"{line_num:4d} | {line.rstrip()}")
                    line_num += 1
            
            result = f"File: {path} ({total_lines} lines)\n"
            result += "─" * 50 + "\n"
            result += "\n".join(numbered_lines)
            
            return True, result
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def _list_dir(self, path: str, max_items: int = 200) -> tuple[bool, str]:
        """List directory contents with file info.
        
        Returns (success, content_or_error)
        """
        full_path = self._resolve_path(path)
        
        if not os.path.exists(full_path):
            return False, f"Directory not found: {path}"
        
        if not os.path.isdir(full_path):
            return False, f"Not a directory: {path}"
        
        try:
            items = sorted(os.listdir(full_path))
            
            # Skip common non-essential items
            skip_patterns = {'.git', '__pycache__', 'node_modules', '.venv', 
                           '.pytest_cache', '.mypy_cache', '.DS_Store', '.env'}
            items = [i for i in items if i not in skip_patterns and not i.endswith('.pyc')]
            
            if len(items) > max_items:
                items = items[:max_items]
                truncated = True
            else:
                truncated = False
            
            lines = [f"Directory: {path} ({len(items)} items)"]
            lines.append("─" * 50)
            
            dirs = []
            files = []
            
            for item in items:
                item_path = os.path.join(full_path, item)
                if os.path.isdir(item_path):
                    dirs.append(f"📁 {item}/")
                else:
                    try:
                        size = os.path.getsize(item_path)
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size // 1024}KB"
                        else:
                            size_str = f"{size // (1024 * 1024)}MB"
                        files.append(f"📄 {item} ({size_str})")
                    except:
                        files.append(f"📄 {item}")
            
            # Directories first, then files
            lines.extend(dirs)
            lines.extend(files)
            
            if truncated:
                lines.append(f"... (truncated, showing first {max_items} items)")
            
            return True, "\n".join(lines)
            
        except Exception as e:
            return False, f"Error listing directory: {str(e)}"
    
    def _write_file(self, path: str, content: str) -> tuple[bool, str]:
        """Write content to a file (creates or overwrites).
        
        Returns (success, message)
        """
        full_path = self._resolve_path(path)
        
        try:
            # Create parent directories if needed
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Check if file exists (for tracking)
            is_new = not os.path.exists(full_path)
            
            # Write the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Count lines written
            line_count = content.count('\n') + (1 if content and not content.endswith('\n') else 0)
            
            action = "Created" if is_new else "Updated"
            return True, f"{action} file: {path} ({line_count} lines, {len(content)} bytes)"
            
        except Exception as e:
            return False, f"Error writing file: {str(e)}"
    
    def _edit_file(self, path: str, old_text: str, new_text: str) -> tuple[bool, str]:
        """Edit a file by replacing old_text with new_text.
        
        Returns (success, message)
        """
        full_path = self._resolve_path(path)
        
        if not os.path.exists(full_path):
            return False, f"File not found: {path}"
        
        if not os.path.isfile(full_path):
            return False, f"Not a file: {path}"
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Check if old_text exists
            if old_text not in content:
                # Try to find similar text for helpful error
                old_lines = old_text.strip().split('\n')
                if old_lines:
                    first_line = old_lines[0].strip()
                    if first_line in content:
                        return False, (
                            f"Exact match not found in {path}. "
                            f"The first line '{first_line[:50]}...' exists but the full text doesn't match. "
                            f"Check whitespace/indentation. Use <read_file> to see exact content."
                        )
                return False, (
                    f"Text not found in {path}. "
                    f"The text you're trying to replace doesn't exist. "
                    f"Use <read_file>{path}</read_file> to see the current file contents."
                )
            
            # Count occurrences
            count = content.count(old_text)
            if count > 1:
                return False, (
                    f"Found {count} occurrences of the text in {path}. "
                    f"Please provide more context to make the match unique."
                )
            
            # Perform replacement
            new_content = content.replace(old_text, new_text, 1)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            lines_removed = old_text.count('\n')
            lines_added = new_text.count('\n')
            
            return True, f"Edited {path}: replaced {len(old_text)} chars with {len(new_text)} chars ({lines_removed} lines → {lines_added} lines)"
            
        except Exception as e:
            return False, f"Error editing file: {str(e)}"
    
    def _detect_file_operations(self, command: str, result: ShellResult) -> tuple[List[str], List[str]]:
        """Detect files created/modified by a command. Best-effort heuristic."""
        created = []
        modified = []
        
        if not result.success:
            return created, modified
        
        command_lower = command.lower().strip()
        
        # Detect file creation via redirect (but not stderr redirect like 2>&1)
        # Match: > filename but not 2> or &>
        redirect_match = re.search(r'(?<![0-9&])\s*>\s*([^\s|&;>]+)', command)
        if redirect_match and '>>' not in command:
            filename = redirect_match.group(1).strip("'\"")
            # Skip if it looks like a number (stderr redirect artifact)
            if not filename.isdigit():
                created.append(filename)
        
        # Detect append
        append_match = re.search(r'>>\s*([^\s|&;]+)', command)
        if append_match:
            filename = append_match.group(1).strip("'\"")
            if not filename.isdigit():
                modified.append(filename)
        
        # Detect touch command
        if command_lower.startswith('touch '):
            # Skip flags
            parts = command[6:].split()
            for part in parts:
                if not part.startswith('-'):
                    created.append(part.strip("'\""))
        
        # Detect mkdir - more careful parsing to skip option values
        if 'mkdir' in command_lower:
            parts = command.split()
            skip_next = False
            for i, part in enumerate(parts):
                if skip_next:
                    skip_next = False
                    continue
                if part == 'mkdir':
                    continue
                # Flags that take a value (-m mode)
                if part in ('-m', '--mode'):
                    skip_next = True
                    continue
                if part.startswith('-'):
                    # Skip flags like -p, -v, etc.
                    # Also skip -m=755 style
                    continue
                dir_name = part.strip("'\"")
                if dir_name and not dir_name.isdigit():
                    self.workspace.register_directory(
                        os.path.join(self.shell.cwd, dir_name),
                        self.agent_id
                    )
        
        # Detect cp destination
        if command_lower.startswith('cp '):
            parts = command.split()
            if len(parts) >= 3:
                dest = parts[-1].strip("'\"")
                if not os.path.isdir(os.path.join(self.shell.cwd, dest)):
                    created.append(dest)
        
        # Detect npm/yarn init
        if 'npm init' in command_lower or 'yarn init' in command_lower:
            created.append('package.json')
        
        return created, modified
    
    async def execute(self, task: Task, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        result = AgentResult(task_id=task.id, success=False)
        context = context or {}
        
        # Track errors for stuck detection
        recent_errors: List[str] = []
        last_intervention_iteration = -10  # Track when we last intervened
        
        # Reset action tracking for this task
        self._action_history = []
        self._consecutive_failures = 0
        self._help_requests = 0
        
        self._debug("start", f"Task: {task.description}")
        self.workspace.set_agent_active(self.agent_id, task.description[:100])
        self.workspace.log_activity(self.agent_id, task.id, "started", f"Started: {task.description[:50]}")
        
        try:
            workspace_context = self.workspace.get_context_for_agent(task.id)
            self._debug("context", f"Workspace: {workspace_context[:200]}...")
            
            dep_context = "None"
            if context:
                dep_parts = []
                for dep_id, dep_data in context.items():
                    if dep_id == '_remediation':
                        continue  # Handle remediation separately
                    
                    # Handle both old format (string) and new format (dict)
                    if isinstance(dep_data, dict):
                        dep_result = dep_data.get('result', str(dep_data))
                        files_created = dep_data.get('files_created', [])
                        files_modified = dep_data.get('files_modified', [])
                        
                        part = f"### {dep_id}:\n"
                        part += f"**Result**: {dep_result}\n"
                        if files_created:
                            part += f"**Files Created**: {', '.join(files_created)}\n"
                        if files_modified:
                            part += f"**Files Modified**: {', '.join(files_modified)}\n"
                        dep_parts.append(part)
                    else:
                        dep_parts.append(f"### {dep_id}:\n{dep_data}")
                
                dep_context = "\n\n".join(dep_parts) if dep_parts else "None"
                self._debug("deps", f"Dependencies: {[k for k in context.keys() if k != '_remediation']}")
            
            system_prompt = AGENT_SYSTEM_PROMPT.format(
                workspace_context=workspace_context,
                task_description=task.description,
                dependency_context=dep_context
            )
            
            self._conversation = [Message(role="system", content=system_prompt)]
            self._conversation.append(Message(role="user", content="Begin working on the task."))
            
            for iteration in range(self.config.max_iterations):
                result.iterations = iteration + 1
                self._debug("iter", f"Iteration {iteration + 1}/{self.config.max_iterations}")
                
                # Check for stuck patterns more frequently when supervisor is enabled
                min_iter = self.supervisor_config.min_iterations_before_check
                check_freq = self.supervisor_config.check_frequency
                
                if (self.supervisor_config.enabled and 
                    iteration >= min_iter and 
                    iteration % check_freq == 0 and 
                    (iteration - last_intervention_iteration) >= check_freq + 1):
                    
                    stuck_reason = self._detect_stuck(result.shell_history, recent_errors)
                    if stuck_reason:
                        self._debug("warn", f"🔄 Stuck detected: {stuck_reason}")
                        
                        # Get supervisor guidance
                        self._debug("supervisor", "Requesting supervisor guidance...")
                        guidance = await self._get_supervisor_guidance(
                            task.description, stuck_reason, result.shell_history, recent_errors
                        )
                        self._debug("supervisor", f"Guidance: {guidance[:200]}...")
                        
                        # Inject guidance into conversation
                        self._conversation.append(Message(
                            role="user",
                            content=f"""⚠️ **AUTOMATIC INTERVENTION - You seem stuck**

Problem detected: {stuck_reason}

**Guidance from supervisor:**
{guidance}

Please try a DIFFERENT approach based on this guidance. If you're still stuck after trying this, use <help>description of problem</help> to get more specific help."""
                        ))
                        last_intervention_iteration = iteration
                        self._consecutive_failures = 0  # Reset after intervention
                
                # Truncate conversation if it's getting too long (every 10 iterations)
                if iteration > 0 and iteration % 10 == 0:
                    self._truncate_conversation(max_messages=30)
                
                self._debug("llm_call", "Calling LLM...")
                try:
                    response = await self.llm.chat(self._conversation)
                except Exception as e:
                    result.error = f"LLM error: {str(e)}"
                    self._debug("error", f"LLM error: {str(e)}")
                    break
                
                # Show full LLM response
                self._debug("llm_response", response)
                
                self._conversation.append(Message(role="assistant", content=response))
                action_type, action_content = self._parse_action(response)
                self._debug("action", f"Parsed action: {action_type}")
                
                if action_type == "done":
                    result.success = True
                    result.result = action_content
                    self._debug("done", action_content)
                    self.workspace.log_activity(self.agent_id, task.id, "completed", f"Completed: {action_content[:50]}")
                    break
                
                elif action_type == "error":
                    result.success = False
                    result.error = action_content
                    self._debug("error", action_content)
                    self.workspace.log_activity(self.agent_id, task.id, "failed", f"Failed: {action_content[:50]}")
                    break
                
                elif action_type == "shell":
                    self._debug("shell_cmd", f"$ {action_content}")
                    if len(result.shell_history) >= self.config.max_shell_commands:
                        self._debug("warn", "Maximum commands reached!")
                        self._conversation.append(Message(
                            role="user",
                            content=f"⚠️ Maximum commands reached. Complete now or report an error."
                        ))
                        continue
                    
                    shell_result = await self.shell.run(action_content)
                    
                    # Track action
                    self._track_action("shell", action_content, shell_result.success, shell_result.output[:200])
                    
                    # Show shell output
                    output_preview = shell_result.output[:500] if shell_result.output else "(no output)"
                    self._debug("shell_out", f"Exit {shell_result.return_code}: {output_preview}")
                    
                    if action_content.strip().startswith('cd '):
                        self.workspace.set_cwd(self.shell.cwd)
                        self._debug("cwd", f"Changed dir to: {self.shell.cwd}")
                    
                    created, modified = self._detect_file_operations(action_content, shell_result)
                    
                    for file_path in created:
                        full_path = os.path.join(self.shell.cwd, file_path)
                        self.workspace.register_file(full_path, self.agent_id)
                        result.files_created.append(file_path)
                        self._debug("file_created", file_path)
                    
                    for file_path in modified:
                        full_path = os.path.join(self.shell.cwd, file_path)
                        self.workspace.register_file(full_path, self.agent_id, is_modification=True)
                        result.files_modified.append(file_path)
                        self._debug("file_modified", file_path)
                    
                    result.shell_history.append({
                        "command": action_content,
                        "output": shell_result.output,
                        "success": shell_result.success,
                        "return_code": shell_result.return_code,
                        "cwd": shell_result.working_directory
                    })
                    
                    # Track errors for stuck detection
                    if not shell_result.success and shell_result.output:
                        recent_errors.append(shell_result.output[:300])
                        # Keep only last 10 errors
                        if len(recent_errors) > 10:
                            recent_errors.pop(0)
                    
                    self.workspace.log_activity(
                        self.agent_id, task.id, "ran_command",
                        f"$ {action_content[:40]}... -> {shell_result.return_code}"
                    )
                    
                    formatted = self._format_shell_result(shell_result)
                    
                    # Suggest help if multiple consecutive failures
                    help_hint = ""
                    if self._should_suggest_help():
                        help_hint = "\n\n💡 **Tip:** You've had several failures. Consider using `<help>what you're trying to do</help>` to get guidance."
                    
                    self._conversation.append(Message(
                        role="user",
                        content=f"Shell result:\n{formatted}{help_hint}\n\nContinue with the task."
                    ))
                
                elif action_type == "read_file":
                    file_path = action_content
                    self._debug("read_file", f"Reading: {file_path}")
                    success, content = self._read_file(file_path)
                    
                    # Track action
                    self._track_action("read_file", file_path, success, content[:100] if not success else "OK")
                    
                    if success:
                        self._debug("file_read", f"Read {file_path} successfully")
                        self.workspace.log_activity(
                            self.agent_id, task.id, "read_file",
                            f"Read: {file_path}"
                        )
                    else:
                        self._debug("file_error", content)
                        recent_errors.append(content[:300])
                    
                    help_hint = ""
                    if not success and self._should_suggest_help():
                        help_hint = "\n\n💡 **Tip:** Having trouble? Use `<help>what you're trying to do</help>` for guidance."
                    
                    self._conversation.append(Message(
                        role="user",
                        content=f"{content}{help_hint}\n\nContinue with the task."
                    ))
                
                elif action_type == "list_dir":
                    dir_path = action_content or "."
                    self._debug("list_dir", f"Listing: {dir_path}")
                    success, content = self._list_dir(dir_path)
                    
                    # Track action
                    self._track_action("list_dir", dir_path, success, content[:100] if not success else "OK")
                    
                    if success:
                        self._debug("dir_listed", f"Listed {dir_path}")
                        self.workspace.log_activity(
                            self.agent_id, task.id, "list_dir",
                            f"Listed: {dir_path}"
                        )
                    else:
                        self._debug("file_error", content)
                    
                    self._conversation.append(Message(
                        role="user",
                        content=f"{content}\n\nContinue with the task."
                    ))
                
                elif action_type == "write_file":
                    # Format: "path|||content"
                    parts = action_content.split("|||", 1)
                    if len(parts) != 2:
                        self._track_action("write_file", "invalid format", False, "Missing path or content")
                        self._conversation.append(Message(
                            role="user",
                            content="Error: Invalid write_file format. Use: <write_file path=\"filepath\">content</write_file>"
                        ))
                        continue
                    
                    file_path, content = parts[0].strip(), parts[1]
                    # Remove leading newline if present (from multiline format)
                    if content.startswith("\n"):
                        content = content[1:]
                    
                    self._debug("write_file", f"Writing: {file_path}")
                    success, message = self._write_file(file_path, content)
                    
                    # Track action
                    self._track_action("write_file", file_path, success, message[:100])
                    
                    if success:
                        full_path = self._resolve_path(file_path)
                        self.workspace.register_file(full_path, self.agent_id)
                        result.files_created.append(file_path)
                        self._debug("file_written", message)
                        self.workspace.log_activity(
                            self.agent_id, task.id, "write_file",
                            f"Wrote: {file_path}"
                        )
                    else:
                        self._debug("file_error", message)
                        recent_errors.append(message[:300])
                    
                    help_hint = ""
                    if not success and self._should_suggest_help():
                        help_hint = "\n\n💡 **Tip:** Having trouble? Use `<help>what you're trying to do</help>` for guidance."
                    
                    self._conversation.append(Message(
                        role="user",
                        content=f"{message}{help_hint}\n\nContinue with the task."
                    ))
                
                elif action_type == "edit_file":
                    # Format: "path|||old_text|||new_text"
                    parts = action_content.split("|||")
                    if len(parts) != 3:
                        self._track_action("edit_file", "invalid format", False, "Wrong number of ||| separators")
                        self._conversation.append(Message(
                            role="user",
                            content=(
                                "Error: Invalid edit_file format. Use:\n"
                                "<edit_file path=\"filepath\">\n"
                                "OLD_TEXT_TO_FIND\n"
                                "|||\n"
                                "NEW_TEXT_TO_REPLACE_WITH\n"
                                "</edit_file>"
                            )
                        ))
                        continue
                    
                    file_path = parts[0].strip()
                    old_text = parts[1]
                    new_text = parts[2]
                    
                    # Remove leading/trailing newlines from separator format
                    if old_text.startswith("\n"):
                        old_text = old_text[1:]
                    if old_text.endswith("\n"):
                        old_text = old_text[:-1]
                    if new_text.startswith("\n"):
                        new_text = new_text[1:]
                    if new_text.endswith("\n"):
                        new_text = new_text[:-1]
                    
                    self._debug("edit_file", f"Editing: {file_path}")
                    success, message = self._edit_file(file_path, old_text, new_text)
                    
                    # Track action
                    self._track_action("edit_file", file_path, success, message[:100])
                    
                    if success:
                        full_path = self._resolve_path(file_path)
                        self.workspace.register_file(full_path, self.agent_id, is_modification=True)
                        if file_path not in result.files_modified:
                            result.files_modified.append(file_path)
                        self._debug("file_edited", message)
                        self.workspace.log_activity(
                            self.agent_id, task.id, "edit_file",
                            f"Edited: {file_path}"
                        )
                    else:
                        self._debug("file_error", message)
                        recent_errors.append(message[:300])
                    
                    help_hint = ""
                    if not success and self._should_suggest_help():
                        help_hint = "\n\n💡 **Tip:** Edit failed. Use `<read_file>` to see actual file contents, or `<help>what you're trying to do</help>` for guidance."
                    
                    self._conversation.append(Message(
                        role="user",
                        content=f"{message}{help_hint}\n\nContinue with the task."
                    ))
                
                elif action_type == "help":
                    # Agent explicitly asks for help
                    help_request = action_content
                    self._debug("help_request", f"Agent asking for help: {help_request[:100]}")
                    self._help_requests += 1
                    
                    # Check if we've exceeded max help requests
                    if self._help_requests > self.supervisor_config.max_help_per_task:
                        self._conversation.append(Message(
                            role="user",
                            content=(
                                "⚠️ You've already received maximum help for this task. "
                                "Try to complete the task with what you've learned, or use <error> if it's truly impossible."
                            )
                        ))
                        continue
                    
                    if not self.supervisor_config.enabled:
                        self._conversation.append(Message(
                            role="user",
                            content="Supervisor is not enabled. Try a different approach or use <error> if the task is impossible."
                        ))
                        continue
                    
                    # Get supervisor guidance with the specific help request
                    self._debug("supervisor", f"Getting help for: {help_request[:100]}")
                    guidance = await self._get_supervisor_guidance(
                        task.description, 
                        f"Agent requested help: {help_request}",
                        result.shell_history, 
                        recent_errors,
                        help_request=help_request
                    )
                    self._debug("supervisor", f"Guidance: {guidance[:200]}...")
                    
                    # Reset consecutive failures after getting help
                    self._consecutive_failures = 0
                    
                    self.workspace.log_activity(
                        self.agent_id, task.id, "requested_help",
                        f"Help: {help_request[:40]}..."
                    )
                    
                    self._conversation.append(Message(
                        role="user",
                        content=f"""🆘 **Supervisor Response** (help request {self._help_requests}/{self.supervisor_config.max_help_per_task})

{guidance}

Now try the suggested approach. You can ask for help again if needed."""
                    ))
                
                elif action_type == "think":
                    self._debug("think", action_content)
                    self._conversation.append(Message(
                        role="user",
                        content="Good reasoning. Now take an action: <read_file>, <write_file>, <edit_file>, <list_dir>, <shell>, <help>, <done>, or <error>."
                    ))
                
                else:
                    self._debug("unknown", f"Unknown action type, raw response: {response[:200]}")
                    self._conversation.append(Message(
                        role="user",
                        content="Please respond with one of: <read_file>, <write_file>, <edit_file>, <list_dir>, <shell>, <think>, <help>, <done>, or <error>."
                    ))
            
            else:
                if not result.success and not result.error:
                    result.error = f"Maximum iterations ({self.config.max_iterations}) reached"
                    self._debug("timeout", result.error)
        
        finally:
            self.workspace.set_agent_inactive(self.agent_id)
            result.completed_at = datetime.now().isoformat()
            self._debug("end", f"Finished task {task.id} - success={result.success}")
        
        return result


class AgentPool:
    """Pool of agents for parallel task execution."""
    
    def __init__(
        self,
        max_agents: int = 3,
        llm: Optional[LLM] = None,
        config: Optional[AgentConfig] = None,
        supervisor_config: Optional[SupervisorConfig] = None,
        workspace: Optional[Workspace] = None,
        debug_callback: Optional[DebugCallback] = None
    ):
        self.max_agents = max_agents
        self.llm = llm or get_llm()
        self.config = config or get_config().agent
        self.supervisor_config = supervisor_config or get_config().supervisor
        self.workspace = workspace or get_workspace()
        self.debug_callback = debug_callback
        self._semaphore = asyncio.Semaphore(max_agents)
        self._agent_counter = 0
    
    def _create_agent(self) -> Agent:
        self._agent_counter += 1
        return Agent(
            agent_id=f"agent_{self._agent_counter}",
            llm=self.llm,
            config=self.config,
            supervisor_config=self.supervisor_config,
            workspace=self.workspace,
            debug_callback=self.debug_callback
        )
    
    async def execute_task(self, task: Task, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        async with self._semaphore:
            agent = self._create_agent()
            return await agent.execute(task, context)
    
    async def execute_tasks(
        self,
        tasks: List[Task],
        contexts: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, AgentResult]:
        contexts = contexts or {}
        
        async def run_task(task: Task) -> tuple[str, AgentResult]:
            context = contexts.get(task.id, {})
            result = await self.execute_task(task, context)
            return task.id, result
        
        results = await asyncio.gather(*[run_task(t) for t in tasks])
        return dict(results)
