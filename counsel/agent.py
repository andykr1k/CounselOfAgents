"""General-purpose worker agent with shell access, LLM reasoning, and workspace awareness."""

import asyncio
import re
import os
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from counsel.config import AgentConfig, get_config
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


AGENT_SYSTEM_PROMPT = """You are a general-purpose AI agent that executes tasks by running shell commands.

You have access to a Linux/Unix shell and can run any command to complete your task.

## Available Actions

Use XML tags to take actions:

1. **Run a shell command**:
   <shell>command here</shell>

2. **Think/reason about the task**:
   <think>your reasoning here</think>

3. **Mark task complete with result**:
   <done>final result or summary</done>

4. **Report an error if task cannot be completed**:
   <error>description of what went wrong</error>

## Workspace Context

{workspace_context}

## CRITICAL: Shell Environment

**Each command runs in a NEW subprocess.** This means:
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
- Or create the venv and immediately use its python:
  - `python -m venv venv && ./venv/bin/pip install requests`

**Shell compatibility:**
- Use `/bin/sh` compatible syntax
- Use `.` instead of `source` if needed (but remember it won't persist)
- Avoid bash-specific syntax like `source`, `[[`, arrays, etc.

## Rules

1. Start by understanding what needs to be done
2. Use <think> to plan before running commands
3. Run ONE shell command at a time and observe output
4. Be aware of files/directories created by other agents (see workspace context)
5. When creating files, register them in your result
6. Use <done> when complete, or <error> if impossible
7. DON'T give up on first error - try alternative approaches!

## File Operations

When creating files, prefer these methods:
- `echo "content" > file.txt` for single lines
- `cat > file.txt << 'EOF'` for multi-line content
- `mkdir -p directory` for directories

## Coordination Notes

- Check the workspace context to see what other agents have done
- Use existing directories/files when they fit your needs
- Your work will be visible to other agents through the shared workspace

## Current Task

{task_description}

## Context from Dependencies

{dependency_context}

Begin working. Start with <think> to plan your approach."""


class Agent:
    """General-purpose worker agent with workspace awareness."""
    
    def __init__(
        self,
        agent_id: str,
        llm: Optional[LLM] = None,
        config: Optional[AgentConfig] = None,
        workspace: Optional[Workspace] = None,
        debug_callback: Optional[DebugCallback] = None
    ):
        self.agent_id = agent_id
        self.llm = llm or get_llm()
        self.config = config or get_config().agent
        self.workspace = workspace or get_workspace()
        self.shell = get_shell()
        self._conversation: List[Message] = []
        self.debug_callback = debug_callback
        
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
    
    def _detect_stuck(self, shell_history: List[Dict], recent_errors: List[str]) -> Optional[str]:
        """Detect if the agent is stuck in a loop.
        
        Returns a description of the stuck pattern if detected, None otherwise.
        """
        if len(shell_history) < 3:
            return None
        
        # Check for repeated commands (same command 3+ times)
        recent_cmds = [h['command'].strip() for h in shell_history[-6:]]
        for cmd in set(recent_cmds):
            if recent_cmds.count(cmd) >= 3:
                return f"Repeating the same command: '{cmd[:50]}...'"
        
        # Check for repeated errors
        if len(recent_errors) >= 3:
            # Check if same error type keeps appearing
            error_patterns = {}
            for err in recent_errors[-5:]:
                # Extract key part of error
                key = err[:100] if err else "unknown"
                error_patterns[key] = error_patterns.get(key, 0) + 1
            
            for pattern, count in error_patterns.items():
                if count >= 3:
                    return f"Repeated error pattern: '{pattern[:80]}...'"
        
        # Check for failed commands loop
        recent_results = [h.get('success', True) for h in shell_history[-5:]]
        if recent_results.count(False) >= 4:
            return "Most recent commands are failing"
        
        return None
    
    async def _get_supervisor_guidance(self, task_description: str, stuck_reason: str, 
                                        shell_history: List[Dict], recent_errors: List[str]) -> str:
        """Get guidance from a supervisor perspective when stuck."""
        
        # Build context of what's been tried
        recent_attempts = []
        for h in shell_history[-5:]:
            cmd = h['command'][:100]
            success = "✓" if h.get('success') else "✗"
            output = (h.get('output', '') or '')[:200]
            recent_attempts.append(f"  {success} `{cmd}`\n    → {output}")
        
        attempts_str = "\n".join(recent_attempts) if recent_attempts else "No commands run yet"
        errors_str = "\n".join(f"  - {e[:150]}" for e in recent_errors[-3:]) if recent_errors else "None"
        
        supervisor_prompt = f"""You are a senior engineer helping a junior developer who is stuck.

## The Task
{task_description}

## Problem
The developer is stuck: {stuck_reason}

## Recent Attempts
{attempts_str}

## Recent Errors
{errors_str}

## Your Job
Analyze what's going wrong and provide SPECIFIC guidance to break out of this loop.
Think about:
1. Is there a fundamental misunderstanding of the task?
2. Are they missing a prerequisite step?
3. Should they try a completely different approach?
4. Is there an environment issue to address first?

Provide concise, actionable guidance (2-4 sentences) that will help them succeed.
Start your response directly with the guidance."""

        try:
            guidance = await self.llm.chat([
                Message(role="system", content="You are a helpful senior engineer. Be direct and specific."),
                Message(role="user", content=supervisor_prompt)
            ])
            return guidance.strip()
        except Exception as e:
            return f"Try a different approach. The current method isn't working: {stuck_reason}"
    
    def _parse_action(self, response: str) -> tuple[str, str]:
        patterns = [
            (r"<shell>(.*?)</shell>", "shell"),
            (r"<think>(.*?)</think>", "think"),
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
        
        self._debug("start", f"Task: {task.description}")
        self.workspace.set_agent_active(self.agent_id, task.description[:100])
        self.workspace.log_activity(self.agent_id, task.id, "started", f"Started: {task.description[:50]}")
        
        try:
            workspace_context = self.workspace.get_context_for_agent(task.id)
            self._debug("context", f"Workspace: {workspace_context[:200]}...")
            
            dep_context = "None"
            if context:
                dep_parts = [f"### {dep_id}:\n{dep_result}" for dep_id, dep_result in context.items()]
                dep_context = "\n\n".join(dep_parts)
                self._debug("deps", f"Dependencies: {list(context.keys())}")
            
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
                
                # Check for stuck patterns (after iteration 5, every 3 iterations, not too soon after last intervention)
                if iteration >= 5 and iteration % 3 == 0 and (iteration - last_intervention_iteration) >= 5:
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
                            content=f"""⚠️ **INTERVENTION - You seem to be stuck**

Problem detected: {stuck_reason}

**Guidance from supervisor:**
{guidance}

Please try a DIFFERENT approach based on this guidance. Don't repeat the same commands that aren't working."""
                        ))
                        last_intervention_iteration = iteration
                
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
                    self._conversation.append(Message(
                        role="user",
                        content=f"Shell result:\n{formatted}\n\nContinue with the task."
                    ))
                
                elif action_type == "think":
                    self._debug("think", action_content)
                    self._conversation.append(Message(
                        role="user",
                        content="Good reasoning. Now take an action: <shell>, <done>, or <error>."
                    ))
                
                else:
                    self._debug("unknown", f"Unknown action type, raw response: {response[:200]}")
                    self._conversation.append(Message(
                        role="user",
                        content="Please respond with: <shell>, <think>, <done>, or <error>."
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
        workspace: Optional[Workspace] = None,
        debug_callback: Optional[DebugCallback] = None
    ):
        self.max_agents = max_agents
        self.llm = llm or get_llm()
        self.config = config or get_config().agent
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
