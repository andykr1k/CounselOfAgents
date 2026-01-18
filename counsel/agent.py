"""General-purpose worker agent with shell access, LLM reasoning, and workspace awareness."""

import asyncio
import re
import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from counsel.config import AgentConfig, get_config
from counsel.llm import LLM, Message, get_llm
from counsel.shell import Shell, ShellResult, get_shell
from counsel.workspace import Workspace, get_workspace
from counsel.task_graph import Task


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

## Rules

1. Start by understanding what needs to be done
2. Use <think> to plan before running commands
3. Run ONE shell command at a time and observe output
4. Be aware of files/directories created by other agents (see workspace context)
5. When creating files, register them in your result
6. Use <done> when complete, or <error> if impossible

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
        workspace: Optional[Workspace] = None
    ):
        self.agent_id = agent_id
        self.llm = llm or get_llm()
        self.config = config or get_config().agent
        self.workspace = workspace or get_workspace()
        self.shell = get_shell()
        self._conversation: List[Message] = []
        
        self.shell._cwd = self.workspace.cwd
    
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
        created = []
        modified = []
        
        if not result.success:
            return created, modified
        
        command_lower = command.lower().strip()
        
        redirect_match = re.search(r'>\s*([^\s|&;]+)', command)
        if redirect_match and '>>' not in command:
            created.append(redirect_match.group(1).strip("'\""))
        
        append_match = re.search(r'>>\s*([^\s|&;]+)', command)
        if append_match:
            modified.append(append_match.group(1).strip("'\""))
        
        if command_lower.startswith('touch '):
            files = command[6:].split()
            created.extend([f.strip("'\"") for f in files])
        
        if 'mkdir' in command_lower:
            parts = command.split()
            for part in parts:
                if part.startswith('-') or part == 'mkdir':
                    continue
                dir_name = part.strip("'\"")
                if dir_name:
                    self.workspace.register_directory(
                        os.path.join(self.shell.cwd, dir_name),
                        self.agent_id
                    )
        
        if command_lower.startswith('cp '):
            parts = command.split()
            if len(parts) >= 3:
                dest = parts[-1].strip("'\"")
                if not os.path.isdir(os.path.join(self.shell.cwd, dest)):
                    created.append(dest)
        
        if 'npm init' in command_lower or 'yarn init' in command_lower:
            created.append('package.json')
        
        return created, modified
    
    async def execute(self, task: Task, context: Optional[Dict[str, Any]] = None) -> AgentResult:
        result = AgentResult(task_id=task.id, success=False)
        context = context or {}
        
        self.workspace.set_agent_active(self.agent_id, task.description[:100])
        self.workspace.log_activity(self.agent_id, task.id, "started", f"Started: {task.description[:50]}")
        
        try:
            workspace_context = self.workspace.get_context_for_agent(task.id)
            
            dep_context = "None"
            if context:
                dep_parts = [f"### {dep_id}:\n{dep_result}" for dep_id, dep_result in context.items()]
                dep_context = "\n\n".join(dep_parts)
            
            system_prompt = AGENT_SYSTEM_PROMPT.format(
                workspace_context=workspace_context,
                task_description=task.description,
                dependency_context=dep_context
            )
            
            self._conversation = [Message(role="system", content=system_prompt)]
            self._conversation.append(Message(role="user", content="Begin working on the task."))
            
            for iteration in range(self.config.max_iterations):
                result.iterations = iteration + 1
                
                try:
                    response = await self.llm.chat(self._conversation)
                except Exception as e:
                    result.error = f"LLM error: {str(e)}"
                    break
                
                self._conversation.append(Message(role="assistant", content=response))
                action_type, action_content = self._parse_action(response)
                
                if action_type == "done":
                    result.success = True
                    result.result = action_content
                    self.workspace.log_activity(self.agent_id, task.id, "completed", f"Completed: {action_content[:50]}")
                    break
                
                elif action_type == "error":
                    result.success = False
                    result.error = action_content
                    self.workspace.log_activity(self.agent_id, task.id, "failed", f"Failed: {action_content[:50]}")
                    break
                
                elif action_type == "shell":
                    if len(result.shell_history) >= self.config.max_shell_commands:
                        self._conversation.append(Message(
                            role="user",
                            content=f"⚠️ Maximum commands reached. Complete now or report an error."
                        ))
                        continue
                    
                    shell_result = await self.shell.run(action_content)
                    
                    if action_content.strip().startswith('cd '):
                        self.workspace.set_cwd(self.shell.cwd)
                    
                    created, modified = self._detect_file_operations(action_content, shell_result)
                    
                    for file_path in created:
                        full_path = os.path.join(self.shell.cwd, file_path)
                        self.workspace.register_file(full_path, self.agent_id)
                        result.files_created.append(file_path)
                    
                    for file_path in modified:
                        full_path = os.path.join(self.shell.cwd, file_path)
                        self.workspace.register_file(full_path, self.agent_id, is_modification=True)
                        result.files_modified.append(file_path)
                    
                    result.shell_history.append({
                        "command": action_content,
                        "output": shell_result.output,
                        "success": shell_result.success,
                        "return_code": shell_result.return_code,
                        "cwd": shell_result.working_directory
                    })
                    
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
                    self._conversation.append(Message(
                        role="user",
                        content="Good reasoning. Now take an action: <shell>, <done>, or <error>."
                    ))
                
                else:
                    self._conversation.append(Message(
                        role="user",
                        content="Please respond with: <shell>, <think>, <done>, or <error>."
                    ))
            
            else:
                if not result.success and not result.error:
                    result.error = f"Maximum iterations ({self.config.max_iterations}) reached"
        
        finally:
            self.workspace.set_agent_inactive(self.agent_id)
            result.completed_at = datetime.now().isoformat()
        
        return result


class AgentPool:
    """Pool of agents for parallel task execution."""
    
    def __init__(
        self,
        max_agents: int = 3,
        llm: Optional[LLM] = None,
        config: Optional[AgentConfig] = None,
        workspace: Optional[Workspace] = None
    ):
        self.max_agents = max_agents
        self.llm = llm or get_llm()
        self.config = config or get_config().agent
        self.workspace = workspace or get_workspace()
        self._semaphore = asyncio.Semaphore(max_agents)
        self._agent_counter = 0
    
    def _create_agent(self) -> Agent:
        self._agent_counter += 1
        return Agent(
            agent_id=f"agent_{self._agent_counter}",
            llm=self.llm,
            config=self.config,
            workspace=self.workspace
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
