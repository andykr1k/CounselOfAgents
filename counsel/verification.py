"""
Task Verification Agent for the Counsel Of Agents Orchestration Platform.

This module provides intelligent verification of completed tasks to ensure
they meet their requirements. The verifier analyzes task outputs, checks
for common issues, and provides actionable feedback for remediation.
"""

import asyncio
import re
import os
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from counsel.config import get_config, AgentConfig
from counsel.llm import LLM, Message, get_llm
from counsel.workspace import Workspace, get_workspace
from counsel.task_graph import Task, TaskStatus


class VerificationStatus(Enum):
    """Status of a task verification."""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class VerificationIssue:
    """An issue found during verification."""
    severity: str  # "critical", "major", "minor", "suggestion"
    category: str  # "missing_output", "incorrect_implementation", "incomplete", etc.
    description: str
    remediation: str  # Specific steps to fix the issue
    file_path: Optional[str] = None
    line_number: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "remediation": self.remediation,
            "file_path": self.file_path,
            "line_number": self.line_number
        }


@dataclass
class VerificationResult:
    """Result of a task verification."""
    task_id: str
    status: VerificationStatus
    score: float  # 0.0 to 1.0 confidence score
    issues: List[VerificationIssue] = field(default_factory=list)
    summary: str = ""
    verified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    verification_time_ms: int = 0
    retry_recommended: bool = False
    remediation_instructions: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        """Check if verification passed."""
        return self.status == VerificationStatus.PASSED
    
    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return any(i.severity == "critical" for i in self.issues)
    
    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "score": self.score,
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
            "verified_at": self.verified_at,
            "verification_time_ms": self.verification_time_ms,
            "retry_recommended": self.retry_recommended,
            "remediation_instructions": self.remediation_instructions
        }


# System prompt for the verification agent
VERIFICATION_SYSTEM_PROMPT = """You are an expert Quality Assurance agent responsible for verifying that tasks have been completed correctly.

Your job is to analyze:
1. The original task description (what was supposed to be done)
2. The agent's reported result (what they claim to have done)
3. The actual workspace state (what files/outputs exist)
4. Any shell command history (what commands were run)

## Your Verification Process

1. **Understand Requirements**: Parse the original task to understand ALL requirements
2. **Check Deliverables**: Verify each expected deliverable exists and is correct
3. **Validate Implementation**: Check that the implementation matches requirements
4. **Identify Issues**: Find any missing, incorrect, or incomplete elements
5. **Provide Remediation**: Give specific, actionable instructions to fix issues

## Output Format

Respond with a JSON object in this exact format:

```json
{{
  "status": "passed|failed|partial",
  "score": 0.85,
  "summary": "Brief summary of verification results",
  "issues": [
    {{
      "severity": "critical|major|minor|suggestion",
      "category": "category_name",
      "description": "What is wrong",
      "remediation": "Specific steps to fix",
      "file_path": "optional/path/to/file",
      "line_number": null
    }}
  ],
  "retry_recommended": true,
  "remediation_instructions": "Overall instructions for the agent to fix all issues"
}}
```

## Issue Categories

- `missing_file`: A required file was not created
- `missing_functionality`: Required functionality is missing
- `incorrect_implementation`: Implementation doesn't match requirements
- `syntax_error`: Code has syntax errors
- `incomplete`: Task was only partially completed
- `test_failure`: Tests are failing
- `missing_dependency`: Required dependencies not installed
- `configuration_error`: Configuration is incorrect
- `documentation_missing`: Required documentation not provided

## Severity Levels

- `critical`: Task fundamentally fails its purpose, must fix
- `major`: Significant issue that impacts functionality
- `minor`: Small issue that doesn't prevent core functionality
- `suggestion`: Optional improvement

## Rules

1. Be thorough but fair - don't fail tasks for trivial issues
2. Always provide specific, actionable remediation steps
3. Consider the task description as the source of truth
4. If the agent claims success but evidence suggests otherwise, investigate
5. A score of 0.8+ with no critical issues = PASSED
6. A score of 0.5-0.8 or minor/major issues = PARTIAL
7. A score below 0.5 or critical issues = FAILED

## Current Context

{workspace_context}

## Task to Verify

**Task ID**: {task_id}
**Description**: {task_description}

## Agent's Reported Result

{agent_result}

## Shell Command History

{shell_history}

## Files Created/Modified

{files_info}

Now analyze and verify this task completion."""


class TaskVerifier:
    """
    Intelligent task verification agent.
    
    Verifies that completed tasks actually meet their requirements by:
    1. Analyzing the task description and requirements
    2. Checking the agent's claimed result
    3. Examining actual workspace state (files, outputs)
    4. Providing detailed feedback and remediation steps
    
    Usage:
        verifier = TaskVerifier()
        result = await verifier.verify(task, agent_result)
        if not result.passed:
            # Handle remediation
    """
    
    def __init__(
        self,
        llm: Optional[LLM] = None,
        workspace: Optional[Workspace] = None,
        config: Optional[AgentConfig] = None
    ):
        self.llm = llm or get_llm()
        self.workspace = workspace or get_workspace()
        self.config = config or get_config().agent
        
    async def verify(
        self,
        task: Task,
        agent_result: Optional[Any] = None,
        shell_history: Optional[List[Dict]] = None,
        files_created: Optional[List[str]] = None,
        files_modified: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Verify that a task has been completed correctly.
        
        Args:
            task: The task that was supposedly completed
            agent_result: The result reported by the agent
            shell_history: History of shell commands run by the agent
            files_created: List of files the agent claims to have created
            files_modified: List of files the agent claims to have modified
            
        Returns:
            VerificationResult with status, issues, and remediation instructions
        """
        start_time = datetime.now()
        
        try:
            # Build context for verification
            workspace_context = self.workspace.get_context_for_agent(task.id)
            
            # Format shell history
            shell_history_str = "No commands recorded"
            if shell_history:
                history_lines = []
                for h in shell_history[-20:]:  # Last 20 commands
                    status = "✓" if h.get('success') else "✗"
                    cmd = h.get('command', '')[:100]
                    output = (h.get('output', '') or '')[:200]
                    history_lines.append(f"{status} $ {cmd}")
                    if output:
                        history_lines.append(f"    → {output[:150]}...")
                shell_history_str = "\n".join(history_lines)
            
            # Format files info
            files_info_parts = []
            all_files = (files_created or []) + (files_modified or [])
            
            if files_created:
                files_info_parts.append(f"Created: {', '.join(files_created)}")
            if files_modified:
                files_info_parts.append(f"Modified: {', '.join(files_modified)}")
            
            # Check if files actually exist and get their contents
            for file_path in all_files[:5]:  # Check first 5 files
                full_path = os.path.join(self.workspace.cwd, file_path)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, 'r') as f:
                            content = f.read(2000)  # First 2000 chars
                            if len(content) == 2000:
                                content += "\n... (truncated)"
                        files_info_parts.append(f"\n--- Content of {file_path} ---\n{content}")
                    except Exception as e:
                        files_info_parts.append(f"\n{file_path}: Could not read ({e})")
                else:
                    files_info_parts.append(f"\n{file_path}: FILE NOT FOUND")
            
            files_info = "\n".join(files_info_parts) if files_info_parts else "No files reported"
            
            # Format agent result
            result_str = str(agent_result) if agent_result else "No result reported"
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "... (truncated)"
            
            # Build the verification prompt
            prompt = VERIFICATION_SYSTEM_PROMPT.format(
                workspace_context=workspace_context,
                task_id=task.id,
                task_description=task.description,
                agent_result=result_str,
                shell_history=shell_history_str,
                files_info=files_info
            )
            
            # Get verification from LLM
            messages = [
                Message(role="system", content="You are an expert QA verification agent. Respond only with valid JSON."),
                Message(role="user", content=prompt)
            ]
            
            response = await self.llm.chat(messages)
            
            # Parse the response
            verification_result = self._parse_verification_response(task.id, response)
            
            # Calculate verification time
            verification_result.verification_time_ms = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            
            return verification_result
            
        except Exception as e:
            return VerificationResult(
                task_id=task.id,
                status=VerificationStatus.ERROR,
                score=0.0,
                summary=f"Verification error: {str(e)}",
                issues=[VerificationIssue(
                    severity="critical",
                    category="verification_error",
                    description=f"Verification process failed: {str(e)}",
                    remediation="Review the task manually"
                )],
                verification_time_ms=int((datetime.now() - start_time).total_seconds() * 1000)
            )
    
    def _parse_verification_response(self, task_id: str, response: str) -> VerificationResult:
        """Parse the LLM's verification response."""
        import json
        
        # Extract JSON from response
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: try to parse the whole response
                json_str = response
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON parsing fails, create a partial result
            return VerificationResult(
                task_id=task_id,
                status=VerificationStatus.PARTIAL,
                score=0.5,
                summary="Could not parse verification response, manual review recommended",
                issues=[VerificationIssue(
                    severity="major",
                    category="verification_error",
                    description="Verification response was not valid JSON",
                    remediation="Review the task output manually"
                )]
            )
        
        # Parse status
        status_str = data.get("status", "partial").lower()
        status_map = {
            "passed": VerificationStatus.PASSED,
            "failed": VerificationStatus.FAILED,
            "partial": VerificationStatus.PARTIAL,
            "skipped": VerificationStatus.SKIPPED
        }
        status = status_map.get(status_str, VerificationStatus.PARTIAL)
        
        # Parse issues
        issues = []
        for issue_data in data.get("issues", []):
            issues.append(VerificationIssue(
                severity=issue_data.get("severity", "major"),
                category=issue_data.get("category", "unknown"),
                description=issue_data.get("description", ""),
                remediation=issue_data.get("remediation", ""),
                file_path=issue_data.get("file_path"),
                line_number=issue_data.get("line_number")
            ))
        
        return VerificationResult(
            task_id=task_id,
            status=status,
            score=float(data.get("score", 0.5)),
            summary=data.get("summary", ""),
            issues=issues,
            retry_recommended=data.get("retry_recommended", False),
            remediation_instructions=data.get("remediation_instructions")
        )
    
    def quick_verify(
        self,
        task: Task,
        files_created: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Perform a quick verification without full LLM analysis.
        
        This is a synchronous check that verifies basic requirements
        without invoking the LLM. Useful for fast pre-checks.
        
        Args:
            task: The task to verify
            files_created: List of files that should exist
            
        Returns:
            Tuple of (passed, reason)
        """
        # Check if any required files exist
        if files_created:
            missing_files = []
            for file_path in files_created:
                full_path = os.path.join(self.workspace.cwd, file_path)
                if not os.path.exists(full_path):
                    missing_files.append(file_path)
            
            if missing_files:
                return False, f"Missing files: {', '.join(missing_files)}"
        
        # Check if task result indicates success
        if task.result:
            result_lower = str(task.result).lower()
            failure_indicators = ['failed', 'error', 'could not', 'unable to', 'exception']
            for indicator in failure_indicators:
                if indicator in result_lower:
                    return False, f"Task result indicates failure: contains '{indicator}'"
        
        return True, "Quick verification passed"


class VerificationManager:
    """
    Manages verification across multiple tasks with retry logic.
    
    Provides:
    - Batch verification of completed tasks
    - Configurable retry policies
    - Verification result aggregation
    - Integration with orchestrator
    """
    
    def __init__(
        self,
        verifier: Optional[TaskVerifier] = None,
        max_retries: int = 2,
        min_passing_score: float = 0.8
    ):
        self.verifier = verifier or TaskVerifier()
        self.max_retries = max_retries
        self.min_passing_score = min_passing_score
        self._verification_cache: Dict[str, VerificationResult] = {}
    
    async def verify_task(
        self,
        task: Task,
        agent_result: Any = None,
        shell_history: Optional[List[Dict]] = None,
        files_created: Optional[List[str]] = None,
        files_modified: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> VerificationResult:
        """
        Verify a single task with caching support.
        """
        cache_key = f"{task.id}_{task.completed_at or ''}"
        
        if use_cache and cache_key in self._verification_cache:
            return self._verification_cache[cache_key]
        
        result = await self.verifier.verify(
            task=task,
            agent_result=agent_result,
            shell_history=shell_history,
            files_created=files_created,
            files_modified=files_modified
        )
        
        self._verification_cache[cache_key] = result
        return result
    
    async def verify_all(
        self,
        tasks: List[Tuple[Task, Any, List[Dict], List[str], List[str]]]
    ) -> Dict[str, VerificationResult]:
        """
        Verify multiple tasks in parallel.
        
        Args:
            tasks: List of (task, agent_result, shell_history, files_created, files_modified) tuples
            
        Returns:
            Dict mapping task_id to VerificationResult
        """
        async def verify_one(item):
            task, result, history, created, modified = item
            return task.id, await self.verify_task(
                task, result, history, created, modified
            )
        
        results = await asyncio.gather(*[verify_one(t) for t in tasks])
        return dict(results)
    
    def get_remediation_prompt(self, verification_result: VerificationResult) -> str:
        """
        Generate a remediation prompt for an agent based on verification results.
        """
        if verification_result.passed:
            return ""
        
        lines = [
            "⚠️ **VERIFICATION FAILED - REMEDIATION REQUIRED**",
            "",
            f"**Status**: {verification_result.status.value.upper()}",
            f"**Score**: {verification_result.score:.0%}",
            f"**Summary**: {verification_result.summary}",
            ""
        ]
        
        if verification_result.issues:
            lines.append("**Issues Found:**")
            for i, issue in enumerate(verification_result.issues, 1):
                severity_icon = {
                    "critical": "🔴",
                    "major": "🟠",
                    "minor": "🟡",
                    "suggestion": "🔵"
                }.get(issue.severity, "⚪")
                
                lines.append(f"\n{i}. {severity_icon} [{issue.severity.upper()}] {issue.category}")
                lines.append(f"   Problem: {issue.description}")
                lines.append(f"   Fix: {issue.remediation}")
                if issue.file_path:
                    lines.append(f"   File: {issue.file_path}")
        
        if verification_result.remediation_instructions:
            lines.append("")
            lines.append("**Overall Remediation Instructions:**")
            lines.append(verification_result.remediation_instructions)
        
        lines.append("")
        lines.append("Please address these issues and complete the task correctly.")
        
        return "\n".join(lines)
    
    def should_retry(self, verification_result: VerificationResult, attempt: int) -> bool:
        """
        Determine if a task should be retried based on verification result.
        """
        if verification_result.passed:
            return False
        
        if attempt >= self.max_retries:
            return False
        
        # Don't retry if score is very low (fundamentally wrong approach)
        if verification_result.score < 0.2:
            return False
        
        # Retry if recommended or if there are fixable issues
        return verification_result.retry_recommended or verification_result.score >= 0.3
    
    def clear_cache(self, task_id: Optional[str] = None):
        """Clear verification cache."""
        if task_id:
            keys_to_remove = [k for k in self._verification_cache if k.startswith(task_id)]
            for key in keys_to_remove:
                del self._verification_cache[key]
        else:
            self._verification_cache.clear()


# Convenience functions for module-level access
_verifier: Optional[TaskVerifier] = None
_verification_manager: Optional[VerificationManager] = None
_verifier_lock = asyncio.Lock()
_manager_lock = asyncio.Lock()


def get_verifier() -> TaskVerifier:
    """Get the global TaskVerifier instance (thread-safe)."""
    global _verifier
    if _verifier is None:
        _verifier = TaskVerifier()
    return _verifier


def get_verification_manager() -> VerificationManager:
    """Get the global VerificationManager instance (thread-safe)."""
    global _verification_manager
    if _verification_manager is None:
        _verification_manager = VerificationManager()
    return _verification_manager


def reset_verification() -> None:
    """Reset verification singletons (for testing)."""
    global _verifier, _verification_manager
    _verifier = None
    _verification_manager = None
