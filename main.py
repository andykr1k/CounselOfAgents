#!/usr/bin/env python3
"""
Counsel Of Agents Orchestration Platform - Command Line Interface

Enterprise-grade multi-agent orchestration for automated task execution.

Usage:
    counsel                           # Interactive shell mode
    counsel "Create a REST API"       # Single task execution
    counsel -w ./project "Add tests"  # With specific workspace
    counsel --verify "Build app"      # With task verification

Environment Variables:
    COUNSEL_MODEL       HuggingFace model ID
    COUNSEL_DEVICE      Device (auto, cuda, mps, cpu)
    COUNSEL_VERIFY      Enable verification by default
    COUNSEL_DEBUG       Enable debug mode
"""

import asyncio
import argparse
import sys
import os
import readline
import atexit
import signal
from typing import Optional, List
from datetime import datetime
from collections import deque
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.live import Live
from rich.prompt import Prompt, Confirm
from rich.columns import Columns
from rich import box

from counsel import (
    Config, get_config, set_config,
    Orchestrator, ExecutionResult,
    TaskGraph, TaskStatus,
    Workspace, get_workspace,
    get_shell,
    get_logger, configure_logging, LogLevel
)
from counsel.shell import cleanup_shells
from counsel.llm import cleanup_llm
from counsel.models import (
    MODEL_CATALOG, ModelInfo, 
    get_model_by_id, get_families, get_suitable_models,
    estimate_system_resources, save_model_selection, load_model_selection,
    clear_model_selection
)
from counsel.jobs import Job, JobStatus, JobManager, get_job_manager
from counsel.verification import VerificationStatus

console = Console()

# History file for command history
HISTORY_FILE = Path.home() / ".counsel_history"

# Track if cleanup has been done
_cleanup_done = False


def cleanup_all():
    """Clean up all resources - processes, threads, etc."""
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    
    try:
        # Clean up shell processes
        cleanup_shells()
    except Exception:
        pass
    
    try:
        # Clean up LLM resources
        cleanup_llm()
    except Exception:
        pass
    
    try:
        # Save command history
        save_history()
    except Exception:
        pass


def signal_handler(signum, frame):
    """Handle SIGINT and SIGTERM."""
    console.print("\n[yellow]Cleaning up...[/yellow]")
    cleanup_all()
    sys.exit(0)


# Register cleanup on normal exit
atexit.register(cleanup_all)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def setup_readline():
    """Setup readline for command history with up/down arrows."""
    # Configure readline
    readline.set_history_length(1000)
    
    # Load history if it exists
    try:
        if HISTORY_FILE.exists():
            readline.read_history_file(str(HISTORY_FILE))
    except Exception:
        pass
    
    # Enable emacs editing mode (for arrow keys)
    try:
        readline.parse_and_bind("set editing-mode emacs")
    except Exception:
        pass  # Some systems don't support this


def save_history():
    """Save command history to file."""
    try:
        readline.write_history_file(str(HISTORY_FILE))
    except Exception:
        pass


def add_to_history(command: str):
    """Add a command to history."""
    if command.strip():
        readline.add_history(command)
        save_history()


def show_model_selection(config: Config) -> bool:
    """Interactive model selection screen. Returns True if a model was selected."""
    
    console.clear()
    console.print()
    console.print(Panel(
        "[bold cyan]🤖 Model Selection[/bold cyan]\n\n"
        "Choose a language model to power your agents.\n"
        "Models are downloaded from HuggingFace on first use.",
        border_style="cyan"
    ))
    
    # Check system resources
    console.print("\n[bold]📊 Detecting system resources...[/bold]")
    resources = estimate_system_resources()
    
    resource_info = []
    resource_info.append(f"[cyan]RAM:[/cyan] {resources['ram_gb']} GB")
    
    if resources["has_cuda"]:
        resource_info.append(f"[green]GPU:[/green] CUDA detected - {resources['vram_gb']} GB VRAM")
    elif resources["has_mps"]:
        resource_info.append(f"[green]GPU:[/green] Apple Silicon (MPS) - uses unified memory")
    else:
        resource_info.append("[yellow]GPU:[/yellow] No GPU detected - will use CPU (slower)")
    
    resource_info.append(f"[dim]Recommended quantization:[/dim] {resources['recommended_quantization']}")
    
    console.print(Panel("\n".join(resource_info), title="[bold]System Info[/bold]", border_style="blue"))
    
    # Check for saved selection
    saved = load_model_selection()
    if saved:
        console.print(f"\n[dim]Previously selected:[/dim] {saved['model_id']}")
        if Confirm.ask("Use previously selected model?", default=True):
            model = get_model_by_id(saved['model_id'])
            if model:
                _apply_model_selection(config, model, saved.get('quantization', '4bit'))
                return True
    
    # Get suitable models based on resources
    use_4bit = resources['recommended_quantization'] in ('4bit', '8bit')
    available_vram = resources['vram_gb'] if resources['has_cuda'] or resources['has_mps'] else 0
    suitable = get_suitable_models(resources['ram_gb'], available_vram, use_4bit)
    
    if not suitable:
        console.print("[yellow]⚠ Your system has limited resources. Showing smallest models.[/yellow]")
        suitable = MODEL_CATALOG[:5]  # Show smallest models
    
    # Display models
    console.print("\n[bold]Available Models:[/bold]\n")
    
    _display_model_table(suitable, use_4bit, resources)
    
    # Selection prompt
    console.print()
    console.print("[dim]Enter a number to select, or:[/dim]")
    console.print("  [cyan]a[/cyan] - Show all models (including ones that may not fit)")
    console.print("  [cyan]f[/cyan] - Filter by family (Qwen, Llama, etc.)")
    console.print("  [cyan]c[/cyan] - Enter custom HuggingFace model ID")
    console.print("  [cyan]q[/cyan] - Quit")
    console.print()
    
    while True:
        choice = Prompt.ask("Select model", default="1")
        
        if choice.lower() == 'q':
            return False
        
        if choice.lower() == 'a':
            console.print("\n[bold]All Models:[/bold]\n")
            _display_model_table(MODEL_CATALOG, use_4bit, resources)
            suitable = MODEL_CATALOG
            continue
        
        if choice.lower() == 'f':
            families = get_families()
            console.print("\n[bold]Model Families:[/bold]")
            for i, fam in enumerate(families, 1):
                console.print(f"  {i}. {fam}")
            fam_choice = Prompt.ask("Select family", default="1")
            try:
                fam_idx = int(fam_choice) - 1
                if 0 <= fam_idx < len(families):
                    from counsel.models import get_models_by_family
                    suitable = get_models_by_family(families[fam_idx])
                    console.print(f"\n[bold]{families[fam_idx]} Models:[/bold]\n")
                    _display_model_table(suitable, use_4bit, resources)
            except ValueError:
                pass
            continue
        
        if choice.lower() == 'c':
            custom_id = Prompt.ask("Enter HuggingFace model ID (e.g., 'org/model-name')")
            if custom_id:
                # Create a custom model entry
                custom_model = ModelInfo(
                    id=custom_id,
                    name=custom_id.split('/')[-1],
                    family="Custom",
                    size="Unknown",
                    ram_fp16="Unknown",
                    ram_4bit="Unknown",
                    vram_fp16="Unknown",
                    vram_4bit="Unknown",
                    context_length=4096,
                    description="Custom model from HuggingFace",
                    recommended_for=["custom"],
                )
                quant = _select_quantization(resources)
                _apply_model_selection(config, custom_model, quant)
                save_model_selection(custom_id, quant)
                return True
            continue
        
        # Try to parse as number
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(suitable):
                model = suitable[idx]
                quant = _select_quantization(resources)
                _apply_model_selection(config, model, quant)
                save_model_selection(model.id, quant)
                console.print(f"\n[green]✓ Selected:[/green] {model.name}")
                console.print(f"[dim]Model will be downloaded on first use.[/dim]\n")
                return True
            else:
                console.print(f"[red]Invalid selection. Enter 1-{len(suitable)}[/red]")
        except ValueError:
            console.print("[red]Invalid input. Enter a number or command.[/red]")


def _display_model_table(models: List[ModelInfo], use_4bit: bool, resources: dict) -> None:
    """Display a table of models."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("VRAM (4bit)" if use_4bit else "VRAM (fp16)", justify="right")
    table.add_column("RAM (4bit)" if use_4bit else "RAM (fp16)", justify="right")
    table.add_column("Context", justify="right")
    table.add_column("Description")
    
    available_vram = resources['vram_gb'] if resources['has_cuda'] or resources['has_mps'] else 0
    available_ram = resources['ram_gb']
    
    for i, model in enumerate(models, 1):
        vram = model.vram_4bit if use_4bit else model.vram_fp16
        ram = model.ram_4bit if use_4bit else model.ram_fp16
        
        # Check if it fits
        try:
            vram_gb = float(vram.split()[0])
            ram_gb = float(ram.split()[0])
        except:
            vram_gb = ram_gb = 999
        
        fits_vram = available_vram == 0 or vram_gb <= available_vram * 0.8
        fits_ram = ram_gb <= available_ram * 0.8
        fits = fits_vram and fits_ram
        
        # Color code based on fit
        if fits:
            style = "green" if "recommended" in model.recommended_for else ""
        else:
            style = "dim red"
        
        # Shorten description
        desc = model.description[:40] + "..." if len(model.description) > 40 else model.description
        
        # Format context length
        ctx = f"{model.context_length // 1000}k"
        
        table.add_row(
            str(i),
            f"[{style}]{model.name}[/{style}]" if style else model.name,
            model.size,
            vram,
            ram,
            ctx,
            desc
        )
    
    console.print(table)


def _select_quantization(resources: dict) -> str:
    """Let user select quantization level."""
    console.print("\n[bold]Quantization (memory/quality tradeoff):[/bold]")
    console.print("  [cyan]1[/cyan]. 4-bit (lowest memory, good quality) [recommended]")
    console.print("  [cyan]2[/cyan]. 8-bit (moderate memory, better quality)")
    console.print("  [cyan]3[/cyan]. fp16 (highest memory, best quality)")
    
    default = "1"
    if resources['recommended_quantization'] == 'fp16':
        default = "3"
    elif resources['recommended_quantization'] == '8bit':
        default = "2"
    
    choice = Prompt.ask("Select quantization", default=default)
    
    if choice == "3":
        return "fp16"
    elif choice == "2":
        return "8bit"
    else:
        return "4bit"


def _apply_model_selection(config: Config, model: ModelInfo, quantization: str) -> None:
    """Apply the model selection to config."""
    config.llm.model_name = model.id
    config.llm.trust_remote_code = model.requires_trust_remote
    
    if quantization == "fp16":
        config.llm.load_in_4bit = False
        config.llm.load_in_8bit = False
    elif quantization == "8bit":
        config.llm.load_in_4bit = False
        config.llm.load_in_8bit = True
    else:  # 4bit
        config.llm.load_in_4bit = True
        config.llm.load_in_8bit = False


class LiveDisplay:
    """Manages a persistent live display showing task graph and debug output."""
    
    def __init__(self, debug: bool = False):
        self.graph: Optional[TaskGraph] = None
        self.status: str = "Initializing..."
        self.debug = debug
        self.debug_logs: deque = deque(maxlen=50)  # Keep more logs for verbose debug
        self._live: Optional[Live] = None
    
    def _create_graph_panel(self) -> Panel:
        """Create the task graph panel."""
        status_styles = {
            TaskStatus.PENDING: "[dim]○[/dim]",
            TaskStatus.READY: "[yellow]◐[/yellow]",
            TaskStatus.RUNNING: "[blue bold]◑[/blue bold]",
            TaskStatus.COMPLETED: "[green]●[/green]",
            TaskStatus.FAILED: "[red]✗[/red]",
            TaskStatus.BLOCKED: "[dim]◌[/dim]"
        }
        
        if not self.graph:
            content = "[dim]No tasks yet...[/dim]"
        else:
            lines = []
            levels = self.graph.get_execution_levels()
            for level_idx, level in enumerate(levels):
                lines.append(f"[cyan]Level {level_idx + 1}:[/cyan]")
                for task_id in level:
                    task = self.graph.get_task(task_id)
                    if task:
                        icon = status_styles.get(task.status, "?")
                        desc = task.description[:45] + "..." if len(task.description) > 45 else task.description
                        deps = f" [dim]← {', '.join(task.dependencies)}[/dim]" if task.dependencies else ""
                        lines.append(f"  {icon} [bold]{task_id}[/bold]: {desc}{deps}")
            
            # Add summary
            summary = self.graph.get_summary()
            lines.append("")
            lines.append(f"[green]● {summary['completed']}[/green] | [blue]◑ {summary['running']}[/blue] | [yellow]◐ {summary['ready']}[/yellow] | [dim]○ {summary['pending']}[/dim] | [red]✗ {summary['failed']}[/red]")
            
            content = "\n".join(lines)
        
        return Panel(content, title=f"[bold cyan]📋 Task Graph[/bold cyan] - {self.status}", border_style="cyan")
    
    def _create_debug_panel(self) -> Panel:
        """Create the debug output panel showing everything."""
        if not self.debug_logs:
            content = "[dim]Waiting for agent activity...[/dim]"
        else:
            lines = []
            # Show last 20 entries to fit in panel
            for log in list(self.debug_logs)[-20:]:
                ts = log['time']
                agent = log['agent']
                event = log['event']
                msg = log['message']
                
                # Format based on event type with icons
                if event == 'start':
                    lines.append(f"[dim]{ts}[/dim] [bold green]▶ {agent}[/bold green] {msg[:100]}")
                elif event == 'end':
                    lines.append(f"[dim]{ts}[/dim] [bold]■ {agent}[/bold] {msg[:100]}")
                elif event == 'iter':
                    lines.append(f"[dim]{ts}[/dim] [blue]↻ {agent}[/blue] {msg}")
                elif event == 'llm_call':
                    lines.append(f"[dim]{ts}[/dim] [magenta]🤖 {agent}[/magenta] {msg}")
                elif event == 'llm_response':
                    # Show truncated LLM response
                    preview = msg[:150].replace('\n', '↵ ')
                    lines.append(f"[dim]{ts}[/dim] [magenta]← {agent}[/magenta] {preview}...")
                elif event == 'action':
                    lines.append(f"[dim]{ts}[/dim] [white]⚡ {agent}[/white] {msg}")
                elif event == 'think':
                    lines.append(f"[dim]{ts}[/dim] [yellow]💭 {agent}[/yellow] {msg[:120]}")
                elif event == 'shell_cmd':
                    lines.append(f"[dim]{ts}[/dim] [cyan]$ {agent}[/cyan] {msg[:100]}")
                elif event == 'shell_out':
                    preview = msg[:120].replace('\n', '↵ ')
                    lines.append(f"[dim]{ts}[/dim] [cyan]  ↳ {agent}[/cyan] {preview}")
                elif event == 'file_created':
                    lines.append(f"[dim]{ts}[/dim] [green]📄 {agent}[/green] Created: {msg}")
                elif event == 'file_modified':
                    lines.append(f"[dim]{ts}[/dim] [yellow]📝 {agent}[/yellow] Modified: {msg}")
                elif event == 'cwd':
                    lines.append(f"[dim]{ts}[/dim] [blue]📁 {agent}[/blue] {msg}")
                elif event == 'done':
                    lines.append(f"[dim]{ts}[/dim] [bold green]✓ {agent}[/bold green] {msg[:100]}")
                elif event == 'error':
                    lines.append(f"[dim]{ts}[/dim] [bold red]✗ {agent}[/bold red] {msg[:100]}")
                elif event == 'warn':
                    lines.append(f"[dim]{ts}[/dim] [yellow]⚠ {agent}[/yellow] {msg}")
                elif event == 'context':
                    lines.append(f"[dim]{ts}[/dim] [dim]📋 {agent}[/dim] {msg[:80]}")
                elif event == 'deps':
                    lines.append(f"[dim]{ts}[/dim] [dim]🔗 {agent}[/dim] {msg}")
                else:
                    lines.append(f"[dim]{ts}[/dim] [dim]{agent}[/dim] [{event}] {msg[:80]}")
            
            content = "\n".join(lines)
        
        return Panel(content, title="[bold yellow]🔍 Debug Output (everything)[/bold yellow]", border_style="yellow")
    
    def render(self) -> Group:
        """Render the full display."""
        panels = [self._create_graph_panel()]
        if self.debug:
            panels.append(self._create_debug_panel())
        return Group(*panels)
    
    def update_graph(self, graph: TaskGraph) -> None:
        """Update the task graph."""
        self.graph = graph
        self._refresh()
    
    def update_status(self, status: str) -> None:
        """Update the status message."""
        self.status = status
        self._refresh()
    
    def add_debug(self, agent_id: str, event: str, content: str) -> None:
        """Add a debug log entry."""
        self.debug_logs.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'agent': agent_id,
            'event': event,
            'message': content.replace('\n', ' ')
        })
        self._refresh()
    
    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self.render())
    
    def __enter__(self) -> 'LiveDisplay':
        self._live = Live(self.render(), console=console, refresh_per_second=4)
        self._live.__enter__()
        return self
    
    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None


def create_graph_tree(graph: TaskGraph) -> Tree:
    """Create a Rich Tree visualization of the task graph."""
    status_styles = {
        TaskStatus.PENDING: "[dim]○[/dim]",
        TaskStatus.READY: "[yellow]◐[/yellow]",
        TaskStatus.RUNNING: "[blue]◑[/blue]",
        TaskStatus.COMPLETED: "[green]●[/green]",
        TaskStatus.FAILED: "[red]✗[/red]",
        TaskStatus.BLOCKED: "[dim]◌[/dim]"
    }
    
    tree = Tree("📋 [bold]Task Graph[/bold]")
    
    levels = graph.get_execution_levels()
    for level_idx, level in enumerate(levels):
        level_branch = tree.add(f"[cyan]Level {level_idx + 1}[/cyan]")
        for task_id in level:
            task = graph.get_task(task_id)
            if task:
                icon = status_styles.get(task.status, "?")
                desc = task.description[:50] + "..." if len(task.description) > 50 else task.description
                deps = f" [dim]← {', '.join(task.dependencies)}[/dim]" if task.dependencies else ""
                level_branch.add(f"{icon} [bold]{task_id}[/bold]: {desc}{deps}")
    
    return tree


def print_banner():
    """Print the welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                    ⚡ COUNSEL OF AGENTS ⚡                      ║
║         Enterprise Multi-Agent Orchestration Platform         ║
║                                                               ║
║   • LLM-Powered Task Planning    • Automatic Verification     ║
║   • Parallel Execution           • Self-Correcting Agents     ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, border_style="cyan"))


def print_result(result: ExecutionResult, workspace: Workspace):
    """Print the execution result with verification status."""
    console.print()
    
    if result.success:
        console.print("[bold green]✅ All tasks completed successfully![/bold green]")
    else:
        console.print("[bold red]❌ Execution had failures[/bold red]")
        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
    
    # Task summary table
    summary = result.task_graph.get_summary()
    table = Table(title="Execution Summary", show_header=True, header_style="bold cyan")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    
    table.add_row("● Completed", f"[green]{summary['completed']}[/green]")
    table.add_row("✗ Failed", f"[red]{summary['failed']}[/red]")
    table.add_row("◌ Blocked", f"[dim]{summary['blocked']}[/dim]")
    
    console.print(table)
    
    # Verification summary if enabled
    verification_summary = result.get_verification_summary()
    if verification_summary.get("enabled"):
        console.print()
        v_table = Table(title="Verification Results", show_header=True, header_style="bold yellow")
        v_table.add_column("Status", style="yellow")
        v_table.add_column("Count", justify="right")
        
        v_table.add_row("✓ Passed", f"[green]{verification_summary['passed']}[/green]")
        v_table.add_row("⚠ Partial", f"[yellow]{verification_summary['partial']}[/yellow]")
        v_table.add_row("✗ Failed", f"[red]{verification_summary['failed']}[/red]")
        v_table.add_row("Pass Rate", f"[cyan]{verification_summary['pass_rate']:.0%}[/cyan]")
        
        console.print(v_table)
        
        # Show retry info if any
        if result.tasks_retried:
            console.print(f"\n[dim]Tasks retried: {', '.join(result.tasks_retried)} ({result.retry_count} total retries)[/dim]")
    
    # Performance metrics
    if result.total_duration_ms > 0:
        console.print(f"\n[dim]⏱ Total: {result.total_duration_ms/1000:.1f}s | Planning: {result.planning_duration_ms/1000:.1f}s | Execution: {result.execution_duration_ms/1000:.1f}s[/dim]")
    
    files_created = result.get_files_created()
    if files_created:
        console.print(f"\n[bold]📁 Files Created ({len(files_created)}):[/bold]")
        for f in files_created[:10]:
            console.print(f"  • {f}")
        if len(files_created) > 10:
            console.print(f"  ... and {len(files_created) - 10} more")
    
    console.print("\n[bold]Task Results:[/bold]")
    for task_id, agent_result in result.results.items():
        status = "[green]✓[/green]" if agent_result.success else "[red]✗[/red]"
        
        # Add verification status if available
        v_status = ""
        if task_id in result.verification_results:
            v_result = result.verification_results[task_id]
            v_status_value = v_result.get("status", "")
            if v_status_value == "passed":
                v_status = " [green]✓ verified[/green]"
            elif v_status_value == "partial":
                v_status = f" [yellow]⚠ partial ({v_result.get('score', 0):.0%})[/yellow]"
            elif v_status_value == "failed":
                v_status = f" [red]✗ verification failed[/red]"
        
        console.print(f"\n{status} [bold]{task_id}[/bold]{v_status}")
        
        if agent_result.result:
            result_text = str(agent_result.result)
            if len(result_text) > 300:
                result_text = result_text[:300] + "..."
            console.print(Panel(
                result_text, 
                title="Result", 
                border_style="green" if agent_result.success else "red"
            ))
        
        if agent_result.error:
            console.print(f"  [red]Error: {agent_result.error}[/red]")
        
        # Show verification issues if any
        if task_id in result.verification_results:
            v_result = result.verification_results[task_id]
            issues = v_result.get("issues", [])
            if issues:
                console.print(f"  [yellow]Issues found:[/yellow]")
                for issue in issues[:3]:  # Show first 3 issues
                    severity_icon = {"critical": "🔴", "major": "🟠", "minor": "🟡"}.get(issue.get("severity", ""), "⚪")
                    console.print(f"    {severity_icon} {issue.get('description', '')[:60]}")
                if len(issues) > 3:
                    console.print(f"    [dim]... and {len(issues) - 3} more issues[/dim]")
        
        if agent_result.shell_history:
            console.print(f"  [dim]Commands executed: {len(agent_result.shell_history)}[/dim]")


def create_status_panel(workspace: Workspace, graph: Optional[TaskGraph] = None) -> Panel:
    """Create a status panel showing workspace and graph state."""
    parts = []
    parts.append(f"[cyan]📁 Working Directory:[/cyan] {workspace.cwd}")
    
    active = workspace.get_active_agents()
    if active:
        parts.append(f"\n[yellow]🤖 Active Agents ({len(active)}):[/yellow]")
        for agent_id, task in active.items():
            parts.append(f"   • {agent_id}: {task[:40]}...")
    
    files = workspace.get_files()[-5:]
    if files:
        parts.append(f"\n[green]📄 Recent Files:[/green]")
        for f in files:
            parts.append(f"   • {f}")
    
    if graph:
        summary = graph.get_summary()
        parts.append(f"\n[blue]📊 Tasks:[/blue] {summary['completed']}/{len(graph)} completed")
    
    return Panel("\n".join(parts), title="[bold]Status[/bold]", border_style="blue")


async def run_orchestrated_task(
    user_request: str,
    config: Config,
    workspace: Workspace,
    verify: bool = False,
    max_retries: int = 2
) -> Optional[ExecutionResult]:
    """
    Run an orchestrated task with live progress display and job persistence.
    
    Args:
        user_request: Natural language task description
        config: Configuration settings
        workspace: Workspace for file operations
        verify: Enable task verification
        max_retries: Maximum retries for failed verifications
    """
    
    # Create job for persistence
    job_manager = get_job_manager()
    job = job_manager.create_job(request=user_request, workspace_cwd=workspace.cwd)
    job.model_used = config.llm.model_name
    job.status = JobStatus.PLANNING
    job_manager.save_job(job)
    
    live_display = LiveDisplay(debug=config.debug)
    
    def progress_callback(status: str, message: str):
        if status in ("task_started", "task_done", "task_failed", "complete", "verifying", "verified", "verification_failed", "retry_needed"):
            live_display.update_status(message)
            if live_display.graph:
                live_display.update_graph(live_display.graph)
        elif status == "executing":
            live_display.update_status(message)
    
    def debug_callback(agent_id: str, event: str, content: str):
        live_display.add_debug(agent_id, event, content)
    
    orchestrator = Orchestrator(
        config=config, 
        workspace=workspace, 
        progress_callback=progress_callback,
        debug_callback=debug_callback if config.debug else None
    )
    
    verify_str = " [yellow](with verification)[/yellow]" if verify else ""
    console.print(f"\n[bold cyan]📝 Task:[/bold cyan] {user_request}{verify_str}")
    console.print(f"[dim]Job ID: {job.short_id}[/dim]\n")
    console.print("[dim]Planning...[/dim]")
    
    try:
        # Plan the task
        graph = await orchestrator.plan(user_request)
        live_display.graph = graph
        live_display.update_status(f"Planned {len(graph)} tasks")
        
        # Update job with task graph
        job.task_graph = graph.to_dict()
        job.status = JobStatus.RUNNING
        job_manager.save_job(job)
        
        console.print(f"[green]✓ Created {len(graph)} tasks[/green]\n")
        
        # Execute with live display
        with live_display:
            result = await orchestrator.execute(
                verify_tasks=verify,
                max_retries=max_retries
            )
            live_display.update_status("Complete!")
            await asyncio.sleep(0.3)  # Brief pause to show final state
        
        # Update job with results
        job.results = {k: v.to_dict() for k, v in result.results.items()}
        job.files_created = result.get_files_created()
        job.total_iterations = sum(r.iterations for r in result.results.values())
        
        if result.success:
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now().isoformat()
        else:
            job.status = JobStatus.FAILED
            job.error = result.error
        
        job_manager.save_job(job)
        
        return result
    
    except asyncio.CancelledError:
        # Handle Ctrl+C during async operations
        job.status = JobStatus.CANCELLED
        job.error = "Cancelled by user"
        job_manager.save_job(job)
        console.print("\n[yellow]Task cancelled.[/yellow]")
        return None
    
    except KeyboardInterrupt:
        # Handle Ctrl+C
        job.status = JobStatus.CANCELLED
        job.error = "Cancelled by user"
        job_manager.save_job(job)
        console.print("\n[yellow]Task cancelled.[/yellow]")
        return None
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        job_manager.save_job(job)
        console.print(f"[red]Error: {e}[/red]")
        raise


async def shell_mode(config: Config, workspace: Workspace, verify_default: bool = False):
    """Interactive shell mode."""
    print_banner()
    
    # Setup readline for command history (up/down arrows)
    setup_readline()
    
    shell = get_shell()
    shell._cwd = workspace.cwd
    
    # Track verification mode
    verify_enabled = verify_default or config.verification.enabled
    
    console.print("\n[bold green]✨ Counsel Of Agents Ready[/bold green]")
    console.print("[dim]Commands:[/dim]")
    console.print("  [cyan]!<command>[/cyan]       - Run shell command directly")
    console.print("  [cyan]@status[/cyan]          - Show workspace status")
    console.print("  [cyan]@files[/cyan]           - List workspace files")
    console.print("  [cyan]@history[/cyan]         - Show agent activities")
    console.print("  [cyan]@debug[/cyan]           - Toggle debug mode")
    console.print("  [cyan]@verify[/cyan]          - Toggle task verification")
    console.print("  [cyan]@model[/cyan]           - Show current model")
    console.print("  [cyan]@jobs[/cyan]            - Show past job history")
    console.print("  [cyan]@delete <id>[/cyan]     - Delete a job by ID")
    console.print("  [cyan]@delete all[/cyan]      - Delete all jobs")
    console.print("  [cyan]help[/cyan]             - Show examples")
    console.print("  [cyan]exit[/cyan]             - Exit the shell")
    console.print("\n  [dim]Or type a task for agents to execute[/dim]")
    console.print("  [dim]Use ↑/↓ arrows to navigate command history[/dim]")
    if verify_enabled:
        console.print("  [yellow]🔍 Task verification is ENABLED[/yellow]")
    console.print()
    
    while True:
        try:
            cwd_display = os.path.basename(workspace.cwd) or workspace.cwd
            debug_indicator = "🔍 " if config.debug else ""
            
            # Use raw input() for readline support (up/down arrows work)
            prompt_text = f"{debug_indicator}{cwd_display} > "
            try:
                user_input = input(prompt_text)
            except EOFError:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                cleanup_all()
                break
            
            if not user_input.strip():
                continue
            
            # Add to history
            add_to_history(user_input)
            
            if user_input.lower() in ('exit', 'quit', 'q'):
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                cleanup_all()
                break
            
            if user_input.startswith('!'):
                cmd = user_input[1:].strip()
                if cmd:
                    result = await shell.run(cmd)
                    if result.stdout:
                        console.print(result.stdout)
                    if result.stderr:
                        console.print(f"[red]{result.stderr}[/red]")
                    if not result.success:
                        console.print(f"[dim]Exit code: {result.return_code}[/dim]")
                    if cmd.strip().startswith('cd '):
                        workspace.set_cwd(shell.cwd)
                continue
            
            if user_input.startswith('@'):
                meta_cmd = user_input[1:].strip().lower()
                
                if meta_cmd == 'status':
                    console.print(create_status_panel(workspace))
                elif meta_cmd == 'files':
                    files = workspace.get_files()
                    if files:
                        console.print("[bold]Files in workspace:[/bold]")
                        for f in files:
                            console.print(f"  📄 {f}")
                    else:
                        console.print("[dim]No files registered yet[/dim]")
                elif meta_cmd == 'dirs':
                    dirs = workspace.get_directories()
                    console.print("[bold]Directories:[/bold]")
                    for d in dirs:
                        console.print(f"  📁 {d}")
                elif meta_cmd == 'history':
                    activities = workspace.get_recent_activities(20)
                    if activities:
                        console.print("[bold]Recent Activities:[/bold]")
                        for act in activities:
                            console.print(f"  [{act.agent_id}] {act.action}: {act.details[:50]}")
                    else:
                        console.print("[dim]No activities yet[/dim]")
                elif meta_cmd == 'clear':
                    console.clear()
                elif meta_cmd == 'context':
                    console.print(workspace.get_context_for_agent())
                elif meta_cmd == 'debug':
                    config.debug = not config.debug
                    if config.debug:
                        console.print("[yellow]🔍 Debug mode ON - will show agent thinking, shell commands[/yellow]")
                    else:
                        console.print("[dim]Debug mode OFF[/dim]")
                elif meta_cmd == 'verify':
                    verify_enabled = not verify_enabled
                    if verify_enabled:
                        console.print("[yellow]✓ Task verification ENABLED - tasks will be verified after completion[/yellow]")
                    else:
                        console.print("[dim]Task verification disabled[/dim]")
                elif meta_cmd == 'model':
                    model_info = get_model_by_id(config.llm.model_name)
                    quant = "4bit" if config.llm.load_in_4bit else ("8bit" if config.llm.load_in_8bit else "fp16")
                    console.print(f"[bold]Current model:[/bold] {config.llm.model_name}")
                    if model_info:
                        console.print(f"[dim]Family: {model_info.family} | Size: {model_info.size} | Quantization: {quant}[/dim]")
                    console.print("\n[dim]To change model, restart with --select-model[/dim]")
                elif meta_cmd == 'jobs':
                    job_manager = get_job_manager()
                    jobs = job_manager.list_jobs(limit=10)
                    if jobs:
                        console.print("[bold]Recent Jobs:[/bold]")
                        for job in jobs:
                            status_icon = {"completed": "✓", "failed": "✗", "running": "●", "cancelled": "○"}.get(job.status.value, "○")
                            status_color = {"completed": "green", "failed": "red", "running": "blue", "cancelled": "dim"}.get(job.status.value, "dim")
                            console.print(f"  [{status_color}]{status_icon}[/{status_color}] [cyan]{job.short_id}[/cyan] {job.name[:50]}")
                        console.print(f"\n[dim]Use @delete <id> to remove a job[/dim]")
                    else:
                        console.print("[dim]No job history yet[/dim]")
                elif meta_cmd.startswith('delete'):
                    job_manager = get_job_manager()
                    parts = meta_cmd.split(maxsplit=1)
                    if len(parts) < 2:
                        console.print("[yellow]Usage: @delete <job_id> or @delete all[/yellow]")
                    elif parts[1] == 'all':
                        jobs = job_manager.list_jobs(limit=1000)
                        if not jobs:
                            console.print("[dim]No jobs to delete[/dim]")
                        else:
                            confirm = Confirm.ask(f"Delete all {len(jobs)} jobs?", default=False)
                            if confirm:
                                deleted = 0
                                for job in jobs:
                                    if job_manager.delete_job(job.id):
                                        deleted += 1
                                console.print(f"[green]✓ Deleted {deleted} jobs[/green]")
                            else:
                                console.print("[dim]Cancelled[/dim]")
                    else:
                        job_id = parts[1].strip()
                        job = job_manager.load_job(job_id) or job_manager.get_job_by_short_id(job_id)
                        if job:
                            if job_manager.delete_job(job.id):
                                console.print(f"[green]✓ Deleted job {job.short_id}[/green]")
                            else:
                                console.print(f"[red]Failed to delete job {job_id}[/red]")
                        else:
                            console.print(f"[red]Job not found: {job_id}[/red]")
                elif meta_cmd.startswith('job '):
                    # Show details of a specific job
                    job_manager = get_job_manager()
                    job_id = meta_cmd[4:].strip()
                    job = job_manager.load_job(job_id) or job_manager.get_job_by_short_id(job_id)
                    if job:
                        status_color = {"completed": "green", "failed": "red", "running": "blue", "cancelled": "dim"}.get(job.status.value, "cyan")
                        console.print(Panel(
                            f"[bold]{job.name}[/bold]\n\n"
                            f"[cyan]ID:[/cyan] {job.id}\n"
                            f"[cyan]Status:[/cyan] [{status_color}]{job.status.value}[/{status_color}]\n"
                            f"[cyan]Created:[/cyan] {job.created_at[:19]}\n"
                            f"[cyan]Model:[/cyan] {job.model_used or 'N/A'}\n"
                            f"[cyan]Files created:[/cyan] {len(job.files_created)}",
                            title=f"[bold]Job {job.short_id}[/bold]",
                            border_style=status_color
                        ))
                        if job.error:
                            console.print(f"[red]Error: {job.error}[/red]")
                    else:
                        console.print(f"[red]Job not found: {job_id}[/red]")
                else:
                    console.print(f"[yellow]Unknown command: @{meta_cmd}[/yellow]")
                    console.print("[dim]Available: @status, @files, @jobs, @delete, @job <id>, @debug, @model[/dim]")
                continue
            
            if user_input.lower() == 'help':
                console.print("\n[bold]Example Tasks:[/bold]")
                console.print("  • [cyan]Create a Python project with a CLI calculator[/cyan]")
                console.print("  • [cyan]Set up an Express.js server with REST endpoints[/cyan]")
                console.print("  • [cyan]Create a bash script that monitors disk usage[/cyan]")
                console.print("\n[bold]Tips:[/bold]")
                console.print("  • Use [cyan]@verify[/cyan] to toggle automatic task verification")
                console.print("  • Use [cyan]@debug[/cyan] to see agent reasoning and shell commands")
                console.print()
                continue
            
            result = await run_orchestrated_task(
                user_input, config, workspace,
                verify=verify_enabled,
                max_retries=config.verification.max_retries
            )
            if result is not None:
                print_result(result, workspace)
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except asyncio.CancelledError:
            console.print("\n[yellow]Task cancelled.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]")
            if config.verbose:
                console.print_exception()
            console.print()
    
    # Final cleanup (in case we exit the loop normally)
    cleanup_all()


async def single_task_mode(
    user_request: str,
    config: Config,
    workspace: Workspace,
    verify: bool = False
) -> bool:
    """Run a single task."""
    print_banner()
    try:
        result = await run_orchestrated_task(
            user_request, config, workspace,
            verify=verify,
            max_retries=config.verification.max_retries
        )
        if result is not None:
            print_result(result, workspace)
            return result.success
        return False  # Cancelled
    except (KeyboardInterrupt, asyncio.CancelledError):
        console.print("\n[yellow]Task cancelled.[/yellow]")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="counsel",
        description="Counsel Of Agents - Enterprise Multi-Agent Orchestration Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    counsel                                    # Interactive mode
    counsel "Create a Python hello world"     # Single task
    counsel --verify "Create a REST API"      # With task verification
    counsel -w ./myproject "Set up Express"   # With workspace
    counsel --debug "Create a todo app"       # Debug mode
    counsel --select-model                    # Change model selection
    counsel --list-models                     # List all available models

Environment Variables:
    COUNSEL_MODEL       HuggingFace model ID
    COUNSEL_VERIFY      Enable verification by default
    COUNSEL_DEBUG       Enable debug mode
    COUNSEL_LOG_FILE    Path to log file
        """
    )
    
    parser.add_argument('task', nargs='?', help='Task to execute')
    parser.add_argument('--select-model', action='store_true', help='Show model selection screen')
    parser.add_argument('--list-models', action='store_true', help='List all available models and exit')
    parser.add_argument('--reset-model', action='store_true', help='Clear saved model selection')
    parser.add_argument('-i', '--interactive', action='store_true', help='Interactive shell mode')
    parser.add_argument('-w', '--workspace', type=str, default=None, help='Working directory')
    parser.add_argument('-m', '--model', type=str, default=None, help='HuggingFace model')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    parser.add_argument('-p', '--parallel', type=int, default=5, help='Max parallel agents (default: 5)')
    parser.add_argument('--no-quantize', action='store_true', help='Disable 4-bit quantization')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode - shows agent thinking and shell commands')
    parser.add_argument('--verify', action='store_true', help='Enable task verification after completion')
    parser.add_argument('--max-retries', type=int, default=2, help='Maximum retries for failed verifications')
    parser.add_argument('--continue-on-failure', action='store_true', help='Continue executing other tasks if one fails')
    
    args = parser.parse_args()
    
    # Handle --list-models
    if args.list_models:
        _show_model_list()
        sys.exit(0)
    
    # Handle --reset-model
    if args.reset_model:
        clear_model_selection()
        console.print("[green]✓ Model selection cleared. Will prompt on next run.[/green]")
        sys.exit(0)
    
    config = Config.from_env()
    
    # Handle model selection
    if args.select_model:
        # Force model selection screen
        if not show_model_selection(config):
            sys.exit(0)
    elif args.model:
        # Model specified via command line
        config.llm.model_name = args.model
    else:
        # Check for saved model or prompt for selection
        saved = load_model_selection()
        if saved:
            model = get_model_by_id(saved['model_id'])
            if model:
                _apply_model_selection(config, model, saved.get('quantization', '4bit'))
            else:
                # Custom model that's not in catalog
                config.llm.model_name = saved['model_id']
                quant = saved.get('quantization', '4bit')
                config.llm.load_in_4bit = (quant == '4bit')
                config.llm.load_in_8bit = (quant == '8bit')
        else:
            # No saved model - show selection on first run
            console.print("[yellow]No model selected yet. Let's choose one![/yellow]\n")
            if not show_model_selection(config):
                sys.exit(0)
    
    if args.device:
        config.llm.device = args.device
    if args.parallel:
        config.execution.max_parallel_agents = args.parallel
    if args.no_quantize:
        config.llm.load_in_4bit = False
        config.llm.load_in_8bit = False
    if args.verbose:
        config.verbose = True
    if args.debug:
        config.debug = True
    if args.verify:
        config.verification.enabled = True
    if args.max_retries:
        config.verification.max_retries = args.max_retries
    if args.continue_on_failure:
        config.execution.continue_on_failure = True
    if args.workspace:
        config.shell.working_directory = os.path.abspath(args.workspace)
    
    # Configure logging based on settings
    log_level = LogLevel.DEBUG if config.debug else LogLevel.INFO
    configure_logging(
        level=log_level,
        log_file=config.logging.file,
        json_output=config.logging.json_format
    )
    
    set_config(config)
    
    workspace_dir = config.shell.working_directory or os.getcwd()
    workspace = get_workspace(workspace_dir)
    
    if args.interactive or not args.task:
        asyncio.run(shell_mode(config, workspace, verify_default=args.verify))
    else:
        success = asyncio.run(single_task_mode(
            args.task, config, workspace, verify=args.verify
        ))
        sys.exit(0 if success else 1)


def _show_model_list():
    """Show a formatted list of all available models."""
    console.print(Panel(
        "[bold cyan]🤖 Available Models[/bold cyan]\n\n"
        "All models are downloaded from HuggingFace on first use.",
        border_style="cyan"
    ))
    
    resources = estimate_system_resources()
    
    # Group by family
    families = {}
    for model in MODEL_CATALOG:
        if model.family not in families:
            families[model.family] = []
        families[model.family].append(model)
    
    for family, models in families.items():
        console.print(f"\n[bold cyan]═══ {family} ═══[/bold cyan]")
        
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Model ID", style="cyan")
        table.add_column("Size")
        table.add_column("VRAM (4bit)")
        table.add_column("RAM (4bit)")
        table.add_column("Context")
        table.add_column("Notes")
        
        for model in models:
            notes = []
            if "recommended" in model.recommended_for:
                notes.append("⭐ Recommended")
            if "coding" in model.recommended_for:
                notes.append("💻 Code")
            
            table.add_row(
                model.id,
                model.size,
                model.vram_4bit,
                model.ram_4bit,
                f"{model.context_length // 1000}k",
                " ".join(notes) if notes else model.description[:30]
            )
        
        console.print(table)
    
    console.print("\n[dim]Use --select-model to interactively choose a model[/dim]")
    console.print("[dim]Use -m MODEL_ID to use a specific model[/dim]")


if __name__ == "__main__":
    main()
