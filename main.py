"""CLI interface for the Agent Orchestration System."""

import asyncio
import argparse
import sys
import os
import signal
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table
from rich.tree import Tree
from rich.layout import Layout
from rich.text import Text
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm

from config import Config, get_config, set_config
from orchestrator import Orchestrator, ExecutionResult
from task_graph import TaskGraph, TaskStatus
from workspace import Workspace, get_workspace
from shell import Shell, get_shell


console = Console()


def create_status_panel(workspace: Workspace, graph: Optional[TaskGraph] = None) -> Panel:
    """Create a status panel showing workspace and graph state."""
    parts = []
    
    # Workspace info
    parts.append(f"[cyan]📁 Working Directory:[/cyan] {workspace.cwd}")
    
    # Active agents
    active = workspace.get_active_agents()
    if active:
        parts.append(f"\n[yellow]🤖 Active Agents ({len(active)}):[/yellow]")
        for agent_id, task in active.items():
            parts.append(f"   • {agent_id}: {task[:40]}...")
    
    # Recent files
    files = workspace.get_files()[-5:]
    if files:
        parts.append(f"\n[green]📄 Recent Files:[/green]")
        for f in files:
            parts.append(f"   • {f}")
    
    # Task graph status
    if graph:
        summary = graph.get_summary()
        parts.append(f"\n[blue]📊 Tasks:[/blue] {summary['completed']}/{len(graph)} completed")
        if summary['running'] > 0:
            parts.append(f"   [yellow]Running: {summary['running']}[/yellow]")
        if summary['failed'] > 0:
            parts.append(f"   [red]Failed: {summary['failed']}[/red]")
    
    return Panel("\n".join(parts), title="[bold]Status[/bold]", border_style="blue")


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
║           🤖 Agent Orchestration System                       ║
║      Multi-agent task execution with shared workspace         ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, border_style="cyan"))


def print_result(result: ExecutionResult, workspace: Workspace):
    """Print the execution result."""
    console.print()
    
    if result.success:
        console.print("[bold green]✅ All tasks completed successfully![/bold green]")
    else:
        console.print("[bold red]❌ Execution had failures[/bold red]")
        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
    
    # Summary table
    summary = result.task_graph.get_summary()
    table = Table(title="Execution Summary", show_header=True, header_style="bold cyan")
    table.add_column("Status", style="cyan")
    table.add_column("Count", justify="right")
    
    table.add_row("● Completed", f"[green]{summary['completed']}[/green]")
    table.add_row("✗ Failed", f"[red]{summary['failed']}[/red]")
    table.add_row("◌ Blocked", f"[dim]{summary['blocked']}[/dim]")
    
    console.print(table)
    
    # Files created
    files_created = result.get_files_created()
    if files_created:
        console.print(f"\n[bold]📁 Files Created ({len(files_created)}):[/bold]")
        for f in files_created[:10]:
            console.print(f"  • {f}")
        if len(files_created) > 10:
            console.print(f"  ... and {len(files_created) - 10} more")
    
    # Individual task results
    console.print("\n[bold]Task Results:[/bold]")
    for task_id, agent_result in result.results.items():
        status = "[green]✓[/green]" if agent_result.success else "[red]✗[/red]"
        console.print(f"\n{status} [bold]{task_id}[/bold]")
        
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
        
        if agent_result.shell_history:
            console.print(f"  [dim]Commands executed: {len(agent_result.shell_history)}[/dim]")


async def run_orchestrated_task(user_request: str, config: Config, workspace: Workspace) -> ExecutionResult:
    """Run an orchestrated task with live progress display."""
    
    current_graph: Optional[TaskGraph] = None
    
    def progress_callback(status: str, message: str):
        nonlocal current_graph
        
        status_icons = {
            "planning": "🔍",
            "planned": "📋",
            "graph": "🌳",
            "executing": "⚙️",
            "task_done": "✓",
            "task_failed": "✗",
            "complete": "✅"
        }
        
        icon = status_icons.get(status, "•")
        
        if status == "graph":
            console.print()
            console.print(Panel(message, title="[bold cyan]Task Graph[/bold cyan]", border_style="cyan"))
            console.print()
        else:
            console.print(f"{icon} {message}")
    
    orchestrator = Orchestrator(
        config=config, 
        workspace=workspace,
        progress_callback=progress_callback
    )
    
    console.print(f"\n[bold cyan]📝 Task:[/bold cyan] {user_request}\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Planning...", total=None)
        
        try:
            graph = await orchestrator.plan(user_request)
            current_graph = graph
            progress.update(task, description=f"Planned {len(graph)} tasks")
            
            console.print()
            tree = create_graph_tree(graph)
            console.print(tree)
            console.print()
            
            progress.update(task, description="Executing tasks...")
            result = await orchestrator.execute()
            
            progress.update(task, description="Complete!", completed=True)
            return result
            
        except Exception as e:
            progress.update(task, description=f"Error: {e}")
            raise


async def shell_mode(config: Config, workspace: Workspace):
    """
    Interactive shell mode.
    
    Allows direct shell commands alongside orchestrated tasks.
    """
    print_banner()
    
    shell = get_shell()
    shell._cwd = workspace.cwd
    
    console.print("\n[bold green]✨ Agent Shell Ready[/bold green]")
    console.print("[dim]Commands:[/dim]")
    console.print("  [cyan]!<command>[/cyan]     - Run shell command directly (e.g., !ls -la)")
    console.print("  [cyan]@status[/cyan]        - Show workspace status")
    console.print("  [cyan]@files[/cyan]         - List files in workspace")
    console.print("  [cyan]@history[/cyan]       - Show recent agent activities")
    console.print("  [cyan]@clear[/cyan]         - Clear the screen")
    console.print("  [cyan]help[/cyan]           - Show example tasks")
    console.print("  [cyan]exit/quit[/cyan]      - Exit the shell")
    console.print("\n  [dim]Or type a task for agents to execute[/dim]\n")
    
    while True:
        try:
            # Show current directory in prompt
            cwd_display = os.path.basename(workspace.cwd) or workspace.cwd
            prompt_text = f"[bold cyan]{cwd_display}[/bold cyan] [bold]>[/bold] "
            user_input = console.input(prompt_text)
            
            if not user_input.strip():
                continue
            
            # Exit commands
            if user_input.lower() in ('exit', 'quit', 'q'):
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            
            # Direct shell command (prefixed with !)
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
                    
                    # Update workspace cwd if cd command
                    if cmd.strip().startswith('cd '):
                        workspace.set_cwd(shell.cwd)
                continue
            
            # Meta commands (prefixed with @)
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
                
                else:
                    console.print(f"[yellow]Unknown command: @{meta_cmd}[/yellow]")
                
                continue
            
            # Help
            if user_input.lower() == 'help':
                console.print("\n[bold]Example Tasks:[/bold]")
                console.print("  • [cyan]Create a Python project with a CLI calculator[/cyan]")
                console.print("  • [cyan]Set up an Express.js server with REST endpoints[/cyan]")
                console.print("  • [cyan]Create a React component with state management[/cyan]")
                console.print("  • [cyan]Build a Python Flask API with SQLite database[/cyan]")
                console.print("  • [cyan]Create a bash script that monitors disk usage[/cyan]")
                console.print("\n[bold]Shell Commands:[/bold]")
                console.print("  • [cyan]!ls -la[/cyan]        - List files")
                console.print("  • [cyan]!cat file.py[/cyan]  - View file contents")
                console.print("  • [cyan]!mkdir mydir[/cyan]  - Create directory")
                console.print("  • [cyan]!pwd[/cyan]          - Print working directory")
                console.print()
                continue
            
            # Run as orchestrated task
            result = await run_orchestrated_task(user_input, config, workspace)
            print_result(result, workspace)
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]")
            if config.verbose:
                console.print_exception()
            console.print()


async def single_task_mode(user_request: str, config: Config, workspace: Workspace) -> bool:
    """Run a single task."""
    print_banner()
    
    result = await run_orchestrated_task(user_request, config, workspace)
    print_result(result, workspace)
    
    return result.success


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agent Orchestration System - Multi-agent task execution with shared workspace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive shell mode
    python main.py
    python main.py -i
    
    # Single task
    python main.py "Create a Python hello world project"
    
    # With specific working directory
    python main.py -w ./my-project "Set up Express.js server"
    
    # With specific model
    python main.py -m "Qwen/Qwen2.5-Coder-7B-Instruct" "Create a REST API"

Shell Commands (in interactive mode):
    !ls -la          Run shell command directly
    @status          Show workspace status
    @files           List workspace files
    @history         Show agent activity history
    help             Show examples
    exit             Exit the shell
        """
    )
    
    parser.add_argument(
        'task',
        nargs='?',
        help='Task to execute (omit for interactive mode)'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive shell mode'
    )
    
    parser.add_argument(
        '-w', '--workspace',
        type=str,
        default=None,
        help='Working directory for agents (default: current directory)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=None,
        help='HuggingFace model (default: Qwen/Qwen2.5-7B-Instruct)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'mps', 'cpu'],
        default='auto',
        help='Device to run model on (default: auto)'
    )
    
    parser.add_argument(
        '-p', '--parallel',
        type=int,
        default=3,
        help='Maximum parallel agents (default: 3)'
    )
    
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Disable 4-bit quantization'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--continue-on-failure',
        action='store_true',
        help='Continue executing even if tasks fail'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = Config.from_env()
    
    if args.model:
        config.llm.model_name = args.model
    if args.device:
        config.llm.device = args.device
    if args.parallel:
        config.execution.max_parallel_agents = args.parallel
    if args.no_quantize:
        config.llm.load_in_4bit = False
    if args.verbose:
        config.verbose = True
    if args.continue_on_failure:
        config.execution.continue_on_failure = True
    if args.workspace:
        config.shell.working_directory = os.path.abspath(args.workspace)
    
    set_config(config)
    
    # Initialize workspace
    workspace_dir = config.shell.working_directory or os.getcwd()
    workspace = get_workspace(workspace_dir)
    
    # Run
    if args.interactive or not args.task:
        asyncio.run(shell_mode(config, workspace))
    else:
        success = asyncio.run(single_task_mode(args.task, config, workspace))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
