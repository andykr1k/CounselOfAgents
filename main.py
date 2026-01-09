"""CLI interface for the Counsel of Agents system."""

import asyncio
import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.table import Table
from agent_registry import AgentRegistry
from orchestrator import Orchestrator
from agent import (
    CodingAgent,
    WritingAgent,
    EvalAgent,
    ReadingAgent,
    ResearchAgent,
    MathAgent,
    TranslationAgent,
    AnalysisAgent,
    FileSystemAgent
)


console = Console()


def create_all_agents(show_progress=True):
    """Create and register all specialized agents."""
    registry = AgentRegistry()
    agents_created = []
    errors = []
    
    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading agents...", total=None)
            
            # File System Agent (uses model to generate commands)
            try:
                progress.update(task, description="Loading File System Agent...")
                registry.register(FileSystemAgent(model_preset="tiny"))
                agents_created.append(("File System Agent", "✓"))
            except Exception as e:
                errors.append(("File System Agent", str(e)))
            
            # Hugging Face agents
            hf_agents = [
                ("Coding Agent", CodingAgent, "code", "StarCoder"),
                ("Writing Agent", WritingAgent, "writing", "GPT-2"),
                ("Evaluation Agent", EvalAgent, "eval", "GPT-2"),
                ("Reading Agent", ReadingAgent, "reading", "BART"),
                ("Research Agent", ResearchAgent, "research", "GPT-2"),
                ("Math Agent", MathAgent, "math", "GPT-2"),
                ("Analysis Agent", AnalysisAgent, "analysis", "GPT-2"),
            ]
            
            for name, agent_class, preset, model_name in hf_agents:
                try:
                    progress.update(task, description=f"Loading {name} ({model_name})...")
                    registry.register(agent_class(model_preset=preset))
                    agents_created.append((name, "✓"))
                except ImportError:
                    errors.append((name, "transformers/torch not installed"))
                except Exception as e:
                    errors.append((name, str(e)[:50]))
            
            # Translation agent (optional)
            try:
                progress.update(task, description="Loading Translation Agent...")
                registry.register(TranslationAgent(model_preset="translation"))
                agents_created.append(("Translation Agent", "✓"))
            except Exception:
                pass  # Translation is optional
    
    else:
        # Quick registration without progress
        registry.register(FileSystemAgent(model_preset="tiny"))
        try:
            registry.register(CodingAgent(model_preset="code"))
            registry.register(WritingAgent(model_preset="writing"))
            registry.register(EvalAgent(model_preset="eval"))
            registry.register(ReadingAgent(model_preset="reading"))
            registry.register(ResearchAgent(model_preset="research"))
            registry.register(MathAgent(model_preset="math"))
            registry.register(AnalysisAgent(model_preset="analysis"))
            try:
                registry.register(TranslationAgent(model_preset="translation"))
            except Exception:
                pass
        except ImportError:
            console.print("[red]Error: transformers/torch not installed[/red]")
            console.print("Install with: [cyan]pip install transformers torch accelerate[/cyan]")
            sys.exit(1)
    
    return registry, agents_created, errors


def print_agent_status(agents_created, errors):
    """Print status of agent creation."""
    if agents_created:
        table = Table(title="Available Agents", show_header=True, header_style="bold cyan")
        table.add_column("Agent", style="cyan")
        table.add_column("Status", style="green")
        
        for name, status in agents_created:
            table.add_row(name, status)
        
        console.print(table)
    
    if errors:
        console.print("\n[yellow]⚠ Some agents could not be loaded:[/yellow]")
        for name, error in errors:
            console.print(f"  • {name}: {error}")


async def process_task(prompt: str, max_parallel: int = 3, registry=None, orchestrator=None):
    """Process a task using the orchestrator with detailed progress updates."""
    if registry is None:
        registry, agents_created, errors = create_all_agents(show_progress=False)
        if not registry.get_all_agents():
            console.print("[red]❌ No agents available![/red]")
            return None, None
    else:
        agents_created, errors = [], []
    
    # Create progress tracker
    progress_tracker = {}
    progress_task = None
    
    def progress_callback(status: str, message: str):
        """Callback for progress updates."""
        nonlocal progress_task
        if progress_task is None:
            return
        
        # Update progress with status and message
        status_icons = {
            "classifying": "🔍",
            "classified": "✓",
            "reasoning": "💭",
            "building": "📋",
            "executing": "⚙️",
            "success": "✓",
            "error": "✗",
            "aggregating": "📊",
            "complete": "✅"
        }
        icon = status_icons.get(status, "•")
        progress_tracker[progress_task].update(
            description=f"{icon} {message}"
        )
    
    if orchestrator is None:
        orchestrator = Orchestrator(registry, progress_callback=progress_callback)
    
    console.print(f"\n[bold cyan]📝 Task:[/bold cyan] {prompt}\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress_task = progress.add_task("Starting...", total=None)
        progress_tracker[progress_task] = progress
        
        result = await orchestrator.process_prompt(prompt, max_parallel=max_parallel)
        progress.update(progress_task, completed=True)
    
    # Display results
    console.print("\n" + "=" * 70)
    
    if result['success']:
        console.print("[bold green]✅ Task Completed Successfully[/bold green]\n")
        
        # Show which agents were used
        if result.get('results', {}).get('task_results'):
            console.print("[bold]🤖 Agents Used:[/bold]")
            for task_id, task_result in result['results']['task_results'].items():
                agent_id = task_result.get('agent_id', 'unknown')
                agent_name = agent_id.replace('_', ' ').title()
                status = "[green]✓[/green]" if task_result.get('success') else "[red]✗[/red]"
                console.print(f"  {status} {agent_name}")
        
        # Show output
        output = result['results'].get('final_output', 'No output')
        console.print("\n[bold]📄 Result:[/bold]")
        # Truncate very long outputs
        display_output = output[:2000] + ("..." if len(output) > 2000 else "")
        console.print(Panel(display_output, border_style="green", title="Output"))
        
        # Show summary
        summary = result['results'].get('summary', {})
        if summary:
            completed = summary.get('completed', 0)
            total = summary.get('total_tasks', 0)
            console.print(f"\n[dim]📊 Summary: {completed}/{total} task(s) completed[/dim]")
    else:
        console.print("[bold red]❌ Task Failed[/bold red]\n")
        console.print(f"[red]{result.get('error', 'Unknown error')}[/red]")
    
    return result


async def interactive_mode():
    """Run in interactive mode with welcome message."""
    # Show welcome banner
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║         🤖 Counsel of Agents - Multi-Agent System        ║
    ╚══════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, border_style="cyan"))
    
    # Load agents
    console.print("\n[dim]Loading specialized agents...[/dim]")
    registry, agents_created, errors = create_all_agents(show_progress=True)
    
    if not registry.get_all_agents():
        console.print("[red]❌ No agents available![/red]")
        console.print("Install dependencies: [cyan]pip install transformers torch accelerate[/cyan]")
        return
    
    print_agent_status(agents_created, errors)
    
    orchestrator = Orchestrator(registry)
    
    # Welcome message
    console.print("\n[bold green]✨ Ready to help![/bold green]")
    console.print("[dim]I can help with coding, writing, analysis, file operations, and more.[/dim]")
    console.print("[dim]Type 'exit' or 'quit' to exit, 'help' for examples[/dim]\n")
    
    while True:
        try:
            # Ask what they need help with
            prompt = console.input("[bold cyan]💬 What can I help you with?[/bold cyan] ")
            
            if prompt.lower() in ['exit', 'quit', 'q']:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            
            if prompt.lower() == 'help':
                console.print("\n[bold]Example tasks I can help with:[/bold]")
                console.print("  • [cyan]Generate a Python function to sort a list[/cyan]")
                console.print("  • [cyan]Write a summary about machine learning[/cyan]")
                console.print("  • [cyan]Read file README.md[/cyan]")
                console.print("  • [cyan]Grep 'def' in main.py[/cyan]")
                console.print("  • [cyan]Evaluate the pros and cons of Python[/cyan]")
                console.print("  • [cyan]Research information about neural networks[/cyan]")
                console.print()
                continue
            
            if not prompt.strip():
                continue
            
            # Process the task with detailed updates
            await process_task(prompt, max_parallel=3, registry=registry, orchestrator=orchestrator)
            
            # Show results
            console.print()
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]")
            console.print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Counsel of Agents - Multi-agent orchestration system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python main.py "Generate a Python function to calculate fibonacci"
        python main.py "Read file README.md"
        python main.py "Grep 'def' in main.py"
        python main.py -i  # Interactive mode
        """
    )
    
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Task prompt to process'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=3,
        help='Maximum parallel tasks (default: 3)'
    )
    
    parser.add_argument(
        '--list-agents',
        action='store_true',
        help='List all available agents and exit'
    )
    
    args = parser.parse_args()
    
    # Show banner (only for non-interactive single commands)
    if args.prompt and not args.interactive:
        banner = """
# Counsel of Agents

Multi-agent orchestration system with specialized agents
        """
        console.print(Panel(Markdown(banner), border_style="cyan"))
    
    # List agents
    if args.list_agents:
        registry, agents_created, errors = create_all_agents(show_progress=False)
        print_agent_status(agents_created, errors)
        return
    
    # Interactive mode (default if no prompt)
    if args.interactive or not args.prompt:
        asyncio.run(interactive_mode())
    else:
        # Process single prompt
        asyncio.run(process_task(args.prompt, max_parallel=args.max_parallel))


if __name__ == "__main__":
    main()
