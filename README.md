# Counsel Of Agents

**Enterprise Multi-Agent Orchestration Platform**

An intelligent orchestration system that breaks down complex tasks into dependency graphs and executes them using self-correcting AI agents with built-in verification.

[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)](https://github.com/your-org/counsel)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

## Key Features

### Core Capabilities
- 🤖 **Intelligent Task Decomposition** - LLM-powered breakdown of complex tasks into executable subtasks
- 📊 **DAG-Based Execution** - Parallel task execution respecting dependencies
- ✅ **Task Verification** - Automatic verification that tasks were completed correctly (ON by default)
- 🔄 **Self-Correcting Agents** - Agents retry with specific remediation when verification fails
- 🆘 **Smart Help System** - Agents can ask for help (`<help>`) and get supervisor guidance (ON by default)
- 🧑‍💼 **Automatic Intervention** - When agents get stuck, supervisor automatically provides guidance
- 📁 **Direct File Operations** - Agents have dedicated tools to read, write, edit, and list files

### Enterprise Features
- 📋 **Job Persistence** - All jobs saved to `~/.counsel/jobs/` for history and recovery
- 📝 **Professional Logging** - Structured logging with telemetry, metrics, and audit trails
- ⚙️ **Configuration Validation** - Comprehensive config validation with environment variable support
- 🔍 **Debug Mode** - See everything agents do: LLM calls, shell commands, thinking
- 🛡️ **Process Cleanup** - Proper cleanup of all subprocesses on exit/interrupt

### Interactive Features
- 💻 **Interactive Shell** - Full control with command history (↑/↓ arrows)
- 🎯 **Model Selection** - Interactive model picker with RAM/VRAM requirements
- 🌳 **File Tree Context** - Agents see visual directory structure

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run (will prompt for model selection on first run)
python main.py
```

### Task Verification (ON by default)

```bash
# Verification is now enabled by default!
python main.py "Create a REST API with user authentication"

# Toggle verification off/on in interactive mode
projects > @verify
✓ Task verification DISABLED

projects > @verify  
✓ Task verification ENABLED
```

## Usage

### Interactive Shell

```bash
python main.py
```

```
⚡ Counsel Of Agents ⚡
Enterprise Multi-Agent Orchestration Platform

✨ Counsel Of Agents Ready

Commands:
  !<command>       - Run shell command directly
  @status          - Show workspace status
  @files           - List workspace files
  @verify          - Toggle task verification
  @debug           - Toggle debug mode
  @model           - Show current model
  @jobs            - Show past job history
  help             - Show examples
  exit             - Exit the shell

  Use ↑/↓ arrows to navigate command history

projects > Create a Python calculator CLI with verification

📝 Task: Create a Python calculator CLI (with verification)
Job ID: a1b2c3d4

Planning...
✓ Created 4 tasks

╭─────────────────── 📋 Task Graph ───────────────────╮
│ Level 1:                                            │
│   ◑ task_1: Create project with venv and deps       │
│ Level 2:                                            │
│   ○ task_2: Create calculator.py with functions     │
│ Level 3:                                            │
│   ○ task_3: Create main.py CLI entry point          │
│ Level 4:                                            │
│   ○ task_4: Test the calculator                     │
╰─────────────────────────────────────────────────────╯
```

### Task Verification

When verification is enabled, each completed task is analyzed to ensure it meets requirements:

```
Verification Results
┌─────────┬───────┐
│ Status  │ Count │
├─────────┼───────┤
│ ✓ Passed│ 3     │
│ ⚠ Partial│ 1    │
│ ✗ Failed│ 0     │
│ Pass Rate│ 75%  │
└─────────┴───────┘

✓ task_1 ✓ verified
✓ task_2 ✓ verified
✓ task_3 ⚠ partial (80%)
  Issues found:
    🟠 Documentation missing - README not created
✓ task_4 ✓ verified
```

### Command Line Options

```bash
python main.py --help

Options:
  task                      Task to execute (optional)
  --verify                  Enable task verification
  --max-retries N           Max retries for failed verifications (default: 2)
  --select-model            Show model selection screen
  --list-models             List all available models
  --reset-model             Clear saved model selection
  -i, --interactive         Force interactive shell mode
  -w, --workspace DIR       Working directory
  -m, --model MODEL         HuggingFace model ID
  --device {auto,cuda,mps,cpu}
  -p, --parallel N          Max parallel agents (default: 5)
  --no-quantize             Disable 4-bit quantization
  -v, --verbose             Verbose output
  -d, --debug               Debug mode (ON by default)
  --continue-on-failure     Continue if tasks fail

Environment Variables:
  COUNSEL_MODEL             HuggingFace model ID
  COUNSEL_DEVICE            Device to use (auto, cuda, mps, cpu)
  COUNSEL_VERIFY            Enable verification by default (1/true/yes)
  COUNSEL_DEBUG             Enable debug mode (1/true/yes)
  COUNSEL_MAX_PARALLEL      Maximum parallel agents
  COUNSEL_LOG_FILE          Path to log file
  COUNSEL_LOG_LEVEL         Logging level (DEBUG, INFO, WARNING, ERROR)
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ Task Planner │   │  Task Graph  │   │ Execution Engine │  │
│  │  (LLM-based) │   │    (DAG)     │   │  (Agent Spawner) │  │
│  └──────────────┘   └──────────────┘   └──────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              VERIFICATION MANAGER                    │    │
│  │  • Task Analysis    • Issue Detection                │    │
│  │  • Remediation      • Retry Logic                    │    │
│  └──────────────────────────────────────────────────────┘    │
└────────────────────────────┬─────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │      SHARED WORKSPACE       │
              │  • File Tree (visual)       │
              │  • Agent Activities         │
              │  • Files Modified Tracking  │
              │  • Shared Variables         │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌───────────┐       ┌───────────┐       ┌───────────┐
   │  Agent 1  │       │  Agent 2  │       │  Agent 3  │
   │           │       │           │       │           │
   │ File Ops: │       │ File Ops: │       │ File Ops: │
   │ read/write│       │ read/write│       │ read/write│
   │ edit/list │       │ edit/list │       │ edit/list │
   │  [shell]  │       │  [shell]  │       │  [shell]  │
   │           │       │           │       │           │
   │ Supervisor│       │ Supervisor│       │ Supervisor│
   │ (if stuck)│       │ (if stuck)│       │ (if stuck)│
   └───────────┘       └───────────┘       └───────────┘
```

## Agent File Operations

Agents have direct file operations tools that are **preferred over shell commands** for file manipulation:

| Tool | Syntax | Use Case |
|------|--------|----------|
| **read_file** | `<read_file>path</read_file>` | Read file contents with line numbers |
| **list_dir** | `<list_dir>path</list_dir>` | List directory contents |
| **write_file** | `<write_file path="path">content</write_file>` | Create or overwrite a file |
| **edit_file** | `<edit_file path="path">old\|\|\|new</edit_file>` | Replace specific text in a file |
| **help** | `<help>description</help>` | Ask supervisor for guidance when stuck |

### Why Direct File Operations?

1. **Reliability** - No shell escaping issues, proper encoding handling
2. **Visibility** - All file changes are tracked and visible to other agents
3. **Error handling** - Clear error messages when files don't exist or edits fail
4. **Line numbers** - When reading files, agents see line numbers for precise editing
5. **Verification** - Edit operations verify the old text exists before replacing

## Multi-Agent Coordination

Counsel uses a sophisticated coordination system to ensure agents work together effectively:

### Task Dependencies
When a task depends on another, the agent receives:
- **Result summary** from the dependency task
- **Files created** list with paths
- **Files modified** list with paths

### Workspace Awareness
Each agent sees:
- **Refreshed file tree** - Always current, shows files from all agents
- **Files by agent** - Who created/modified each file
- **Active agents** - What other agents are working on
- **Recent activities** - Latest actions across all agents
- **Task results** - Outcomes from completed tasks with file lists

### Example Flow
```
task_1: Create project structure
  └─ Creates: project/, project/venv/, project/src/
  
task_2 (depends on task_1): Write main.py
  └─ Sees: task_1 created project/, project/src/
  └─ Reads existing files before writing
  └─ Creates: project/src/main.py
  
task_3 (depends on task_2): Add tests
  └─ Sees: task_1 and task_2 results + all created files
  └─ Can import and test the code from task_2
```

## Help System

Agents have a built-in help system that provides guidance when they get stuck:

### Automatic Intervention
The supervisor automatically detects when agents are stuck (repeated failures, same errors) and injects guidance into the conversation.

### Explicit Help Requests
Agents can explicitly ask for help using the `<help>` action:
```xml
<help>I'm trying to install dependencies but pip keeps failing with permission errors</help>
```

### Configuration
```python
from counsel import Config, SupervisorConfig

config = Config(
    supervisor=SupervisorConfig(
        enabled=True,                      # Enable supervisor (default: True)
        failures_before_intervention=2,    # Failures before suggesting help
        max_help_per_task=5,              # Max help requests per task
    )
)
```

### How It Works
1. **Action Tracking** - All agent actions are tracked for context
2. **Failure Detection** - Consecutive failures trigger help suggestions
3. **Automatic Intervention** - After N iterations with failures, supervisor intervenes
4. **Explicit Help** - Agents can ask `<help>` anytime they're stuck
5. **Contextual Guidance** - Supervisor sees full action history and provides specific steps

## Configuration

### Default Configuration (v1.3.0+)

By default, Counsel now ships with production-ready defaults:
- ✅ **Task Verification** - Enabled by default
- 🆘 **Supervisor Help** - Enabled by default  
- 🔍 **Debug Mode** - Enabled by default

### Production Configuration

```python
from counsel import Config

config = Config.for_production()
# Includes:
# - Task verification enabled
# - Supervisor help enabled
# - Optimized model settings
# - Audit logging enabled
# - Telemetry enabled
```

### Custom Configuration

```python
from counsel import Config

config = Config(
    llm=LLMConfig(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        temperature=0.5,
    ),
    verification=VerificationConfig(
        enabled=True,
        max_retries=3,
        min_passing_score=0.85,
    ),
    execution=ExecutionConfig(
        max_parallel_agents=5,
        continue_on_failure=False,
    ),
)
```

### Environment Variables

```bash
export COUNSEL_MODEL="Qwen/Qwen2.5-14B-Instruct"
export COUNSEL_DEVICE="cuda"
export COUNSEL_VERIFY="true"
export COUNSEL_MAX_PARALLEL=5
export COUNSEL_LOG_FILE="/var/log/counsel/counsel.log"
export COUNSEL_LOG_LEVEL="INFO"
```

## Recommended Models

| Model | Size | VRAM (4-bit) | Best For |
|-------|------|--------------|----------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~1.5 GB | Testing, simple tasks |
| `Qwen/Qwen2.5-14B-Instruct` | 7B | ~5 GB | General use ⭐ |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ~5 GB | Code-heavy tasks ⭐ |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | ~9 GB | Complex reasoning |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~2.5 GB | Long context (128k) |

## Project Structure

```
CounselOfAgents/
├── counsel/                 # Main package
│   ├── __init__.py         # Package exports
│   ├── agent.py            # Worker agents + supervisor intervention
│   ├── config.py           # Configuration with validation
│   ├── jobs.py             # Job persistence
│   ├── llm.py              # LLM interface
│   ├── logging.py          # Professional logging system
│   ├── models.py           # Model catalog
│   ├── orchestrator.py     # Task coordination + verification integration
│   ├── shell.py            # Shell execution + process tracking
│   ├── task_graph.py       # DAG management
│   ├── verification.py     # Task verification system
│   └── workspace.py        # Shared state + file tree
├── tests/                  # Test suite
├── projects/               # Agent working directory
├── main.py                 # CLI entry point
├── Dockerfile
├── Dockerfile.cuda
├── docker-compose.yml
├── docker-compose.cuda.yml
├── Makefile
├── requirements.txt
└── README.md
```

## Programmatic Usage

```python
import asyncio
from counsel import Orchestrator, Config

async def main():
    config = Config.from_env()
    orchestrator = Orchestrator(config=config)
    
    # With verification
    result = await orchestrator.run(
        "Create a REST API with user authentication",
        verify_tasks=True,
        max_retries=2
    )
    
    if result.success:
        print("All tasks completed and verified!")
        print(f"Files created: {result.get_files_created()}")
        
        # Check verification summary
        v_summary = result.get_verification_summary()
        print(f"Verification pass rate: {v_summary['pass_rate']:.0%}")
    else:
        print(f"Execution failed: {result.error}")
        
        # Check verification issues
        for task_id, v_result in result.verification_results.items():
            if v_result['status'] != 'passed':
                print(f"Task {task_id} issues:")
                for issue in v_result.get('issues', []):
                    print(f"  - {issue['description']}")

asyncio.run(main())
```

## Docker

### Build & Run

```bash
# CPU version
make docker-build
make docker-run

# GPU/CUDA version
make docker-cuda
make docker-gpu
```

### Docker Compose

```bash
# CPU
docker-compose up -d
docker-compose exec counsel-agents python main.py

# GPU
docker-compose -f docker-compose.cuda.yml up -d
```

## Requirements

- Python 3.10+
- ~6GB RAM (with 4-bit quantization for 7B model)
- NVIDIA GPU recommended (works on CPU/MPS)
- Docker (optional)

## License

MIT License

---

**Counsel Of Agents** - Enterprise Multi-Agent Orchestration Platform  
Built for reliability. Designed for scale.
