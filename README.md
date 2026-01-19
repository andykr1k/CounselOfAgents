# Counsel of Agents

A multi-agent orchestration system that breaks down complex tasks into a dependency graph (DAG) and executes them in parallel using LLM-powered agents with shell access.

## Features

### Core
- 🤖 **Intelligent Task Decomposition** - LLM breaks down complex tasks into executable subtasks
- 📊 **DAG-Based Execution** - Parallel task execution respecting dependencies
- 🔄 **Shared Workspace** - Agents coordinate through a shared file/activity tracker
- 💻 **Interactive Shell** - Full control with command history (↑/↓ arrows)
- 🐚 **Shell Access** - Agents execute real commands in your environment

### New Features
- 🎯 **Model Selection** - Interactive model picker on first run with RAM/VRAM requirements
- 📋 **Job Persistence** - All jobs saved to `~/.counsel/jobs/` for history and recovery
- 🔍 **Debug Mode** - See everything agents do: LLM calls, shell commands, thinking
- 🌳 **File Tree Context** - Agents see visual directory structure, not just file lists
- 🧑‍💼 **Supervisor Intervention** - When agents get stuck, a supervisor provides fresh guidance
- 🛡️ **Process Cleanup** - Proper cleanup of all subprocesses on exit/interrupt
- ⌨️ **Command History** - Up/down arrows navigate previous commands (saved to `~/.counsel_history`)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run (will prompt for model selection on first run)
python main.py
```

### First Run - Model Selection

On first run, you'll see an interactive model selection screen:

```
🤖 Model Selection

Choose a language model to power your agents.

📊 System Info
RAM: 32 GB
GPU: CUDA detected - 12 GB VRAM

Available Models:
 #  Model                    Size   VRAM    RAM    Context  Description
 1  Qwen 2.5 0.5B            0.5B   0.8 GB  1 GB   32k      Ultra-lightweight
 2  Qwen 2.5 1.5B            1.5B   1.5 GB  2 GB   32k      Lightweight but capable
 3  Qwen 2.5 3B              3B     2.5 GB  3 GB   32k      Good for coding
 4  Qwen 2.5 7B ⭐           7B     5 GB    6 GB   32k      Recommended default
 ...

Select model [1]: 4
```

## Usage

### Interactive Shell

```bash
python main.py
```

```
✨ Agent Shell Ready

Commands:
  !<command>       - Run shell command directly
  @status          - Show workspace status
  @files           - List workspace files
  @history         - Show agent activities
  @debug           - Toggle debug mode
  @model           - Show current model
  @jobs            - Show past job history
  @delete <id>     - Delete a job by ID
  @delete all      - Delete all jobs
  help             - Show examples
  exit             - Exit the shell

  Use ↑/↓ arrows to navigate command history

projects > Create a Python calculator CLI

📝 Task: Create a Python calculator CLI
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
│                                                     │
│ ● 0 | ◑ 1 | ◐ 0 | ○ 3 | ✗ 0                        │
╰─────────────────────────────────────────────────────╯
```

### Debug Mode (ON by default)

Shows everything agents are doing:

```
╭──────────────── 🔍 Debug Output ────────────────────╮
│ 19:35:02 ▶ agent_1 Task: Create project with venv   │
│ 19:35:02 💭 agent_1 Planning approach...            │
│ 19:35:05 $ agent_1 $ mkdir -p calculator            │
│ 19:35:05   ↳ agent_1 Exit 0: (no output)            │
│ 19:35:05 $ agent_1 $ python -m venv calculator/venv │
│ 19:35:07   ↳ agent_1 Exit 0: (no output)            │
│ 19:35:07 📄 agent_1 Created: calculator/venv        │
│ 19:35:10 ✓ agent_1 Created project structure        │
╰─────────────────────────────────────────────────────╯
```

### Shell Commands

| Command | Description |
|---------|-------------|
| `!<cmd>` | Run shell command directly |
| `@status` | Show workspace status |
| `@files` | List workspace files |
| `@history` | Show agent activities |
| `@debug` | Toggle debug mode |
| `@model` | Show current model |
| `@jobs` | List job history |
| `@job <id>` | Show job details |
| `@delete <id>` | Delete a job |
| `@delete all` | Delete all jobs |
| `help` | Show examples |
| `exit` | Exit |

### Single Task Mode

```bash
python main.py "Create a React todo app"
```

### Command Line Options

```bash
python main.py --help

Options:
  task                      Task to execute (optional)
  --select-model            Show model selection screen
  --list-models             List all available models
  --reset-model             Clear saved model selection
  --jobs                    List all jobs
  --job ID                  Show specific job details
  -i, --interactive         Interactive shell mode
  -w, --workspace DIR       Working directory
  -m, --model MODEL         HuggingFace model ID
  --device {auto,cuda,mps,cpu}
  -p, --parallel N          Max parallel agents (default: 3)
  --no-quantize             Disable 4-bit quantization
  -v, --verbose             Verbose output
  -d, --debug               Debug mode (ON by default)
  --continue-on-failure     Continue if tasks fail
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ Task Planner │   │  Task Graph  │   │ Execution Engine │  │
│  │  (LLM-based) │   │    (DAG)     │   │  (Agent Spawner) │  │
│  └──────────────┘   └──────────────┘   └──────────────────┘  │
└────────────────────────────┬─────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │      SHARED WORKSPACE       │
              │  • File Tree (visual)       │
              │  • Agent Activities         │
              │  • Shared Variables         │
              │  • Real-time Coordination   │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌───────────┐       ┌───────────┐       ┌───────────┐
   │  Agent 1  │       │  Agent 2  │       │  Agent 3  │
   │  [shell]  │       │  [shell]  │       │  [shell]  │
   │           │       │           │       │           │
   │ Supervisor│       │ Supervisor│       │ Supervisor│
   │ (if stuck)│       │ (if stuck)│       │ (if stuck)│
   └───────────┘       └───────────┘       └───────────┘
```

## Agent Context

Agents receive rich context including a visual file tree:

```
==================================================
PROJECT: calculator
ROOT PATH: /home/user/projects/calculator
CURRENT DIRECTORY: /home/user/projects/calculator
==================================================

## File Tree (actual filesystem)
```
calculator/
├── src/
│   └── calc.py ← created by agent_1
├── tests/
│   └── test_calc.py ← created by agent_3
├── main.py ← created by agent_2
└── venv/
    └── bin/
        └── python
```

## Other Agents Working Now:
  • agent_3: Running tests...

## Results from Completed Tasks:
  • task_1: Created project structure
  • task_2: Implemented calculator functions
```

## Job Persistence

All jobs are automatically saved to `~/.counsel/jobs/`:

```bash
# List past jobs
python main.py --jobs

# Or in interactive mode
projects > @jobs

Recent Jobs:
  ✓ a1b2c3d4 Create a Python calculator CLI
  ✓ e5f6g7h8 Set up Express.js server
  ✗ i9j0k1l2 Create React app (failed)

# View job details
projects > @job a1b2

# Delete old jobs
projects > @delete a1b2
projects > @delete all
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

## Configuration

### Environment Variables

```bash
export AGENT_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export AGENT_LLM_DEVICE="cuda"
export AGENT_MAX_PARALLEL=5
export AGENT_DEBUG=1
```

### Recommended Models

| Model | Size | VRAM (4-bit) | Best For |
|-------|------|--------------|----------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~1.5 GB | Testing, simple tasks |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~5 GB | General use ⭐ |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ~5 GB | Code-heavy tasks ⭐ |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | ~9 GB | Complex reasoning |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~2.5 GB | Long context (128k) |

## Project Structure

```
CounselOfAgents/
├── counsel/                 # Main package
│   ├── __init__.py
│   ├── agent.py            # Worker agents + supervisor intervention
│   ├── config.py           # Configuration
│   ├── jobs.py             # Job persistence
│   ├── llm.py              # LLM interface
│   ├── models.py           # Model catalog
│   ├── orchestrator.py     # Task coordination
│   ├── shell.py            # Shell execution + process tracking
│   ├── task_graph.py       # DAG management
│   └── workspace.py        # Shared state + file tree
├── tests/
├── projects/               # Agent working directory
├── main.py                 # CLI entry point
├── Dockerfile
├── Dockerfile.cuda
├── docker-compose.yml
├── docker-compose.cuda.yml
├── Makefile
├── requirements.txt
├── README.md
└── NEXTSTEPS.md           # Roadmap
```

## Requirements

- Python 3.10+
- ~6GB RAM (with 4-bit quantization for 7B model)
- NVIDIA GPU recommended (works on CPU/MPS)
- Docker (optional)

## License

MIT License
