# Agent Orchestration System

A multi-agent system that breaks down complex, long-horizon tasks into a dependency graph and executes them in parallel. Features a **shared workspace** for agent coordination and an **interactive shell** for direct control.

## Quick Start

### Using Docker (Recommended)

```bash
# CPU version
make docker-build
make docker-run

# GPU/CUDA version
make docker-cuda
make docker-gpu
```

### Local Installation

```bash
pip install -r requirements.txt
python main.py
```

## Features

- 🤖 **Single Orchestrator** - Intelligent task decomposition
- 📊 **DAG-Based Execution** - Parallel task execution with dependencies
- 🔄 **Shared Workspace** - Agents see each other's files and activities
- 💻 **Interactive Shell** - Run commands alongside orchestrated tasks
- 🐚 **Full Shell Access** - Agents can run any command

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
              │  • Files & Directories      │
              │  • Agent Activities         │
              │  • Shared Context           │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌───────────┐       ┌───────────┐       ┌───────────┐
   │  Agent 1  │       │  Agent 2  │       │  Agent 3  │
   │  [shell]  │       │  [shell]  │       │  [shell]  │
   └───────────┘       └───────────┘       └───────────┘
```

## Usage

### Interactive Shell Mode

```bash
python main.py
```

```
myproject > Create a Python Flask API with authentication

🔍 Analyzing task and creating execution plan...
📋 Created 4 tasks

📋 Task Graph
├── Level 1
│   ├── ● task_1: Create project directory
├── Level 2
│   ├── ◑ task_2: Set up Flask application
│   ├── ◑ task_3: Create user model
├── Level 3
│   └── ○ task_4: Add authentication routes

✓ task_1 completed
✓ task_2 completed
✓ task_3 completed
✓ task_4 completed

✅ All tasks completed successfully!

myproject > !ls -la           # Run shell command
myproject > @status           # Show workspace status
myproject > @files            # List files created
```

### Shell Commands

| Command | Description |
|---------|-------------|
| `!<cmd>` | Run shell command directly |
| `@status` | Show workspace status |
| `@files` | List workspace files |
| `@history` | Show agent activities |
| `@clear` | Clear screen |
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
  -i, --interactive         Interactive shell mode
  -w, --workspace DIR       Working directory for agents
  -m, --model MODEL         HuggingFace model name
  --device {auto,cuda,mps,cpu}
  -p, --parallel N          Max parallel agents (default: 3)
  --no-quantize             Disable 4-bit quantization
  -v, --verbose             Verbose output
```

## Docker

### Build Images

```bash
# CPU version
make docker-build

# CUDA/GPU version  
make docker-cuda
```

### Run Containers

```bash
# CPU - Interactive mode
make docker-run

# GPU - Interactive mode
make docker-gpu

# Run tests in Docker
make docker-test
make docker-test-gpu
```

### Docker Compose

```bash
# CPU version
docker-compose up -d
docker-compose exec counsel-agents python main.py

# GPU version
docker-compose -f docker-compose.cuda.yml up -d
docker-compose -f docker-compose.cuda.yml exec counsel-agents python main.py
```

### Docker with Custom Task

```bash
docker run -it --rm \
  -v $(pwd)/projects:/app/projects \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  counsel-agents:latest \
  python main.py "Create a hello world project"
```

### Docker with GPU

```bash
docker run -it --rm --gpus all \
  -v $(pwd)/projects:/app/projects \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  counsel-agents:cuda \
  python main.py -i
```

## Project Structure

```
CounselOfAgents/
├── counsel/                 # Main package
│   ├── __init__.py
│   ├── agent.py            # Worker agents
│   ├── config.py           # Configuration
│   ├── llm.py              # LLM interface
│   ├── orchestrator.py     # Task coordination
│   ├── shell.py            # Shell execution
│   ├── task_graph.py       # DAG management
│   └── workspace.py        # Shared state
├── tests/                   # Test suite
│   ├── __init__.py
│   └── test_basic.py
├── projects/                # Agent working directory (gitignored)
├── main.py                  # CLI entry point
├── Dockerfile               # CPU Docker image
├── Dockerfile.cuda          # GPU Docker image
├── docker-compose.yml       # CPU compose
├── docker-compose.cuda.yml  # GPU compose
├── Makefile                 # Build commands
├── pyproject.toml           # Package config
├── requirements.txt
└── README.md
```

## Testing

```bash
# Local tests
make test

# With coverage
make test-cov

# In Docker
make docker-test

# In Docker with GPU
make docker-test-gpu
```

## Configuration

### Environment Variables

```bash
export AGENT_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export AGENT_LLM_DEVICE="cuda"
export AGENT_MAX_PARALLEL=5
export AGENT_VERBOSE=1
export AGENT_NO_QUANTIZE=0
```

### Recommended Models

| Model | Size | Memory | Best For |
|-------|------|--------|----------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~4GB | Testing, simple tasks |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~8GB | General use (default) |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ~8GB | Code-heavy tasks |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | ~12GB | Complex reasoning |

## How Agent Coordination Works

Agents share context through the **Workspace**:

1. **File Tracking**: Agent 1 creates `src/app.py` → Agent 2 sees it
2. **Activity Log**: Real-time visibility into what agents are doing
3. **Shared Variables**: Pass data between dependent tasks
4. **Project Structure**: All agents understand the directory layout

```python
# Agent 2 receives this context:
"""
## Project Structure
Root: /app/projects/my-api

### Files in workspace:
  - package.json (by agent_1)
  - src/index.js (by agent_1)

### Other agents currently working:
  - agent_3: Setting up database...

### Recent activities:
  - [agent_1] ran_command: npm init -y
  - [agent_1] created_file: package.json
"""
```

## Requirements

- Python 3.10+
- ~8GB RAM (with 4-bit quantization)
- NVIDIA GPU recommended (works on CPU/MPS)
- Docker (optional, for containerized usage)

## License

MIT License
