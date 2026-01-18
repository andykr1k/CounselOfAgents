# Agent Orchestration System

A multi-agent system that breaks down complex, long-horizon tasks into a dependency graph and executes them in parallel. Features a **shared workspace** for agent coordination and an **interactive shell** for direct control.

## Key Features

- **🤖 Single Orchestrator** - One intelligent agent that decomposes complex tasks
- **📊 DAG-Based Execution** - Tasks run in parallel, respecting dependencies  
- **🔄 Shared Workspace** - Agents see each other's files, directories, and activities
- **💻 Interactive Shell** - Run shell commands alongside orchestrated tasks
- **🔌 General-Purpose Agents** - No specialized agents; all workers are identical
- **🐚 Full Shell Access** - Agents can run any command (ls, git, npm, etc.)

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
              │  • Shared Variables         │
              │  • Project Context          │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌───────────┐       ┌───────────┐       ┌───────────┐
   │  Agent 1  │       │  Agent 2  │       │  Agent 3  │
   │  [shell]  │◄─────►│  [shell]  │◄─────►│  [shell]  │
   └───────────┘       └───────────┘       └───────────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                    Coordination via Workspace
```

## How Agent Coordination Works

Agents share context through the **Workspace**:

1. **File Tracking**: When Agent 1 creates `src/index.js`, Agent 2 knows about it
2. **Activity Log**: Agents see what others are doing in real-time
3. **Shared Variables**: Tasks can pass data to dependent tasks
4. **Project Structure**: All agents understand the current directory layout

```python
# Example: Agent 2 receives this context
"""
## Project Structure
Root: /home/user/my-project
Current directory: /home/user/my-project

### Files in workspace:
  - package.json (by agent_1)
  - src/index.js (by agent_1)

### Other agents currently working:
  - agent_3: Setting up database configuration...

### Recent activities:
  - [agent_1] created_file: package.json
  - [agent_1] ran_command: npm init -y
"""
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CounselOfAgents.git
cd CounselOfAgents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Interactive Shell Mode (Recommended)

```bash
python main.py
```

This opens an interactive shell where you can:
- Run orchestrated tasks by typing them
- Execute shell commands directly with `!` prefix
- Check workspace status with `@` commands

```
my-project > Create a Python Flask API with user authentication

🔍 Analyzing task and creating execution plan...
📋 Created 4 tasks

📋 Task Graph
├── Level 1
│   ├── ● task_1: Create project directory and initialize Flask app
├── Level 2
│   ├── ◑ task_2: Install Flask and dependencies
│   ├── ◑ task_3: Create user model and database setup
├── Level 3
│   └── ○ task_4: Create authentication routes and middleware

✓ task_1 completed
✓ task_2 completed
✓ task_3 completed  
✓ task_4 completed

✅ All tasks completed successfully!

my-project > !ls -la
total 24
drwxr-xr-x  5 user  staff   160 Jan 18 10:30 .
-rw-r--r--  1 user  staff   245 Jan 18 10:30 app.py
-rw-r--r--  1 user  staff   512 Jan 18 10:30 models.py
-rw-r--r--  1 user  staff   128 Jan 18 10:30 requirements.txt

my-project > @status
╭─────────────────────────────────────────────╮
│ Status                                      │
│ 📁 Working Directory: /home/user/my-project │
│                                             │
│ 📄 Recent Files:                            │
│    • app.py                                 │
│    • models.py                              │
│    • requirements.txt                       │
│                                             │
│ 📊 Tasks: 4/4 completed                     │
╰─────────────────────────────────────────────╯
```

### Single Task Mode

```bash
python main.py "Create a React todo app with local storage"
```

### Shell Commands Reference

| Command | Description |
|---------|-------------|
| `!<cmd>` | Run shell command directly (e.g., `!ls -la`) |
| `@status` | Show workspace status (files, agents, tasks) |
| `@files` | List all files in workspace |
| `@dirs` | List all directories |
| `@history` | Show recent agent activities |
| `@context` | Show full workspace context |
| `@clear` | Clear the screen |
| `help` | Show example tasks |
| `exit` | Exit the shell |

### Command Line Options

```bash
python main.py --help

Options:
  -i, --interactive         Interactive shell mode
  -w, --workspace DIR       Working directory for agents
  -m, --model MODEL         HuggingFace model name
  --device {auto,cuda,mps,cpu}  Device to run on
  -p, --parallel N          Max parallel agents (default: 3)
  --no-quantize             Disable 4-bit quantization
  -v, --verbose             Enable verbose output
  --continue-on-failure     Continue even if tasks fail
```

### Examples

```bash
# Start in specific directory
python main.py -w ./my-new-project

# Use coding-optimized model
python main.py -m "Qwen/Qwen2.5-Coder-7B-Instruct"

# More parallel agents
python main.py -p 5 "Build a microservices architecture"

# Continue even if some tasks fail
python main.py --continue-on-failure "Set up CI/CD pipeline"
```

## Project Structure

```
CounselOfAgents/
├── main.py              # CLI with interactive shell
├── orchestrator.py      # Task planning and coordination
├── agent.py             # Worker agents with shell access
├── task_graph.py        # DAG for task dependencies
├── workspace.py         # Shared state for coordination
├── llm.py               # HuggingFace LLM interface
├── shell.py             # Safe shell execution
├── config.py            # Configuration management
└── requirements.txt     # Dependencies
```

## Configuration

### Environment Variables

```bash
export AGENT_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
export AGENT_LLM_DEVICE="cuda"
export AGENT_MAX_PARALLEL=5
export AGENT_VERBOSE=1
```

### Recommended Models

| Model | Size | Memory | Best For |
|-------|------|--------|----------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~4GB | Simple tasks, testing |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | ~8GB | General use (default) |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B | ~8GB | Code-heavy tasks |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | ~12GB | Complex reasoning |

## Safety Features

- **Blocked Commands**: Dangerous patterns like `rm -rf /` are blocked
- **No Sudo**: Sudo disabled by default
- **Timeouts**: Commands timeout after 120 seconds
- **Output Limits**: Large outputs are truncated

## How It Works

1. **User Request** → "Build a REST API with Express"

2. **Planning Phase** → Orchestrator uses LLM to decompose:
   ```
   task_1: Create directory and npm init
   task_2: Install express (depends on task_1)
   task_3: Create routes (depends on task_2)
   task_4: Add error handling (depends on task_3)
   ```

3. **Execution Phase** → Agents execute in parallel where possible:
   ```
   Level 1: [task_1] ────────────────►
   Level 2:          [task_2] ───────►
   Level 3:                   [task_3]
   Level 4:                            [task_4]
   ```

4. **Coordination** → Via shared workspace:
   - Agent 1 creates files → Workspace tracks them
   - Agent 2 starts → Sees Agent 1's files in context
   - Agent 3 starts → Sees both agents' work

5. **Results** → Aggregated and displayed

## Requirements

- Python 3.10+
- ~8GB RAM (with 4-bit quantization)
- CUDA GPU recommended (works on CPU/MPS too)

## License

MIT License
