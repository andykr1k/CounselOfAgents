# Counsel of Agents

An intelligent multi-agent orchestration system that uses real Hugging Face models to route and execute tasks across specialized agents. Simply describe what you need, and the system automatically selects the right agents and performs all necessary actions.

## 🚀 Quick Start

### Installation

**Option 1: Docker (Recommended)**

```bash
# Build and run with Docker Compose (CPU)
docker-compose up --build

# Or build manually
docker build -t counsel-of-agents .
docker run -it -v $(pwd):/app/workspace counsel-of-agents

# For GPU support (requires NVIDIA Docker)
docker-compose -f docker-compose.cuda.yml up --build
```

**Option 2: Local Installation**

```bash
# Install dependencies
pip install -r requirements.txt

# Install as command-line tool (optional)
pip install -e .
```

### Usage

**Interactive Mode (Recommended):**
```bash
agent
# or
python main.py
```

You'll see:
```
💬 What can I help you with?
```

Just type your task and watch the system work!

**Single Command:**
```bash
python main.py "Generate a Python function to calculate fibonacci"
```

**Docker Usage:**
```bash
# Interactive mode
docker run -it -v $(pwd):/app/workspace counsel-of-agents

# Single command
docker run -it -v $(pwd):/app/workspace counsel-of-agents python main.py "Your task here"
```

## ✨ Features

- **🤖 9 Specialized Agents** - Each using real Hugging Face models:
  - **Coding Agent** - Code generation (StarCoder)
  - **Writing Agent** - Content creation (GPT-2)
  - **Evaluation Agent** - Critical analysis (GPT-2)
  - **Reading Agent** - Summarization & comprehension (BART)
  - **Research Agent** - Information gathering (GPT-2)
  - **Math Agent** - Problem solving (GPT-2)
  - **Analysis Agent** - General analysis (GPT-2)
  - **Translation Agent** - Language translation (Helsinki-NLP)
  - **File System Agent** - File operations & shell commands (GPT-2)

- **🧠 Intelligent Routing** - LLM-based task classifier understands all agents and routes tasks intelligently
- **⚡ Parallel Execution** - Multiple agents work simultaneously on independent tasks
- **📊 Real-time Progress** - See exactly what's happening with live updates
- **🔌 Plug-and-Play** - Add new agents easily - they self-describe and are automatically discovered

## 📖 How It Works

1. **You describe your task** - "Generate a Python function" or "Read file README.md"
2. **LLM analyzes the task** - The classifier understands all agents and their specializations
3. **Best agent(s) are selected** - System routes to the most appropriate agent(s)
4. **Tasks execute with progress** - You see real-time updates as agents work
5. **Results are returned** - Formatted output with agent usage summary

## 🎯 Example Tasks

### Code Generation
```bash
agent
💬 What can I help you with? Generate a Python function to sort a list
```

### File Operations
```bash
💬 What can I help you with? Read file README.md
💬 What can I help you with? Grep 'def' in main.py
💬 What can I help you with? List files in current directory
```

### Writing & Analysis
```bash
💬 What can I help you with? Write a summary about machine learning
💬 What can I help you with? Evaluate the pros and cons of Python
💬 What can I help you with? Research information about neural networks
```

## 🏗️ Architecture

The system consists of 6 core files:

- **`agent.py`** - Base agent framework and all specialized agent implementations
- **`agent_registry.py`** - Dynamic agent discovery and management
- **`task_classifier.py`** - LLM-based intelligent task routing
- **`task_graph.py`** - Dependency management and parallel execution
- **`orchestrator.py`** - Main coordinator that orchestrates the workflow
- **`main.py`** - Beautiful CLI interface with progress indicators

## 🔧 Adding Your Own Agent

Create a new agent by inheriting from `HuggingFaceAgent` or `BaseAgent`:

```python
from agent import HuggingFaceAgent, AgentCapability, ModelConfig, ModelType

class MyCustomAgent(HuggingFaceAgent):
    def __init__(self):
        config = ModelConfig(
            model_name="your-model/name",
            model_type=ModelType.CAUSAL_LM,
            max_length=512,
            temperature=0.7
        )
        super().__init__(
            "my_agent",
            "My Custom Agent",
            [AgentCapability.ANALYSIS],
            config,
            "Description of what this agent does and when to use it"
        )
```

Then register it:
```python
registry.register(MyCustomAgent())
```

The agent automatically appears in the classifier's knowledge - no configuration needed!

## 📋 Requirements

- Python 3.8+
- `transformers` - Hugging Face model library
- `torch` - PyTorch for model execution
- `accelerate` - Efficient model loading
- `rich` - Beautiful terminal output

See `requirements.txt` for full list.

## 🎨 What You See

When you run a task, you'll see:

```
📝 Task: Generate a Python function to sort a list

🔍 Analyzing task and selecting appropriate agents...
✓ Identified 1 task(s) to execute
💭 [Reasoning from LLM about why coding_agent was selected]
📋 Building task execution plan...
⚙️ Level 1/1: Executing 1 task(s)...
⚙️ Using Coding Agent for: Generate a Python function...
✓ Coding Agent completed task
📊 Aggregating results...
✅ Task completed

✅ Task Completed Successfully

🤖 Agents Used:
  ✓ Coding Agent

📄 Result:
[Generated code displayed here]
```

## 🔍 How Task Routing Works

The system uses an LLM (GPT-2 by default) that:

1. **Knows all agents** - Receives information about every registered agent
2. **Understands specializations** - Each agent provides its own description
3. **Reasons about tasks** - Analyzes your task and determines the best agent(s)
4. **Provides reasoning** - Explains why it selected specific agents

No keywords, no hardcoded rules - pure LLM reasoning based on agent descriptions.

## 🛠️ Advanced Usage

### Custom Model Configuration

```python
from agent import CodingAgent, ModelConfig, ModelType

# Use a different model
agent = CodingAgent(model_preset="code", max_length=2048, temperature=0.2)
```

### List Available Agents

```bash
python main.py --list-agents
```

### Adjust Parallelism

```bash
python main.py "Your task" --max-parallel 5
```

## 📝 Project Structure

```
CounselOfAgents/
├── agent.py              # Agent framework & implementations
├── agent_registry.py     # Agent discovery & management
├── task_classifier.py    # LLM-based task routing
├── task_graph.py         # Dependency & execution management
├── orchestrator.py         # Main workflow coordinator
├── main.py               # CLI interface
├── setup.py              # Installation script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🤝 Contributing

To add a new agent:

1. Create a class inheriting from `HuggingFaceAgent` or `BaseAgent`
2. Provide a clear description of what it does
3. Register it in your code
4. That's it! The system automatically discovers and uses it.

## 📄 License

Open source - modify and extend as needed.

---

**Ready to use?** Just run `agent` and start describing what you need! 🚀
