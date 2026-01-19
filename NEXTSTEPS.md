# Counsel Of Agents - Roadmap & Next Steps

This document outlines planned features and improvements for the Counsel Of Agents Orchestration Platform.

## 🚀 High Priority

### Multi-Job Management
- [ ] **Job Dashboard** - Multi-panel terminal UI showing:
  - Active job panel with task graph
  - Command/input panel for user interaction
  - Debug panel for agent activity
  - Job list sidebar
- [ ] **Job Switching** - Switch between jobs with `@switch <id>`
- [ ] **In-Flight Modifications** - Send messages to running jobs: `@<job_id> add error handling`
- [ ] **Job Resume** - Continue failed/paused jobs from where they left off
- [ ] **Parallel Jobs** - Run multiple jobs simultaneously

### Enhanced Verification
- [ ] **Quick Verification First** - Run fast checks before LLM verification
- [ ] **Custom Verification Rules** - Define project-specific verification criteria
- [ ] **Verification Reports** - Generate detailed HTML/PDF reports
- [ ] **Verification Caching** - Don't re-verify unchanged tasks
- [ ] **Partial Fix Mode** - Fix only failed issues, keep passing ones

### Better Terminal Visualizations
- [ ] **Rich Live Dashboard** - Using Rich's Layout for multi-panel display
- [ ] **Progress Bars** - Per-task progress indicators
- [ ] **Agent Status Icons** - Visual indicators for agent states
- [ ] **Collapsible Sections** - Expand/collapse task details
- [ ] **Color Themes** - Light/dark mode support
- [ ] **Task Timeline** - Visual timeline of task execution

### Agent Tools
- [ ] **Web Search** - Search the internet for documentation, examples, solutions
  - DuckDuckGo API integration
  - Google Custom Search API option
  - Results summarization
- [ ] **File Reading** - Dedicated tool to read files intelligently
  - Read specific line ranges
  - Search within files
  - Parse structured files (JSON, YAML, etc.)
- [ ] **Code Analysis** - Understand existing codebases
  - AST parsing for Python
  - Dependency analysis
  - Function/class extraction
- [ ] **Git Integration** - Version control operations
  - Commit changes
  - Create branches
  - Push to remotes

## 🔧 Medium Priority

### Improved Agent Intelligence
- [ ] **Learning from Mistakes** - Remember what didn't work in this session
- [ ] **Pattern Recognition** - Detect common failure patterns earlier
- [ ] **Better Supervisor** - More intelligent intervention strategies
- [ ] **Context Compression** - Smarter truncation that preserves key information
- [ ] **Multi-Model Routing** - Use different models for different task types
- [ ] **Skill-Based Agent Allocation** - Match agents to tasks based on requirements

### Enhanced Error Handling
- [ ] **Error Classification** - Categorize errors for better recovery
- [ ] **Rollback Support** - Undo changes from failed tasks
- [ ] **Checkpoint/Restore** - Save state at key points for recovery
- [ ] **Error Aggregation** - Group similar errors across tasks

### Enhanced Workspace
- [ ] **File Watching** - Real-time detection of file changes
- [ ] **Smart Ignore** - Better filtering of irrelevant files
- [ ] **Project Templates** - Recognize project types (Python, Node, etc.)
- [ ] **Dependency Detection** - Understand project dependencies
- [ ] **Code Quality Metrics** - Track complexity, test coverage

### API & Integration
- [ ] **REST API** - HTTP API for remote control
- [ ] **WebSocket Updates** - Real-time status streaming
- [ ] **Webhook Notifications** - Notify external systems on completion
- [ ] **Plugin System** - Extensible tool/capability system
- [ ] **SDK Libraries** - Python/JavaScript/Go SDKs

## 🎯 Future Ideas

### Advanced Features
- [ ] **Multi-User Support** - Multiple users sharing a server
- [ ] **Team Collaboration** - Agents can delegate to specialized sub-agents
- [ ] **Memory System** - Long-term memory across sessions
- [ ] **Fine-Tuning Integration** - Learn from successful task completions
- [ ] **Cost Estimation** - Estimate tokens/compute before execution

### UI/UX
- [ ] **Web Interface** - Browser-based dashboard
- [ ] **VS Code Extension** - Integration with VS Code
- [ ] **Mobile App** - Monitor jobs from phone
- [ ] **Slack/Discord Bot** - Control via chat

### Infrastructure
- [ ] **Distributed Execution** - Agents on multiple machines
- [ ] **Cloud Deployment** - Easy deployment to AWS/GCP/Azure
- [ ] **Kubernetes Support** - Scalable container orchestration
- [ ] **Cost Tracking** - Track API/compute costs
- [ ] **Usage Analytics** - Dashboard for usage patterns

## 🐛 Known Issues to Fix

- [ ] **Context Length Hardcoded** - Should read from model config dynamically
- [ ] **Workspace Context Rebuilding** - Gets rebuilt on every call, should cache with invalidation
- [ ] **File Content Truncation** - Large files may lose important context
- [ ] **Error Messages** - Some error messages could be more descriptive

## ✅ Recently Completed (v1.0.0)

- [x] ~~**Task Verification System**~~ - Automatic verification of task completion
  - LLM-based verification with detailed issue reporting
  - Automatic retry with remediation instructions
  - Configurable verification thresholds
  - Quick verification for fast pre-checks
- [x] ~~**Professional Logging**~~ - Structured logging system
  - JSON log format support
  - Metrics and telemetry
  - Audit logging
  - Colored console output
- [x] ~~**Configuration Validation**~~ - Comprehensive config validation
  - Environment variable support (COUNSEL_*)
  - Production/testing presets
  - Validation with helpful error messages
- [x] ~~**Job Persistence**~~ - Jobs saved to `~/.counsel/jobs/`
- [x] ~~**Debug Mode**~~ - Full visibility into agent activity
- [x] ~~**Command History**~~ - Up/down arrow navigation
- [x] ~~**Job Deletion**~~ - `@delete` command
- [x] ~~**File Tree Context**~~ - Visual tree instead of flat list
- [x] ~~**Supervisor Intervention**~~ - Help when agents get stuck
- [x] ~~**Process Cleanup**~~ - Proper cleanup on exit/interrupt
- [x] ~~**Model Selection**~~ - Interactive model picker
- [x] ~~**Shell Environment Docs**~~ - Agents understand subprocess limitations
- [x] ~~**Better Task Planning**~~ - Orchestrator creates more practical tasks
- [x] ~~**Performance Metrics**~~ - Timing for planning, execution, verification

## Contributing

Want to help? Pick an item from the roadmap and:

1. Open an issue to discuss the approach
2. Fork the repo and create a branch
3. Implement the feature with tests
4. Submit a PR

## Architecture Notes for Contributors

### Key Files
```
counsel/
├── agent.py           # Worker agents + supervisor intervention
├── config.py          # Configuration with validation
├── jobs.py            # Job persistence layer
├── llm.py             # LLM interface (HuggingFace)
├── logging.py         # Professional logging system
├── models.py          # Model catalog
├── orchestrator.py    # Task planning + coordination + verification
├── shell.py           # Safe shell execution
├── task_graph.py      # DAG management
├── verification.py    # Task verification system
└── workspace.py       # Shared state + file tree
```

### Adding a New Agent Tool
```python
# In agent.py, add a new action type:

# 1. Add to _parse_action patterns:
(r"<search>(.*?)</search>", "search"),

# 2. Handle in execute() loop:
elif action_type == "search":
    query = action_content
    results = await self._web_search(query)
    self._conversation.append(Message(
        role="user",
        content=f"Search results:\n{results}\n\nContinue with the task."
    ))

# 3. Implement the tool method:
async def _web_search(self, query: str) -> str:
    # Implementation here
    pass
```

### Adding a New Shell Command
```python
# In main.py shell_mode():

elif meta_cmd == 'mycommand':
    # Handle @mycommand
    do_something()
```

### Adding Configuration Options
```python
# In config.py:

@dataclass
class MyNewConfig:
    option1: str = "default"
    option2: int = 10
    
    def validate(self) -> List[str]:
        errors = []
        if self.option2 < 0:
            errors.append("option2 must be non-negative")
        return errors

# Add to Config dataclass:
my_new: MyNewConfig = field(default_factory=MyNewConfig)
```

## Version History

- **v1.0.0** - Task verification, professional logging, configuration validation, metrics
- **v0.3.0** - Job persistence, debug mode, supervisor intervention
- **v0.2.0** - Model selection, parallel execution
- **v0.1.0** - Initial release

## License

MIT License
