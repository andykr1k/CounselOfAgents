"""Configuration management for the Agent Orchestration System."""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import os


@dataclass
class LLMConfig:
    """Configuration for the LLM backend."""
    
    # Model settings
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    
    # Generation settings
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Quantization (for memory efficiency)
    load_in_8bit: bool = False
    load_in_4bit: bool = True  # Recommended for 7B+ models
    
    # Trust remote code (required for some models)
    trust_remote_code: bool = True


@dataclass
class ShellConfig:
    """Configuration for shell execution."""
    
    # Working directory (None = current directory)
    working_directory: Optional[str] = None
    
    # Timeout for commands (seconds)
    default_timeout: int = 120
    
    # Maximum output size (characters)
    max_output_size: int = 50000
    
    # Blocked command patterns (for safety)
    blocked_patterns: List[str] = field(default_factory=lambda: [
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf ~/",
        "> /dev/sd",
        "dd if=",
        "mkfs",
        "fdisk",
        ":(){:|:&};:",  # Fork bomb
        "chmod -R 777 /",
        "chown -R",
        "format c:",
    ])
    
    # Whether to allow sudo commands
    allow_sudo: bool = False


@dataclass 
class AgentConfig:
    """Configuration for worker agents."""
    
    # Maximum iterations per task (prevent infinite loops)
    max_iterations: int = 15
    
    # Maximum shell commands per task
    max_shell_commands: int = 50
    
    # Whether agent can create files
    can_create_files: bool = True
    
    # Whether agent can delete files
    can_delete_files: bool = True
    
    # Whether agent can make network requests
    can_network: bool = True


@dataclass
class ExecutionConfig:
    """Configuration for task execution."""
    
    # Maximum parallel agents
    max_parallel_agents: int = 3
    
    # Task timeout (seconds)
    task_timeout: int = 600
    
    # Whether to persist state for recovery
    persist_state: bool = True
    
    # State file location
    state_file: str = ".agent_state.json"
    
    # Whether to continue on task failure
    continue_on_failure: bool = False


@dataclass
class Config:
    """Main configuration container."""
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    shell: ShellConfig = field(default_factory=ShellConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    
    # Logging
    verbose: bool = False
    log_file: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()
        
        # LLM settings
        if model := os.getenv("AGENT_LLM_MODEL"):
            config.llm.model_name = model
        if device := os.getenv("AGENT_LLM_DEVICE"):
            config.llm.device = device
        
        # Execution settings
        if max_parallel := os.getenv("AGENT_MAX_PARALLEL"):
            config.execution.max_parallel_agents = int(max_parallel)
        
        # Verbosity
        if os.getenv("AGENT_VERBOSE", "").lower() in ("1", "true", "yes"):
            config.verbose = True
        
        return config
    
    @classmethod
    def for_testing(cls) -> "Config":
        """Create a lightweight config for testing."""
        return cls(
            llm=LLMConfig(
                model_name="Qwen/Qwen2.5-1.5B-Instruct",
                load_in_4bit=False,
                load_in_8bit=False,
            ),
            execution=ExecutionConfig(
                max_parallel_agents=2,
                task_timeout=120,
                persist_state=False,
            )
        )


# Global default config
DEFAULT_CONFIG = Config()


def get_config() -> Config:
    """Get the current configuration."""
    return DEFAULT_CONFIG


def set_config(config: Config) -> None:
    """Set the global configuration."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config
