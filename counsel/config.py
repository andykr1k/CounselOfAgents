"""
Configuration Management for the Counsel Of Agents Orchestration Platform.

This module provides comprehensive configuration for all components of the
Counsel platform including LLM settings, shell execution, agents, verification,
and execution parameters.

Configuration can be loaded from:
    - Environment variables (COUNSEL_*)
    - Code (Config dataclass)
    - Saved preferences (~/.counsel/)

Environment Variables:
    COUNSEL_MODEL: HuggingFace model ID
    COUNSEL_DEVICE: Device to use (auto, cuda, mps, cpu)
    COUNSEL_MAX_PARALLEL: Maximum parallel agents
    COUNSEL_DEBUG: Enable debug mode (1/true/yes)
    COUNSEL_VERIFY: Enable task verification (1/true/yes)
    COUNSEL_LOG_FILE: Path to log file
    COUNSEL_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class LLMConfig:
    """
    Configuration for the LLM backend.
    
    Controls model selection, generation parameters, and quantization settings.
    """
    
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
    
    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []
        if not self.model_name:
            errors.append("model_name is required")
        if self.max_new_tokens < 1:
            errors.append("max_new_tokens must be positive")
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            errors.append("top_p must be between 0 and 1")
        return errors


@dataclass
class ShellConfig:
    """
    Configuration for shell execution.
    
    Controls timeouts, output limits, and security restrictions.
    """
    
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
    
    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []
        if self.default_timeout < 1:
            errors.append("default_timeout must be positive")
        if self.max_output_size < 1000:
            errors.append("max_output_size should be at least 1000")
        return errors


@dataclass 
class AgentConfig:
    """
    Configuration for worker agents.
    
    Controls iteration limits and agent capabilities.
    """
    
    # Maximum iterations per task (prevent infinite loops)
    max_iterations: int = 100
    
    # Maximum shell commands per task
    max_shell_commands: int = 100
    
    # Whether agent can create files
    can_create_files: bool = True
    
    # Whether agent can delete files
    can_delete_files: bool = True
    
    # Whether agent can make network requests
    can_network: bool = True
    
    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []
        if self.max_iterations < 1:
            errors.append("max_iterations must be positive")
        if self.max_shell_commands < 1:
            errors.append("max_shell_commands must be positive")
        return errors


@dataclass
class SupervisorConfig:
    """
    Configuration for the supervisor/helper system.
    
    The supervisor provides guidance when agents get stuck.
    """
    
    # Enable supervisor intervention (ON by default)
    enabled: bool = True
    
    # Number of consecutive failures before automatic intervention
    failures_before_intervention: int = 2
    
    # Number of iterations before checking for stuck patterns
    min_iterations_before_check: int = 3
    
    # Check frequency (every N iterations after min)
    check_frequency: int = 2
    
    # Max help requests per task (to prevent infinite help loops)
    max_help_per_task: int = 5
    
    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []
        if self.failures_before_intervention < 1:
            errors.append("failures_before_intervention must be positive")
        if self.max_help_per_task < 1:
            errors.append("max_help_per_task must be positive")
        return errors


@dataclass
class VerificationConfig:
    """
    Configuration for task verification.
    
    Controls automatic verification and retry behavior.
    """
    
    # Enable task verification (ON by default for reliability)
    enabled: bool = True
    
    # Maximum retries for failed verifications
    max_retries: int = 2
    
    # Minimum score to consider task passed (0.0 to 1.0)
    min_passing_score: float = 0.8
    
    # Verify all tasks or only critical ones
    verify_all_tasks: bool = True
    
    # Quick verification before full LLM verification
    enable_quick_verify: bool = True
    
    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []
        if self.max_retries < 0:
            errors.append("max_retries cannot be negative")
        if not 0 <= self.min_passing_score <= 1:
            errors.append("min_passing_score must be between 0 and 1")
        return errors


@dataclass
class ExecutionConfig:
    """
    Configuration for task execution.
    
    Controls parallelism, persistence, and failure handling.
    """
    
    # Maximum parallel agents
    max_parallel_agents: int = 5
    
    # Task timeout (seconds)
    task_timeout: int = 600
    
    # Whether to persist state for recovery
    persist_state: bool = True
    
    # State file location
    state_file: str = ".agent_state.json"
    
    # Whether to continue on task failure
    continue_on_failure: bool = False
    
    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []
        if self.max_parallel_agents < 1:
            errors.append("max_parallel_agents must be at least 1")
        if self.task_timeout < 10:
            errors.append("task_timeout should be at least 10 seconds")
        return errors


@dataclass
class LoggingConfig:
    """
    Configuration for logging and telemetry.
    """
    
    # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    level: str = "INFO"
    
    # Log file path (None = console only)
    file: Optional[str] = None
    
    # JSON output format
    json_format: bool = False
    
    # Enable telemetry/metrics
    enable_telemetry: bool = True
    
    # Enable audit logging
    enable_audit: bool = True
    
    def validate(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            errors.append(f"level must be one of {valid_levels}")
        return errors


@dataclass
class Config:
    """
    Main configuration container for the Counsel platform.
    
    Example:
        # Load from environment
        config = Config.from_env()
        
        # Create with custom settings
        config = Config(
            llm=LLMConfig(model_name="Qwen/Qwen2.5-14B-Instruct"),
            verification=VerificationConfig(enabled=True)
        )
    """
    
    llm: LLMConfig = field(default_factory=LLMConfig)
    shell: ShellConfig = field(default_factory=ShellConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Legacy fields for backward compatibility
    verbose: bool = False
    log_file: Optional[str] = None
    
    # Debug mode - shows agent thinking, LLM responses, shell commands
    debug: bool = True  # ON by default
    
    def validate(self) -> List[str]:
        """
        Validate all configuration sections.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        errors.extend(self.llm.validate())
        errors.extend(self.shell.validate())
        errors.extend(self.agent.validate())
        errors.extend(self.execution.validate())
        errors.extend(self.verification.validate())
        errors.extend(self.supervisor.validate())
        errors.extend(self.logging.validate())
        return errors
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0
    
    @classmethod
    def from_env(cls) -> "Config":
        """
        Create config from environment variables.
        
        Supports both COUNSEL_* and legacy AGENT_* prefixes.
        """
        config = cls()
        
        # LLM settings (support both prefixes)
        if model := os.getenv("COUNSEL_MODEL") or os.getenv("AGENT_LLM_MODEL"):
            config.llm.model_name = model
        if device := os.getenv("COUNSEL_DEVICE") or os.getenv("AGENT_LLM_DEVICE"):
            config.llm.device = device
        
        # Execution settings
        if max_parallel := os.getenv("COUNSEL_MAX_PARALLEL") or os.getenv("AGENT_MAX_PARALLEL"):
            config.execution.max_parallel_agents = int(max_parallel)
        
        # Quantization
        if os.getenv("COUNSEL_NO_QUANTIZE", os.getenv("AGENT_NO_QUANTIZE", "")).lower() in ("1", "true", "yes"):
            config.llm.load_in_4bit = False
            config.llm.load_in_8bit = False
        
        # Verbosity
        if os.getenv("COUNSEL_VERBOSE", os.getenv("AGENT_VERBOSE", "")).lower() in ("1", "true", "yes"):
            config.verbose = True
        
        # Debug mode
        if os.getenv("COUNSEL_DEBUG", os.getenv("AGENT_DEBUG", "")).lower() in ("1", "true", "yes"):
            config.debug = True
        
        # Verification
        if os.getenv("COUNSEL_VERIFY", "").lower() in ("1", "true", "yes"):
            config.verification.enabled = True
        
        # Logging
        if log_file := os.getenv("COUNSEL_LOG_FILE"):
            config.logging.file = log_file
            config.log_file = log_file  # Legacy
        if log_level := os.getenv("COUNSEL_LOG_LEVEL"):
            config.logging.level = log_level.upper()
        
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
            ),
            verification=VerificationConfig(
                enabled=False,
            )
        )
    
    @classmethod
    def for_production(cls) -> "Config":
        """Create a production-optimized config."""
        return cls(
            llm=LLMConfig(
                model_name="Qwen/Qwen2.5-7B-Instruct",
                load_in_4bit=True,
                temperature=0.5,  # Lower for more consistent output
            ),
            execution=ExecutionConfig(
                max_parallel_agents=3,
                task_timeout=600,
                persist_state=True,
                continue_on_failure=False,
            ),
            verification=VerificationConfig(
                enabled=True,
                max_retries=2,
            ),
            logging=LoggingConfig(
                level="INFO",
                enable_telemetry=True,
                enable_audit=True,
            ),
            debug=False,
        )


# Global default config
_DEFAULT_CONFIG: Optional[Config] = None


def get_config() -> Config:
    """
    Get the current global configuration.
    
    Returns the existing config or creates a new one from environment variables.
    """
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = Config.from_env()
    return _DEFAULT_CONFIG


def set_config(config: Config) -> None:
    """
    Set the global configuration.
    
    Args:
        config: Configuration to use globally
        
    Raises:
        ValueError: If configuration is invalid
    """
    global _DEFAULT_CONFIG
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {', '.join(errors)}")
    _DEFAULT_CONFIG = config


def reset_config() -> None:
    """Reset the global configuration to default."""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = None
