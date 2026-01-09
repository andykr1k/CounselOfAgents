"""Base Agent class and Hugging Face integration for the Counsel of Agents system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Optional Hugging Face imports (only if transformers is installed)
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        pipeline
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class AgentCapability(Enum):
    """Enumeration of agent capabilities for classification."""
    TEXT_ANALYSIS = "text_analysis"
    CODE_GENERATION = "code_generation"
    DATA_PROCESSING = "data_processing"
    RESEARCH = "research"
    WRITING = "writing"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    MATH = "math"
    REASONING = "reasoning"
    FILE_OPERATIONS = "file_operations"


class ModelType(Enum):
    """Types of Hugging Face models."""
    CAUSAL_LM = "causal_lm"  # GPT-style generation
    SEQ2SEQ = "seq2seq"  # BART/T5-style
    CHAT = "chat"  # Conversational models


@dataclass
class ModelConfig:
    """Configuration for a Hugging Face model."""
    model_name: str
    model_type: ModelType
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    system_prompt: Optional[str] = None
    
    @staticmethod
    def preset(name: str, **overrides) -> 'ModelConfig':
        """Get a preset configuration with specialized models."""
        presets = {
            # Code generation models
            "code": ModelConfig(
                "bigcode/starcoderbase-1b",
                ModelType.CAUSAL_LM,
                max_length=2048,
                temperature=0.2,
                system_prompt="You are an expert programmer. Generate clean, well-commented code."
            ),
            "code_small": ModelConfig(
                "bigcode/starcoderbase-1b",
                ModelType.CAUSAL_LM,
                max_length=1024,
                temperature=0.2
            ),
            
            # Writing models
            "writing": ModelConfig(
                "gpt2",
                ModelType.CAUSAL_LM,
                max_length=512,
                temperature=0.8,
                system_prompt="You are a professional writer. Create engaging, well-structured content."
            ),
            "writing_medium": ModelConfig(
                "gpt2-medium",
                ModelType.CAUSAL_LM,
                max_length=1024,
                temperature=0.8
            ),
            
            # Analysis/Evaluation models
            "eval": ModelConfig(
                "gpt2",
                ModelType.CAUSAL_LM,
                max_length=512,
                temperature=0.3,
                system_prompt="You are an expert evaluator. Analyze and provide critical assessment."
            ),
            "analysis": ModelConfig(
                "gpt2",
                ModelType.CAUSAL_LM,
                max_length=512,
                temperature=0.5,
                system_prompt="You are an analytical assistant. Break down problems and provide insights."
            ),
            
            # Reading/Comprehension models
            "reading": ModelConfig(
                "facebook/bart-large-cnn",
                ModelType.SEQ2SEQ,
                max_length=512,
                temperature=0.3,
                system_prompt="You are a reading comprehension expert. Summarize and extract key information."
            ),
            "summarization": ModelConfig(
                "facebook/bart-large-cnn",
                ModelType.SEQ2SEQ,
                max_length=512,
                temperature=0.3
            ),
            
            # Research models
            "research": ModelConfig(
                "gpt2",
                ModelType.CAUSAL_LM,
                max_length=512,
                temperature=0.7,
                system_prompt="You are a research assistant. Provide comprehensive information on topics."
            ),
            
            # Math/Reasoning models
            "math": ModelConfig(
                "gpt2",
                ModelType.CAUSAL_LM,
                max_length=256,
                temperature=0.1,
                system_prompt="You are a math expert. Solve problems step by step."
            ),
            
            # Translation
            "translation": ModelConfig(
                "Helsinki-NLP/opus-mt-en-es",
                ModelType.SEQ2SEQ,
                max_length=512
            ),
            
            # General purpose (small, fast)
            "tiny": ModelConfig("gpt2", ModelType.CAUSAL_LM, max_length=256),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
        config = presets[name]
        for key, value in overrides.items():
            setattr(config, key, value)
        return config


@dataclass
class Task:
    """Represents a task to be executed by an agent."""
    id: str
    description: str
    required_capabilities: List[AgentCapability]
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
    result: Any = None
    status: str = "pending"
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentResult:
    """Result from an agent execution."""
    task_id: str
    agent_id: str
    result: Any
    metadata: Dict[str, Any] = None
    success: bool = True
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """Base class for all specialized agents."""
    
    def __init__(self, agent_id: str, name: str, capabilities: List[AgentCapability], description: str):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier
            name: Human-readable name
            capabilities: List of capabilities this agent has
            description: Description of what this agent specializes in and when to use it
        """
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.description = description
        self.metadata = {}
    
    def can_handle(self, task: Task) -> bool:
        """Check if this agent can handle the given task."""
        return any(cap in self.capabilities for cap in task.required_capabilities)
    
    @abstractmethod
    async def execute(self, task: Task) -> AgentResult:
        """Execute a task and return the result."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "capabilities": [cap.value for cap in self.capabilities],
            "description": self.description,
            "metadata": self.metadata
        }


class HuggingFaceAgent(BaseAgent):
    """Agent that uses Hugging Face models."""
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        capabilities: List[AgentCapability],
        model_config: ModelConfig,
        description: str
    ):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required for HuggingFaceAgent. Install with: pip install transformers torch")
        
        super().__init__(agent_id, name, capabilities, description)
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = self._determine_device()
        self._load_model()
    
    def _determine_device(self) -> str:
        """Determine the best device to use."""
        if self.model_config.device != "auto":
            return self.model_config.device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_config.model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name,
            trust_remote_code=self.model_config.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            "trust_remote_code": self.model_config.trust_remote_code,
            "device_map": "auto" if self.device == "cuda" else None
        }
        
        if self.model_config.load_in_8bit or self.model_config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=self.model_config.load_in_8bit,
                    load_in_4bit=self.model_config.load_in_4bit
                )
            except ImportError:
                print("Warning: bitsandbytes not available, skipping quantization")
        
        if self.model_config.model_type == ModelType.SEQ2SEQ:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_config.model_name, **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_name, **model_kwargs
            )
        
        if model_kwargs.get("device_map") is None:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        task = "text2text-generation" if self.model_config.model_type == ModelType.SEQ2SEQ else "text-generation"
        self.pipeline = pipeline(
            task,
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        print("Model loaded successfully!")
    
    def _format_prompt(self, text: str) -> str:
        """Format prompt with system prompt if available."""
        if self.model_config.system_prompt:
            return f"{self.model_config.system_prompt}\n\n{text}"
        return text
    
    async def _generate(self, prompt: str) -> str:
        """Generate text from the model."""
        formatted = self._format_prompt(prompt)
        
        try:
            result = self.pipeline(
                formatted,
                max_length=self.model_config.max_length,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                do_sample=self.model_config.do_sample,
                return_full_text=False
            )
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict):
                    return result[0].get("generated_text", result[0].get("summary_text", str(result[0])))
                return str(result[0])
            return str(result)
        except Exception as e:
            # Fallback to manual generation
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.model_config.max_length,
                    temperature=self.model_config.temperature,
                    top_p=self.model_config.top_p,
                    do_sample=self.model_config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if generated.startswith(formatted):
                generated = generated[len(formatted):].strip()
            return generated
    
    async def execute(self, task: Task) -> AgentResult:
        """Execute a task using the Hugging Face model."""
        try:
            result = await self._generate(task.description)
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                result=result,
                success=True,
                metadata={"model": self.model_config.model_name, "device": self.device}
            )
        except Exception as e:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                result=None,
                success=False,
                error=str(e),
                metadata={"model": self.model_config.model_name}
            )


# Specialized Agent Classes
class CodingAgent(HuggingFaceAgent):
    """Specialized agent for code generation and programming tasks."""
    
    def __init__(self, model_preset: str = "code", **config_overrides):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        config = ModelConfig.preset(model_preset, **config_overrides)
        super().__init__(
            "coding_agent",
            "Coding Agent",
            [AgentCapability.CODE_GENERATION, AgentCapability.REASONING],
            config,
            "Expert at generating, writing, and analyzing code. Best for programming tasks, code generation, software development, debugging, and implementing algorithms."
        )


class WritingAgent(HuggingFaceAgent):
    """Specialized agent for writing and content creation."""
    
    def __init__(self, model_preset: str = "writing", **config_overrides):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        config = ModelConfig.preset(model_preset, **config_overrides)
        super().__init__(
            "writing_agent",
            "Writing Agent",
            [AgentCapability.WRITING, AgentCapability.TEXT_ANALYSIS],
            config,
            "Specializes in content creation, writing articles, essays, documentation, and creative writing. Best for generating written content, drafting documents, and composing text."
        )


class EvalAgent(HuggingFaceAgent):
    """Specialized agent for evaluation and critical analysis."""
    
    def __init__(self, model_preset: str = "eval", **config_overrides):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        config = ModelConfig.preset(model_preset, **config_overrides)
        super().__init__(
            "eval_agent",
            "Evaluation Agent",
            [AgentCapability.ANALYSIS, AgentCapability.REASONING],
            config,
            "Expert evaluator for critical analysis, pros/cons, assessments, and evaluations. Best for evaluating options, analyzing trade-offs, and providing critical assessments."
        )


class ReadingAgent(HuggingFaceAgent):
    """Specialized agent for reading comprehension and summarization."""
    
    def __init__(self, model_preset: str = "reading", **config_overrides):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        config = ModelConfig.preset(model_preset, **config_overrides)
        super().__init__(
            "reading_agent",
            "Reading Agent",
            [AgentCapability.SUMMARIZATION, AgentCapability.TEXT_ANALYSIS, AgentCapability.QUESTION_ANSWERING],
            config,
            "Specializes in reading comprehension, summarization, and extracting key information from text. Best for summarizing documents, extracting main points, and understanding content."
        )


class ResearchAgent(HuggingFaceAgent):
    """Specialized agent for research and information gathering."""
    
    def __init__(self, model_preset: str = "research", **config_overrides):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        config = ModelConfig.preset(model_preset, **config_overrides)
        super().__init__(
            "research_agent",
            "Research Agent",
            [AgentCapability.RESEARCH, AgentCapability.QUESTION_ANSWERING, AgentCapability.ANALYSIS],
            config,
            "Expert researcher for finding and providing information on topics. Best for research tasks, answering questions, and gathering comprehensive information."
        )


class MathAgent(HuggingFaceAgent):
    """Specialized agent for mathematical problem solving."""
    
    def __init__(self, model_preset: str = "math", **config_overrides):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        config = ModelConfig.preset(model_preset, **config_overrides)
        super().__init__(
            "math_agent",
            "Math Agent",
            [AgentCapability.MATH, AgentCapability.REASONING],
            config,
            "Mathematical problem solver. Best for calculations, solving equations, mathematical reasoning, and numerical analysis."
        )


class TranslationAgent(HuggingFaceAgent):
    """Specialized agent for translation tasks."""
    
    def __init__(self, model_preset: str = "translation", **config_overrides):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        config = ModelConfig.preset(model_preset, **config_overrides)
        super().__init__(
            "translation_agent",
            "Translation Agent",
            [AgentCapability.TRANSLATION],
            config,
            "Language translation specialist. Best for translating text between languages, converting content from one language to another."
        )


class AnalysisAgent(HuggingFaceAgent):
    """Specialized agent for general analysis tasks."""
    
    def __init__(self, model_preset: str = "analysis", **config_overrides):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        config = ModelConfig.preset(model_preset, **config_overrides)
        super().__init__(
            "analysis_agent",
            "Analysis Agent",
            [AgentCapability.ANALYSIS, AgentCapability.TEXT_ANALYSIS],
            config,
            "General analysis expert. Best for analyzing data, trends, patterns, and providing insights from information."
        )


class FileSystemAgent(HuggingFaceAgent):
    """Agent for file system operations that generates and executes shell commands."""
    
    def __init__(self, model_preset: str = "tiny", **config_overrides):
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch are required. Install with: pip install transformers torch")
        
        # Use a small, fast model for command generation
        config = ModelConfig.preset(
            model_preset,
            system_prompt="""You are a shell command generator. Given a task description, generate ONLY the shell command needed to complete it.
            Return ONLY the command, nothing else. Examples:
            - Task: "read file README.md" -> Command: "cat README.md"
            - Task: "grep 'def' in main.py" -> Command: "grep -n 'def' main.py"
            - Task: "list files in current directory" -> Command: "ls -la"
            - Task: "delete file test.txt" -> Command: "rm test.txt"
            - Task: "write 'hello' to file.txt" -> Command: "echo 'hello' > file.txt"
            Generate the command:""",
            max_length=128,
            temperature=0.1,  # Low temperature for deterministic command generation
            **config_overrides
        )
        super().__init__(
            "filesystem_agent",
            "File System Agent",
            [AgentCapability.FILE_OPERATIONS, AgentCapability.DATA_PROCESSING],
            config,
            "File system operations expert. Generates and executes shell commands for file operations like reading, writing, deleting, searching (grep), listing files, and running commands. Best for any file system or shell command tasks."
        )
    
    async def execute(self, task: Task) -> AgentResult:
        """Generate a shell command from the task and execute it."""
        import subprocess
        import os
        
        try:
            # Generate command using the model
            prompt = f"Task: {task.description}\nCommand:"
            generated = await self._generate(prompt)
            
            # Extract command (clean up model output)
            command = generated.strip()
            # Remove any quotes if the model added them
            if command.startswith('"') and command.endswith('"'):
                command = command[1:-1]
            elif command.startswith("'") and command.endswith("'"):
                command = command[1:-1]
            
            # Remove any explanatory text (model might add context)
            # Take first line only, as that should be the command
            command = command.split('\n')[0].strip()
            
            # Safety check: don't allow dangerous commands
            dangerous_commands = ['rm -rf /', 'format', 'dd if=', 'mkfs', 'fdisk']
            if any(danger in command.lower() for danger in dangerous_commands):
                return AgentResult(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    result=f"Error: Potentially dangerous command blocked: {command}",
                    success=False,
                    error="Dangerous command detected",
                    metadata={"generated_command": command, "operation": "filesystem"}
                )
            
            # Execute the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )
            
            # Format result
            if result.returncode == 0:
                output = result.stdout if result.stdout else "Command executed successfully (no output)"
                if result.stderr:
                    output += f"\n\nNote: {result.stderr}"
                
                return AgentResult(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    result=f"Command: {command}\n\nOutput:\n{output}",
                    success=True,
                    metadata={"generated_command": command, "operation": "filesystem"}
                )
            else:
                error_msg = result.stderr if result.stderr else "Command failed with non-zero exit code"
                return AgentResult(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    result=f"Command: {command}\n\nError: {error_msg}",
                    success=False,
                    error=error_msg,
                    metadata={"generated_command": command, "operation": "filesystem"}
                )
        
        except subprocess.TimeoutExpired:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                result="Error: Command execution timed out",
                success=False,
                error="Timeout",
                metadata={"operation": "filesystem"}
            )
        except Exception as e:
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                result=f"Error executing file operation: {str(e)}",
                success=False,
                error=str(e),
                metadata={"operation": "filesystem"}
            )
