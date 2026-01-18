"""LLM abstraction layer for HuggingFace models (Qwen, etc.)."""

import asyncio
from typing import List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

from counsel.config import LLMConfig, get_config


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str


class LLM:
    """
    LLM interface using HuggingFace transformers.
    
    Thread-safe singleton that can be shared across agents.
    Uses a lock to ensure only one inference runs at a time
    (since GPU memory is shared), but allows true parallelism
    for shell commands.
    """
    
    _instance: Optional["LLM"] = None
    _lock = threading.Lock()
    
    def __new__(cls, config: Optional[LLMConfig] = None):
        """Singleton pattern - only one LLM instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM."""
        if self._initialized:
            return
        
        self.config = config or get_config().llm
        self.model = None
        self.tokenizer = None
        self.device = None
        self._inference_lock = asyncio.Lock()
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
        self._initialized = True
        
    def load(self) -> None:
        """Load the model and tokenizer."""
        if self.model is not None:
            return
        
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch accelerate"
            )
        
        print(f"Loading model: {self.config.model_name}...")
        
        # Determine device
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        quantization_config = None
        if self.device == "cuda":
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if self.device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if model_kwargs.get("device_map") is None and self.device != "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully on {self.device}!")
    
    def _format_messages(self, messages: List[Message]) -> str:
        """Format messages for the model using chat template."""
        message_dicts = [{"role": m.role, "content": m.content} for m in messages]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                message_dicts,
                tokenize=False,
                add_generation_prompt=True
            )
        
        # Fallback formatting
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"<|system|>\n{msg.content}\n"
            elif msg.role == "user":
                formatted += f"<|user|>\n{msg.content}\n"
            elif msg.role == "assistant":
                formatted += f"<|assistant|>\n{msg.content}\n"
        formatted += "<|assistant|>\n"
        return formatted
    
    def _generate_sync(self, prompt: str) -> str:
        """Synchronous generation (runs in thread pool)."""
        import torch
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )
        
        # Move to device
        if self.device != "cpu" and hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        elif self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_length = inputs["input_ids"].shape[1]
        generated = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        return generated.strip()
    
    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        if self.model is None:
            self.load()
        
        # Use lock to prevent concurrent GPU inference
        async with self._inference_lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._thread_pool,
                self._generate_sync,
                prompt
            )
        
        return result
    
    async def chat(self, messages: List[Message]) -> str:
        """Generate a response from chat messages."""
        prompt = self._format_messages(messages)
        return await self.generate(prompt)
    
    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Complete a prompt with optional system context."""
        messages = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=prompt))
        return await self.chat(messages)
    
    def unload(self) -> None:
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        LLM._instance = None
        self._initialized = False


def get_llm(config: Optional[LLMConfig] = None) -> LLM:
    """Get the singleton LLM instance."""
    return LLM(config)
