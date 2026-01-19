"""Model catalog for the Agent Orchestration System."""

from dataclasses import dataclass
from typing import List, Optional
import os
import json


@dataclass
class ModelInfo:
    """Information about a model."""
    id: str  # HuggingFace model ID
    name: str  # Display name
    family: str  # Model family (Qwen, Llama, Mistral, etc.)
    size: str  # Parameter count (1.5B, 7B, etc.)
    ram_fp16: str  # RAM needed for fp16
    ram_4bit: str  # RAM needed for 4-bit quantization
    vram_fp16: str  # VRAM needed for fp16
    vram_4bit: str  # VRAM needed for 4-bit quantization
    context_length: int  # Context window size
    description: str  # Brief description
    recommended_for: List[str]  # What it's good for
    requires_trust_remote: bool = True
    supports_4bit: bool = True
    supports_8bit: bool = True
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "size": self.size,
            "ram_fp16": self.ram_fp16,
            "ram_4bit": self.ram_4bit,
            "vram_fp16": self.vram_fp16,
            "vram_4bit": self.vram_4bit,
            "context_length": self.context_length,
            "description": self.description,
            "recommended_for": self.recommended_for,
        }


# Model Catalog - curated list of recommended models
MODEL_CATALOG: List[ModelInfo] = [
    # ===== QWEN 2.5 FAMILY =====
    ModelInfo(
        id="Qwen/Qwen2.5-0.5B-Instruct",
        name="Qwen 2.5 0.5B",
        family="Qwen",
        size="0.5B",
        ram_fp16="2 GB",
        ram_4bit="1 GB",
        vram_fp16="1.5 GB",
        vram_4bit="0.8 GB",
        context_length=32768,
        description="Ultra-lightweight, fast responses. Good for simple tasks.",
        recommended_for=["low-memory", "fast", "simple-tasks"],
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-1.5B-Instruct",
        name="Qwen 2.5 1.5B",
        family="Qwen",
        size="1.5B",
        ram_fp16="4 GB",
        ram_4bit="2 GB",
        vram_fp16="3.5 GB",
        vram_4bit="1.5 GB",
        context_length=32768,
        description="Lightweight but capable. Great balance of speed and quality.",
        recommended_for=["low-memory", "fast", "general"],
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-3B-Instruct",
        name="Qwen 2.5 3B",
        family="Qwen",
        size="3B",
        ram_fp16="7 GB",
        ram_4bit="3 GB",
        vram_fp16="6.5 GB",
        vram_4bit="2.5 GB",
        context_length=32768,
        description="Good reasoning, reasonable speed. Excellent for coding tasks.",
        recommended_for=["coding", "general", "balanced"],
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-7B-Instruct",
        name="Qwen 2.5 7B ⭐",
        family="Qwen",
        size="7B",
        ram_fp16="16 GB",
        ram_4bit="6 GB",
        vram_fp16="15 GB",
        vram_4bit="5 GB",
        context_length=32768,
        description="Recommended default. Strong reasoning and coding abilities.",
        recommended_for=["recommended", "coding", "reasoning", "general"],
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-14B-Instruct",
        name="Qwen 2.5 14B",
        family="Qwen",
        size="14B",
        ram_fp16="30 GB",
        ram_4bit="10 GB",
        vram_fp16="28 GB",
        vram_4bit="9 GB",
        context_length=32768,
        description="High quality reasoning. Needs more resources.",
        recommended_for=["high-quality", "complex-tasks", "reasoning"],
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-32B-Instruct",
        name="Qwen 2.5 32B",
        family="Qwen",
        size="32B",
        ram_fp16="68 GB",
        ram_4bit="20 GB",
        vram_fp16="65 GB",
        vram_4bit="18 GB",
        context_length=32768,
        description="Top-tier quality. Requires high-end GPU.",
        recommended_for=["best-quality", "complex-reasoning"],
    ),
    
    # ===== QWEN 2.5 CODER FAMILY =====
    ModelInfo(
        id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        name="Qwen 2.5 Coder 1.5B",
        family="Qwen-Coder",
        size="1.5B",
        ram_fp16="4 GB",
        ram_4bit="2 GB",
        vram_fp16="3.5 GB",
        vram_4bit="1.5 GB",
        context_length=32768,
        description="Code-specialized, lightweight. Fast code generation.",
        recommended_for=["coding", "fast", "low-memory"],
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-Coder-7B-Instruct",
        name="Qwen 2.5 Coder 7B ⭐",
        family="Qwen-Coder",
        size="7B",
        ram_fp16="16 GB",
        ram_4bit="6 GB",
        vram_fp16="15 GB",
        vram_4bit="5 GB",
        context_length=32768,
        description="Excellent for coding tasks. Specialized training on code.",
        recommended_for=["recommended", "coding", "programming"],
    ),
    ModelInfo(
        id="Qwen/Qwen2.5-Coder-14B-Instruct",
        name="Qwen 2.5 Coder 14B",
        family="Qwen-Coder",
        size="14B",
        ram_fp16="30 GB",
        ram_4bit="10 GB",
        vram_fp16="28 GB",
        vram_4bit="9 GB",
        context_length=32768,
        description="Premium code model. Complex refactoring and architecture.",
        recommended_for=["coding", "high-quality", "complex-code"],
    ),
    
    # ===== LLAMA 3.2 FAMILY =====
    ModelInfo(
        id="meta-llama/Llama-3.2-1B-Instruct",
        name="Llama 3.2 1B",
        family="Llama",
        size="1B",
        ram_fp16="3 GB",
        ram_4bit="1.5 GB",
        vram_fp16="2.5 GB",
        vram_4bit="1 GB",
        context_length=128000,
        description="Meta's compact model. Long context window.",
        recommended_for=["low-memory", "long-context"],
        requires_trust_remote=False,
    ),
    ModelInfo(
        id="meta-llama/Llama-3.2-3B-Instruct",
        name="Llama 3.2 3B",
        family="Llama",
        size="3B",
        ram_fp16="7 GB",
        ram_4bit="3 GB",
        vram_fp16="6.5 GB",
        vram_4bit="2.5 GB",
        context_length=128000,
        description="Excellent long-context support. Good general performance.",
        recommended_for=["long-context", "general"],
        requires_trust_remote=False,
    ),
    
    # ===== MISTRAL FAMILY =====
    ModelInfo(
        id="mistralai/Mistral-7B-Instruct-v0.3",
        name="Mistral 7B v0.3",
        family="Mistral",
        size="7B",
        ram_fp16="16 GB",
        ram_4bit="6 GB",
        vram_fp16="15 GB",
        vram_4bit="5 GB",
        context_length=32768,
        description="Strong open model. Good reasoning and instruction following.",
        recommended_for=["reasoning", "general"],
        requires_trust_remote=False,
    ),
    
    # ===== DEEPSEEK FAMILY =====
    ModelInfo(
        id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        name="DeepSeek Coder V2 Lite",
        family="DeepSeek",
        size="16B (2.4B active)",
        ram_fp16="8 GB",
        ram_4bit="4 GB",
        vram_fp16="7 GB",
        vram_4bit="3.5 GB",
        context_length=128000,
        description="MoE architecture. Efficient coding model.",
        recommended_for=["coding", "efficient", "long-context"],
    ),
    
    # ===== PHI FAMILY =====
    ModelInfo(
        id="microsoft/Phi-3.5-mini-instruct",
        name="Phi 3.5 Mini",
        family="Phi",
        size="3.8B",
        ram_fp16="9 GB",
        ram_4bit="4 GB",
        vram_fp16="8 GB",
        vram_4bit="3 GB",
        context_length=128000,
        description="Microsoft's compact powerhouse. Great reasoning for size.",
        recommended_for=["reasoning", "efficient", "long-context"],
    ),
    
    # ===== GEMMA FAMILY =====
    ModelInfo(
        id="google/gemma-2-2b-it",
        name="Gemma 2 2B",
        family="Gemma",
        size="2B",
        ram_fp16="5 GB",
        ram_4bit="2 GB",
        vram_fp16="4.5 GB",
        vram_4bit="1.5 GB",
        context_length=8192,
        description="Google's efficient model. Good instruction following.",
        recommended_for=["efficient", "general"],
        requires_trust_remote=False,
    ),
    ModelInfo(
        id="google/gemma-2-9b-it",
        name="Gemma 2 9B",
        family="Gemma",
        size="9B",
        ram_fp16="20 GB",
        ram_4bit="7 GB",
        vram_fp16="19 GB",
        vram_4bit="6 GB",
        context_length=8192,
        description="Strong performance from Google. Quality responses.",
        recommended_for=["high-quality", "general"],
        requires_trust_remote=False,
    ),
]


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """Get model info by HuggingFace ID."""
    for model in MODEL_CATALOG:
        if model.id == model_id:
            return model
    return None


def get_models_by_family(family: str) -> List[ModelInfo]:
    """Get all models from a specific family."""
    return [m for m in MODEL_CATALOG if m.family.lower() == family.lower()]


def get_models_by_tag(tag: str) -> List[ModelInfo]:
    """Get models recommended for a specific use case."""
    return [m for m in MODEL_CATALOG if tag.lower() in [t.lower() for t in m.recommended_for]]


def get_recommended_models() -> List[ModelInfo]:
    """Get the recommended models (marked with star)."""
    return get_models_by_tag("recommended")


def get_families() -> List[str]:
    """Get list of all model families."""
    return sorted(set(m.family for m in MODEL_CATALOG))


def estimate_system_resources() -> dict:
    """Estimate available system resources."""
    import shutil
    
    result = {
        "ram_gb": 0,
        "vram_gb": 0,
        "has_cuda": False,
        "has_mps": False,
        "recommended_quantization": "4bit",
    }
    
    # Check RAM
    try:
        import psutil
        result["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        # Fallback: try to read /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        result["ram_gb"] = round(kb / (1024**2), 1)
                        break
        except:
            result["ram_gb"] = 8  # Assume 8GB as fallback
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            result["has_cuda"] = True
            result["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["has_mps"] = True
            # MPS uses unified memory, so VRAM = RAM
            result["vram_gb"] = result["ram_gb"]
    except ImportError:
        pass
    
    # Determine recommended quantization
    if result["vram_gb"] >= 24:
        result["recommended_quantization"] = "fp16"
    elif result["vram_gb"] >= 8:
        result["recommended_quantization"] = "8bit"
    else:
        result["recommended_quantization"] = "4bit"
    
    return result


def get_suitable_models(ram_gb: float, vram_gb: float, use_4bit: bool = True) -> List[ModelInfo]:
    """Get models that fit within the available memory."""
    suitable = []
    for model in MODEL_CATALOG:
        # Parse memory requirement
        if use_4bit:
            req = model.vram_4bit if vram_gb > 0 else model.ram_4bit
        else:
            req = model.vram_fp16 if vram_gb > 0 else model.ram_fp16
        
        # Parse the requirement string (e.g., "6 GB" -> 6.0)
        try:
            req_gb = float(req.split()[0])
        except:
            req_gb = 999
        
        available = vram_gb if vram_gb > 0 else ram_gb
        
        # Leave some headroom (80% of available)
        if req_gb <= available * 0.8:
            suitable.append(model)
    
    return suitable


# Config file for persisting model selection
CONFIG_FILE = os.path.expanduser("~/.counsel_model_config.json")


def save_model_selection(model_id: str, quantization: str = "4bit") -> None:
    """Save the selected model to config file."""
    config = {
        "model_id": model_id,
        "quantization": quantization,
        "saved_at": __import__("datetime").datetime.now().isoformat(),
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except:
        pass  # Silently fail if can't write


def load_model_selection() -> Optional[dict]:
    """Load previously selected model from config file."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return None


def clear_model_selection() -> None:
    """Clear the saved model selection."""
    try:
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
    except:
        pass
