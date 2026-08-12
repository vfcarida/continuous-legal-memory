"""
Adapters package providing infrastructure implementations of domain interfaces.
"""

from continuous_legal_memory.adapters.encoders import HuggingFaceEncoderAdapter
from continuous_legal_memory.adapters.ollama import OllamaGemmaAdapter

__all__ = ["HuggingFaceEncoderAdapter", "OllamaGemmaAdapter"]
