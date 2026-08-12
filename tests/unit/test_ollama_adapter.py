"""
Unit tests for Ollama Gemma Edge Adapter and strict privacy mode.
"""

import pytest

from continuous_legal_memory.adapters.ollama import OllamaGemmaAdapter
from continuous_legal_memory.domain.exceptions import EncoderInferenceError


def test_ollama_adapter_strict_privacy_mode() -> None:
    """Verify that strict privacy mode blocks non-local remote URLs."""
    # Valid local loopback URL should succeed
    adapter = OllamaGemmaAdapter(host_url="http://localhost:11434", strict_privacy_mode=True)
    assert adapter.embedding_dim == 768

    # Non-local remote endpoint in strict privacy mode must raise EncoderInferenceError
    with pytest.raises(EncoderInferenceError):
        OllamaGemmaAdapter(host_url="https://api.remote-llm-cloud.com", strict_privacy_mode=True)


def test_ollama_adapter_offline_embedding_generation() -> None:
    """Verify offline fallback vector generation when local Ollama service is unavailable."""
    adapter = OllamaGemmaAdapter(host_url="http://localhost:11434", embedding_dim=128, strict_privacy_mode=True)
    embeds = adapter.get_embedding(["Test offline legal prompt"])

    assert embeds.ndim == 2
    assert embeds.size(0) == 1
    assert embeds.size(1) == 128
