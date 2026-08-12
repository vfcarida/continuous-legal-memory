"""
Domain package containing core entities, value objects, domain interfaces, and exceptions.
"""

from continuous_legal_memory.domain.exceptions import (
    ContextWindowExceededError,
    EncoderInferenceError,
    InvalidMemoryVectorError,
    LegalMemoryError,
    MemoryContradictionError,
    TemporalInvalidationError,
)
from continuous_legal_memory.domain.interfaces import BaseEncoderPort, BaseMemoryStorePort
from continuous_legal_memory.domain.models import MemoryRecord, PredictionResult

__all__ = [
    "LegalMemoryError",
    "ContextWindowExceededError",
    "MemoryContradictionError",
    "TemporalInvalidationError",
    "EncoderInferenceError",
    "InvalidMemoryVectorError",
    "BaseEncoderPort",
    "BaseMemoryStorePort",
    "MemoryRecord",
    "PredictionResult",
]
