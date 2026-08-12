"""
Unit tests for domain models, exception hierarchy, and validation rules.
"""

from datetime import datetime, timedelta, timezone

import torch

from continuous_legal_memory.domain.exceptions import (
    ContextWindowExceededError,
    EncoderInferenceError,
    InvalidMemoryVectorError,
    LegalMemoryError,
    MemoryContradictionError,
    TemporalInvalidationError,
)
from continuous_legal_memory.domain.models import MemoryRecord, PredictionResult


def test_exception_hierarchy() -> None:
    """Verify that domain exceptions inherit cleanly from base LegalMemoryError."""
    err = InvalidMemoryVectorError("Dimension mismatch", payload={"expected": 768})
    assert isinstance(err, LegalMemoryError)
    assert "Dimension mismatch" in str(err)
    assert err.payload == {"expected": 768}

    assert issubclass(ContextWindowExceededError, LegalMemoryError)
    assert issubclass(MemoryContradictionError, LegalMemoryError)
    assert issubclass(TemporalInvalidationError, LegalMemoryError)
    assert issubclass(EncoderInferenceError, LegalMemoryError)


def test_memory_record_temporal_validity() -> None:
    """Verify memory record temporal validity logic."""
    now = datetime.now(timezone.utc)
    past = now - timedelta(days=10)
    future = now + timedelta(days=10)

    # Currently valid record
    record = MemoryRecord(
        text="Sample rule",
        key_vector=torch.zeros(1, 10),
        value_vector=torch.zeros(1, 2),
        valid_from=past,
        valid_to=future,
    )
    assert record.is_temporally_valid(now) is True

    # Expired record
    expired_record = MemoryRecord(
        text="Expired rule",
        key_vector=torch.zeros(1, 10),
        value_vector=torch.zeros(1, 2),
        valid_from=past,
        valid_to=past + timedelta(days=5),
    )
    assert expired_record.is_temporally_valid(now) is False


def test_prediction_result_initialization() -> None:
    """Verify PredictionResult DTO structure."""
    res = PredictionResult(
        query="Test query",
        predicted_action_vector=[0.8, 0.2],
        most_relevant_rule="Rule 1",
        confidence=0.95,
    )
    assert res.query == "Test query"
    assert res.predicted_action_vector == [0.8, 0.2]
    assert res.confidence == 0.95
