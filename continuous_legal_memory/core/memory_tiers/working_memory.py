"""
Working Memory Component.

Provides a sliding-window cache for short-term contextual tracking during active legal reasoning sessions.
"""

from collections import deque
from datetime import datetime, timezone

import torch

from continuous_legal_memory.domain.exceptions import ContextWindowExceededError
from continuous_legal_memory.domain.models import MemoryRecord, MemoryTier


class WorkingMemory:
    """
    Working Memory sliding-window context cache.

    Rationale:
        Manages short-term contextual turn tracking for active legal interactions.
        When session turns exceed capacity bounds, enforces sliding-window eviction or throws
        `ContextWindowExceededError` depending on strict capacity policy flags.
    """

    def __init__(self, capacity: int = 10, strict_capacity: bool = False) -> None:
        """
        Initialize WorkingMemory.

        Args:
            capacity: Maximum number of active context items held in memory.
            strict_capacity: If True, raises `ContextWindowExceededError` when capacity is breached.
                             If False, automatically evicts oldest items in FIFO order.
        """
        self.capacity = capacity
        self.strict_capacity = strict_capacity
        self._records: deque[MemoryRecord] = deque()

    def add(self, text: str, key_vector: torch.Tensor, value_vector: torch.Tensor, metadata: dict | None = None) -> MemoryRecord:
        """
        Add a new short-term interaction record to Working Memory.

        Args:
            text: Text snippet of the turn or active legal context.
            key_vector: Semantic vector representation.
            value_vector: Action target decision vector.
            metadata: Optional metadata dictionary.

        Returns:
            The created `MemoryRecord`.

        Raises:
            ContextWindowExceededError: If strict capacity is enabled and capacity is exceeded.
        """
        if len(self._records) >= self.capacity:
            if self.strict_capacity:
                raise ContextWindowExceededError(
                    f"Working memory capacity of {self.capacity} exceeded.",
                    payload={"current_size": len(self._records), "capacity": self.capacity},
                )
            # FIFO sliding window eviction
            self._records.popleft()

        record = MemoryRecord(
            text=text,
            key_vector=key_vector,
            value_vector=value_vector,
            tier=MemoryTier.WORKING,
            metadata=metadata or {},
        )
        self._records.append(record)
        return record

    def get_active_context(self, at_time: datetime | None = None) -> list[MemoryRecord]:
        """
        Retrieve all non-expired memory records currently active in Working Memory.

        Args:
            at_time: Datetime timestamp to evaluate validity against.

        Returns:
            List of valid `MemoryRecord` instances.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        return [rec for rec in self._records if rec.is_temporally_valid(eval_time)]

    def clear(self) -> None:
        """Purge all working memory records."""
        self._records.clear()

    def __len__(self) -> int:
        return len(self._records)
