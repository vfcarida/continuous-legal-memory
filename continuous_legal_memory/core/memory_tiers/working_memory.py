"""
Working Memory Component.

Provides a sliding-window cache for short-term contextual tracking during active legal reasoning sessions,
enforcing multi-tenant isolation.
"""

from __future__ import annotations

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
        Enforces tenant partitioning to prevent cross-tenant contextual bleeding.
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

    def add(
        self,
        text: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        metadata: dict | None = None,
        tenant_id: str | None = None,
    ) -> MemoryRecord:
        """
        Add a new short-term interaction record to Working Memory.

        Args:
            text: Text snippet of the turn or active legal context.
            key_vector: Semantic vector representation.
            value_vector: Action target decision vector.
            metadata: Optional metadata dictionary.
            tenant_id: Optional tenant identifier. If omitted, checks metadata['tenant_id']
                       or defaults to 'default'.

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

        tenant = (
            tenant_id
            or (metadata.get("tenant_id") if metadata else None)
            or (metadata.get("tenant") if metadata else None)
            or "default"
        )

        record = MemoryRecord(
            text=text,
            key_vector=key_vector,
            value_vector=value_vector,
            tier=MemoryTier.WORKING,
            metadata=metadata or {},
            tenant_id=tenant,
        )
        self._records.append(record)
        return record

    def get_active_context(
        self,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[MemoryRecord]:
        """
        Retrieve all non-expired memory records currently active in Working Memory,
        optionally filtered by tenant namespace.

        Args:
            at_time: Datetime timestamp to evaluate validity against.
            tenant_id: Optional tenant identifier to enforce multi-tenant isolation.

        Returns:
            List of valid `MemoryRecord` instances.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        return [
            rec
            for rec in self._records
            if rec.is_temporally_valid(eval_time)
            and (tenant_id is None or rec.tenant_id == tenant_id)
        ]

    def clear(self, tenant_id: str | None = None) -> None:
        """
        Purge working memory records, optionally restricted to a specific tenant.

        Args:
            tenant_id: Optional tenant identifier. If None, purges entire working memory.
        """
        if tenant_id is None:
            self._records.clear()
        else:
            self._records = deque(rec for rec in self._records if rec.tenant_id != tenant_id)

    def remove_by_text(self, text: str, tenant_id: str | None = None) -> None:
        """
        Purge working memory records matching the provided text and optional tenant filter.

        Args:
            text: Text snippet to match for removal.
            tenant_id: Optional tenant identifier.
        """
        self._records = deque(
            rec
            for rec in self._records
            if not (rec.text == text and (tenant_id is None or rec.tenant_id == tenant_id))
        )

    def __len__(self) -> int:
        return len(self._records)

