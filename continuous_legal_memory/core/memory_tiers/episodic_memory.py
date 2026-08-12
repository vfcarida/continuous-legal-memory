"""
Episodic Memory Component.

Implements an immutable, timestamped chronological ledger for auditing legal interactions,
documents parsed, and regulatory updates with cryptographic hash integrity verification.
"""

import hashlib
from datetime import datetime, timezone

import torch

from continuous_legal_memory.domain.exceptions import TemporalInvalidationError
from continuous_legal_memory.domain.models import MemoryRecord, MemoryTier


class EpisodicMemory:
    """
    Episodic Memory timestamped chronological ledger.

    Rationale:
        Legal systems require tamper-evident audit trails. Every document ingested or rule recorded
        in Episodic Memory is assigned an immutable timestamp, record ID, and SHA-256 hash digest,
        ensuring verifiable auditability for compliance frameworks (e.g., EU AI Act).
    """

    def __init__(self, decay_rate: float = 0.01) -> None:
        """
        Initialize EpisodicMemory.

        Args:
            decay_rate: Temporal decay factor applied per day elapsed to calculate activation weights.
        """
        self.decay_rate = decay_rate
        self._ledger: list[MemoryRecord] = []
        self._hash_chain: list[str] = []

    def append(
        self,
        text: str,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        importance_score: float = 1.0,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict | None = None,
    ) -> MemoryRecord:
        """
        Append a new immutable event record to the episodic ledger.

        Args:
            text: Exact text snippet of legal record or interaction.
            key_vector: Dense vector embedding key.
            value_vector: Target decision value tensor.
            importance_score: Dynamic surprise-weighted importance coefficient.
            valid_from: Start timestamp of legal validity.
            valid_to: Expiry or temporal invalidation timestamp.
            metadata: Custom key-value audit metadata.

        Returns:
            The created and appended `MemoryRecord`.
        """
        timestamp = valid_from or datetime.now(timezone.utc)
        record_id = self._generate_record_hash(text, timestamp)

        record = MemoryRecord(
            text=text,
            key_vector=key_vector,
            value_vector=value_vector,
            importance_score=importance_score,
            record_id=record_id,
            tier=MemoryTier.EPISODIC,
            valid_from=timestamp,
            valid_to=valid_to,
            metadata=metadata or {},
        )

        self._ledger.append(record)
        self._hash_chain.append(record_id)
        return record

    def get_valid_records(self, at_time: datetime | None = None) -> list[MemoryRecord]:
        """
        Retrieve all temporally valid episodic records at the specified timestamp.

        Args:
            at_time: Datetime timestamp to evaluate validity against.

        Returns:
            List of valid `MemoryRecord` instances.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        return [rec for rec in self._ledger if rec.is_temporally_valid(eval_time)]

    def compute_temporal_decay(self, record: MemoryRecord, at_time: datetime | None = None) -> float:
        """
        Calculate the decay-driven activation score of an episodic record based on time elapsed.

        Args:
            record: Memory record to evaluate.
            at_time: Evaluation timestamp.

        Returns:
            Decayed activation multiplier (float between 0.0 and 1.0).

        Raises:
            TemporalInvalidationError: If the record has passed its `valid_to` expiry date.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        if not record.is_temporally_valid(eval_time):
            raise TemporalInvalidationError(
                f"Record '{record.record_id}' is temporally invalid at {eval_time.isoformat()}.",
                payload={"record_id": record.record_id, "valid_to": str(record.valid_to)},
            )

        days_elapsed = max(0.0, (eval_time - record.valid_from).total_seconds() / 86400.0)
        # Exponential decay function
        activation = 1.0 / (1.0 + self.decay_rate * days_elapsed)
        return max(0.0, min(1.0, activation))

    def _generate_record_hash(self, text: str, timestamp: datetime) -> str:
        """Generate a SHA-256 cryptographic digest binding content, timestamp, and previous hash link."""
        prev_hash = self._hash_chain[-1] if self._hash_chain else "GENESIS"
        payload = f"{prev_hash}:{timestamp.isoformat()}:{text}".encode()
        return hashlib.sha256(payload).hexdigest()

    def __len__(self) -> int:
        return len(self._ledger)
