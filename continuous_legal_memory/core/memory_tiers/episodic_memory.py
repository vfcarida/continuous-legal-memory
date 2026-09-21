"""
Episodic Memory Component.

Implements an immutable, timestamped chronological ledger for auditing legal interactions,
documents parsed, and regulatory updates with cryptographic hash integrity verification.
"""

import hashlib
from datetime import datetime, timezone

import torch

from continuous_legal_memory.domain.exceptions import TemporalInvalidationError
from continuous_legal_memory.domain.interfaces import BaseMemoryStorePort
from continuous_legal_memory.domain.models import MemoryRecord, MemoryTier


class EpisodicMemory(BaseMemoryStorePort):
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
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        transaction_time: datetime | None = None,
        personal_data: bool = False,
        tenant_id: str = "default",
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
            authority_rank: Legal authority rank integer (higher = higher authority).
            jurisdiction: Jurisdictional scope identifier.
            transaction_time: System ingestion timestamp.
            personal_data: Whether this record contains personal data.
            tenant_id: Tenant or workspace identifier for multi-tenant isolation.

        Returns:
            The created and appended `MemoryRecord`.
        """
        timestamp = valid_from or datetime.now(timezone.utc)
        trans_time = transaction_time or datetime.now(timezone.utc)
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
            authority_rank=authority_rank,
            jurisdiction=jurisdiction,
            transaction_time=trans_time,
            personal_data=personal_data,
            tenant_id=tenant_id,
        )

        self._ledger.append(record)
        self._hash_chain.append(record_id)
        return record

    def get_valid_records(
        self,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[MemoryRecord]:
        """
        Retrieve all temporally valid episodic records at the specified timestamp.

        Args:
            at_time: Datetime timestamp to evaluate validity against.
            tenant_id: Optional tenant identifier to enforce multi-tenant isolation.

        Returns:
            List of valid `MemoryRecord` instances.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        return [
            rec for rec in self._ledger
            if rec.is_temporally_valid(eval_time) and (tenant_id is None or rec.tenant_id == tenant_id)
        ]

    def get_valid_indices(
        self,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[int]:
        """
        Retrieve ledger indices of all temporally valid episodic records at the specified timestamp.

        Args:
            at_time: Datetime timestamp to evaluate validity against.
            tenant_id: Optional tenant identifier to enforce multi-tenant isolation.

        Returns:
            List of integer indices corresponding to valid records in the ledger.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        return [
            i for i, rec in enumerate(self._ledger)
            if rec.is_temporally_valid(eval_time) and (tenant_id is None or rec.tenant_id == tenant_id)
        ]

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

    def add_record(self, record: MemoryRecord) -> None:
        """
        Store a new legal memory record in the persistent index (BaseMemoryStorePort).

        Args:
            record: MemoryRecord to append to the episodic ledger.
        """
        if record.record_id is None:
            record.record_id = self._generate_record_hash(record.text, record.valid_from)
        self._ledger.append(record)
        self._hash_chain.append(record.record_id)

    def get_records(self, tenant_id: str | None = None) -> list[MemoryRecord]:
        """
        Retrieve all currently registered memory records (BaseMemoryStorePort).

        Args:
            tenant_id: Optional tenant identifier to filter records by tenant.

        Returns:
            List of MemoryRecord instances in the ledger.
        """
        if tenant_id is not None:
            return [rec for rec in self._ledger if rec.tenant_id == tenant_id]
        return list(self._ledger)

    def clear(self) -> None:
        """Purge all stored memory records (BaseMemoryStorePort)."""
        self._ledger.clear()
        self._hash_chain.clear()

    def record_tombstone(self, record_id: str, timestamp: datetime | None = None) -> str:
        """
        Record a cryptographic tombstone entry in the episodic hash chain.

        Args:
            record_id: Identifier of the erased record.
            timestamp: Invalidation/deletion timestamp.

        Returns:
            The SHA-256 tombstone digest string appended to the hash chain.
        """
        ts = timestamp or datetime.now(timezone.utc)
        prev_hash = self._hash_chain[-1] if self._hash_chain else "GENESIS"
        payload = f"TOMBSTONE:{prev_hash}:{ts.isoformat()}:{record_id}".encode()
        tombstone_hash = hashlib.sha256(payload).hexdigest()
        self._hash_chain.append(tombstone_hash)
        return tombstone_hash

    def delete_record(self, record_id: str, timestamp: datetime | None = None) -> MemoryRecord:
        """
        Delete a record from the ledger and record an immutable tombstone in the hash chain.

        Args:
            record_id: Identifier of record to delete.
            timestamp: Erasure timestamp.

        Returns:
            The deleted MemoryRecord instance.

        Raises:
            KeyError: If record_id is not found in the ledger.
        """
        for i, rec in enumerate(self._ledger):
            if rec.record_id == record_id:
                deleted_rec = self._ledger.pop(i)
                self.record_tombstone(record_id, timestamp)
                return deleted_rec
        raise KeyError(f"Record with ID '{record_id}' not found in episodic ledger.")

