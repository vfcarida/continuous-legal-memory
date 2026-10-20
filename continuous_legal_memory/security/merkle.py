"""
Sparse Merkle Tree (SMT) and Append-Only Merkle Log for Verifiable GDPR Art. 17 Erasure Audits.

Implements a 256-bit cryptographically domain-separated Sparse Merkle Tree (RFC 6962 leaf/node
separation) supporting:
1. O(256) dynamic leaf insertion and physical erasure (`remove`).
2. Succinct Inclusion Proofs (`is_member=True`) for active normative rules.
3. Succinct Non-Membership / Erasure Proofs (`is_member=False`) proving that a deleted
   `rule_id` maps to the canonical empty leaf (`0x00...00`) under the signed state root,
   without revealing the contents or identifiers of other retained rules in memory.
4. Append-Only Merkle Audit Log chaining every state root transition (`INGEST` / `ERASE`).
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from continuous_legal_memory.security import ed25519
from continuous_legal_memory.security.pq_hybrid import (
    PQ_HYBRID_ALGORITHM,
    pq_hybrid_sign,
    pq_hybrid_verify,
)

TREE_DEPTH = 256
EMPTY_LEAF_BYTES = b"\x00" * 32
EMPTY_LEAF_HEX = EMPTY_LEAF_BYTES.hex()


def _hash_leaf(rule_id: str, payload_bytes: bytes) -> bytes:
    """Domain-separated leaf digest (0x00 prefix per RFC 6962)."""
    h = hashlib.sha256()
    h.update(b"\x00")
    h.update(rule_id.encode("utf-8"))
    h.update(b":")
    h.update(payload_bytes)
    return h.digest()


def _hash_internal(left: bytes, right: bytes) -> bytes:
    """Domain-separated internal node digest (0x01 prefix per RFC 6962)."""
    h = hashlib.sha256()
    h.update(b"\x01")
    h.update(left)
    h.update(right)
    return h.digest()


def _precompute_default_hashes() -> list[bytes]:
    """
    Precompute default empty-subtree hashes for depths 0..256.
    Depth 0 is leaf level; Depth 256 is the root.
    """
    defaults = [EMPTY_LEAF_BYTES]
    for _ in range(TREE_DEPTH):
        prev = defaults[-1]
        h = hashlib.sha256()
        h.update(b"\x01")
        h.update(prev)
        h.update(prev)
        defaults.append(h.digest())
    return defaults


DEFAULT_SUBTREE_HASHES: list[bytes] = _precompute_default_hashes()


def rule_id_to_key_int(rule_id: str) -> int:
    """Map an arbitrary string `rule_id` to its deterministic 256-bit integer path."""
    digest = hashlib.sha256(rule_id.encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big")


@dataclass
class SparseMerkleProof:
    """
    Succinct cryptographic inclusion or non-membership proof in a 256-bit Sparse Merkle Tree.

    Attributes:
        rule_id: Target rule identifier being proven present or erased.
        leaf_key_hex: 64-char SHA-256 hex key determining the 256-bit path in the tree.
        leaf_value_hash: 64-char hex leaf hash (`00` * 32 for non-membership / erased rules).
        is_member: True if proving inclusion of an active rule; False if proving erasure/absence.
        root_hash: 64-char hex Sparse Merkle Tree state root.
        siblings: Compact list of `(depth, sibling_hash_hex)` for non-default siblings along the path.
        timestamp: UTC ISO-8601 generation timestamp.
        signature: Optional HMAC-SHA256 or Ed25519 signature over the canonical proof commitment.
        algorithm: Optional signature algorithm identifier ('HMAC-SHA256' | 'Ed25519').
        public_key: Optional hex public key when signed with Ed25519.
    """

    rule_id: str
    leaf_key_hex: str
    leaf_value_hash: str
    is_member: bool
    root_hash: str
    siblings: list[tuple[int, str]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signature: str | None = None
    algorithm: str | None = None
    public_key: str | None = None

    def canonical_commitment(self) -> bytes:
        """Canonical byte string committed to by cryptographic signatures."""
        member_flag = "1" if self.is_member else "0"
        return (
            f"SMT_PROOF|RULE:{self.rule_id}|KEY:{self.leaf_key_hex}|"
            f"LEAF:{self.leaf_value_hash}|MEMBER:{member_flag}|"
            f"ROOT:{self.root_hash}|TS:{self.timestamp}"
        ).encode()

    def to_dict(self) -> dict[str, Any]:
        """Serialize proof into a JSON-compatible dictionary."""
        return {
            "rule_id": self.rule_id,
            "leaf_key_hex": self.leaf_key_hex,
            "leaf_value_hash": self.leaf_value_hash,
            "is_member": self.is_member,
            "root_hash": self.root_hash,
            "siblings": [[int(d), str(h)] for d, h in self.siblings],
            "non_default_sibling_count": len(self.siblings),
            "tree_depth": TREE_DEPTH,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "algorithm": self.algorithm,
            "public_key": self.public_key,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SparseMerkleProof:
        """Deserialize a `SparseMerkleProof` from a dictionary."""
        raw_siblings = data.get("siblings", [])
        siblings = [(int(item[0]), str(item[1])) for item in raw_siblings]
        return cls(
            rule_id=str(data["rule_id"]),
            leaf_key_hex=str(data["leaf_key_hex"]),
            leaf_value_hash=str(data["leaf_value_hash"]),
            is_member=bool(data["is_member"]),
            root_hash=str(data["root_hash"]),
            siblings=siblings,
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            signature=data.get("signature"),
            algorithm=data.get("algorithm"),
            public_key=data.get("public_key"),
        )


@dataclass
class MerkleLogEntry:
    """Single immutable entry in the append-only Merkle state transition log."""

    seq_no: int
    operation: str
    rule_id: str
    smt_root_hash: str
    prev_entry_hash: str
    entry_hash: str
    timestamp: str
    tenant_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq_no": self.seq_no,
            "operation": self.operation,
            "rule_id": self.rule_id,
            "smt_root_hash": self.smt_root_hash,
            "prev_entry_hash": self.prev_entry_hash,
            "entry_hash": self.entry_hash,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
        }


class AppendOnlyMerkleLog:
    """
    Append-only cryptographic state transition log chaining every Sparse Merkle Tree root update.

    Guarantees that historical ingestion and erasure events form a tamper-evident hash chain
    auditable by external regulators.
    """

    def __init__(self) -> None:
        self._entries: list[MerkleLogEntry] = []

    @property
    def entries(self) -> list[MerkleLogEntry]:
        return list(self._entries)

    @property
    def head_hash(self) -> str:
        if not self._entries:
            return "00" * 32
        return self._entries[-1].entry_hash

    def append(
        self,
        operation: str,
        rule_id: str,
        smt_root_hash: str,
        tenant_id: str = "default",
        timestamp: datetime | None = None,
    ) -> MerkleLogEntry:
        """Append a new state transition event and compute its chained SHA-256 digest."""
        seq_no = len(self._entries) + 1
        prev_hash = self.head_hash
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        payload = f"{seq_no}|{operation}|{rule_id}|{smt_root_hash}|{prev_hash}|{ts}|{tenant_id}".encode()
        entry_hash = hashlib.sha256(payload).hexdigest()
        entry = MerkleLogEntry(
            seq_no=seq_no,
            operation=operation,
            rule_id=rule_id,
            smt_root_hash=smt_root_hash,
            prev_entry_hash=prev_hash,
            entry_hash=entry_hash,
            timestamp=ts,
            tenant_id=tenant_id,
        )
        self._entries.append(entry)
        return entry

    def verify_chain(self) -> bool:
        """Verify cryptographic integrity of the entire append-only log from genesis."""
        expected_prev = "00" * 32
        for idx, entry in enumerate(self._entries, start=1):
            if entry.seq_no != idx or entry.prev_entry_hash != expected_prev:
                return False
            payload = (
                f"{entry.seq_no}|{entry.operation}|{entry.rule_id}|"
                f"{entry.smt_root_hash}|{entry.prev_entry_hash}|{entry.timestamp}|{entry.tenant_id}"
            ).encode()
            recomputed = hashlib.sha256(payload).hexdigest()
            if not hmac.compare_digest(recomputed, entry.entry_hash):
                return False
            expected_prev = entry.entry_hash
        return True

    def get_root_as_of(
        self,
        as_of_transaction_time: datetime | str,
        tenant_id: str | None = None,
    ) -> str:
        """
        Reconstruct the historical Sparse Merkle Tree root hash as of `as_of_transaction_time`.

        Args:
            as_of_transaction_time: Transaction-time cutoff (datetime or ISO-8601 string).
            tenant_id: Optional tenant scope filter.

        Returns:
            64-character hex root hash active at `as_of_transaction_time` (or empty tree root if prior to genesis).
        """
        cutoff = (
            datetime.fromisoformat(as_of_transaction_time)
            if isinstance(as_of_transaction_time, str)
            else as_of_transaction_time
        )
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)

        latest_root = DEFAULT_SUBTREE_HASHES[TREE_DEPTH].hex()
        for entry in self._entries:
            if tenant_id is not None and entry.tenant_id != tenant_id:
                continue
            entry_dt = datetime.fromisoformat(entry.timestamp)
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            if entry_dt <= cutoff:
                latest_root = entry.smt_root_hash
        return latest_root


class SparseMerkleTree:
    """
    256-bit Sparse Merkle Tree supporting O(256) updates, deletions, and succinct
    inclusion / non-membership proofs.

    Level 0 stores leaf hashes keyed by 256-bit integer `int.from_bytes(SHA256(rule_id), 'big')`.
    Level 256 stores the single root hash.
    """

    def __init__(self) -> None:
        self._nodes: list[dict[int, bytes]] = [{} for _ in range(TREE_DEPTH + 1)]
        self._active_rules: dict[str, bytes] = {}
        self.audit_log = AppendOnlyMerkleLog()

    @property
    def root_bytes(self) -> bytes:
        """Return current 32-byte root hash."""
        return self._nodes[TREE_DEPTH].get(0, DEFAULT_SUBTREE_HASHES[TREE_DEPTH])

    @property
    def root_hex(self) -> str:
        """Return current 64-character hex root hash."""
        return self.root_bytes.hex()

    @property
    def active_count(self) -> int:
        """Number of currently active (non-erased) rules in the Sparse Merkle Tree."""
        return len(self._active_rules)

    def get_root_as_of(
        self,
        as_of_transaction_time: datetime | str | None = None,
        tenant_id: str | None = None,
    ) -> str:
        """
        Return the Sparse Merkle Tree root hash as of `as_of_transaction_time`
        (or current `root_hex` if `as_of_transaction_time` is None).
        """
        if as_of_transaction_time is None:
            return self.root_hex
        return self.audit_log.get_root_as_of(as_of_transaction_time, tenant_id=tenant_id)

    def _set_leaf(self, key_int: int, leaf_hash: bytes) -> None:
        if leaf_hash == DEFAULT_SUBTREE_HASHES[0]:
            self._nodes[0].pop(key_int, None)
        else:
            self._nodes[0][key_int] = leaf_hash

        curr_idx = key_int
        curr_hash = leaf_hash
        for depth in range(TREE_DEPTH):
            sibling_idx = curr_idx ^ 1
            sibling_hash = self._nodes[depth].get(sibling_idx, DEFAULT_SUBTREE_HASHES[depth])
            if (curr_idx & 1) == 0:
                parent_hash = _hash_internal(curr_hash, sibling_hash)
            else:
                parent_hash = _hash_internal(sibling_hash, curr_hash)

            parent_idx = curr_idx >> 1
            if parent_hash == DEFAULT_SUBTREE_HASHES[depth + 1]:
                self._nodes[depth + 1].pop(parent_idx, None)
            else:
                self._nodes[depth + 1][parent_idx] = parent_hash

            curr_idx = parent_idx
            curr_hash = parent_hash

    def update(
        self,
        rule_id: str,
        payload: bytes | str,
        tenant_id: str = "default",
        record_log: bool = True,
        timestamp: datetime | None = None,
    ) -> str:
        """
        Insert or update a rule leaf in the Sparse Merkle Tree and return the new root hex.
        """
        payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else payload
        leaf_hash = _hash_leaf(rule_id, payload_bytes)
        key_int = rule_id_to_key_int(rule_id)
        self._active_rules[rule_id] = leaf_hash
        self._set_leaf(key_int, leaf_hash)
        root = self.root_hex
        if record_log:
            self.audit_log.append("INGEST", rule_id, root, tenant_id=tenant_id, timestamp=timestamp)
        return root

    def remove(
        self,
        rule_id: str,
        tenant_id: str = "default",
        record_log: bool = True,
        timestamp: datetime | None = None,
    ) -> str:
        """
        Physically erase a rule leaf from the Sparse Merkle Tree (resetting its leaf slot to
        `EMPTY_LEAF_BYTES`) and return the new root hex.
        """
        key_int = rule_id_to_key_int(rule_id)
        self._active_rules.pop(rule_id, None)
        self._set_leaf(key_int, DEFAULT_SUBTREE_HASHES[0])
        root = self.root_hex
        if record_log:
            self.audit_log.append("ERASE", rule_id, root, tenant_id=tenant_id, timestamp=timestamp)
        return root

    def generate_proof(
        self,
        rule_id: str,
        attestor: Any | None = None,
    ) -> SparseMerkleProof:
        """
        Generate a succinct inclusion proof (if `rule_id` is active) or non-membership proof
        (if `rule_id` was erased or never existed).
        """
        key_int = rule_id_to_key_int(rule_id)
        key_hex = f"{key_int:064x}"
        leaf_hash = self._nodes[0].get(key_int, DEFAULT_SUBTREE_HASHES[0])
        is_member = leaf_hash != DEFAULT_SUBTREE_HASHES[0]

        siblings: list[tuple[int, str]] = []
        curr_idx = key_int
        for depth in range(TREE_DEPTH):
            sibling_idx = curr_idx ^ 1
            sibling_hash = self._nodes[depth].get(sibling_idx, DEFAULT_SUBTREE_HASHES[depth])
            if sibling_hash != DEFAULT_SUBTREE_HASHES[depth]:
                siblings.append((depth, sibling_hash.hex()))
            curr_idx >>= 1

        proof = SparseMerkleProof(
            rule_id=rule_id,
            leaf_key_hex=key_hex,
            leaf_value_hash=leaf_hash.hex(),
            is_member=is_member,
            root_hash=self.root_hex,
            siblings=siblings,
        )

        if attestor is not None:
            eff_att = getattr(attestor, "active_attestor", attestor)
            commitment = proof.canonical_commitment()
            if (
                getattr(eff_att, "algorithm", None) == PQ_HYBRID_ALGORITHM
                and getattr(eff_att, "_private_key", None) is not None
            ):
                proof.signature = pq_hybrid_sign(commitment, eff_att._private_key)
                proof.algorithm = PQ_HYBRID_ALGORITHM
                proof.public_key = eff_att.public_key_hex
            elif hasattr(eff_att, "_private_key") and getattr(eff_att, "_private_key", None) is not None:
                sig_bytes = ed25519.sign(commitment, eff_att._private_key)
                proof.signature = sig_bytes.hex()
                proof.algorithm = "Ed25519"
                proof.public_key = eff_att.public_key_hex
            elif hasattr(eff_att, "_secret_key"):
                proof.signature = hmac.new(eff_att._secret_key, commitment, hashlib.sha256).hexdigest()
                proof.algorithm = "HMAC-SHA256"

        return proof

    def prove_non_membership(self, rule_id: str, attestor: Any | None = None) -> SparseMerkleProof:
        """
        Generate a cryptographic non-membership (erasure) proof for `rule_id`.

        Raises:
            ValueError: If `rule_id` is still an active member of the tree.
        """
        proof = self.generate_proof(rule_id, attestor=attestor)
        if proof.is_member:
            raise ValueError(f"Cannot prove non-membership: rule '{rule_id}' is currently active in the tree.")
        return proof

    def prove_inclusion(self, rule_id: str, attestor: Any | None = None) -> SparseMerkleProof:
        """
        Generate a cryptographic inclusion proof for an active `rule_id`.

        Raises:
            KeyError: If `rule_id` is not an active member of the tree.
        """
        proof = self.generate_proof(rule_id, attestor=attestor)
        if not proof.is_member:
            raise KeyError(f"Cannot prove inclusion: rule '{rule_id}' is not present in the tree.")
        return proof


def verify_sparse_merkle_proof(
    proof: SparseMerkleProof | dict[str, Any],
    expected_root_hash: str | None = None,
    attestor: Any | None = None,
) -> bool:
    """
    Standalone cryptographic verifier for Sparse Merkle Tree inclusion and non-membership proofs.

    Verifies:
        1. `leaf_key_hex` matches `SHA256(rule_id)`.
        2. `is_member` is consistent with `leaf_value_hash` (`False` iff `EMPTY_LEAF_HEX`).
        3. Reconstructed 256-level Merkle root matches `proof.root_hash` (and `expected_root_hash` if supplied).
        4. Optional cryptographic signature (`Ed25519` or `HMAC-SHA256`) if present.
    """
    p = SparseMerkleProof.from_dict(proof) if isinstance(proof, dict) else proof

    expected_key_int = rule_id_to_key_int(p.rule_id)
    expected_key_hex = f"{expected_key_int:064x}"
    if not hmac.compare_digest(p.leaf_key_hex, expected_key_hex):
        return False

    if p.is_member and hmac.compare_digest(p.leaf_value_hash, EMPTY_LEAF_HEX):
        return False
    if (not p.is_member) and (not hmac.compare_digest(p.leaf_value_hash, EMPTY_LEAF_HEX)):
        return False

    if expected_root_hash is not None and not hmac.compare_digest(p.root_hash, expected_root_hash):
        return False

    try:
        curr_hash = bytes.fromhex(p.leaf_value_hash)
    except ValueError:
        return False

    sibling_map: dict[int, bytes] = {}
    for depth, hex_str in p.siblings:
        if depth < 0 or depth >= TREE_DEPTH:
            return False
        try:
            sibling_map[depth] = bytes.fromhex(hex_str)
        except ValueError:
            return False

    curr_idx = expected_key_int
    for depth in range(TREE_DEPTH):
        sibling_hash = sibling_map.get(depth, DEFAULT_SUBTREE_HASHES[depth])
        if (curr_idx & 1) == 0:
            curr_hash = _hash_internal(curr_hash, sibling_hash)
        else:
            curr_hash = _hash_internal(sibling_hash, curr_hash)
        curr_idx >>= 1

    if not hmac.compare_digest(curr_hash.hex(), p.root_hash):
        return False

    if p.signature is not None:
        eff_att = getattr(attestor, "active_attestor", attestor) if attestor is not None else None
        commitment = p.canonical_commitment()
        if p.algorithm == PQ_HYBRID_ALGORITHM:
            pk = p.public_key or (getattr(eff_att, "public_key_hex", None) if eff_att else None)
            if not pk:
                return False
            if not pq_hybrid_verify(p.signature, commitment, pk):
                return False
        elif p.algorithm == "Ed25519":
            pk = p.public_key or (getattr(eff_att, "public_key_hex", None) if eff_att else None)
            if not pk:
                return False
            if not ed25519.verify(p.signature, commitment, pk):
                return False
        elif p.algorithm == "HMAC-SHA256" and eff_att is not None and hasattr(eff_att, "_secret_key"):
            expected_sig = hmac.new(eff_att._secret_key, commitment, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(p.signature, expected_sig):
                return False

    return True
