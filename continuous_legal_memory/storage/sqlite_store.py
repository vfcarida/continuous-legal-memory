"""
SQLite Memory Store Implementation.

Provides durable, cross-session ACID persistence for:
1. Multi-tier memory ledger (EpisodicMemory with bitemporal timestamps and authority metadata).
2. Cryptographic hash chain audit trails (SHA-256 block digests and tombstone records).
3. Entity-typed Semantic Knowledge Graph nodes and edges.
4. Neural continuum buffers (keys, values, surprise momentum) and MLP model weights (fast_net / slow_net).
"""

from __future__ import annotations

import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from continuous_legal_memory.domain.interfaces import BaseMemoryStorePort
from continuous_legal_memory.domain.models import (
    EntityType,
    GraphEdge,
    GraphNode,
    MemoryRecord,
    MemoryTier,
    RelationType,
)


class SqliteMemoryStore(BaseMemoryStorePort):
    """
    SQLite-backed durable memory store implementing BaseMemoryStorePort.

    Rationale:
        Legal AI agents operate in persistent environments requiring auditable cross-process state.
        SqliteMemoryStore provides full serialization of both cognitive symbolic records (ledger, graph)
        and subsymbolic neural parameters (keys/values buffers and dual-timescale MLP state dicts).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        """
        Initialize SqliteMemoryStore.

        Args:
            db_path: Path to the SQLite database file or ':memory:' for transient in-memory testing.
        """
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Episodic Ledger Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodic_ledger (
                    record_id TEXT PRIMARY KEY,
                    seq INTEGER,
                    text TEXT NOT NULL,
                    key_vector BLOB NOT NULL,
                    value_vector BLOB NOT NULL,
                    importance_score REAL NOT NULL,
                    tier TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    transaction_time TEXT NOT NULL,
                    authority_rank INTEGER NOT NULL,
                    jurisdiction TEXT,
                    personal_data INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL,
                    tenant_id TEXT NOT NULL DEFAULT 'default'
                )
            """)
            # Migration check for existing databases
            cursor.execute("PRAGMA table_info(episodic_ledger)")
            cols = [row[1] for row in cursor.fetchall()]
            if "tenant_id" not in cols:
                cursor.execute("ALTER TABLE episodic_ledger ADD COLUMN tenant_id TEXT NOT NULL DEFAULT 'default'")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodic_tenant ON episodic_ledger(tenant_id)")

            # 2. Hash Chain Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hash_chain (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash_val TEXT NOT NULL
                )
            """)

            # 3. Semantic Graph Nodes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_nodes (
                    node_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    description TEXT NOT NULL,
                    embedding BLOB,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    decay_factor REAL NOT NULL
                )
            """)

            # 4. Semantic Graph Edges Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    weight REAL NOT NULL
                )
            """)

            # 5. Neural State Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS neural_state (
                    key TEXT PRIMARY KEY,
                    tensor_blob BLOB,
                    meta_text TEXT
                )
            """)

            # 6. Orchestrator Metadata Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orchestrator_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.commit()

    # --- BaseMemoryStorePort Implementation ---

    def add_record(self, record: MemoryRecord) -> None:
        """Store a new legal memory record in the persistent index."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM episodic_ledger")
            next_seq = cursor.fetchone()[0]

            key_blob = self._tensor_to_blob(record.key_vector)
            val_blob = self._tensor_to_blob(record.value_vector)

            cursor.execute("""
                INSERT OR REPLACE INTO episodic_ledger (
                    record_id, seq, text, key_vector, value_vector, importance_score,
                    tier, valid_from, valid_to, transaction_time, authority_rank,
                    jurisdiction, personal_data, metadata_json, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.record_id or f"rec_{next_seq}",
                next_seq,
                record.text,
                key_blob,
                val_blob,
                record.importance_score,
                record.tier.value if hasattr(record.tier, "value") else str(record.tier),
                record.valid_from.isoformat(),
                record.valid_to.isoformat() if record.valid_to else None,
                record.transaction_time.isoformat(),
                record.authority_rank,
                record.jurisdiction,
                1 if record.personal_data else 0,
                json.dumps(record.metadata),
                record.tenant_id,
            ))
            conn.commit()

    def get_records(self, tenant_id: str | None = None) -> list[MemoryRecord]:
        """Retrieve all currently registered memory records, optionally filtered by tenant."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if tenant_id is not None:
                cursor.execute("SELECT * FROM episodic_ledger WHERE tenant_id = ? ORDER BY seq ASC", (tenant_id,))
            else:
                cursor.execute("SELECT * FROM episodic_ledger ORDER BY seq ASC")
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    def clear(self) -> None:
        """Purge all stored memory records and state."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM episodic_ledger")
            cursor.execute("DELETE FROM hash_chain")
            cursor.execute("DELETE FROM semantic_nodes")
            cursor.execute("DELETE FROM semantic_edges")
            cursor.execute("DELETE FROM neural_state")
            cursor.execute("DELETE FROM orchestrator_meta")
            conn.commit()

    # --- Full Orchestrator Persistence ---

    def save_orchestrator(self, orchestrator: Any) -> None:
        """
        Serialize entire LegalMemoryOrchestrator state to SQLite.

        Args:
            orchestrator: LegalMemoryOrchestrator instance to persist.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Clear existing state in DB
            cursor.execute("DELETE FROM episodic_ledger")
            cursor.execute("DELETE FROM hash_chain")
            cursor.execute("DELETE FROM semantic_nodes")
            cursor.execute("DELETE FROM semantic_edges")
            cursor.execute("DELETE FROM neural_state")
            cursor.execute("DELETE FROM orchestrator_meta")

            # 2. Metadata
            cursor.execute("INSERT INTO orchestrator_meta VALUES (?, ?)", ("seed", str(orchestrator.seed)))
            cursor.execute("INSERT INTO orchestrator_meta VALUES (?, ?)", ("value_dim", str(orchestrator.value_dim)))

            # 3. Episodic Ledger
            ledger = orchestrator.episodic_memory.get_records()
            for seq, rec in enumerate(ledger):
                key_blob = self._tensor_to_blob(rec.key_vector)
                val_blob = self._tensor_to_blob(rec.value_vector)
                cursor.execute("""
                    INSERT INTO episodic_ledger (
                        record_id, seq, text, key_vector, value_vector, importance_score,
                        tier, valid_from, valid_to, transaction_time, authority_rank,
                        jurisdiction, personal_data, metadata_json, tenant_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rec.record_id or f"rec_{seq}",
                    seq,
                    rec.text,
                    key_blob,
                    val_blob,
                    rec.importance_score,
                    rec.tier.value if hasattr(rec.tier, "value") else str(rec.tier),
                    rec.valid_from.isoformat(),
                    rec.valid_to.isoformat() if rec.valid_to else None,
                    rec.transaction_time.isoformat(),
                    rec.authority_rank,
                    rec.jurisdiction,
                    1 if rec.personal_data else 0,
                    json.dumps(rec.metadata),
                    rec.tenant_id,
                ))

            # 4. Hash Chain
            for h in orchestrator.episodic_memory._hash_chain:
                cursor.execute("INSERT INTO hash_chain (hash_val) VALUES (?)", (h,))

            # 5. Semantic Knowledge Graph
            for node in orchestrator.semantic_graph.nodes.values():
                emb_blob = self._tensor_to_blob(node.embedding) if node.embedding is not None else None
                cursor.execute("""
                    INSERT INTO semantic_nodes VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    node.node_id,
                    node.entity_type.value if hasattr(node.entity_type, "value") else str(node.entity_type),
                    node.label,
                    node.description,
                    emb_blob,
                    node.valid_from.isoformat(),
                    node.valid_to.isoformat() if node.valid_to else None,
                    node.decay_factor,
                ))

            for edge in orchestrator.semantic_graph.edges:
                cursor.execute("""
                    INSERT INTO semantic_edges VALUES (?, ?, ?, ?)
                """, (
                    edge.source_id,
                    edge.target_id,
                    edge.relation_type.value if hasattr(edge.relation_type, "value") else str(edge.relation_type),
                    edge.weight,
                ))

            # 6. Neural Continuum State & Weights
            mem = orchestrator.hope_module.memory
            cursor.execute("INSERT INTO neural_state VALUES (?, ?, ?)", ("keys", self._tensor_to_blob(mem.keys), None))
            cursor.execute("INSERT INTO neural_state VALUES (?, ?, ?)", ("values", self._tensor_to_blob(mem.values), None))
            cursor.execute("INSERT INTO neural_state VALUES (?, ?, ?)", ("rule_importance", self._tensor_to_blob(mem.rule_importance), None))
            cursor.execute("INSERT INTO neural_state VALUES (?, ?, ?)", ("surprise_momentum", self._tensor_to_blob(mem.surprise_momentum), None))
            cursor.execute("INSERT INTO neural_state VALUES (?, ?, ?)", ("texts", None, json.dumps(mem.texts)))

            fast_buf = io.BytesIO()
            torch.save(mem.fast_net.state_dict(), fast_buf)
            cursor.execute("INSERT INTO neural_state VALUES (?, ?, ?)", ("fast_net", fast_buf.getvalue(), None))

            slow_buf = io.BytesIO()
            torch.save(mem.slow_net.state_dict(), slow_buf)
            cursor.execute("INSERT INTO neural_state VALUES (?, ?, ?)", ("slow_net", slow_buf.getvalue(), None))

            conn.commit()

    def load_into_orchestrator(self, orchestrator: Any) -> None:
        """
        Reload full state from SQLite into an instantiated LegalMemoryOrchestrator.

        Args:
            orchestrator: LegalMemoryOrchestrator instance to populate.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Restore Metadata
            cursor.execute("SELECT key, value FROM orchestrator_meta")
            meta_dict = dict(cursor.fetchall())
            if "seed" in meta_dict and meta_dict["seed"] != "None":
                seed_val = int(meta_dict["seed"])
                orchestrator.seed = seed_val
                torch.manual_seed(seed_val)

            # 2. Restore Episodic Ledger
            cursor.execute("SELECT * FROM episodic_ledger ORDER BY seq ASC")
            records = [self._row_to_record(row) for row in cursor.fetchall()]
            orchestrator.episodic_memory._ledger = records

            # 3. Restore Hash Chain
            cursor.execute("SELECT hash_val FROM hash_chain ORDER BY seq ASC")
            orchestrator.episodic_memory._hash_chain = [row[0] for row in cursor.fetchall()]

            # 4. Restore Semantic Knowledge Graph
            cursor.execute("SELECT * FROM semantic_nodes")
            orchestrator.semantic_graph.nodes.clear()
            for row in cursor.fetchall():
                emb = self._blob_to_tensor(row["embedding"]) if row["embedding"] else None
                node = GraphNode(
                    node_id=row["node_id"],
                    entity_type=EntityType(row["entity_type"]),
                    label=row["label"],
                    description=row["description"],
                    embedding=emb,
                    valid_from=datetime.fromisoformat(row["valid_from"]),
                    valid_to=datetime.fromisoformat(row["valid_to"]) if row["valid_to"] else None,
                    decay_factor=row["decay_factor"],
                )
                orchestrator.semantic_graph.nodes[node.node_id] = node

            cursor.execute("SELECT * FROM semantic_edges")
            orchestrator.semantic_graph.edges.clear()
            for row in cursor.fetchall():
                edge = GraphEdge(
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relation_type=RelationType(row["relation_type"]),
                    weight=row["weight"],
                )
                orchestrator.semantic_graph.edges.append(edge)

            # 5. Restore Neural Continuum State
            cursor.execute("SELECT key, tensor_blob, meta_text FROM neural_state")
            state_map = {row["key"]: (row["tensor_blob"], row["meta_text"]) for row in cursor.fetchall()}

            mem = orchestrator.hope_module.memory
            if "keys" in state_map and state_map["keys"][0]:
                mem.keys = self._blob_to_tensor(state_map["keys"][0])
            if "values" in state_map and state_map["values"][0]:
                mem.values = self._blob_to_tensor(state_map["values"][0])
            if "rule_importance" in state_map and state_map["rule_importance"][0]:
                mem.rule_importance = self._blob_to_tensor(state_map["rule_importance"][0])
            if "surprise_momentum" in state_map and state_map["surprise_momentum"][0]:
                mem.surprise_momentum = self._blob_to_tensor(state_map["surprise_momentum"][0])
            if "texts" in state_map and state_map["texts"][1]:
                mem.texts = json.loads(state_map["texts"][1])

            if "fast_net" in state_map and state_map["fast_net"][0]:
                fast_dict = torch.load(io.BytesIO(state_map["fast_net"][0]), weights_only=True)
                mem.fast_net.load_state_dict(fast_dict)
            if "slow_net" in state_map and state_map["slow_net"][0]:
                slow_dict = torch.load(io.BytesIO(state_map["slow_net"][0]), weights_only=True)
                mem.slow_net.load_state_dict(slow_dict)

            # 6. Re-populate Working Memory from latest records
            orchestrator.working_memory.clear()
            for rec in records[-orchestrator.working_memory.capacity:]:
                orchestrator.working_memory.add(
                    text=rec.text,
                    key_vector=rec.key_vector,
                    value_vector=rec.value_vector,
                    metadata=rec.metadata,
                )

    # --- Serialization Helpers ---

    @staticmethod
    def _tensor_to_blob(tensor: torch.Tensor) -> bytes:
        buf = io.BytesIO()
        torch.save(tensor, buf)
        return buf.getvalue()

    @staticmethod
    def _blob_to_tensor(blob: bytes) -> torch.Tensor:
        buf = io.BytesIO(blob)
        return torch.load(buf, weights_only=True)

    @classmethod
    def _row_to_record(cls, row: sqlite3.Row) -> MemoryRecord:
        key_vec = cls._blob_to_tensor(row["key_vector"])
        val_vec = cls._blob_to_tensor(row["value_vector"])
        meta = json.loads(row["metadata_json"])

        vf = datetime.fromisoformat(row["valid_from"])
        if vf.tzinfo is None:
            vf = vf.replace(tzinfo=timezone.utc)

        vt = datetime.fromisoformat(row["valid_to"]) if row["valid_to"] else None
        if vt is not None and vt.tzinfo is None:
            vt = vt.replace(tzinfo=timezone.utc)

        tt = datetime.fromisoformat(row["transaction_time"])
        if tt.tzinfo is None:
            tt = tt.replace(tzinfo=timezone.utc)

        tenant_id = row["tenant_id"] if "tenant_id" in row else "default"  # noqa: SIM401

        return MemoryRecord(
            text=row["text"],
            key_vector=key_vec,
            value_vector=val_vec,
            importance_score=row["importance_score"],
            record_id=row["record_id"],
            tier=MemoryTier(row["tier"]),
            valid_from=vf,
            valid_to=vt,
            metadata=meta,
            authority_rank=row["authority_rank"],
            jurisdiction=row["jurisdiction"],
            transaction_time=tt,
            personal_data=bool(row["personal_data"]),
            tenant_id=tenant_id,
        )
