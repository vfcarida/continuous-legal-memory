"""
Hexagonal Application Service (Legal Memory Orchestrator).

Serves as the central entry point for the Multi-Tier Continuous Legal Memory engine, coordinating
Working Memory, Episodic Ledger, Semantic Knowledge Graph, and neural continuum adaptation networks.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from continuous_legal_memory.adapters.encoders import HuggingFaceEncoderAdapter
from continuous_legal_memory.adapters.ollama import OllamaGemmaAdapter
from continuous_legal_memory.core.hope_module import HopeModule
from continuous_legal_memory.core.memory_tiers.episodic_memory import EpisodicMemory
from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.core.memory_tiers.working_memory import WorkingMemory
from continuous_legal_memory.domain.exceptions import (
    EncoderInferenceError,
    InvalidMemoryVectorError,
)
from continuous_legal_memory.domain.interfaces import (
    BaseEncoderPort,
    BaseMemoryStorePort,
    BaseRetrieverPort,
)
from continuous_legal_memory.domain.models import (
    EntityType,
    GraphNode,
    MemoryRecord,
    MemoryTier,
    PredictionResult,
    RelationType,
)
from continuous_legal_memory.security.attestation import KeyedHashAttestationModule
from continuous_legal_memory.storage.sqlite_store import SqliteMemoryStore
from continuous_legal_memory.telemetry.observability import TelemetryLogger


class LegalMemoryOrchestrator:
    """
    Main Hexagonal Application Orchestrator for Continuous Legal Memory operations.

    Rationale:
        Synthesizes multi-tiered cognitive memory (Working, Episodic, Semantic) with neural associative
        learning (`HopeModule` / `ContinuousMemory`). Supports strict privacy mode for local edge execution.
    """

    def __init__(
        self,
        encoder: BaseEncoderPort | None = None,
        value_dim: int = 2,
        temperature: float = 0.05,
        strict_privacy_mode: bool = False,
        seed: int | None = None,
        allow_pseudo_embeddings: bool = False,
        retriever: BaseRetrieverPort | None = None,
        episodic_store: BaseMemoryStorePort | None = None,
        telemetry: TelemetryLogger | None = None,
        enable_telemetry: bool = False,
        attestor: KeyedHashAttestationModule | None = None,
        enable_attestation: bool = False,
        engine_mode: str = "structured",
    ) -> None:
        """
        Initialize the LegalMemoryOrchestrator.

        Args:
            encoder: Instance of `BaseEncoderPort`. If None, defaults to `HuggingFaceEncoderAdapter`
                     or `OllamaGemmaAdapter` when `strict_privacy_mode` is True.
            value_dim: Target decision vector dimension.
            temperature: Softmax attention scaling temperature.
            strict_privacy_mode: Enforces local offline execution.
            seed: Optional integer random seed for deterministic neural initialization.
            allow_pseudo_embeddings: Forwarded to OllamaGemmaAdapter when strict_privacy_mode is used.
            retriever: Optional BaseRetrieverPort implementation.
            episodic_store: Optional BaseMemoryStorePort instance for episodic ledger.
            telemetry: Optional TelemetryLogger instance for operational metrics tracing.
            enable_telemetry: If True and telemetry is None, instantiates a default TelemetryLogger.
            attestor: Optional KeyedHashAttestationModule for cryptographic state attestation.
            enable_attestation: If True and attestor is None, instantiates a default KeyedHashAttestationModule.
            engine_mode: Core memory engine strategy ('structured' [default production engine]
                         or 'hybrid' [experimental dual-timescale neural adaptation head]).
        """
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.strict_privacy_mode = strict_privacy_mode
        self.allow_pseudo_embeddings = allow_pseudo_embeddings

        if encoder is not None:
            self.encoder = encoder
        elif self.strict_privacy_mode:
            self.encoder = OllamaGemmaAdapter(
                strict_privacy_mode=True,
                allow_pseudo_embeddings=allow_pseudo_embeddings,
            )
        else:
            self.encoder = HuggingFaceEncoderAdapter()

        self.value_dim = value_dim
        self.engine_mode = engine_mode.lower()
        if self.engine_mode not in ("structured", "hybrid"):
            raise ValueError(f"Unknown engine_mode '{engine_mode}'. Supported modes: 'structured', 'hybrid'.")

        self.hope_module = HopeModule(
            embed_dim=self.encoder.embedding_dim,
            value_dim=value_dim,
            temperature=temperature,
            retriever=retriever,
            engine_mode=self.engine_mode,
        )

        # Multi-Tier Cognitive Memory System
        self.working_memory = WorkingMemory(capacity=10)
        self.episodic_memory: EpisodicMemory = episodic_store if episodic_store is not None else EpisodicMemory()
        self.semantic_graph = SemanticKnowledgeGraph()

        # Telemetry & Observability
        if telemetry is not None:
            self.telemetry: TelemetryLogger | None = telemetry
        elif enable_telemetry:
            self.telemetry = TelemetryLogger()
        else:
            self.telemetry = None

        # Cryptographic Attestation
        if attestor is not None:
            self.attestor: KeyedHashAttestationModule | None = attestor
        elif enable_attestation:
            self.attestor = KeyedHashAttestationModule()
        else:
            self.attestor = None

        self._current_tenant_id: str | None = None

    def update_memory(
        self,
        rule_text: str,
        action_vector: list[float],
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        personal_data: bool = False,
        tenant_id: str = "default",
    ) -> MemoryRecord:
        """
        Ingest a new legal directive into active neural memory and multi-tier ledgers.

        Args:
            rule_text: Human-readable text string of the legal rule or statute.
            action_vector: Target decision outputs list.
            valid_from: Start timestamp of legal validity.
            valid_to: Expiry or temporal invalidation timestamp.
            metadata: Custom audit metadata dictionary.
            authority_rank: Hierarchical legal authority rank integer.
            jurisdiction: Jurisdictional scope identifier.
            personal_data: Whether this record contains personal data requiring right-to-erasure.
            tenant_id: Tenant or workspace identifier for multi-tenant isolation.

        Returns:
            The created and appended MemoryRecord.

        Raises:
            InvalidMemoryVectorError: If input parameters violate dimension or text bounds.
            EncoderInferenceError: If key embedding generation fails.
        """
        if self.telemetry is not None:
            res, _ = self.telemetry.trace_operation(
                "update_memory",
                self._execute_update_memory,
                rule_text,
                action_vector,
                valid_from,
                valid_to,
                metadata,
                authority_rank,
                jurisdiction,
                personal_data,
                tenant_id,
            )
            return res
        return self._execute_update_memory(
            rule_text, action_vector, valid_from, valid_to, metadata,
            authority_rank, jurisdiction, personal_data, tenant_id
        )

    def _execute_update_memory(
        self,
        rule_text: str,
        action_vector: list[float],
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        personal_data: bool = False,
        tenant_id: str = "default",
    ) -> MemoryRecord:
        if not isinstance(rule_text, str) or len(rule_text.strip()) == 0:
            raise InvalidMemoryVectorError("Rule text must be a non-empty string.")

        if not isinstance(action_vector, list) or len(action_vector) != self.value_dim:
            raise InvalidMemoryVectorError(
                f"Action vector must be a list of floats with length equal to {self.value_dim}.",
                payload={"action_vector": action_vector, "expected_length": self.value_dim},
            )

        auth_rank = metadata.get("authority_rank", authority_rank) if metadata else authority_rank
        juris = metadata.get("jurisdiction", jurisdiction) if metadata else jurisdiction
        is_personal = metadata.get("personal_data", personal_data) if metadata else personal_data
        default_tenant = getattr(self, "_current_tenant_id", None) or "default"
        explicit_tenant = tenant_id if tenant_id != "default" else default_tenant
        tenant = (
            metadata.get("tenant_id", metadata.get("tenant", explicit_tenant))
            if metadata else explicit_tenant
        )

        try:
            key_embed = self.encoder.get_embedding([rule_text])
        except Exception as e:
            if isinstance(e, EncoderInferenceError):
                raise
            raise EncoderInferenceError(f"Failed to generate embedding for rule text: {e}") from e

        val_tensor = torch.tensor([action_vector], dtype=torch.float32)

        # 1. Adapt neural associative continuum memory
        # In structured mode (default) or for personal data, bypass online gradient backprop on fast_net
        skip_param = is_personal or (self.engine_mode == "structured")
        self.hope_module.memory.add_memory(
            key_embed,
            val_tensor,
            rule_text,
            skip_parametric=skip_param,
            authority_rank=auth_rank,
        )

        # 2. Record in Episodic Memory ledger
        if hasattr(self.episodic_memory, "append"):
            rec = self.episodic_memory.append(
                text=rule_text,
                key_vector=key_embed,
                value_vector=val_tensor,
                valid_from=valid_from,
                valid_to=valid_to,
                metadata=metadata,
                authority_rank=auth_rank,
                jurisdiction=juris,
                personal_data=is_personal,
                tenant_id=tenant,
            )
        else:
            rec = MemoryRecord(
                text=rule_text,
                key_vector=key_embed,
                value_vector=val_tensor,
                valid_from=valid_from or datetime.now(timezone.utc),
                valid_to=valid_to,
                metadata=metadata or {},
                authority_rank=auth_rank,
                jurisdiction=juris,
                personal_data=is_personal,
                tenant_id=tenant,
            )
            self.episodic_memory.add_record(rec)

        # 3. Add to Working Memory
        self.working_memory.add(
            text=rule_text,
            key_vector=key_embed,
            value_vector=val_tensor,
            metadata=metadata,
            tenant_id=tenant,
        )

        # 4. Ingest into Semantic Knowledge Graph
        entity_type = EntityType.STATUTE
        if metadata and "entity_type" in metadata:
            raw_et = metadata["entity_type"]
            if isinstance(raw_et, EntityType):
                entity_type = raw_et
            elif isinstance(raw_et, str):
                try:
                    entity_type = EntityType(raw_et)
                except ValueError:
                    entity_type = EntityType.STATUTE

        node_id = str(metadata.get("node_id")) if (metadata and "node_id" in metadata) else (rec.record_id or f"node_{len(self.semantic_graph.nodes) + 1}")
        label = metadata.get("label", rule_text[:30]) if metadata else rule_text[:30]

        self.semantic_graph.add_node(
            node_id=node_id,
            entity_type=entity_type,
            label=label,
            description=rule_text,
            embedding=key_embed,
            valid_from=valid_from,
            valid_to=valid_to,
            tenant_id=tenant,
        )

        if metadata and "relations" in metadata:
            for rel in metadata["relations"]:
                target = rel.get("target_id")
                rel_type = rel.get("relation_type", RelationType.DEPENDS_ON)
                if isinstance(rel_type, str):
                    rel_type = RelationType(rel_type)
                weight = rel.get("weight", 1.0)
                if target and target in self.semantic_graph.nodes:
                    self.semantic_graph.add_edge(node_id, target, rel_type, weight, tenant_id=tenant)

        return rec

    def predict(
        self,
        query_text: str,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> PredictionResult:
        """
        Predict decision vectors across multi-tier memory networks for a query string.

        Args:
            query_text: Legal query or case scenario text string.
            at_time: Datetime timestamp to evaluate temporal validity.
            tenant_id: Optional tenant identifier to enforce multi-tenant isolation.

        Returns:
            A populated `PredictionResult` domain object.

        Raises:
            InvalidMemoryVectorError: If query text is empty.
            EncoderInferenceError: If query embedding generation fails.
        """
        eff_tenant = tenant_id if tenant_id is not None else getattr(self, "_current_tenant_id", None)
        if self.telemetry is not None:
            result, _ = self.telemetry.trace_operation(
                "predict",
                self._execute_predict,
                query_text,
                at_time,
                eff_tenant,
            )
        else:
            result = self._execute_predict(query_text, at_time, eff_tenant)

        if self.attestor is not None:
            token = self.attestor.sign_attestation(result, retrieved_text=result.most_relevant_rule)
            result.attestation_token = token

        return result

    def _execute_predict(
        self,
        query_text: str,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> PredictionResult:
        if not isinstance(query_text, str) or len(query_text.strip()) == 0:
            raise InvalidMemoryVectorError("Query text must be a non-empty string.")

        eval_time = at_time or datetime.now(timezone.utc)
        valid_indices = self.episodic_memory.get_valid_indices(eval_time, tenant_id=tenant_id)
        valid_records = self.episodic_memory.get_valid_records(eval_time, tenant_id=tenant_id)

        try:
            query_embed = self.encoder.get_embedding([query_text])
        except Exception as e:
            if isinstance(e, EncoderInferenceError):
                raise
            raise EncoderInferenceError(f"Failed to generate embedding for query text: {e}") from e

        # If no valid records are present, return an empty-memory zero vector result
        if not valid_indices:
            return PredictionResult(
                query=query_text,
                predicted_action_vector=[0.0] * self.value_dim,
                fast_slow_gate=None,
                source_tier=MemoryTier.SEMANTIC if self.semantic_graph.nodes else MemoryTier.EPISODIC,
                most_relevant_rule=None,
                confidence=None,
                attention_weights=None,
                tenant_id=tenant_id or "default",
            )

        predicted_action, attention_weights, fast_slow_gate = self.hope_module(
            query_embed,
            valid_indices=valid_indices,
            query_text=query_text,
            records=valid_records,
        )

        pred_action_list = predicted_action.squeeze().tolist()
        if isinstance(pred_action_list, float):
            pred_action_list = [pred_action_list]

        result = PredictionResult(
            query=query_text,
            predicted_action_vector=pred_action_list,
            fast_slow_gate=fast_slow_gate,
            source_tier=MemoryTier.SEMANTIC if self.semantic_graph.nodes else MemoryTier.EPISODIC,
            retrieved_snippets=self.hope_module.last_snippets,
            tenant_id=tenant_id or "default",
        )

        if attention_weights is not None:
            weights = attention_weights.squeeze().tolist()
            if isinstance(weights, float):
                weights = [weights]

            result.attention_weights = weights
            top_local_idx = int(torch.argmax(attention_weights, dim=-1).item())
            top_global_idx = valid_indices[top_local_idx]
            if valid_records and top_local_idx < len(valid_records):
                result.most_relevant_rule = valid_records[top_local_idx].text
            else:
                result.most_relevant_rule = self.hope_module.memory.texts[top_global_idx]
            result.confidence = weights[top_local_idx]

        return result

    def delete_rule(self, record_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        """
        Execute Right-to-Erasure (GDPR Art. 17 / CLM-R7) across all memory tiers.

        Workflow:
            1. Locates the record in EpisodicMemory; appends an immutable cryptographic
               tombstone hash to the hash chain and deletes the record from the ledger.
            2. Purges any matching items from WorkingMemory context cache.
            3. Removes entity node and associated edges from SemanticKnowledgeGraph.
            4. Purges corresponding entries from HopeModule/ContinuousMemory retrieval buffers.
            5. If the deleted rule was non-personal (having entered neural weights), re-consolidates
               the neural memory head from the clean retained records buffer to ensure zero parametric remnant.

        Args:
            record_id: Identifier of the rule/record to permanently erase.
            tenant_id: Optional tenant identifier to enforce authorization.

        Returns:
            Audit record dictionary confirming multi-tier erasure metrics.

        Raises:
            KeyError: If record_id is not found in the episodic ledger or tenant mismatch.
        """
        deleted_rec = None
        for rec in self.episodic_memory.get_records():
            if rec.record_id == record_id:
                deleted_rec = rec
                break

        if deleted_rec is None:
            raise KeyError(f"Rule with ID '{record_id}' not found in episodic ledger.")

        if tenant_id is not None and deleted_rec.tenant_id != tenant_id:
            raise KeyError(f"Rule with ID '{record_id}' does not belong to tenant '{tenant_id}'.")

        # 1. Delete from Episodic Memory (records tombstone in hash chain)
        if hasattr(self.episodic_memory, "delete_record"):
            self.episodic_memory.delete_record(record_id)
        else:
            self.episodic_memory.record_tombstone(record_id)

        # 2. Clear from Working Memory
        self.working_memory.remove_by_text(deleted_rec.text, tenant_id=deleted_rec.tenant_id)

        # 3. Clear from Semantic Knowledge Graph
        matching_nodes = [
            nid for nid, node in self.semantic_graph.nodes.items()
            if (nid == record_id or node.description == deleted_rec.text)
            and getattr(node, "tenant_id", "default") == deleted_rec.tenant_id
        ]
        for nid in matching_nodes:
            self.semantic_graph.remove_node(nid, tenant_id=deleted_rec.tenant_id)

        # 4. Clear from Retrieval Buffers
        buf_idx = None
        for i, txt in enumerate(self.hope_module.memory.texts):
            if txt == deleted_rec.text:
                buf_idx = i
                break
        if buf_idx is not None:
            self.hope_module.memory.delete_buffer_index(buf_idx)

        # 5. Parametric influence cleanup / re-consolidation
        parametric_rebuilt = False
        if not deleted_rec.personal_data:
            retained = self.episodic_memory.get_records()
            self.hope_module.memory.rebuild_from_records(retained, seed=self.seed)
            parametric_rebuilt = True

        tombstone = self.episodic_memory._hash_chain[-1] if self.episodic_memory._hash_chain else ""

        return {
            "record_id": record_id,
            "deleted_at": datetime.now(timezone.utc).isoformat(),
            "tombstone_hash": tombstone,
            "personal_data": deleted_rec.personal_data,
            "status": "ERASED",
            "tiers_cleared": ["working", "episodic", "semantic", "neural_buffer"],
            "parametric_rebuilt": parametric_rebuilt,
        }

    def save_to_disk(self, db_path: str | Path) -> None:
        """Persist complete multi-tier state and model weights to a SQLite database."""
        store = SqliteMemoryStore(db_path)
        store.save_orchestrator(self)

    def load_from_disk(self, db_path: str | Path) -> None:
        """Restore complete multi-tier state and model weights from a SQLite database."""
        store = SqliteMemoryStore(db_path)
        store.load_into_orchestrator(self)

    def get_working_memory_context(
        self,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[MemoryRecord]:
        """
        Retrieve active working memory records, optionally filtered by tenant.

        Args:
            at_time: Datetime timestamp to evaluate validity.
            tenant_id: Optional tenant identifier to enforce multi-tenant isolation.
        """
        return self.working_memory.get_active_context(at_time=at_time, tenant_id=tenant_id)

    def get_semantic_graph_nodes(self, tenant_id: str | None = None) -> list[GraphNode]:
        """
        Retrieve semantic knowledge graph nodes, optionally filtered by tenant.

        Args:
            tenant_id: Optional tenant identifier.
        """
        return self.semantic_graph.get_nodes(tenant_id=tenant_id)

    @contextmanager
    def tenant(self, tenant_id: str):
        """
        Context manager scoping orchestrator actions to a specific tenant.

        Usage:
            with orchestrator.tenant("client_alpha"):
                res = orchestrator.predict("Query text")
        """
        prev_tenant = getattr(self, "_current_tenant_id", None)
        self._current_tenant_id = tenant_id
        try:
            yield tenant_id
        finally:
            self._current_tenant_id = prev_tenant


