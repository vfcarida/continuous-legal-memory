"""
Hexagonal Application Service (Legal Memory Orchestrator).

Serves as the central entry point for the Multi-Tier Continuous Legal Memory engine, coordinating
Working Memory, Episodic Ledger, Semantic Knowledge Graph, and neural continuum adaptation networks.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F

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
    BaseRerankerPort,
    BaseRetrieverPort,
)
from continuous_legal_memory.domain.models import (
    GraphNode,
    MemoryRecord,
    MemoryTier,
    PredictionResult,
)
from continuous_legal_memory.reasoning.amendment_impact import (
    AmendmentImpactReport,
    StatutoryAmendmentAnalyzer,
)
from continuous_legal_memory.reasoning.deontic_checker import (
    DeonticConsistencyChecker,
    DeonticConsistencyReport,
)
from continuous_legal_memory.reasoning.drafting_copilot import (
    DraftingCopilotReport,
    NormativeDraftingCopilot,
)
from continuous_legal_memory.reasoning.modal_verifier import (
    DeonticModalVerifier,
    PreFlightSimulationReport,
)
from continuous_legal_memory.reasoning.norm_extractor import NormativeGraphExtractor
from continuous_legal_memory.reasoning.proof_trace import build_defeasible_proof_trace
from continuous_legal_memory.reasoning.smtlib_exporter import (
    SMTLibDeonticExporter,
    SMTLibExportBundle,
)
from continuous_legal_memory.reasoning.temporal_simulator import (
    RegulatoryTimelineReport,
    TemporalHorizonSimulator,
)
from continuous_legal_memory.retrieval.choice_of_law import (
    ChoiceOfLawRouter,
    ConnectingFactors,
)
from continuous_legal_memory.retrieval.conformal import SplitConformalCalibrator
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
)
from continuous_legal_memory.security.federated_sync import (
    CrossPartyAlignmentReport,
    FederatedComplianceVerifier,
    MerkleStateDiff,
)
from continuous_legal_memory.security.keyring import (
    CryptographicKeyRing,
    KeyMetadata,
)
from continuous_legal_memory.security.merkle import (
    SparseMerkleProof,
    SparseMerkleTree,
    verify_sparse_merkle_proof,
)
from continuous_legal_memory.security.pq_hybrid import (
    PQ_HYBRID_ALGORITHM,
    PostQuantumHybridAttestationModule,
)
from continuous_legal_memory.security.selective_disclosure import (
    SelectiveDisclosureBundle,
    SelectiveDisclosureComplianceProof,
    SelectiveDisclosureProver,
)
from continuous_legal_memory.security.standalone_verifier import (
    StandaloneAuditVerifierExporter,
    StandaloneVerifierArtifact,
)
from continuous_legal_memory.security.tee_attestation import (
    TEEAttestationProvider,
    TEEAttestationQuote,
    TEEQuoteVerificationReport,
)
from continuous_legal_memory.security.timestamp_gossip import (
    GossipCheckpoint,
    SplitViewAuditReport,
    TimestampToken,
    TransparencyGossipAuditor,
)
from continuous_legal_memory.storage.portable_bundle import (
    DistributedVectorGraphExporter,
    PortableBundleManager,
    SignedComplianceBundle,
)
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
        attestor: Any | None = None,
        enable_attestation: bool = False,
        engine_mode: str = "structured",
        min_similarity_threshold: float | None = None,
        reranker: BaseRerankerPort | None = None,
        conformal_calibrator: SplitConformalCalibrator | None = None,
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
            attestor: Optional KeyedHashAttestationModule or AsymmetricAttestationModule for state attestation.
            enable_attestation: If True and attestor is None, instantiates a default KeyedHashAttestationModule.
            engine_mode: Core memory engine strategy ('structured' [default production engine]
                         or 'hybrid' [experimental dual-timescale neural adaptation head]).
            min_similarity_threshold: Optional minimum cosine similarity threshold for epistemic abstention
                                      (EU AI Act Art. 14 guardrail). If the best matching active rule falls
                                      below this threshold, `predict()` abstains with `abstained=True`.
            reranker: Optional `BaseRerankerPort` second-stage cross-encoder reranker.
            conformal_calibrator: Optional `SplitConformalCalibrator` providing `(1 - alpha)` coverage
                                  prediction sets and distribution-free epistemic abstention.
        """
        self.seed = seed
        self.min_similarity_threshold = min_similarity_threshold
        self.reranker = reranker
        self.conformal_calibrator = conformal_calibrator
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
        if self.reranker is not None and hasattr(self.hope_module.retriever, "reranker"):
            self.hope_module.retriever.reranker = self.reranker

        # Multi-Tier Cognitive Memory System
        self.working_memory = WorkingMemory(capacity=10)
        self.episodic_memory: Any = (
            episodic_store if episodic_store is not None else EpisodicMemory()
        )
        self.semantic_graph = SemanticKnowledgeGraph()
        self.merkle_tree = SparseMerkleTree()
        self.choice_of_law_router = ChoiceOfLawRouter()
        self.deontic_checker = DeonticConsistencyChecker()
        self.modal_verifier = DeonticModalVerifier()
        self.drafting_copilot = NormativeDraftingCopilot(modal_verifier=self.modal_verifier)
        self.smtlib_exporter = SMTLibDeonticExporter()
        self.temporal_simulator = TemporalHorizonSimulator(consistency_checker=self.deontic_checker)
        self.standalone_verifier_exporter = StandaloneAuditVerifierExporter()
        self.amendment_analyzer = StatutoryAmendmentAnalyzer()

        # Telemetry & Observability
        if telemetry is not None:
            self.telemetry: TelemetryLogger | None = telemetry
        elif enable_telemetry:
            self.telemetry = TelemetryLogger()
        else:
            self.telemetry = None

        # Cryptographic Attestation & Key Lifecycle Ring
        if attestor is not None:
            self.attestor: Any | None = attestor
        elif enable_attestation:
            self.attestor = KeyedHashAttestationModule()
        else:
            self.attestor = None

        self.keyring: CryptographicKeyRing = (
            attestor if isinstance(attestor, CryptographicKeyRing) else CryptographicKeyRing()
        )
        self.tee_provider = TEEAttestationProvider()
        self.abstained_predictions_count: int = 0

        self._current_tenant_id: str | None = None
        self._lock = threading.RLock()

    def update_memory(
        self,
        rule_text: str,
        action_vector: list[float] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        personal_data: bool = False,
        tenant_id: str = "default",
        transaction_time: datetime | None = None,
    ) -> MemoryRecord:
        """
        Ingest a new legal directive into active neural memory and multi-tier ledgers.

        Args:
            rule_text: Human-readable text string of the legal rule or statute.
            action_vector: Optional target decision outputs list (defaults to zero vector).
            valid_from: Start timestamp of legal validity.
            valid_to: Expiry or temporal invalidation timestamp.
            metadata: Custom audit metadata dictionary.
            authority_rank: Hierarchical legal authority rank integer.
            jurisdiction: Jurisdictional scope identifier.
            personal_data: Whether this record contains personal data requiring right-to-erasure.
            tenant_id: Tenant or workspace identifier for multi-tenant isolation.
            transaction_time: Optional explicit system ingestion timestamp for bi-temporal auditing.

        Returns:
            The created and appended MemoryRecord.

        Raises:
            InvalidMemoryVectorError: If input parameters violate dimension or text bounds.
            EncoderInferenceError: If key embedding generation fails.
        """
        effective_action = [0.0] * self.value_dim if action_vector is None else action_vector
        if self.telemetry is not None:
            res, _ = self.telemetry.trace_operation(
                "update_memory",
                self._execute_update_memory,
                rule_text,
                effective_action,
                valid_from,
                valid_to,
                metadata,
                authority_rank,
                jurisdiction,
                personal_data,
                tenant_id,
                transaction_time,
            )
            return cast(MemoryRecord, res)
        return self._execute_update_memory(
            rule_text, effective_action, valid_from, valid_to, metadata,
            authority_rank, jurisdiction, personal_data, tenant_id, transaction_time
        )

    def _execute_update_memory(
        self,
        rule_text: str,
        action_vector: list[float] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        personal_data: bool = False,
        tenant_id: str = "default",
        transaction_time: datetime | None = None,
    ) -> MemoryRecord:
        if not isinstance(rule_text, str) or len(rule_text.strip()) == 0:
            raise InvalidMemoryVectorError("Rule text must be a non-empty string.")

        if action_vector is None:
            action_vector = [0.0] * self.value_dim

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
        tx_time = transaction_time
        if tx_time is None and metadata and "transaction_time" in metadata:
            raw_tx = metadata["transaction_time"]
            if isinstance(raw_tx, datetime):
                tx_time = raw_tx
            elif isinstance(raw_tx, str):
                tx_time = datetime.fromisoformat(raw_tx)

        try:
            key_embed = self.encoder.get_embedding([rule_text])
        except Exception as e:
            if isinstance(e, EncoderInferenceError):
                raise
            raise EncoderInferenceError(f"Failed to generate embedding for rule text: {e}") from e

        val_tensor = torch.tensor([action_vector], dtype=torch.float32)

        with self._lock:
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
                    transaction_time=tx_time,
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
                    transaction_time=tx_time or datetime.now(timezone.utc),
                    personal_data=is_personal,
                    tenant_id=tenant,
                )
                self.episodic_memory.add_record(rec)

            if metadata and ("record_id" in metadata or "rule_id" in metadata):
                custom_id = str(metadata.get("record_id") or metadata.get("rule_id"))
                rec.record_id = custom_id

            # 3. Add to Working Memory
            self.working_memory.add(
                text=rule_text,
                key_vector=key_embed,
                value_vector=val_tensor,
                metadata=metadata,
                tenant_id=tenant,
                authority_rank=auth_rank,
                valid_from=valid_from,
                valid_to=valid_to,
                jurisdiction=juris,
                record_id=rec.record_id,
            )

            # 4. Ingest into Semantic Knowledge Graph via NormativeGraphExtractor
            node_id = str(metadata.get("node_id")) if (metadata and "node_id" in metadata) else (rec.record_id or f"node_{len(self.semantic_graph.nodes) + 1}")
            label = metadata.get("label", rule_text[:30]) if metadata else rule_text[:30]
            extracted = NormativeGraphExtractor.extract(
                node_id=node_id,
                text=rule_text,
                metadata=metadata,
                graph=self.semantic_graph,
                tenant_id=tenant,
            )

            self.semantic_graph.add_node(
                node_id=node_id,
                entity_type=extracted.entity_type,
                label=label,
                description=rule_text,
                embedding=key_embed,
                valid_from=valid_from,
                valid_to=valid_to,
                tenant_id=tenant,
                transaction_time=rec.transaction_time,
            )

            for target, rel_type, weight in extracted.edges:
                if target in self.semantic_graph.nodes:
                    self.semantic_graph.add_edge(node_id, target, rel_type, weight, tenant_id=tenant)

            # 5. Commit active record to 256-bit Sparse Merkle Tree & Append-Only Audit Log
            smt_id = rec.record_id or node_id
            vec_payload = ",".join(f"{float(v):.6f}" for v in action_vector)
            payload_str = f"{rule_text}|{vec_payload}|{auth_rank}|{tenant}"
            self.merkle_tree.update(
                rule_id=smt_id,
                payload=payload_str,
                tenant_id=tenant,
                timestamp=rec.transaction_time,
            )

            return cast(MemoryRecord, rec)

    def predict(
        self,
        query_text: str,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
        min_similarity_threshold: float | None = None,
        jurisdiction: str | list[str] | None = None,
        connecting_factors: ConnectingFactors | dict[str, Any] | None = None,
        as_of_transaction_time: datetime | None = None,
    ) -> PredictionResult:
        """
        Predict decision vectors across multi-tier memory networks for a query string.

        Args:
            query_text: Legal query or case scenario text string.
            at_time: Datetime timestamp to evaluate valid-time temporal validity.
            tenant_id: Optional tenant identifier to enforce multi-tenant isolation.
            min_similarity_threshold: Optional per-query cosine similarity threshold override
                                      for epistemic abstention.
            jurisdiction: Optional single jurisdiction or list of jurisdictions to scope retrieval.
            connecting_factors: Optional Private International Law `ConnectingFactors` (or dict)
                                resolving cross-jurisdictional conflicts and overriding mandatory norms.
            as_of_transaction_time: Optional transaction-time cutoff for bi-temporal 'as-of' replay.

        Returns:
            A populated `PredictionResult` domain object.

        Raises:
            InvalidMemoryVectorError: If query text is empty.
            EncoderInferenceError: If query embedding generation fails.
        """
        eff_tenant = tenant_id if tenant_id is not None else getattr(self, "_current_tenant_id", None)
        eff_threshold = (
            min_similarity_threshold
            if min_similarity_threshold is not None
            else self.min_similarity_threshold
        )
        if self.telemetry is not None:
            result, _ = self.telemetry.trace_operation(
                "predict",
                self._execute_predict,
                query_text,
                at_time,
                eff_tenant,
                eff_threshold,
                jurisdiction,
                connecting_factors,
                as_of_transaction_time,
            )
            result = cast(PredictionResult, result)
        else:
            result = self._execute_predict(
                query_text,
                at_time,
                eff_tenant,
                eff_threshold,
                jurisdiction,
                connecting_factors,
                as_of_transaction_time,
            )

        if result.abstained:
            with self._lock:
                self.abstained_predictions_count += 1

        if self.attestor is not None:
            token = self.attestor.sign_attestation(result, retrieved_text=result.most_relevant_rule)
            result.attestation_token = token

        return result

    def _execute_predict(
        self,
        query_text: str,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
        min_similarity_threshold: float | None = None,
        jurisdiction: str | list[str] | None = None,
        connecting_factors: ConnectingFactors | dict[str, Any] | None = None,
        as_of_transaction_time: datetime | None = None,
    ) -> PredictionResult:
        if not isinstance(query_text, str) or len(query_text.strip()) == 0:
            raise InvalidMemoryVectorError("Query text must be a non-empty string.")

        try:
            query_embed = self.encoder.get_embedding([query_text])
        except Exception as e:
            if isinstance(e, EncoderInferenceError):
                raise
            raise EncoderInferenceError(f"Failed to generate embedding for query text: {e}") from e

        eval_time = at_time or datetime.now(timezone.utc)

        with self._lock:
            all_tenant_records = (
                self.episodic_memory.get_records(tenant_id=tenant_id)
                if hasattr(self.episodic_memory, "get_records")
                else []
            )
            if as_of_transaction_time is not None and isinstance(self.episodic_memory, EpisodicMemory):
                valid_indices = self.episodic_memory.get_valid_indices(
                    eval_time,
                    tenant_id=tenant_id,
                    as_of_transaction_time=as_of_transaction_time,
                )
                valid_records = self.episodic_memory.get_valid_records(
                    eval_time,
                    tenant_id=tenant_id,
                    as_of_transaction_time=as_of_transaction_time,
                )
            else:
                valid_indices = self.episodic_memory.get_valid_indices(eval_time, tenant_id=tenant_id)
                valid_records = self.episodic_memory.get_valid_records(eval_time, tenant_id=tenant_id)

            smt_root = self.merkle_tree.get_root_as_of(
                as_of_transaction_time,
                tenant_id=tenant_id,
            )

            col_resolution = None
            if (jurisdiction is not None or connecting_factors is not None) and valid_indices:
                valid_indices, routed_recs, bonuses, col_resolution = (
                    self.choice_of_law_router.route_candidates(
                        valid_indices,
                        valid_records,
                        jurisdiction=jurisdiction,
                        connecting_factors=connecting_factors,
                    )
                )
                valid_records = []
                for rec, bonus in zip(routed_recs, bonuses):
                    if bonus != 0.0:
                        cloned_meta = dict(rec.metadata) if isinstance(rec.metadata, dict) else {}
                        cloned_meta["_choice_of_law_bonus"] = bonus
                        rec_copy = MemoryRecord(
                            text=rec.text,
                            key_vector=rec.key_vector,
                            value_vector=rec.value_vector,
                            importance_score=rec.importance_score,
                            record_id=rec.record_id,
                            tier=rec.tier,
                            valid_from=rec.valid_from,
                            valid_to=rec.valid_to,
                            metadata=cloned_meta,
                            authority_rank=rec.authority_rank,
                            jurisdiction=rec.jurisdiction,
                            transaction_time=rec.transaction_time,
                            personal_data=rec.personal_data,
                            tenant_id=rec.tenant_id,
                        )
                        valid_records.append(rec_copy)
                    else:
                        valid_records.append(rec)

            # If no valid records are present, return an empty-memory zero vector result
            if not valid_indices:
                abstained = min_similarity_threshold is not None
                reason = "no_valid_records_in_memory" if abstained else None
                empty_trace = build_defeasible_proof_trace(
                    query=query_text,
                    all_tenant_records=all_tenant_records,
                    active_records=[],
                    attention_weights=None,
                    top_local_idx=None,
                    abstained=abstained,
                    abstention_reason=reason,
                    at_valid_time=eval_time,
                    as_of_transaction_time=as_of_transaction_time,
                )
                return PredictionResult(
                    query=query_text,
                    predicted_action_vector=[0.0] * self.value_dim,
                    fast_slow_gate=None,
                    source_tier=MemoryTier.SEMANTIC if self.semantic_graph.nodes else MemoryTier.EPISODIC,
                    most_relevant_rule=None,
                    confidence=None,
                    attention_weights=None,
                    tenant_id=tenant_id or "default",
                    abstained=abstained,
                    abstention_reason=reason,
                    max_similarity=None,
                    choice_of_law=col_resolution.to_dict() if col_resolution else None,
                    proof_trace=empty_trace.to_dict(),
                    merkle_root=smt_root,
                )

            keys_slice = self.hope_module.memory.keys[valid_indices]
            q_norm = F.normalize(query_embed, p=2, dim=-1)
            k_norm = F.normalize(keys_slice, p=2, dim=-1)
            raw_sims = torch.matmul(q_norm, k_norm.T)
            max_sim_val = float(raw_sims.max().item())

            if min_similarity_threshold is not None and max_sim_val < min_similarity_threshold:
                abst_reason = (
                    f"max_similarity ({max_sim_val:.4f}) below epistemic threshold "
                    f"({min_similarity_threshold:.4f})"
                )
                abst_trace = build_defeasible_proof_trace(
                    query=query_text,
                    all_tenant_records=all_tenant_records,
                    active_records=valid_records,
                    attention_weights=None,
                    top_local_idx=None,
                    abstained=True,
                    abstention_reason=abst_reason,
                    at_valid_time=eval_time,
                    as_of_transaction_time=as_of_transaction_time,
                )
                return PredictionResult(
                    query=query_text,
                    predicted_action_vector=[0.0] * self.value_dim,
                    fast_slow_gate=0.0,
                    source_tier=MemoryTier.SEMANTIC if self.semantic_graph.nodes else MemoryTier.EPISODIC,
                    most_relevant_rule=None,
                    confidence=0.0,
                    attention_weights=None,
                    retrieved_snippets=None,
                    tenant_id=tenant_id or "default",
                    abstained=True,
                    abstention_reason=abst_reason,
                    max_similarity=max_sim_val,
                    choice_of_law=col_resolution.to_dict() if col_resolution else None,
                    proof_trace=abst_trace.to_dict(),
                    merkle_root=smt_root,
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
            abstained=False,
            abstention_reason=None,
            max_similarity=max_sim_val,
            choice_of_law=col_resolution.to_dict() if col_resolution else None,
            merkle_root=smt_root,
        )

        top_local_idx: int | None = None
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

        if self.conformal_calibrator is not None:
            conf_out = self.conformal_calibrator.predict_set(
                result.predicted_action_vector,
                max_similarity=max_sim_val,
            )
            result.conformal_set = conf_out.prediction_set
            result.conformal_threshold = conf_out.quantile_threshold
            result.conformal_alpha = conf_out.alpha
            if conf_out.is_empty and self.conformal_calibrator.abstain_on_empty:
                result.abstained = True
                result.abstention_reason = "conformal_empty_set"
                result.predicted_action_vector = [0.0] * self.value_dim
                result.most_relevant_rule = None
                result.confidence = 0.0
            elif conf_out.is_ambiguous and self.conformal_calibrator.abstain_on_ambiguous:
                result.abstained = True
                result.abstention_reason = "conformal_ambiguous_set"
                result.predicted_action_vector = [0.0] * self.value_dim
                result.most_relevant_rule = None
                result.confidence = 0.0

        trace = build_defeasible_proof_trace(
            query=query_text,
            all_tenant_records=all_tenant_records,
            active_records=valid_records,
            attention_weights=result.attention_weights,
            top_local_idx=top_local_idx,
            abstained=result.abstained,
            abstention_reason=result.abstention_reason,
            at_valid_time=eval_time,
            as_of_transaction_time=as_of_transaction_time,
        )
        result.proof_trace = trace.to_dict()

        return result

    def audit_deontic_consistency(
        self,
        tenant_id: str | None = None,
        at_time: datetime | None = None,
        register_contradiction_edges: bool = True,
    ) -> DeonticConsistencyReport:
        """
        Audit active legal memory records and semantic knowledge graph for deontic cycles
        and unresolved horizontal antinomies.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            records = self.episodic_memory.get_records(tenant_id=eff_tenant)
            return self.deontic_checker.audit(
                records=records,
                graph=self.semantic_graph,
                tenant_id=eff_tenant,
                at_time=at_time,
                register_contradiction_edges=register_contradiction_edges,
            )

    def simulate_rule_ingestion(
        self,
        rule_text: str,
        action_vector: list[float] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        tenant_id: str | None = None,
    ) -> PreFlightSimulationReport:
        """
        Execute a non-mutating Pre-Flight Standard Deontic Logic (KD) & Defeasible Precedence
        simulation for a candidate rule prior to committing it via `update_memory()`.
        """
        if not isinstance(rule_text, str) or len(rule_text.strip()) == 0:
            raise InvalidMemoryVectorError("Rule text must be a non-empty string.")
        eff_action = [0.0] * self.value_dim if action_vector is None else action_vector
        if len(eff_action) != self.value_dim:
            raise InvalidMemoryVectorError(
                f"Action vector must be a list of floats with length equal to {self.value_dim}."
            )
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        auth_rank = metadata.get("authority_rank", authority_rank) if metadata else authority_rank
        juris = metadata.get("jurisdiction", jurisdiction) if metadata else jurisdiction

        key_embed = self.encoder.get_embedding([rule_text])
        val_tensor = torch.tensor([eff_action], dtype=torch.float32)
        cand_id = (
            str(metadata.get("record_id") or metadata.get("rule_id"))
            if (metadata and ("record_id" in metadata or "rule_id" in metadata))
            else "candidate_rule"
        )
        candidate = MemoryRecord(
            text=rule_text,
            key_vector=key_embed,
            value_vector=val_tensor,
            record_id=cand_id,
            valid_from=valid_from or datetime.now(timezone.utc),
            valid_to=valid_to,
            metadata=metadata or {},
            authority_rank=auth_rank,
            jurisdiction=juris,
            tenant_id=eff_tenant,
        )
        with self._lock:
            existing_records = self.episodic_memory.get_records(tenant_id=eff_tenant)
            return self.modal_verifier.simulate_candidate(
                candidate=candidate,
                existing_records=existing_records,
                graph=self.semantic_graph,
            )

    def propose_norm_repairs(
        self,
        rule_text: str,
        action_vector: list[float] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        tenant_id: str | None = None,
    ) -> DraftingCopilotReport:
        """
        Synthesize and pre-verify legislative or contractual drafting repair proposals
        (`EXPLICIT_SUPERSESSION`, `LEX_SPECIALIS_CARVEOUT`, `TEMPORAL_PARTITION`,
        `DAG_CYCLE_PRUNING`) for a candidate rule via `NormativeDraftingCopilot`.
        """
        if not isinstance(rule_text, str) or len(rule_text.strip()) == 0:
            raise InvalidMemoryVectorError("Rule text must be a non-empty string.")
        eff_action = [0.0] * self.value_dim if action_vector is None else action_vector
        if len(eff_action) != self.value_dim:
            raise InvalidMemoryVectorError(
                f"Action vector must be a list of floats with length equal to {self.value_dim}."
            )
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        auth_rank = metadata.get("authority_rank", authority_rank) if metadata else authority_rank
        juris = metadata.get("jurisdiction", jurisdiction) if metadata else jurisdiction

        key_embed = self.encoder.get_embedding([rule_text])
        val_tensor = torch.tensor([eff_action], dtype=torch.float32)
        cand_id = (
            str(metadata.get("record_id") or metadata.get("rule_id"))
            if (metadata and ("record_id" in metadata or "rule_id" in metadata))
            else "candidate_rule"
        )
        candidate = MemoryRecord(
            text=rule_text,
            key_vector=key_embed,
            value_vector=val_tensor,
            record_id=cand_id,
            valid_from=valid_from or datetime.now(timezone.utc),
            valid_to=valid_to,
            metadata=metadata or {},
            authority_rank=auth_rank,
            jurisdiction=juris,
            tenant_id=eff_tenant,
        )
        with self._lock:
            existing_records = self.episodic_memory.get_records(tenant_id=eff_tenant)
            return self.drafting_copilot.propose_repairs(
                candidate=candidate,
                existing_records=existing_records,
                graph=self.semantic_graph,
            )

    def delete_rule(
        self,
        record_id: str | None = None,
        tenant_id: str | None = None,
        rule_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute Right-to-Erasure (GDPR Art. 17) across all memory tiers.

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
            rule_id: Optional alias for record_id.

        Returns:
            Audit record dictionary confirming multi-tier erasure metrics.

        Raises:
            KeyError: If record_id is not found in the episodic ledger or tenant mismatch.
        """
        target_id = record_id or rule_id
        if not target_id:
            raise KeyError("A non-empty record_id or rule_id must be provided for erasure.")

        with self._lock:
            deleted_rec = None
            for rec in self.episodic_memory.get_records():
                if rec.record_id == target_id:
                    deleted_rec = rec
                    break

            if deleted_rec is None:
                raise KeyError(f"Rule with ID '{target_id}' not found in episodic ledger.")

            if tenant_id is not None and deleted_rec.tenant_id != tenant_id:
                raise KeyError(f"Rule with ID '{target_id}' does not belong to tenant '{tenant_id}'.")

            # 1. Delete from Episodic Memory (records tombstone in hash chain)
            if hasattr(self.episodic_memory, "delete_record"):
                self.episodic_memory.delete_record(target_id)
            else:
                self.episodic_memory.record_tombstone(target_id)

            # 2. Clear from Working Memory
            self.working_memory.remove_by_text(deleted_rec.text, tenant_id=deleted_rec.tenant_id)

            # 3. Clear from Semantic Knowledge Graph
            matching_nodes = [
                nid for nid, node in self.semantic_graph.nodes.items()
                if (nid == target_id or node.description == deleted_rec.text)
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
            if not deleted_rec.personal_data and self.engine_mode != "structured":
                retained = self.episodic_memory.get_records()
                self.hope_module.memory.rebuild_from_records(
                    retained,
                    seed=self.seed,
                    skip_parametric=False,
                )
                parametric_rebuilt = True

            tombstone = self.episodic_memory._hash_chain[-1] if self.episodic_memory._hash_chain else ""

            # 6. Remove from Sparse Merkle Tree and generate cryptographic Non-Membership Proof
            for nid in matching_nodes:
                if nid != target_id:
                    self.merkle_tree.remove(nid, tenant_id=deleted_rec.tenant_id, record_log=False)
            new_smt_root = self.merkle_tree.remove(target_id, tenant_id=deleted_rec.tenant_id)
            non_membership_proof = self.merkle_tree.prove_non_membership(
                target_id,
                attestor=self.attestor,
            )

            return {
                "record_id": target_id,
                "deleted_at": datetime.now(timezone.utc).isoformat(),
                "tombstone_hash": tombstone,
                "personal_data": deleted_rec.personal_data,
                "status": "ERASED",
                "tiers_cleared": ["working", "episodic", "semantic", "neural_buffer", "sparse_merkle_tree"],
                "parametric_rebuilt": parametric_rebuilt,
                "merkle_root": new_smt_root,
                "merkle_non_membership_proof": non_membership_proof.to_dict(),
            }

    def prove_erasure(self, rule_id: str) -> SparseMerkleProof:
        """
        Generate a cryptographic Sparse Merkle Tree non-membership proof certifying that
        `rule_id` is absent from active memory under the current state root.
        """
        with self._lock:
            return self.merkle_tree.prove_non_membership(rule_id, attestor=self.attestor)

    def prove_inclusion(self, rule_id: str) -> SparseMerkleProof:
        """
        Generate a cryptographic Sparse Merkle Tree inclusion proof certifying that
        `rule_id` is present in active memory under the current state root.
        """
        with self._lock:
            return self.merkle_tree.prove_inclusion(rule_id, attestor=self.attestor)

    def verify_merkle_proof(
        self,
        proof: SparseMerkleProof | dict[str, Any],
        expected_root_hash: str | None = None,
    ) -> bool:
        """
        Verify a Sparse Merkle Tree inclusion or non-membership erasure proof.
        """
        return verify_sparse_merkle_proof(
            proof=proof,
            expected_root_hash=expected_root_hash,
            attestor=self.attestor,
        )

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

    get_active_context = get_working_memory_context

    def get_semantic_graph_nodes(self, tenant_id: str | None = None) -> list[GraphNode]:
        """
        Retrieve semantic knowledge graph nodes, optionally filtered by tenant.

        Args:
            tenant_id: Optional tenant identifier.
        """
        return self.semantic_graph.get_nodes(tenant_id=tenant_id)

    @contextmanager
    def tenant(self, tenant_id: str) -> Iterator[str]:
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

    def associate_statutes(
        self,
        seed_nodes: list[str] | dict[str, float],
        damping: float = 0.85,
        max_results: int = 10,
        tenant_id: str | None = None,
        at_time: datetime | None = None,
        as_of_transaction_time: datetime | None = None,
        deontic_polarity: bool = True,
    ) -> list[tuple[GraphNode, float]]:
        """
        Execute HippoRAG-style Personalized PageRank over the Semantic Knowledge Graph to discover
        multi-hop related legal entities and prerequisite clauses.

        Args:
            seed_nodes: List of seed node IDs or dict mapping node_id -> initial relevance weight.
            damping: Teleportation damping factor (default 0.85).
            max_results: Maximum number of ranked node results to return.
            tenant_id: Optional tenant identifier. If omitted, uses active tenant context.
            at_time: Temporal evaluation timestamp.
            as_of_transaction_time: Optional transaction-time cutoff for bi-temporal replay.
            deontic_polarity: Whether to enforce deontic edge polarity (`SUPERSEDES` redirection
                              and `CONTRADICTS` blocking).

        Returns:
            List of (GraphNode, ppr_score) tuples ordered by associative relevance.
        """
        eff_tenant = tenant_id if tenant_id is not None else getattr(self, "_current_tenant_id", None)
        scores = self.semantic_graph.personalized_pagerank(
            seed_weights=seed_nodes,
            damping=damping,
            tenant_id=eff_tenant,
            at_time=at_time,
            as_of_transaction_time=as_of_transaction_time,
            deontic_polarity=deontic_polarity,
        )
        results: list[tuple[GraphNode, float]] = []
        for nid, score in scores.items():
            if nid in self.semantic_graph.nodes:
                results.append((self.semantic_graph.nodes[nid], score))
            if len(results) >= max_results:
                break
        return results

    def _get_or_default_attestor(
        self,
    ) -> (
        AsymmetricAttestationModule
        | KeyedHashAttestationModule
        | PostQuantumHybridAttestationModule
    ):
        if isinstance(self.attestor, CryptographicKeyRing):
            return self.attestor.active_attestor
        if isinstance(
            self.attestor,
            (
                AsymmetricAttestationModule,
                KeyedHashAttestationModule,
                PostQuantumHybridAttestationModule,
            ),
        ):
            return self.attestor
        if not hasattr(self, "_default_asymmetric_attestor"):
            self._default_asymmetric_attestor = AsymmetricAttestationModule()
        return self._default_asymmetric_attestor

    def rotate_signing_key(
        self,
        new_key_id: str | None = None,
        algorithm: str | None = None,
        key_material: bytes | str | None = None,
    ) -> KeyMetadata:
        """
        Atomically rotate the active signing key in `self.keyring`, retiring the previous active key
        while preserving historical token verification. Also binds `self.attestor` to `self.keyring`.
        """
        with self._lock:
            meta = self.keyring.rotate_key(
                new_key_id=new_key_id,
                algorithm=algorithm,
                key_material=key_material,
            )
            self.attestor = self.keyring
            return meta

    def revoke_signing_key(self, key_id: str) -> KeyMetadata:
        """
        Permanently revoke `key_id` in `self.keyring` so any attestation signed by it is rejected.
        """
        with self._lock:
            meta = self.keyring.revoke_key(key_id)
            self.attestor = self.keyring
            return meta

    def list_signing_keys(self) -> list[dict[str, Any]]:
        """Return lifecycle metadata for all signing keys managed by `self.keyring`."""
        with self._lock:
            return self.keyring.list_keys()

    def generate_tee_quote(
        self,
        nonce: str | None = None,
        platform: str = "AWS_NITRO",
        tenant_id: str | None = None,
        signature_algorithm: str | None = None,
    ) -> TEEAttestationQuote:
        """
        Generate a hardware-anchored TEE Remote Attestation Quote (`AWS_NITRO` or `INTEL_TDX`)
        binding the current `SparseMerkleTree` root and active `CryptographicKeyRing` signing key
        to the enclave measurement registers.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            eff_alg = signature_algorithm
            if eff_alg is None:
                active_alg = getattr(self.keyring.active_attestor, "algorithm", "Ed25519")
                eff_alg = (
                    PQ_HYBRID_ALGORITHM
                    if active_alg == PQ_HYBRID_ALGORITHM
                    else "Ed25519"
                )
            return self.tee_provider.generate_quote(
                merkle_root=self.merkle_tree.root_hex,
                active_key_id=self.keyring.active_key_id,
                public_key_hex=self.keyring.public_key_hex,
                nonce=nonce,
                platform=platform,
                tenant_id=eff_tenant,
                signature_algorithm=eff_alg,
            )

    def verify_tee_quote(
        self,
        quote: TEEAttestationQuote | dict[str, Any],
        expected_nonce: str | None = None,
        expected_merkle_root: str | None = None,
        expected_measurements: dict[str, str] | None = None,
    ) -> TEEQuoteVerificationReport:
        """
        Cryptographically verify a `TEEAttestationQuote` against expected nonce, Sparse Merkle root,
        and enclave measurement registers (`PCR0..2` or `MRTD`/`RTMR0..1`).
        """
        return self.tee_provider.verify_quote(
            quote=quote,
            expected_nonce=expected_nonce,
            expected_merkle_root=expected_merkle_root,
            expected_measurements=expected_measurements,
        )

    def generate_compliance_proof(
        self,
        record_id: str,
        query: str = "",
        min_required_authority_rank: int = 1,
        at_time: datetime | None = None,
        as_of_transaction_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> SelectiveDisclosureBundle:
        """
        Generate a Zero-Knowledge / Selective-Disclosure Normative Compliance Proof (EU AI Act Art. 12 & 78)
        for `record_id` without exposing the underlying confidential rule text or query in plaintext.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            records = self.episodic_memory.get_records(tenant_id=eff_tenant)
            target_rec = next((r for r in records if r.record_id == record_id), None)
            if target_rec is None:
                raise KeyError(f"Record '{record_id}' not found in tenant '{eff_tenant}'.")
            prover = SelectiveDisclosureProver(attestation_module=self._get_or_default_attestor())
            return prover.generate_proof(
                record=target_rec,
                merkle_tree=self.merkle_tree,
                query=query,
                min_required_authority_rank=min_required_authority_rank,
                at_time=at_time,
                as_of_transaction_time=as_of_transaction_time,
            )

    def verify_compliance_proof(
        self,
        proof: SelectiveDisclosureComplianceProof | dict[str, Any],
        expected_merkle_root: str | None = None,
        min_authority_rank: int | None = None,
        required_jurisdiction: str | None = None,
    ) -> bool:
        """
        Verify a Selective-Disclosure Normative Compliance Proof without accessing plaintext text.
        """
        prover = SelectiveDisclosureProver(attestation_module=self._get_or_default_attestor())
        return prover.verify_proof(
            proof=proof,
            expected_merkle_root=expected_merkle_root,
            min_authority_rank=min_authority_rank,
            required_jurisdiction=required_jurisdiction,
        )

    def export_compliance_bundle(
        self,
        tenant_id: str | None = None,
        redact_personal_data: bool = False,
    ) -> SignedComplianceBundle:
        """
        Export a cryptographically signed `.clm.bundle.json` snapshot for `tenant_id`.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            records = self.episodic_memory.get_records(tenant_id=eff_tenant)
            nodes = self.semantic_graph.get_nodes(tenant_id=eff_tenant)
            edges = self.semantic_graph.get_edges(tenant_id=eff_tenant)
            manager = PortableBundleManager(attestation_module=self._get_or_default_attestor())
            return manager.export_bundle(
                records=records,
                graph_nodes=nodes,
                graph_edges=edges,
                merkle_tree=self.merkle_tree,
                tenant_id=eff_tenant,
                redact_personal_data=redact_personal_data,
            )

    def verify_compliance_bundle(
        self,
        bundle: SignedComplianceBundle | dict[str, Any],
    ) -> bool:
        """
        Cryptographically verify a `SignedComplianceBundle` signature, Sparse Merkle root, and audit chain.
        """
        manager = PortableBundleManager(attestation_module=self._get_or_default_attestor())
        return manager.verify_bundle(bundle)

    def diff_merkle_states(
        self,
        other_state: SparseMerkleTree | SignedComplianceBundle | dict[str, Any],
    ) -> MerkleStateDiff:
        """
        Compute the exact cryptographic leaf-level diff between this orchestrator's
        `SparseMerkleTree` and `other_state`.
        """
        with self._lock:
            verifier = FederatedComplianceVerifier(attestation_module=self._get_or_default_attestor())
            return verifier.diff_merkle_states(self.merkle_tree, other_state)

    def verify_cross_party_alignment(
        self,
        proofs: list[SelectiveDisclosureComplianceProof | dict[str, Any]],
        min_authority_rank: int = 1,
        min_action_alignment: float = 0.0,
        require_temporal_validity: bool = True,
    ) -> CrossPartyAlignmentReport:
        """
        Verify multi-party `SelectiveDisclosureComplianceProof` certificates for cryptographic
        validity, authority rank, temporal validity, and pairwise action vector alignment.
        """
        verifier = FederatedComplianceVerifier(attestation_module=self._get_or_default_attestor())
        return verifier.verify_cross_party_alignment(
            proofs=proofs,
            min_authority_rank=min_authority_rank,
            min_action_alignment=min_action_alignment,
            require_temporal_validity=require_temporal_validity,
        )

    def restore_compliance_bundle(
        self,
        bundle: SignedComplianceBundle | dict[str, Any],
    ) -> SignedComplianceBundle:
        """
        Verify and restore a `SignedComplianceBundle` into this orchestrator's multi-tier state.
        """
        manager = PortableBundleManager(attestation_module=self._get_or_default_attestor())
        records, nodes, edges, smt = manager.restore_bundle(bundle)
        parsed = SignedComplianceBundle.from_dict(bundle) if isinstance(bundle, dict) else bundle
        with self._lock:
            self.episodic_memory._ledger = list(records)
            self.semantic_graph.nodes = {n.node_id: n for n in nodes}
            self.semantic_graph.edges = list(edges)
            self.merkle_tree = smt
            self.hope_module.memory.rebuild_from_records(
                records,
                seed=self.seed,
                skip_parametric=(self.engine_mode == "structured"),
            )
        return parsed

    def export_to_qdrant(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """Export tenant records as Qdrant multi-vector point structures."""
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            records = self.episodic_memory.get_records(tenant_id=eff_tenant)
            return DistributedVectorGraphExporter.to_qdrant_points(records, self.merkle_tree)

    def export_to_pgvector(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """Export tenant records as PostgreSQL `pgvector` parameterized row dicts."""
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            records = self.episodic_memory.get_records(tenant_id=eff_tenant)
            return DistributedVectorGraphExporter.to_pgvector_rows(records, self.merkle_tree)

    def export_to_neo4j(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """Export tenant Semantic Knowledge Graph nodes and edges as parameterized Neo4j Cypher statements."""
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            nodes = self.semantic_graph.get_nodes(tenant_id=eff_tenant)
            edges = self.semantic_graph.get_edges(tenant_id=eff_tenant)
            return DistributedVectorGraphExporter.to_neo4j_cypher(nodes, edges)

    # Intuitive method alias
    ingest_rule = update_memory

    async def async_update_memory(
        self,
        rule_text: str,
        action_vector: list[float] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        personal_data: bool = False,
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> MemoryRecord:
        """
        Asynchronously update cognitive memory without blocking the active asyncio event loop.
        """
        if "is_personal_data" in kwargs:
            personal_data = bool(kwargs["is_personal_data"])
        return await asyncio.to_thread(
            self.update_memory,
            rule_text=rule_text,
            action_vector=action_vector,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
            authority_rank=authority_rank,
            jurisdiction=jurisdiction,
            personal_data=personal_data,
            tenant_id=tenant_id,
        )

    async def async_ingest_rule(
        self,
        rule_text: str,
        action_vector: list[float] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        personal_data: bool = False,
        tenant_id: str = "default",
        **kwargs: Any,
    ) -> MemoryRecord:
        """Alias for async_update_memory."""
        return await self.async_update_memory(
            rule_text=rule_text,
            action_vector=action_vector,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
            authority_rank=authority_rank,
            jurisdiction=jurisdiction,
            personal_data=personal_data,
            tenant_id=tenant_id,
            **kwargs,
        )

    async def async_predict(
        self,
        query_text: str,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
        min_similarity_threshold: float | None = None,
        jurisdiction: str | list[str] | None = None,
        connecting_factors: ConnectingFactors | dict[str, Any] | None = None,
        as_of_transaction_time: datetime | None = None,
    ) -> PredictionResult:
        """
        Asynchronously predict decision vectors without blocking the active asyncio event loop.
        """
        return await asyncio.to_thread(
            self.predict,
            query_text=query_text,
            at_time=at_time,
            tenant_id=tenant_id,
            min_similarity_threshold=min_similarity_threshold,
            jurisdiction=jurisdiction,
            connecting_factors=connecting_factors,
            as_of_transaction_time=as_of_transaction_time,
        )

    async def async_delete_rule(
        self,
        record_id: str | None = None,
        tenant_id: str | None = None,
        rule_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously erase a rule across all memory tiers without blocking the active asyncio event loop.
        """
        target_id = record_id or rule_id
        return await asyncio.to_thread(
            self.delete_rule,
            record_id=target_id,
            tenant_id=tenant_id,
        )

    async def async_associate_statutes(
        self,
        seed_nodes: list[str] | dict[str, float],
        damping: float = 0.85,
        max_results: int = 10,
        tenant_id: str | None = None,
        at_time: datetime | None = None,
        as_of_transaction_time: datetime | None = None,
        deontic_polarity: bool = True,
    ) -> list[tuple[GraphNode, float]]:
        """
        Asynchronously execute HippoRAG Personalized PageRank over the legal graph.
        """
        return await asyncio.to_thread(
            self.associate_statutes,
            seed_nodes=seed_nodes,
            damping=damping,
            max_results=max_results,
            tenant_id=tenant_id,
            at_time=at_time,
            as_of_transaction_time=as_of_transaction_time,
            deontic_polarity=deontic_polarity,
        )

    async def async_simulate_rule_ingestion(
        self,
        rule_text: str,
        action_vector: list[float] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        tenant_id: str | None = None,
    ) -> PreFlightSimulationReport:
        """
        Asynchronously execute a pre-flight deontic modal simulation without blocking the event loop.
        """
        return await asyncio.to_thread(
            self.simulate_rule_ingestion,
            rule_text=rule_text,
            action_vector=action_vector,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
            authority_rank=authority_rank,
            jurisdiction=jurisdiction,
            tenant_id=tenant_id,
        )

    async def async_propose_norm_repairs(
        self,
        rule_text: str,
        action_vector: list[float] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        authority_rank: int = 1,
        jurisdiction: str | None = None,
        tenant_id: str | None = None,
    ) -> DraftingCopilotReport:
        """
        Asynchronously synthesize and verify normative drafting repair proposals without blocking the event loop.
        """
        return await asyncio.to_thread(
            self.propose_norm_repairs,
            rule_text=rule_text,
            action_vector=action_vector,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata,
            authority_rank=authority_rank,
            jurisdiction=jurisdiction,
            tenant_id=tenant_id,
        )

    def export_smtlib(
        self,
        query_text: str | None = None,
        tenant_id: str | None = None,
        at_time: datetime | None = None,
    ) -> SMTLibExportBundle:
        """
        Export the active tenant's normative rules and optional query `DefeasibleProofTrace`
        as a formal SMT-LIB 2.6 specification (`QF_UFLIA`) for Z3 / cvc5 verification.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        proof_trace: dict[str, Any] | None = None
        if query_text and query_text.strip():
            pred = self.predict(query_text=query_text, at_time=at_time, tenant_id=eff_tenant)
            proof_trace = pred.proof_trace
        with self._lock:
            eval_time = at_time or datetime.now(timezone.utc)
            records = self.episodic_memory.get_valid_records(eval_time, tenant_id=eff_tenant)
            return self.smtlib_exporter.export_tenant_theory(
                records=records,
                proof_trace=proof_trace,
                tenant_id=eff_tenant,
            )

    async def async_export_smtlib(
        self,
        query_text: str | None = None,
        tenant_id: str | None = None,
        at_time: datetime | None = None,
    ) -> SMTLibExportBundle:
        """
        Asynchronously export the tenant's normative rules as an SMT-LIB 2.6 specification.
        """
        return await asyncio.to_thread(
            self.export_smtlib,
            query_text=query_text,
            tenant_id=tenant_id,
            at_time=at_time,
        )

    def simulate_temporal_horizon(
        self,
        query_text: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> RegulatoryTimelineReport:
        """
        Simulate how legal compliance outcomes, prevailing statutes, deontic modalities,
        and horizontal antinomies evolve across all `valid_from` / `valid_to` boundaries
        within `[start_time, end_time]`.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            records = self.episodic_memory.get_records(tenant_id=eff_tenant)
            return self.temporal_simulator.simulate_horizon(
                records=records,
                graph=self.semantic_graph,
                query_text=query_text,
                start_time=start_time,
                end_time=end_time,
                tenant_id=eff_tenant,
                predict_fn=lambda q, dt, t: self.predict(query_text=q, at_time=dt, tenant_id=t),
            )

    async def async_simulate_temporal_horizon(
        self,
        query_text: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> RegulatoryTimelineReport:
        """
        Asynchronously simulate the bi-temporal regulatory horizon across `[start_time, end_time]`.
        """
        return await asyncio.to_thread(
            self.simulate_temporal_horizon,
            query_text=query_text,
            start_time=start_time,
            end_time=end_time,
            tenant_id=tenant_id,
        )

    def export_standalone_verifier(
        self,
        tenant_id: str | None = None,
        redact_personal_data: bool = False,
        query_text: str | None = None,
    ) -> StandaloneVerifierArtifact:
        """
        Export a self-contained, zero-dependency HTML5 + WebCrypto single-file audit verifier
        embedding the tenant's signed compliance bundle and SMT-LIB 2.6 specification.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        bundle = self.export_compliance_bundle(
            tenant_id=eff_tenant,
            redact_personal_data=redact_personal_data,
        )
        smt_bundle = self.export_smtlib(query_text=query_text, tenant_id=eff_tenant)
        return self.standalone_verifier_exporter.export_html_verifier(
            bundle=bundle,
            smtlib_script=smt_bundle.smtlib_script,
        )

    async def async_export_standalone_verifier(
        self,
        tenant_id: str | None = None,
        redact_personal_data: bool = False,
        query_text: str | None = None,
    ) -> StandaloneVerifierArtifact:
        """
        Asynchronously export a self-contained HTML5 + WebCrypto standalone audit verifier.
        """
        return await asyncio.to_thread(
            self.export_standalone_verifier,
            tenant_id=tenant_id,
            redact_personal_data=redact_personal_data,
            query_text=query_text,
        )

    def issue_timestamp_token(
        self,
        tenant_id: str | None = None,
        nonce: str | None = None,
        policy_oid: str = TransparencyGossipAuditor.DEFAULT_POLICY_OID,
        issued_at: datetime | None = None,
    ) -> TimestampToken:
        """
        Issue an RFC 3161-inspired cryptographic `TimestampToken` binding the tenant's
        current 256-bit `SparseMerkleTree` root and `AppendOnlyMerkleLog` head hash.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            root = self.merkle_tree.root_hex
            entries = self.merkle_tree.audit_log.entries
            seq = len(entries)
            head_hash = entries[-1].entry_hash if entries else "0" * 64
            auditor = TransparencyGossipAuditor(attestor=self._get_or_default_attestor())
            return auditor.issue_timestamp_token(
                tenant_id=eff_tenant,
                merkle_root=root,
                log_head_hash=head_hash,
                sequence_number=seq,
                nonce=nonce,
                policy_oid=policy_oid,
                issued_at=issued_at,
            )

    def verify_timestamp_token(
        self,
        token: TimestampToken | dict[str, Any],
        expected_merkle_root: str | None = None,
        expected_nonce: str | None = None,
    ) -> bool:
        """
        Verify the cryptographic signature and message imprint of an RFC 3161-style `TimestampToken`.
        """
        auditor = TransparencyGossipAuditor(attestor=self._get_or_default_attestor())
        return auditor.verify_timestamp_token(
            token=token,
            expected_merkle_root=expected_merkle_root,
            expected_nonce=expected_nonce,
        )

    def audit_transparency_gossip(
        self,
        checkpoints: list[GossipCheckpoint | dict[str, Any]],
        tenant_id: str | None = None,
    ) -> SplitViewAuditReport:
        """
        Audit cross-witness transparency log checkpoints against the local `AppendOnlyMerkleLog`
        to detect split-view equivocation or history rollback attacks.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            auditor = TransparencyGossipAuditor(attestor=self._get_or_default_attestor())
            return auditor.audit_gossip_checkpoints(
                tenant_id=eff_tenant,
                local_log=self.merkle_tree.audit_log,
                checkpoints=checkpoints,
            )

    def analyze_statutory_amendment(
        self,
        target_rule_id: str,
        amended_text: str,
        amended_action_vector: list[float] | None = None,
        amended_authority_rank: int | None = None,
        tenant_id: str | None = None,
    ) -> AmendmentImpactReport:
        """
        Perform clause-level statutory amendment diffing on `target_rule_id` and trace
        the transitive compliance blast radius across subordinate policies and contracts.
        """
        eff_tenant = (
            tenant_id
            if tenant_id is not None
            else (getattr(self, "_current_tenant_id", None) or "default")
        )
        with self._lock:
            records = self.episodic_memory.get_records(tenant_id=eff_tenant)
            target_rec = next(
                (r for r in records if r.record_id == target_rule_id),
                None,
            )
            if target_rec is None:
                raise KeyError(
                    f"Active rule '{target_rule_id}' not found in tenant '{eff_tenant}'."
                )
            return self.amendment_analyzer.analyze_amendment_impact(
                target_record=target_rec,
                amended_text=amended_text,
                amended_action_vector=amended_action_vector,
                active_records=records,
                graph=self.semantic_graph,
                tenant_id=eff_tenant,
                amended_authority_rank=amended_authority_rank,
            )

    async def async_issue_timestamp_token(
        self,
        tenant_id: str | None = None,
        nonce: str | None = None,
        policy_oid: str = TransparencyGossipAuditor.DEFAULT_POLICY_OID,
    ) -> TimestampToken:
        """Asynchronously issue an RFC 3161-inspired cryptographic `TimestampToken`."""
        return await asyncio.to_thread(
            self.issue_timestamp_token,
            tenant_id=tenant_id,
            nonce=nonce,
            policy_oid=policy_oid,
        )

    async def async_audit_transparency_gossip(
        self,
        checkpoints: list[GossipCheckpoint | dict[str, Any]],
        tenant_id: str | None = None,
    ) -> SplitViewAuditReport:
        """Asynchronously audit cross-witness transparency log checkpoints."""
        return await asyncio.to_thread(
            self.audit_transparency_gossip,
            checkpoints=checkpoints,
            tenant_id=tenant_id,
        )

    async def async_analyze_statutory_amendment(
        self,
        target_rule_id: str,
        amended_text: str,
        amended_action_vector: list[float] | None = None,
        amended_authority_rank: int | None = None,
        tenant_id: str | None = None,
    ) -> AmendmentImpactReport:
        """Asynchronously analyze the clause-level diff and downstream impact of a statutory amendment."""
        return await asyncio.to_thread(
            self.analyze_statutory_amendment,
            target_rule_id=target_rule_id,
            amended_text=amended_text,
            amended_action_vector=amended_action_vector,
            amended_authority_rank=amended_authority_rank,
            tenant_id=tenant_id,
        )





