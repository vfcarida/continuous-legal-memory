# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] - 2026-10-19

### Added
- **Cryptographic Timestamp Authority (RFC 3161) & Transparency Log Gossip (`continuous_legal_memory/security/timestamp_gossip.py`)**:
  - Implemented `TimestampToken`, `GossipCheckpoint`, `SplitViewAuditReport`, and `TransparencyGossipAuditor` (`orchestrator.issue_timestamp_token()`, `orchestrator.verify_timestamp_token()`, `orchestrator.audit_transparency_gossip()`, and async counterparts).
  - Binds the tenant's 256-bit `SparseMerkleTree` state root and `AppendOnlyMerkleLog` head hash into a signed RFC 3161-style timestamp token (`PQ-Hybrid-Ed25519-SHA256`, `Ed25519`, or `HMAC-SHA256`) and detects cross-witness split-view equivocation and history rollback attacks.
- **Automated Statutory Diff & Downstream Amendment Impact Analyzer (`continuous_legal_memory/reasoning/amendment_impact.py`)**:
  - Implemented `ClauseDiffSpan`, `DownstreamObligationImpact`, `AmendmentImpactReport`, and `StatutoryAmendmentAnalyzer` (`orchestrator.analyze_statutory_amendment()` and `orchestrator.async_analyze_statutory_amendment()`).
  - Performs clause-level legislative amendment diffing, detects deontic modality reversals (`OBLIGATION` $\leftrightarrow$ `PROHIBITION` $\leftrightarrow$ `PERMISSION`), and traces the transitive compliance blast radius across subordinate contracts and corporate policies via the Semantic Knowledge Graph (`DEPENDS_ON`, `DERIVES_FROM`), shared statutory citations, and subject-matter overlap.
- **OpenAPI 3.1.0 Specification & Full REST, MCP, and CLI Parity (`server.py`, `mcp.py`, `cli.py`)**:
  - Added `generate_openapi_spec()` and `GET /v1/openapi.json` to `server.py`.
  - Added REST endpoints `POST /v1/merkle/timestamp`, `POST /v1/merkle/gossip-audit`, and `POST /v1/rules/amendment-impact`.
  - Added MCP tools `legal_memory_issue_timestamp`, `legal_memory_gossip_audit`, and `legal_memory_amendment_impact`, and CLI subcommands `clm timestamp-root`, `clm gossip-audit`, and `clm amendment-impact`.
- **Consolidated References & Subpackage Exports (`docs/REFERENCES.md`, `reasoning/__init__.py`, `security/__init__.py`, `retrieval/__init__.py`)**:
  - Added `docs/REFERENCES.md` and exported all public reasoning, security, and retrieval symbols in subpackage `__all__` manifests.

---

## [1.1.0] - 2026-10-19

### Added
- **End-to-End Post-Quantum Hybrid Attestation (`PQ-Hybrid-Ed25519-SHA256`) Across All Cryptographic Subsystems (`selective_disclosure.py`, `portable_bundle.py`, `federated_sync.py`, `tee_attestation.py`, `merkle.py`)**:
  - Extended `SelectiveDisclosureProver`, `PortableBundleManager`, `FederatedComplianceVerifier`, `TEEAttestationProvider`, and `SparseMerkleTree.generate_proof()` / `verify_sparse_merkle_proof()` to sign and verify with `PostQuantumHybridAttestationModule` (`Ed25519 + WOTS+-SHA256`).
- **Bi-Temporal Regulatory Horizon Simulator (`continuous_legal_memory/reasoning/temporal_simulator.py`)**:
  - Implemented `TemporalHorizonSimulator`, `TimelineEpochTransition`, and `RegulatoryTimelineReport` (`orchestrator.simulate_temporal_horizon()`, `orchestrator.async_simulate_temporal_horizon()`), sweeping across all `valid_from` phase-in and `valid_to` sunset boundaries to forecast deontic modality shifts and future horizontal antinomies.
- **Global Multi-Jurisdictional Expansion (`UK`, `SG`, `JP`, `CH`) & Multilingual Deontic Extraction (`choice_of_law.py`, `hybrid_lexical.py`, `modal_verifier.py`, `norm_extractor.py`)**:
  - Added `SUPPORTED_JURISDICTION_REGIMES` for `UK` (UK GDPR / DPA 2018 / FCA), `SG` (PDPA / MAS TRM), `JP` (APPI Art. 171), and `CH` (Swiss FADP / revDSG SR 235.1 & PILA Art. 18), extended statutory pin-cite extraction (`SR 235.1`, `s. 24`, `Artikel`), and added multilingual deontic modality regexes (`EN`, `PT`, `FR`, `DE`).
- **Client-Side Standalone Audit Verifier Exporter (`continuous_legal_memory/security/standalone_verifier.py`) & REST/MCP/CLI Parity**:
  - Implemented `StandaloneAuditVerifierExporter` and `StandaloneVerifierArtifact` (`orchestrator.export_standalone_verifier()`, `orchestrator.async_export_standalone_verifier()`), generating a self-contained, zero-dependency HTML5 + WebCrypto single-file audit bundle & SMT-LIB inspector.
  - Added REST endpoints `POST /v1/rules/timeline` and `POST /v1/audit/standalone-verifier`, MCP tools `legal_memory_simulate_timeline` and `legal_memory_export_verifier`, and CLI subcommands `clm simulate-timeline` and `clm export-verifier`.

---

## [1.0.0] - 2026-10-19

### Added
- **Post-Quantum Hybrid Cryptographic Attestation (`continuous_legal_memory/security/pq_hybrid.py` & `keyring.py`)**:
  - Implemented `PostQuantumHybridAttestationModule` (`algorithm="PQ-Hybrid-Ed25519-SHA256"`), combining classical RFC 8032 `Ed25519` digital signatures with a 67-chain ($w=16$) Winternitz **WOTS+-SHA256** (RFC 8391 / NIST FIPS 205 SLH-DSA building block) post-quantum hash-based signature.
  - Integrated `"PQ-Hybrid-Ed25519-SHA256"` natively into `CryptographicKeyRing` (`rotate_signing_key`, `verify_attestation`, and SQLite WAL persistence), requiring both classical and post-quantum signatures to verify.
- **Formal SMT-LIB 2.6 Export Bridge & `DefeasibleProofTrace.winning_canon` (`continuous_legal_memory/reasoning/smtlib_exporter.py` & `proof_trace.py`)**:
  - Implemented `SMTLibDeonticExporter` and `SMTLibExportBundle` (`orchestrator.export_smtlib()`, `orchestrator.async_export_smtlib()`), compiling active tenant rules and `DefeasibleProofTrace` derivations into standard SMT-LIB 2.6 scripts (`QF_UFLIA`) with Standard Deontic Logic ($\mathbf{KD}$) axioms and pairwise Tri-Canon precedence for Z3 and cvc5 verification.
  - Added top-level `winning_canon` summary field (`LEX_SUPERIOR`, `LEX_SPECIALIS`, `LEX_POSTERIOR`, `EXPLICIT_SUPERSEDES`, `LOIS_DE_POLICE`, `SEMANTIC_RELEVANCE`, `UNCONTESTED_PRIMARY_NORM`, `ABSTAINED`) to `DefeasibleProofTrace`.
- **Extended OpenMetrics / Prometheus Governance Telemetry (`continuous_legal_memory/telemetry/observability.py`)**:
  - Added `clm_keyring_keys_total{status="ACTIVE|RETIRED|REVOKED"}`, `clm_conformal_calibrated`, `clm_conformal_q_hat`, and `clm_predictions_abstained_total` to `export_prometheus_metrics()`.
- **REST API, MCP Server & CLI `v1.0.0` GA Parity (`server.py`, `mcp.py`, `cli.py`)**:
  - Added REST endpoint `POST /v1/audit/smtlib`, MCP tool `legal_memory_export_smtlib`, and CLI subcommand `clm export-smtlib`, and updated `clm key-rotate` / `legal_memory_rotate_key` to support `"PQ-Hybrid-Ed25519-SHA256"`.

---

## [0.9.0] - 2026-10-19

### Added
- **Hardware/TEE Enclave Remote Attestation Quotes (`continuous_legal_memory/security/tee_attestation.py`)**:
  - Implemented `TEEAttestationProvider`, `TEEAttestationQuote`, `TEEQuoteVerificationReport`, and `TEEPlatform` (`AWS_NITRO`, `INTEL_TDX`).
  - Binds the tenant's 256-bit `SparseMerkleTree` state root, active `CryptographicKeyRing` signing key, and anti-replay challenge `nonce` into hardware-anchored measurement registers (`PCR0..PCR2` for AWS Nitro Enclaves and `MRTD`/`RTMR0..1` for Intel TDX) via `orchestrator.generate_tee_quote()` and `orchestrator.verify_tee_quote()`.
- **Formal Legislative & Contract Normative Drafting Co-Pilot (`continuous_legal_memory/reasoning/drafting_copilot.py`)**:
  - Implemented `NormativeDraftingCopilot`, `DraftingRepairOption`, `DraftingRepairStrategy`, and `DraftingCopilotReport` (`orchestrator.propose_norm_repairs()`, `orchestrator.async_propose_norm_repairs()`).
  - Synthesizes and automatically re-verifies `EXPLICIT_SUPERSESSION`, `LEX_SPECIALIS_CARVEOUT`, `TEMPORAL_PARTITION`, and `DAG_CYCLE_PRUNING` repair options through `DeonticModalVerifier` to guarantee `post_repair_satisfiable == True` and zero unintended shadowing.
- **Upgraded Ecosystem Integrations (`continuous_legal_memory/integrations/langchain.py`, `llamaindex.py`, `letta.py`)**:
  - Enhanced `ContinuousLegalMemoryLangChain` with `include_proof_trace`, `include_merkle_root`, `get_last_prediction()`, bi-temporal/choice-of-law inputs, and `[EPISTEMIC ABSTENTION]` formatting.
  - Enhanced `ContinuousLegalMemoryLlamaRetriever` with `merkle_root`, `proof_trace`, `conformal_set`, and `include_conformal_set` multi-node expansion.
  - Enhanced `ContinuousLegalMemoryBlock` and `create_letta_tools()` with `legal_memory_simulate`, `legal_memory_repair`, and `legal_memory_erase` agent tools.
- **REST API, MCP Server & CLI Parity (`server.py`, `mcp.py`, `cli.py`)**:
  - Added REST endpoints `POST /v1/rules/repair`, `POST /v1/tee/quote`, and `POST /v1/tee/verify`.
  - Added MCP tools `legal_memory_propose_repairs` and `legal_memory_tee_quote`, and CLI subcommands `clm repair-rule` and `clm tee-quote`.

---

## [0.8.0] - 2026-10-19

### Added
- **Cross-Tenant Federated Normative Verification & Merkle State Diff (`continuous_legal_memory/security/federated_sync.py`)**:
  - Implemented `FederatedComplianceVerifier`, `MerkleStateDiff`, and `CrossPartyAlignmentReport` (`orchestrator.diff_merkle_states()`, `orchestrator.verify_cross_party_alignment()`).
  - Enables exact cryptographic leaf-level state reconciliation (`added_rule_ids`, `removed_rule_ids`, `modified_rule_ids`, `unchanged_rule_ids`) and multi-party zero-knowledge compliance alignment verification across sovereign regulatory nodes.
- **Full SQLite WAL Persistence for `CryptographicKeyRing` & `SplitConformalCalibrator` (`continuous_legal_memory/storage/sqlite_store.py`)**:
  - Added deterministic `to_dict()` / `from_dict()` serialization to `CryptographicKeyRing` and `SplitConformalCalibrator` and persisted both in `SqliteMemoryStore.save_orchestrator()` / `load_into_orchestrator()`.
- **Automated Normative Entity & Deontic Graph Extractor (`continuous_legal_memory/reasoning/norm_extractor.py`)**:
  - Implemented `NormativeGraphExtractor` and `ExtractedNormStructure`, automatically inferring `EntityType` (`OBLIGATION`, `PROHIBITION`, `PERMISSION`, `STATUTE`) from deontic modal cues and wiring acyclic `RelationType.SUPERSEDES` (from textual supersession clauses) and `RelationType.DEPENDS_ON` (from shared statutory citations) edges during `update_memory()`.
- **Key Lifecycle & Federated Verification Endpoints (`server.py`, `mcp.py`, `cli.py`)**:
  - Added REST endpoints `GET /v1/keys`, `POST /v1/keys/rotate`, `POST /v1/keys/revoke`, and `POST /v1/federated/verify`.
  - Added MCP tools `legal_memory_rotate_key`, `legal_memory_revoke_key`, and `legal_memory_federated_verify`, and CLI subcommands `clm key-list`, `clm key-rotate`, and `clm key-revoke`.

---

## [0.7.0] - 2026-10-19

### Added
- **Formal Deontic Modal Satisfiability ($\mathbf{KD}$) & Pre-Flight Rule Ingestion Simulator (`continuous_legal_memory/reasoning/modal_verifier.py`)**:
  - Implemented `DeonticModality` (`OBLIGATION`, `PROHIBITION`, `PERMISSION`, `NEUTRAL`), `DeonticModalVerifier`, `ModalConflictFinding`, and `PreFlightSimulationReport`.
  - Enables non-mutating pre-flight verification (`orchestrator.simulate_rule_ingestion()`, `POST /v1/rules/simulate`, MCP `legal_memory_simulate_rule`, CLI `clm simulate-rule`) to detect Standard Deontic Logic ($\mathbf{KD}$) contradictions ($\mathcal{O}\varphi \land \mathcal{F}\varphi$, $\mathcal{F}\varphi \land \mathcal{P}\varphi$), DAG cycles, and defeasible shadowing prior to rule ingestion.
- **Cryptographic Key Lifecycle & Rotation Ring (`continuous_legal_memory/security/keyring.py`)**:
  - Implemented `CryptographicKeyRing` and `KeyMetadata` supporting atomic key rotation (`ACTIVE` $\rightarrow$ `RETIRED`), historical token verification across retired keys, and immediate revocation (`REVOKED` / Key Revocation List) for both `Ed25519` and `HMAC-SHA256` keys (`orchestrator.rotate_signing_key()`, `orchestrator.revoke_signing_key()`, `orchestrator.list_signing_keys()`).
- **Full Bi-Temporal `GraphNode.transaction_time` Persistence in SQLite WAL (`continuous_legal_memory/storage/sqlite_store.py`)**:
  - Extended `semantic_nodes` SQLite schema with automatic backward-compatible column migration (`ALTER TABLE semantic_nodes ADD COLUMN transaction_time TEXT`) so `GraphNode.transaction_time` persists and reloads identically across `save_to_disk()` / `load_from_disk()`.
- **CLI Subcommands & Prometheus Sparse Merkle Gauges (`continuous_legal_memory/cli.py`, `continuous_legal_memory/telemetry/observability.py`)**:
  - Added `clm compliance-proof`, `clm export-bundle`, `clm restore-bundle`, and `clm simulate-rule` CLI subcommands.
  - Added `clm_merkle_active_leaves`, `clm_merkle_audit_log_entries_total`, and `clm_merkle_chain_verified` gauges to OpenMetrics / Prometheus telemetry export.

---

## [0.6.0] - 2026-10-19

### Added
- **Selective-Disclosure / Zero-Knowledge Normative Compliance Proofs (`continuous_legal_memory/security/selective_disclosure.py`)**:
  - Implemented `SelectiveDisclosureProver`, `SelectiveDisclosureComplianceProof`, and `SelectiveDisclosureBundle` aligned with EU AI Act Article 12 (Record-Keeping) and Article 78 (Confidentiality).
  - Enables third-party auditors to cryptographically verify 256-bit Sparse Merkle Tree inclusion, minimum Kelsenian `authority_rank`, jurisdiction, and bi-temporal validity (`at_time` $\times$ `as_of_transaction_time`) via Ed25519 or HMAC-SHA256 signatures over salted SHA-256 commitments (`H(salt || utf8(plaintext))`) without exposing confidential contract text or queries, with optional selective unblinding (`verify_disclosed_text`, `verify_disclosed_query`).
- **Deontic Polarity-Aware Graph Diffusion & Bi-Temporal Graph Nodes (`continuous_legal_memory/core/memory_tiers/semantic_memory.py`)**:
  - Extended `GraphNode` with `transaction_time` and `is_known_as_of(as_of_transaction_time)`.
  - Upgraded `SemanticKnowledgeGraph.personalized_pagerank()` with `deontic_polarity=True` (redirecting random-walk and teleport activation mass from superseded norms along `RelationType.SUPERSEDES` to their active superseding norms while blocking positive walk leakage across `RelationType.CONTRADICTS` edges) and `as_of_transaction_time` filtering.
- **Signed Portable Compliance Bundles & Distributed Vector/Graph Exporters (`continuous_legal_memory/storage/portable_bundle.py`)**:
  - Implemented `SignedComplianceBundle` and `PortableBundleManager` for tamper-evident `.clm.bundle.json` snapshot export (with optional GDPR personal-data redaction), Ed25519/HMAC signature + Sparse Merkle root verification, and state restoration.
  - Implemented `DistributedVectorGraphExporter` providing structured exporters for Qdrant (`to_qdrant_points`), PostgreSQL `pgvector` (`to_pgvector_rows`), and parameterized Neo4j Cypher `MERGE` statements (`to_neo4j_cypher`).
- **Refined `HybridLegalRetriever` (`continuous_legal_memory/retrieval/hybrid_retriever.py`) & REST/MCP Parity**:
  - Updated `HybridLegalRetriever.retrieve()` to return precedence-scaled scores (`scaled_scores`) and statutory citation pinpointing (`compute_lexical_citation_scores`), and upgraded `extract_character_snippet()` to center snippets on matched statutory citations or highest keyword-density windows.
  - Added REST endpoints `POST /v1/compliance/prove`, `POST /v1/compliance/verify`, `POST /v1/bundle/export`, and `POST /v1/bundle/restore` (`server.py`) and MCP tools `legal_memory_compliance_prove`, `legal_memory_compliance_verify`, and `legal_memory_bundle_export` (`mcp.py`).

---

## [0.5.0] - 2026-10-19

### Added
- **Strict Bi-Temporal 'As-Of' Auditing & Historical Merkle Root Replay (`at_time` $\times$ `as_of_transaction_time`)**:
  - Extended `MemoryRecord.is_known_as_of()`, `MemoryRecord.is_temporally_valid()`, `EpisodicMemory.get_valid_records()`, `EpisodicMemory.get_valid_indices()`, and `LegalMemoryOrchestrator.predict()` / `async_predict()` with orthogonal `as_of_transaction_time` filtering for point-in-time regulatory replay (*ex-tunc* vs. *ex-nunc* retroactive amendments).
  - Added `AppendOnlyMerkleLog.get_root_as_of()` and `SparseMerkleTree.get_root_as_of()` to reconstruct historical 256-bit Merkle roots as of any transaction timestamp.
- **Hybrid Lexical-Dense Retrieval & Statutory Citation Pinpointing (`continuous_legal_memory/retrieval/hybrid_lexical.py`)**:
  - Implemented `HybridLexicalCitationScorer` and `extract_legal_citations` (Okapi BM25+ sublinear lexical scoring and canonical multi-jurisdictional citation extraction across EU Articles/Regulations, US Code/CFR/Rules, and BR Artigos/Leis), integrated into `TemporalStructuredRetriever`.
- **Defeasible Proof-Tree Explanation & Auditable Reasoning Trace (`continuous_legal_memory/reasoning/proof_trace.py`)**:
  - Implemented `DefeasibleProofTrace`, `CandidateProofNode`, and `build_defeasible_proof_trace` satisfying EU AI Act Art. 13 & Art. 86 explainability, detailing per-candidate defeat canons (`PREVAILED`, `DEFEATED_BY_SUPERSEDES`, `DEFEATED_BY_LOIS_DE_POLICE`, `DEFEATED_BY_LEX_SUPERIOR`, `DEFEATED_BY_LEX_SPECIALIS`, `DEFEATED_BY_LEX_POSTERIOR`, `EXCLUDED_TRANSACTION_TIME`, `EXCLUDED_TEMPORAL_VALIDITY`) with a deterministic SHA-256 `trace_hash`.
- **Full REST Microservice (`server.py`) & CLI (`cli.py`) Parity**:
  - Added `GET /v1/merkle/root`, `GET /v1/merkle/proof/<rule_id>`, `POST /v1/merkle/verify`, `POST /v1/audit/consistency`, and `POST /v1/conformal/calibrate` endpoints to `server.py`.
  - Added `clm audit` and `clm merkle-proof` subcommands and `--as-of-tx` / `--jurisdiction` query flags to `cli.py`.

---

## [0.4.0] - 2026-10-19

### Added
- **Formal Deontic Consistency Checker & Antinomy Detector (`continuous_legal_memory/reasoning/deontic_checker.py`)**:
  - Implemented `DeonticConsistencyChecker`, `DeonticAntinomy`, and `DeonticConsistencyReport` using Tarjan's Strongly Connected Components ($O(|V| + |E|)$) for cycle detection and simultaneous horizontal antinomy auditing.
  - Added ingestion-time acyclic graph guards on `SUPERSEDES`, `DEPENDS_ON`, and `DERIVES_FROM` relations in `SemanticKnowledgeGraph.add_edge()`.
  - Exposed `orchestrator.audit_deontic_consistency()` and MCP tool `legal_memory_audit_consistency`.
- **Private International Law Choice-of-Law Router (`continuous_legal_memory/retrieval/choice_of_law.py`)**:
  - Implemented `ChoiceOfLawRouter`, `ConnectingFactors`, and `ChoiceOfLawResolution` supporting *lex loci protectionis* (`data_subject_domicile`), *lex loci solutionis* (`place_of_performance`), *lex voluntatis* (`choice_of_forum`), and Overriding Mandatory Provisions (*lois de police*, Rome I Art. 9 / GDPR Art. 3(2) / LGPD Art. 3).
  - Integrated `jurisdiction` and `connecting_factors` into `orchestrator.predict()`, `orchestrator.async_predict()`, and `MCPServer` `legal_memory_predict`.
- **Full SQLite WAL Persistence for Sparse Merkle Tree & Audit Log (`continuous_legal_memory/storage/sqlite_store.py`)**:
  - Added `merkle_leaves` and `merkle_audit_log` SQLite tables so the 256-bit `SparseMerkleTree` state root and `AppendOnlyMerkleLog` hash chain persist and reload bit-for-bit across `save_to_disk` / `load_from_disk`.
- **Property-Based Algebraic & Cryptographic Invariant Suite (`tests/unit/test_deontic_choice_of_law_invariants.py`)**:
  - Added randomized property tests proving 256-bit Sparse Merkle Tree permutation invariance, erasure reversibility, precedence monotonicity, and finite-sample conformal coverage lower bounds.

---

## [0.3.0] - 2026-10-19

### Added
- **Split-Conformal Legal Prediction & Distribution-Free Abstention (`continuous_legal_memory/retrieval/conformal.py`)**:
  - Implemented `SplitConformalCalibrator` and `ConformalPredictionOutput` providing finite-sample marginal coverage guarantees ($P(Y \in \mathcal{C}_\alpha(X)) \ge 1 - \alpha$) and similarity-gated empty/ambiguous set abstention (`conformal_empty_set`, `conformal_ambiguous_set`).
  - Extended `PredictionResult` and `MCPServer` `legal_memory_predict` with `conformal_set`, `conformal_threshold`, and `conformal_alpha`.
- **Calibrated Cross-Encoder Legal Rerankers (`continuous_legal_memory/retrieval/reranker.py`)**:
  - Defined `BaseRerankerPort` in `continuous_legal_memory/domain/interfaces.py`.
  - Implemented `DeonticCrossEncoderReranker` (combining lexical content recall/F1, pinpoint statutory citation matching, and deontic modal alignment) and optional `HuggingFaceCrossEncoderReranker`, integrated into `TemporalStructuredRetriever` and `LegalMemoryOrchestrator`.
- **540-Case Multi-Jurisdictional Statutory Graph Benchmark & RAGAS/DeepEval Adapters (`continuous_legal_memory/evaluation/multijurisdictional.py`)**:
  - Added `generate_multijurisdictional_benchmark` and `evaluate_multijurisdictional_suite` spanning `EU` (GDPR, EU AI Act, DORA), `US` (HIPAA 45 CFR, SEC 17a-4, GLBA), and `BR` (LGPD, Marco Civil, BACEN) across 6 normative conflict categories (540 total cases).
  - Added `to_ragas_records` and `to_deepeval_test_cases` zero-dependency evaluation export adapters.
- **256-Bit Sparse Merkle Tree Non-Membership Erasure Proofs & Append-Only Merkle Log (`continuous_legal_memory/security/merkle.py`)**:
  - Implemented RFC 6962 domain-separated 256-bit `SparseMerkleTree`, `SparseMerkleProof`, `AppendOnlyMerkleLog`, and `verify_sparse_merkle_proof`.
  - Integrated succinct cryptographic non-membership proofs (`merkle_non_membership_proof`) into `LegalMemoryOrchestrator.delete_rule()`, `prove_erasure()`, `prove_inclusion()`, `verify_merkle_proof()`, and `MCPServer` `legal_memory_verify`.

---

## [0.2.0] - 2026-10-19

### Added
- **Tri-Canon Deontic Precedence Engine (`continuous_legal_memory/retrieval/precedence.py`)**:
  - Formalized all three classical statutory interpretation canons across candidate pools: *lex superior derogat legi inferiori* (`authority_rank`), *lex specialis derogat legi generali* (`specificity`), and *lex posterior derogat legi priori* (chronological `valid_from` / ingestion order *in pari materia*), alongside explicit `SUPERSEDES` graph suppression.
  - Eliminated random `slow_net` forward passes when `skip_parametric=True`, ensuring 100% deterministic unit `rule_importance` in `structured` mode across arbitrary seeds.
- **Epistemic Abstention Guardrails (`continuous_legal_memory/orchestrator.py`, `domain/models.py`)**:
  - Added calibrated `min_similarity_threshold` support (constructor and per-query override) for EU AI Act Article 14 human-oversight compliance.
  - Extended `PredictionResult` with `abstained: bool`, `abstention_reason: str | None`, and `max_similarity: float | None`.
- **Model Context Protocol (MCP) JSON-RPC 2.0 Server (`continuous_legal_memory/mcp.py`)**:
  - Added zero-dependency `MCPServer` supporting `initialize`, `ping`, `tools/list`, `tools/call` (`legal_memory_ingest`, `legal_memory_predict`, `legal_memory_erase`, `legal_memory_associate`, `legal_memory_verify`), `resources/list`, and `resources/read` (`clm://context/{tenant_id}`, `clm://graph/{tenant_id}`, `clm://metrics`).
  - Wired `clm mcp` stdio subcommand in `continuous_legal_memory/cli.py` and `POST /v1/mcp` HTTP endpoint in `continuous_legal_memory/server.py`.
- **Canonical Root Package Public Surface (`continuous_legal_memory/__init__.py`)**:
  - Exported `LegalMemoryOrchestrator`, `MCPServer`, `MemoryRecord`, `PredictionResult`, `GraphNode`, `GraphEdge`, `MemoryTier`, `EntityType`, `RelationType`, `PrecedenceConfig`, `apply_legal_precedence`, `KeyedHashAttestationModule`, `AsymmetricAttestationModule`, and `MemoryAttestationToken` via explicit `__all__`.
- **Governance Roadmap & Citation Metadata**:
  - Added `ROADMAP.md` defining the Maturity Model & Quality Gates matrix and next-horizon frontier tracks.
  - Added `CITATION.cff` for machine-readable academic and enterprise citation.
- **Asynchronous Orchestrator API & Concurrency**:
  - Added non-blocking asynchronous coroutine methods to `LegalMemoryOrchestrator`: `async_ingest_rule`, `async_update_memory`, `async_predict`, `async_delete_rule`, and `async_associate_statutes`.
  - Configured `SqliteMemoryStore` connection pragmas with Write-Ahead Logging (`PRAGMA journal_mode = WAL`) and reentrant locking (`threading.RLock`).
- **REST API Microservice, CLI & Ecosystem Integrations**:
  - Added multi-threaded JSON REST API server (`continuous_legal_memory/server.py`), unified `clm` CLI (`continuous_legal_memory/cli.py`), OpenMetrics Prometheus exporter, and adapters for LangChain, LlamaIndex, and Letta (`MemGPT`).
  - Added HippoRAG Personalized PageRank graph diffusion (`SemanticKnowledgeGraph.personalized_pagerank` and `orchestrator.associate_statutes`).
  - Added `LegalBenchEvaluator` and `LegalBenchRAGDataset` benchmark suite.

### Changed
- Translated `examples/legacy_poc/poc_continuous_memory.py` into 100% international technical English and aligned default encoder with `BAAI/bge-small-en-v1.5`.
- Reconciled `README.md`, `docs/evaluation_report.md`, `docs/BASELINE.md`, and `docs/reference/api-orchestrator.md` with shipped multi-tenant isolation (`0.00%` leakage) and Ed25519 attestation capabilities.

### Fixed
- Fixed async unit tests (`tests/unit/test_async_orchestrator.py` and `tests/unit/test_examples.py`) to execute deterministically via `asyncio.run` without failing when external pytest plugins are absent.
- Removed tracked `.pyc` binary cache files from git index.

---

## [0.1.0] - 2026-09-25

### Added
- **Multi-Tenant Isolation**:
  - Enforced tenant boundaries across `WorkingMemory`, `SemanticKnowledgeGraph`, and `SqliteMemoryStore` with `orchestrator.tenant(tenant_id)` context manager.
- **Temporal Structured Retrieval Engine**:
  - Introduced `TemporalStructuredRetriever` as the deterministic default retrieval engine and isolated `ContinuousMemoryResearchHead` under `continuous_legal_memory.experimental`.
- **Asymmetric Cryptographic Attestation**:
  - Implemented RFC 8032 **Ed25519** digital signatures in `AsymmetricAttestationModule` alongside `KeyedHashAttestationModule` (HMAC-SHA256).
- **Open Source Governance & Fast Offline Test Suite**:
  - Added `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SemanticMockEncoder`, and `DeterministicTokenEncoder`.
