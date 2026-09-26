# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **LegalBench-RAG Evaluation Benchmark Suite (ECO-03 / Phase 3)**:
  - Added `LegalBenchEvaluator` and `LegalBenchRAGDataset` evaluating character-level snippet precision, recall, and hierarchical precedence resolution over real-world commercial contracts.
  - Added comprehensive benchmark guide `docs/evaluation/legalbench.md` and registered it in `mkdocs.yml`.
  - Added unit test suite in `tests/unit/test_legalbench_eval.py`.
- **Letta (MemGPT) Agent Memory Integration (ECO-01 / Phase 3)**:
  - Added `ContinuousLegalMemoryBlock` providing dynamic context window compilation, token budgeting, and active directive summaries.
  - Added `create_letta_tools` exposing `legal_memory_search`, `legal_memory_insert`, and `legal_memory_associate` for autonomous LLM agent execution.
  - Added comprehensive guide `docs/guides/letta-integration.md` and registered it in `mkdocs.yml`.
  - Added unit test suite in `tests/unit/test_integrations.py`.
- **Thread-Safe Concurrency & Multi-Tenant Locking (ARCH-03)**:
  - Added reentrant mutex locking (`threading.RLock`) across `LegalMemoryOrchestrator` critical sections (`_execute_update_memory`, `_execute_predict`, `delete_rule`).
  - Added multi-threaded test suite in `tests/unit/test_concurrency.py` verifying race-free execution under concurrent writers, readers, and multi-tenant workloads.
- **Sub-10ms Right-to-Erasure Execution (Defect 1)**:
  - Fixed CPU blocking on rule deletion in `structured` mode by bypassing unnecessary neural re-consolidation loops.
  - Verified sub-50ms CPU execution in `test_delete_rule_sub_10ms_latency_in_structured_mode`.
- **HippoRAG-Style Personalized PageRank (ECO-02)**:
  - Added `SemanticKnowledgeGraph.personalized_pagerank` implementing power iteration diffusion across legal dependency edges.
  - Added `LegalMemoryOrchestrator.associate_statutes` for top-level multi-hop associative retrieval of prerequisite statutes.
  - Added dedicated unit test suite in `tests/unit/test_personalized_pagerank.py`.
- **Working Memory Fidelity & Precedence Preservation**:
  - Propagated `authority_rank`, `jurisdiction`, and temporal boundaries (`valid_from`, `valid_to`) into `WorkingMemory` records and SQLite restoration routines.
- **Precedence Configuration Calibration (Defect 3)**:
  - Added `PrecedenceConfig` dataclass parameterizing superseded penalty logit docking, authority penalty scales, similarity margins, and optional boolean (`-inf`) attention masking.
  - Exported `PrecedenceConfig` and `apply_legal_precedence` in `continuous_legal_memory.retrieval`.
- **Ecosystem & Integration Guides (COMM-02 & ECO-01)**:
  - Added guides for LangChain memory, LlamaIndex retriever, Letta memory, HippoRAG graph diffusion, and multi-tenant partitioning in `docs/guides/`.
  - Added GitHub issue templates (`bug_report.md`, `feature_request.md`) and pull request template (`PULL_REQUEST_TEMPLATE.md`).
- **Strict Static Typing & Test Optimization**:
  - Maintained 100% error-free static analysis (`mypy continuous_legal_memory` passing cleanly across all 38 source files).
  - Resolved local test fixture shadowing in `test_precedence_erasure_persistence.py`, eliminating socket timeouts and achieving 400x speedup in zero-network unit tests.

---

## [0.1.0] - 2026-09-25

### Added
- **Multi-Tenant Isolation (SEC-01)**:
  - Enforced tenant boundaries across `WorkingMemory`, `SemanticKnowledgeGraph`, and `SqliteMemoryStore`.
  - Added `orchestrator.tenant(tenant_id)` context manager and tenant-scoped query accessors.
  - Implemented schema migrations adding `tenant_id` column to SQLite relational stores.
- **Temporal Structured Retrieval Engine (ARCH-01 & ARCH-02)**:
  - Introduced `TemporalStructuredRetriever` as the high-throughput, deterministic default retrieval engine.
  - Bypassed online neural gradient backpropagation during default memory updates, reducing ingestion latency from ~450ms to sub-10ms.
  - Demoted parametric neural head to optional research module under `continuous_legal_memory.experimental.neural_head.ContinuousMemoryResearchHead` with `ParametricNeuralHeadWarning` documenting academic findings.
- **Asymmetric Cryptographic Attestation (SEC-02)**:
  - Implemented RFC 8032 **Ed25519** digital signatures with extended projective coordinates for high-speed, zero-dependency offline signing.
  - Created `AsymmetricAttestationModule` enabling third-party zero-trust verification (e.g. judicial or regulatory audits) using only public keys without secret disclosure.
  - Added full cross-module interoperability between Ed25519 tokens and symmetric HMAC-SHA256 tokens.
- **Open Source Governance (COMM-01 & STAB-01)**:
  - Added standalone `LICENSE` (MIT License).
  - Added `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`, and `CHANGELOG.md`.
- **Offline-Safe Fast Test Suite (STAB-02 & STAB-03)**:
  - Added `SemanticMockEncoder`, `DeterministicTokenEncoder`, and zero-network test fixtures in `tests/conftest.py`.
  - Decoupled slow statistical benchmark sweeps into `@pytest.mark.benchmark`, reducing standard `pytest` run times to sub-15s.

### Fixed
- **Python 3.9 Type Compatibility (STAB-05)**: Added `from __future__ import annotations` across domain exceptions and models.
- **Linter Compliance (STAB-04)**: Resolved 73 Ruff linting violations in legacy PoC and configured exclusion rules in `pyproject.toml`.
