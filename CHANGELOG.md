# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- LangChain `BaseMemory` and LlamaIndex `BaseRetriever` integration adapters.
- Real-world LegalBench-RAG evaluation benchmark integration.
- HippoRAG-style Personalized PageRank graph traversal over the Semantic Knowledge Graph.

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
