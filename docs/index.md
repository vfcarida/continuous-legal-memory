# Continuous Legal Memory

> **Production-grade, deterministic cognitive memory framework for legal AI agents under the rule of law.**

[![CI](https://github.com/vfcarida/continuous-legal-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/vfcarida/continuous-legal-memory/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## What is Continuous Legal Memory?

General-purpose agent memory architectures (e.g. standard vector databases or key-value caches) treat all retrieved knowledge as flat, unstructured chunks. When legal norms conflict, standard RAG systems hallucinate, arbitrarily blending opposing legal mandates or failing to recognize that a higher statutory amendment superseded an earlier policy.

**Continuous Legal Memory (CLM)** is purpose-built for legal and regulatory compliance engineering. It integrates formal jurisprudence principles directly into the retrieval and memory consolidation mathematics:

- **Formal Legal Precedence & Modal Deontic Resolvers**: Encodes *lex superior* (constitutional/statutory authority tiers strictly override subordinate contracts), *lex posterior* (recency within authority tier), and *lex specialis* alongside an embedded **SMT-LIB 2.6 Formal Deontic Solver** (`QF_LIA`).
- **Statutory Amendment Diffing & Blast-Radius Analysis**: Automatically diffs proposed statutory or regulatory amendments at the clause level and computes the transitive downstream compliance blast radius across the `SemanticKnowledgeGraph` (`DEPENDS_ON`, `DERIVES_FROM`, `SUPERSEDES`), shared statutory citations, and subordinate contractual policies.
- **Bi-Temporal Validity Tracking**: Disentangles **valid time** (when a law or contractual clause was legally effective) from **transaction time** (when the agent ingested the record).
- **GDPR Article 17 Machine Unlearning & ZK-Ready Proofs**: Mathematically provable right-to-erasure across working, episodic, and semantic tiers, leaving irreversible SHA-256 cryptographic tombstones and issuing zero-knowledge non-membership witness bundles.
- **Post-Quantum Hybrid Attestation, RFC 3161 Timestamping & Gossip Auditing**: Every retrieved decision state and Merkle root is signed via **PQ-Hybrid-Ed25519-SHA256** or RFC 8032 **Ed25519**, anchored with RFC 3161-inspired timestamp tokens, and verified across independent witness nodes for split-view equivocation detection.
- **Strict Multi-Tenant Boundary Partitioning & OpenAPI 3.1.0 Parity**: Enforces per-tenant isolation across all memory tiers and exposes 100% parity across the Python SDK, OpenAPI 3.1.0 REST Server (`/v1/openapi.json`), Model Context Protocol (`MCPServer`), and `clm` CLI.

See [Architecture Baseline](BASELINE.md) and [Scientific & Governance References](REFERENCES.md) for formal specifications and primary-source citations.

---

## Quick Installation

```bash
pip install continuous-legal-memory
```

---

## Minimal Example

```python
from continuous_legal_memory import LegalMemoryOrchestrator
from continuous_legal_memory.adapters.encoders import HuggingFaceEncoderPort

# Initialize orchestrator with high-speed deterministic structured retrieval
orchestrator = LegalMemoryOrchestrator(
    encoder=HuggingFaceEncoderPort(model_name="BAAI/bge-small-en-v1.5"),
    value_dim=2,  # e.g. [DELETION, RETENTION]
    engine_mode="structured",
)

# Ingest baseline contractual policy (rank 2)
orchestrator.update_memory(
    "Policy A: User transaction logs are purged after 90 days.",
    action_vector=[1.0, 0.0],
    authority_rank=2,
    metadata={"rule_id": "policy_a"},
)

# Ingest statutory regulatory mandate (rank 5 - overrides rank 2)
orchestrator.update_memory(
    "AML Statute 104: Financial transaction records must be retained for 5 years.",
    action_vector=[0.0, 1.0],
    authority_rank=5,
    metadata={"rule_id": "aml_104"},
)

# Querying with tenant scoping
with orchestrator.tenant("firm_acme"):
    result = orchestrator.predict("Can we delete customer transaction records older than 90 days?")
    print(f"Action: {result.predicted_action_vector}")  # Resolves to [0.0, 1.0] (RETAIN)
    print(f"Governing Rule: {result.most_relevant_rule}")
```
