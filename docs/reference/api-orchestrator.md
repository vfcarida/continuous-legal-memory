# API Reference: LegalMemoryOrchestrator

::: continuous_legal_memory.orchestrator.LegalMemoryOrchestrator

## Overview

`LegalMemoryOrchestrator` is the central hexagonal application service coordinating Working Memory, the Episodic Memory Ledger, the Semantic Knowledge Graph, tri-canon legal precedence scoring, epistemic abstention guardrails, and cryptographic attestation.

### Constructor Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `encoder` | `BaseEncoderPort \| None` | `None` | Embedding adapter. Defaults to `HuggingFaceEncoderAdapter("BAAI/bge-small-en-v1.5")` or `OllamaGemmaAdapter` when `strict_privacy_mode=True`. |
| `value_dim` | `int` | `2` | Dimensionality of the target legal decision / action vector. |
| `temperature` | `float` | `0.05` | Attention temperature for softmax scaling. |
| `strict_privacy_mode` | `bool` | `False` | Enforces local offline execution via `OllamaGemmaAdapter`. |
| `seed` | `int \| None` | `None` | Optional random seed for deterministic reproducibility. |
| `allow_pseudo_embeddings` | `bool` | `False` | Allows deterministic hash fallback in `OllamaGemmaAdapter` when offline. |
| `retriever` | `BaseRetrieverPort \| None` | `None` | Custom retriever port. Defaults to `TemporalStructuredRetriever` in `"structured"` mode. |
| `episodic_store` | `BaseMemoryStorePort \| None` | `None` | Custom episodic ledger store (defaults to `EpisodicMemory`). |
| `telemetry` | `TelemetryLogger \| None` | `None` | Optional telemetry logger instance. |
| `enable_telemetry` | `bool` | `False` | Instantiates a default `TelemetryLogger` when `True`. |
| `attestor` | `KeyedHashAttestationModule \| AsymmetricAttestationModule \| None` | `None` | Optional cryptographic attestation provider (HMAC-SHA256 or Ed25519). |
| `enable_attestation` | `bool` | `False` | Instantiates a default `KeyedHashAttestationModule` when `True`. |
| `engine_mode` | `str` | `"structured"` | Execution mode: `"structured"` (deterministic tri-canon precedence) or `"hybrid"` (experimental neural research head). |
| `min_similarity_threshold` | `float \| None` | `None` | Minimum cosine similarity required to emit a prediction; triggers epistemic abstention (`abstained=True`) when `max_similarity` is below threshold. |

### Key Synchronous & Asynchronous Methods

- `update_memory(rule_text, action_vector, ...)` / `ingest_rule(...)`: Ingests a legal rule into all cognitive memory tiers and the 256-bit `SparseMerkleTree`.
- `predict(query_text, at_time=None, tenant_id=None, min_similarity_threshold=None, jurisdiction=None, connecting_factors=None, as_of_transaction_time=None)`: Evaluates a legal query using tri-canon precedence, Private International Law choice-of-law routing (`EU`, `US`, `BR`, `UK`, `SG`, `JP`, `CH`), Split-Conformal prediction sets, and calibrated epistemic abstention.
- `delete_rule(record_id, tenant_id=None)`: Executes GDPR Art. 17 right-to-erasure across all memory tiers and generates a 256-bit Sparse Merkle Tree non-membership proof.
- `associate_statutes(seed_nodes, damping=0.85, max_results=10, tenant_id=None, deontic_polarity=True)`: Runs deontic polarity-aware HippoRAG Personalized PageRank over the Semantic Knowledge Graph.
- `simulate_rule_ingestion(rule_text, ...)` / `propose_norm_repairs(rule_text, ...)`: Executes non-mutating $\mathbf{KD}$ modal satisfiability simulation and synthesizes verified drafting repairs.
- `simulate_temporal_horizon(query_text=None, start_time=None, end_time=None, tenant_id=None)`: Sweeps across all `valid_from` and `valid_to` boundaries to forecast deontic modality transitions and future antinomies.
- `analyze_statutory_amendment(target_rule_id, amended_text, amended_action_vector=None, amended_authority_rank=None, tenant_id=None)`: Computes clause-level statutory diffs (`ClauseDiffSpan`) and traces the transitive compliance blast radius across subordinate contracts and policies.
- `export_smtlib(query_text=None, tenant_id=None, at_time=None)` / `export_standalone_verifier(tenant_id=None, redact_personal_data=False, query_text=None)`: Exports formal SMT-LIB 2.6 (`QF_UFLIA`) theories for Z3/cvc5 or a zero-dependency HTML5 + WebCrypto standalone audit verifier.
- `issue_timestamp_token(tenant_id=None, nonce=None)` / `audit_transparency_gossip(checkpoints, tenant_id=None)`: Issues RFC 3161-inspired cryptographic `TimestampToken`s and audits cross-witness gossip checkpoints for split-view equivocation.
- `generate_tee_quote(nonce=None, platform="AWS_NITRO", tenant_id=None)` / `verify_tee_quote(quote, ...)`: Generates and verifies hardware-anchored TEE remote attestation quotes (`AWS_NITRO`, `INTEL_TDX`).
- `rotate_signing_key(new_key_id=None, algorithm=None)` / `revoke_signing_key(key_id)`: Manages cryptographic key rotation and revocation (`PQ-Hybrid-Ed25519-SHA256`, `Ed25519`, `HMAC-SHA256`).
- `tenant(tenant_id)`: Context manager scoping operations to a specific tenant namespace.
- `save_to_disk(db_path)` / `load_from_disk(db_path)`: Persists and restores complete multi-tier state via `SqliteMemoryStore` (WAL mode).
- `async_*`: Non-blocking `asyncio` coroutine counterparts for all core ingestion, prediction, erasure, simulation, and verification operations.
