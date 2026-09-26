# Architecture Overview

Continuous Legal Memory adheres to **Hexagonal Architecture (Ports and Adapters)** to decouple core legal cognition logic from storage, embedding models, and external frameworks.

```
       +-----------------------------------------------------------+
       |                  LegalMemoryOrchestrator                  |
       +-----------------------------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
    +----------------------+                   +-----------------------+
    |  BaseRetrieverPort   |                   |  BaseMemoryStorePort  |
    +----------------------+                   +-----------------------+
               |                                           |
       +-------+-------+                           +-------+-------+
       |               |                           |               |
       v               v                           v               v
TemporalStructured  HybridRetriever           SqliteMemoryStore  In-Memory
```

---

## Core Principles

1. **Deterministic Rule of Law**:
   Precedence resolution follows formal doctrines (*lex superior*, *lex posterior*, statutory amendment supersession) rather than ungrounded neural weight drifting.
2. **Pluggable Retrieval Engine**:
   Default production runs on `TemporalStructuredRetriever`, eliminating CPU-blocking online gradient descent while achieving sub-10ms update latency.
3. **Strict Multi-Tenancy**:
   Working memory, semantic graphs, and storage backends are partitioned by tenant IDs.
4. **Non-Repudiable Cryptographic Attestation**:
   Decision outputs are verifiable via Ed25519 digital signatures.
