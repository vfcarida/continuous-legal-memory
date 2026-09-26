# Command-Line Interface (CLI) Reference

Continuous Legal Memory provides a unified command-line utility, registered as `clm`, for managing cognitive memory instances, ingesting statutes, querying decisions, auditing cryptographic state proofs, and running evaluation benchmarks.

---

## Global Syntax & Options

```bash
clm [-h] [--verbose] <subcommand> [subcommand-options]
```

| Flag | Short | Description |
|---|---|---|
| `--help` | `-h` | Display help synopsis and command list. |
| `--verbose` | `-v` | Enable detailed debug logging to standard error. |

---

## 1. `clm serve`

Starts the multi-threaded JSON REST API HTTP server and Prometheus metrics exporter.

```bash
clm serve [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--host HOST` | `0.0.0.0` | Host IP address to bind. |
| `--port PORT` | `8000` | TCP port number. |
| `--db-path PATH` | `None` | Path to persistent SQLite database for loading and storing state. |
| `--engine-mode MODE` | `structured` | Retrieval engine mode (`structured` or `hybrid`). |
| `--strict-privacy` | `False` | Enforce on-device Ollama execution; blocks external calls. |
| `--enable-attestation` | `False` | Enable automatic RFC 8032 Ed25519 digital signature generation. |
| `--enable-telemetry` | `False` | Enable Prometheus OpenMetrics telemetry tracking. |
| `--mock-encoder` | `False` | Use offline `SemanticMockEncoder` for test environments. |

### Example

```bash
clm serve --host 127.0.0.1 --port 8000 --db-path /data/legal_memory.db --enable-telemetry
```

---

## 2. `clm ingest`

Ingests an individual legal directive or a batch JSONL file into active episodic memory, working memory, and semantic knowledge graph tiers.

```bash
clm ingest --db-path PATH [options]
```

### Options

| Flag | Required | Description |
|---|---|---|
| `--db-path PATH` | **Yes** | Path to the SQLite memory store. |
| `--text TEXT` | Conditional | Text content of the legal rule or statute. |
| `--action FLOATS...`| Conditional | Target decision vector outputs (e.g. `1.0 0.0`). |
| `--file PATH` | Conditional | Path to JSONL file containing rule directives. |
| `--tenant TENANT` | No (`default`) | Multi-tenant namespace identifier. |
| `--authority-rank RANK` | No (`1`) | Hierarchical authority rank (1 = branch policy, 10 = federal statute). |
| `--jurisdiction JURIS` | No (`None`) | Jurisdictional identifier (e.g. `FEDERAL`, `INTERNAL`). |
| `--personal-data` | No (`False`) | Flags record as personal data subject to GDPR Art. 17 right-to-erasure. |
| `--mock-encoder` | No (`False`) | Use offline mock encoder for testing. |

### Examples

```bash
# Ingest single federal statute
clm ingest \
  --db-path /data/memory.db \
  --text "Federal Banking Act: Retain client financial records for 5 years." \
  --action 0.0 1.0 \
  --authority-rank 8 \
  --jurisdiction FEDERAL \
  --tenant law_corp

# Batch ingestion from JSONL file
clm ingest --db-path /data/memory.db --file statutes.jsonl
```

---

## 3. `clm query`

Evaluates a legal inquiry against multi-tier memory networks, resolving temporal validity, precedence hierarchies, and returning the governing decision vector.

```bash
clm query "QUERY_TEXT" --db-path PATH [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `QUERY_TEXT` | (Positional) | Query scenario text string. |
| `--db-path PATH` | **Required** | SQLite database path. |
| `--tenant TENANT` | `default` | Tenant partition scope. |
| `--at-time TIMESTAMP` | Current UTC | ISO 8601 evaluation timestamp for point-in-time reasoning. |
| `--mock-encoder` | `False` | Use offline mock encoder for testing. |

### Output Example

```json
{
  "query": "Client requests deletion of loan transaction logs.",
  "predicted_action_vector": [0.0, 1.0],
  "source_tier": "semantic",
  "most_relevant_rule": "Federal Banking Act: Retain client financial records for 5 years.",
  "confidence": 0.982,
  "tenant_id": "law_corp",
  "attestation_token": {
    "state_hash": "a5e9...",
    "integrity_tag": "3f8b...",
    "algorithm": "Ed25519"
  }
}
```

---

## 4. `clm erase`

Executes deterministic Right-to-Erasure (GDPR Art. 17 / CLM-R7) across all memory tiers, recording an immutable cryptographic tombstone.

```bash
clm erase RECORD_ID --db-path PATH [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `RECORD_ID` | (Positional) | Identifier of the rule or client directive to erase. |
| `--db-path PATH` | **Required** | SQLite database path. |
| `--tenant TENANT` | `default` | Tenant partition scope. |
| `--mock-encoder` | `False` | Use offline mock encoder for testing. |

### Output Example

```json
{
  "record_id": "rec_3a8f10b2",
  "deleted_at": "2026-09-26T18:30:00Z",
  "tombstone_hash": "b2c3d4e5...",
  "personal_data": true,
  "status": "ERASED",
  "tiers_cleared": [
    "working",
    "episodic",
    "semantic",
    "neural_buffer"
  ],
  "parametric_rebuilt": false
}
```

---

## 5. `clm associate`

Runs HippoRAG Personalized PageRank random walks over the Semantic Knowledge Graph to discover multi-hop associative statutes, cross-references, and prerequisite clauses.

```bash
clm associate SEED_NODES... --db-path PATH [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `SEED_NODES` | (Positional) | One or more seed node IDs. |
| `--db-path PATH` | **Required** | SQLite database path. |
| `--damping DAMPING` | `0.85` | Random walk teleportation damping factor. |
| `--max-results N` | `10` | Maximum number of ranked entities to return. |
| `--tenant TENANT` | `default` | Tenant partition scope. |

---

## 6. `clm verify`

Verifies the cryptographic integrity and state proof of an attestation token generated for a prediction result using public key cryptography.

```bash
clm verify --token-file PATH --query TEXT --action FLOATS... [options]
```

### Options

| Flag | Required | Description |
|---|---|---|
| `--token-file PATH` | **Yes** | Path to JSON file containing `MemoryAttestationToken`. |
| `--query TEXT` | **Yes** | Exact canonical query text. |
| `--action FLOATS...`| **Yes** | Exact canonical decision vector floats. |
| `--rule TEXT` | No | Exact retrieved legal rule snippet. |
| `--public-key HEX` | No | Ed25519 public key hex string (if not embedded in token). |

---

## 7. `clm evaluate`

Executes the LegalBench-RAG evaluation benchmark suite, calculating decision accuracy, temporal precision, attestation coverage, and erasure latencies.

```bash
clm evaluate [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--benchmark PATH` | `tests/fixtures/synthetic_legal_benchmark.jsonl` | Benchmark dataset filepath. |
| `--engine-mode MODE` | `structured` | Retrieval engine mode (`structured` or `hybrid`). |
| `--output PATH` | `None` | Optional JSON file destination to export metrics. |
