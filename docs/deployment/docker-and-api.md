# Deployment & REST API Microservice

Continuous Legal Memory provides a high-throughput, multi-threaded JSON REST API microservice and Prometheus OpenMetrics exporter implemented using standard library networking. This allows deployment as an on-premise service, a Kubernetes sidecar, or an isolated container in air-gapped law firm networks without imposing heavy external web framework dependencies.

---

## 1. Quickstart: Running the REST API

### Via CLI

The `clm` CLI tool provides the `serve` subcommand:

```bash
# Launch the server with default in-memory storage
clm serve --host 0.0.0.0 --port 8000

# Launch with persistent SQLite database and telemetry
clm serve --host 0.0.0.0 --port 8000 --db-path /data/memory.db --enable-telemetry
```

### Via Docker

Run the official multi-stage, rootless container image:

```bash
# Pull and run the container with persistent volume
docker run -d \
  --name clm-server \
  -p 8000:8000 \
  -v clm-data:/data \
  continuous-legal-memory:latest
```

### Via Docker Compose

A production-ready `docker-compose.yml` is provided at the repository root:

```yaml
services:
  clm-server:
    build:
      context: .
      dockerfile: Dockerfile
    image: continuous-legal-memory:latest
    container_name: clm-server
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - CLM_ATTESTATION_SECRET=secure-production-secret-key
    volumes:
      - clm-data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://127.0.0.1:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s

volumes:
  clm-data:
    driver: local
```

Start the service with:

```bash
docker compose up -d
```

---

## 2. API Endpoints Reference

### Health & Readiness

```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "continuous-legal-memory",
  "version": "0.1.0",
  "engine_mode": "structured",
  "value_dim": 2,
  "stored_records_total": 42,
  "semantic_nodes_total": 38
}
```

---

### Prometheus OpenMetrics Telemetry

```http
GET /metrics
```

**Response (200 OK, `text/plain`):**
```text
# HELP clm_service_info Metadata info about the continuous-legal-memory service.
# TYPE clm_service_info gauge
clm_service_info{service="continuous-legal-memory",engine_mode="structured"} 1
# HELP clm_episodic_records_total Total number of stored immutable episodic records.
# TYPE clm_episodic_records_total gauge
clm_episodic_records_total 42
# HELP clm_working_memory_active_items Active records in working memory sliding window.
# TYPE clm_working_memory_active_items gauge
clm_working_memory_active_items 10
# HELP clm_semantic_graph_nodes_total Total entity nodes in the semantic knowledge graph.
# TYPE clm_semantic_graph_nodes_total gauge
clm_semantic_graph_nodes_total 38
# HELP clm_hash_chain_length Current height of the cryptographic hash chain.
# TYPE clm_hash_chain_length gauge
clm_hash_chain_length 42
# HELP clm_operations_total Total count of executed memory pipeline operations.
# TYPE clm_operations_total counter
clm_operations_total{operation="predict"} 128
clm_operations_total{operation="update_memory"} 42
```

---

### Ingest Rule Directive

```http
POST /v1/rules
Content-Type: application/json
X-Tenant-ID: tenant_alpha
```

**Request Body:**
```json
{
  "rule_text": "Federal Statute 404: Consumer credit history must be preserved for 5 years.",
  "action_vector": [0.0, 1.0],
  "authority_rank": 8,
  "jurisdiction": "FEDERAL",
  "personal_data": false,
  "valid_from": "2023-01-01T00:00:00Z",
  "valid_to": null,
  "metadata": {
    "node_id": "statute_404",
    "category": "banking"
  }
}
```

**Response (201 Created):**
```json
{
  "status": "INGESTED",
  "record_id": "rec_3a8f10b2...",
  "tenant_id": "tenant_alpha",
  "authority_rank": 8,
  "personal_data": false,
  "created_at": "2023-01-01T00:00:00Z"
}
```

---

### Query Decision & Context Prediction

```http
POST /v1/predict
Content-Type: application/json
X-Tenant-ID: tenant_alpha
```

**Request Body:**
```json
{
  "query_text": "What is the retention duration for consumer credit transaction logs?",
  "at_time": "2024-01-15T00:00:00Z"
}
```

**Response (200 OK):**
```json
{
  "query": "What is the retention duration for consumer credit transaction logs?",
  "predicted_action_vector": [0.0, 1.0],
  "source_tier": "semantic",
  "most_relevant_rule": "Federal Statute 404: Consumer credit history must be preserved for 5 years.",
  "confidence": 0.9642,
  "attention_weights": [0.9642, 0.0358],
  "tenant_id": "tenant_alpha",
  "retrieved_snippets": [
    "Federal Statute 404: Consumer credit history must be preserved for 5 years."
  ],
  "attestation_token": {
    "timestamp": "2024-01-15T00:00:00Z",
    "state_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "integrity_tag": "f512c...",
    "key_id": "legal-memory-ed25519-key",
    "algorithm": "Ed25519",
    "public_key": "302a3005..."
  }
}
```

---

### Right-to-Erasure (GDPR Art. 17)

```http
DELETE /v1/rules/{record_id}
X-Tenant-ID: tenant_alpha
```

**Response (200 OK):**
```json
{
  "status": "ERASED",
  "record_id": "rec_3a8f10b2...",
  "deleted_at": "2026-09-26T18:00:00Z",
  "tombstone_hash": "a1b2c3d4...",
  "personal_data": true,
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

### HippoRAG Graph Association

```http
POST /v1/associate
Content-Type: application/json
X-Tenant-ID: tenant_alpha
```

**Request Body:**
```json
{
  "seed_nodes": ["statute_404"],
  "damping": 0.85,
  "max_results": 5
}
```

**Response (200 OK):**
```json
{
  "total": 2,
  "results": [
    {
      "node_id": "statute_404",
      "label": "Consumer credit history",
      "entity_type": "statute",
      "score": 0.582
    },
    {
      "node_id": "clause_prerequisite",
      "label": "Audit Trail Prerequisite",
      "entity_type": "clause",
      "score": 0.418
    }
  ]
}
```

---

### Cryptographic Attestation Verification

```http
POST /v1/attestation/verify
Content-Type: application/json
```

**Request Body:**
```json
{
  "token": {
    "timestamp": "2024-01-15T00:00:00Z",
    "state_hash": "e3b0c442...",
    "integrity_tag": "f512c...",
    "key_id": "legal-memory-ed25519-key",
    "algorithm": "Ed25519",
    "public_key": "302a3005..."
  },
  "result": {
    "query": "What is the retention duration for consumer credit transaction logs?",
    "predicted_action_vector": [0.0, 1.0],
    "most_relevant_rule": "Federal Statute 404: Consumer credit history must be preserved for 5 years."
  },
  "retrieved_text": "Federal Statute 404: Consumer credit history must be preserved for 5 years."
}
```

**Response (200 OK):**
```json
{
  "valid": true,
  "algorithm": "Ed25519",
  "state_hash": "e3b0c442...",
  "verified_at": "2026-09-26T18:00:00Z"
}
```
