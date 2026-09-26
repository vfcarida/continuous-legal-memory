# API Reference: Domain Models & Exceptions

## Core Models

### `MemoryRecord`
Encapsulates a single legal norm or policy item:
- `record_id`: Unique identifier (UUID or statutory citation).
- `text`: Natural language text of the legal rule.
- `key_vector`: Embedding representation.
- `value_vector`: Target legal action vector.
- `authority_rank`: Statutory authority tier (1 to 10).
- `valid_from`: Start of temporal legal validity.
- `valid_to`: Expiration of temporal legal validity.
- `tenant_id`: Partition identifier for multi-tenant isolation.
- `personal_data`: Boolean flag indicating GDPR Art. 17 data protection scope.

### `PredictionResult`
Container for prediction outputs:
- `query`: The user or agent query string.
- `predicted_action_vector`: Attention-weighted legal decision vector.
- `most_relevant_rule`: Text of the governing legal rule.
- `fast_slow_gate`: Gating ratio between fast short-term and slow long-term memory.
- `retrieved_snippets`: Exact character-level extracted legal snippets.

---

## Domain Exceptions

- `LegalMemoryException`: Root exception for all library errors.
- `InvalidMemoryVectorError`: Raised on dimensionality mismatch or invalid vectors.
- `MemoryContradictionError`: Raised when attempting to insert contradictory knowledge graph edges.
- `MemoryCapacityExceededError`: Raised when storage limits are exceeded.
- `StrictPrivacyViolationError`: Raised when network security boundaries are breached.
