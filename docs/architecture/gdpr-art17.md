# GDPR Article 17 Right to Erasure

In standard fine-tuned neural models, deleting personal data requires computationally prohibitive machine unlearning or complete model re-training. In Continuous Legal Memory:

---

## 1. Personal Data Confinement

Rules tagged with `personal_data=True` are strictly quarantined from parametric weight updates. They reside exclusively in non-parametric retrieval buffers, ensuring deterministic erasure capability.

---

## 2. Cross-Tier Purging

When `orchestrator.delete_rule(record_id)` is invoked:
1. **Working Memory**: Clears matching context slots immediately.
2. **Episodic Memory**: Purges the record from the episodic buffer and SQLite storage.
3. **Semantic Knowledge Graph**: Removes corresponding graph nodes and attached edges.

---

## 3. Cryptographic Tombstone

To satisfy compliance auditing under GDPR Art. 17 while proving that the erasure was executed:
- The raw personal text and embedding are purged.
- A one-way **SHA-256 cryptographic tombstone** recording the erasure timestamp and transaction hash is preserved for regulatory audit verification.
