# Legal Precedence & Conflict Resolution

Standard semantic search cannot distinguish between two contradictory clauses: it merely computes dot products. Continuous Legal Memory implements formal legal doctrines directly into the retrieval scoring function:

---

## 1. Explicit Supersession (Statutory Amendments)

When a new statutory enactment or higher court ruling explicitly supersedes an existing rule:
- Metadata field: `{"supersedes": "prior_rule_id"}` or relation edge `RelationType.SUPERSEDES`.
- If the superseding rule has equal or greater authority rank, the target rule is assigned an attention suppression penalty (`-1000.0`), dropping its softmax attention probability to near-zero.

---

## 2. Hierarchical Authority Tiers (*Lex Superior Derogat Legi Inferiori*)

Statutory laws and constitutional mandates strictly override subordinate administrative circulars or private contractual agreements:
- Every rule possesses an `authority_rank` (integer from 1 to 10).
- When a query activates candidates across multiple authority tiers within the relevant candidate pool, subordinate candidates are penalized proportional to the authority difference:
  $$\Delta \text{score} = -(\text{max\_authority} - \text{rank}_i) \times 20.0$$

---

## 3. Recency Within Authority Tier (*Lex Posterior Derogat Legi Priori*)

When two rules occupy the same authority rank and no explicit supersession relationship is declared:
- Rules ingested later in time are given priority within matching authority tiers.
