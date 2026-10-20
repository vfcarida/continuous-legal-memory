"""
Defeasible Proof-Tree Explanation & Auditable Legal Reasoning Trace.

Implements machine-verifiable derivation traces satisfying EU AI Act Article 13
(Transparency) and Article 86 (Right to Explanation of Individual Decision-Making).
For every prediction, records which candidate norms were evaluated, which records
were excluded on bi-temporal or jurisdictional grounds, and the exact defeasible
canon (*lex superior*, *lex specialis*, *lex posterior*, explicit `SUPERSEDES`,
or *lois de police*) by which the prevailing norm defeated competing norms.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.retrieval.precedence import (
    _extract_specificity as _compute_specificity,
)


def _record_identifiers(record: MemoryRecord) -> set[str]:
    ids: set[str] = set()
    if record.record_id:
        ids.add(str(record.record_id))
    if isinstance(record.metadata, dict):
        for k in ("record_id", "rule_id", "node_id"):
            val = record.metadata.get(k)
            if val:
                ids.add(str(val))
    return ids


def _collect_superseded_identifiers(records: list[MemoryRecord]) -> set[str]:
    superseded: set[str] = set()
    for rec in records:
        meta = rec.metadata if isinstance(rec.metadata, dict) else {}
        sup = meta.get("supersedes")
        if isinstance(sup, str):
            superseded.add(sup)
        elif isinstance(sup, (list, tuple, set)):
            for item in sup:
                superseded.add(str(item))
        rels = meta.get("relations")
        if isinstance(rels, list):
            for rel in rels:
                if isinstance(rel, dict):
                    rtype = str(rel.get("relation_type", "")).lower()
                    target = rel.get("target_id")
                    if target and ("supersede" in rtype or "override" in rtype):
                        superseded.add(str(target))
    return superseded


def _extract_deontic_modality(record: MemoryRecord) -> str:
    if isinstance(record.metadata, dict) and record.metadata.get("deontic_modality"):
        return str(record.metadata["deontic_modality"]).lower()
    lower = record.text.lower()
    if any(w in lower for w in ("prohibited", "shall not", "must not", "forbidden", "vedado", "proibido")):
        return "prohibition"
    if any(w in lower for w in ("shall", "must", "obligated", "mandatory", "deve", "obrigatório")):
        return "obligation"
    if any(w in lower for w in ("may", "permitted", "allowed", "facultado", "permitido")):
        return "permission"
    return "unspecified"


@dataclass
class CandidateProofNode:
    """
    Single candidate norm evaluation node in a defeasible legal derivation trace.
    """

    record_id: str
    rule_text: str
    status: str
    defeated_by_record_id: str | None
    authority_rank: int
    jurisdiction: str | None
    deontic_modality: str
    specificity: float
    attention_weight: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "rule_text": self.rule_text,
            "status": self.status,
            "defeated_by_record_id": self.defeated_by_record_id,
            "authority_rank": self.authority_rank,
            "jurisdiction": self.jurisdiction,
            "deontic_modality": self.deontic_modality,
            "specificity": round(self.specificity, 4),
            "attention_weight": round(self.attention_weight, 6),
            "rationale": self.rationale,
        }


@dataclass
class DefeasibleProofTrace:
    """
    Structured, cryptographically hashed derivation trace for a legal memory prediction.
    """

    query: str
    prevailing_record_id: str | None
    abstained: bool
    abstention_reason: str | None
    at_valid_time: str
    as_of_transaction_time: str | None
    candidates: list[CandidateProofNode] = field(default_factory=list)
    excluded_records: list[dict[str, Any]] = field(default_factory=list)
    winning_canon: str = "NONE"
    trace_hash: str = ""

    def compute_trace_hash(self) -> str:
        """Compute deterministic SHA-256 digest over the canonical derivation tree."""
        canonical = {
            "query": self.query,
            "prevailing_record_id": self.prevailing_record_id,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "at_valid_time": self.at_valid_time,
            "as_of_transaction_time": self.as_of_transaction_time,
            "candidates": [
                {
                    "record_id": c.record_id,
                    "status": c.status,
                    "defeated_by_record_id": c.defeated_by_record_id,
                    "authority_rank": c.authority_rank,
                }
                for c in self.candidates
            ],
            "excluded_records": [
                {
                    "record_id": ex.get("record_id"),
                    "status": ex.get("status"),
                }
                for ex in self.excluded_records
            ],
        }
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.trace_hash = hashlib.sha256(payload).hexdigest()
        return self.trace_hash

    def to_dict(self) -> dict[str, Any]:
        if not self.trace_hash:
            self.compute_trace_hash()
        return {
            "query": self.query,
            "prevailing_record_id": self.prevailing_record_id,
            "winning_canon": self.winning_canon,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "at_valid_time": self.at_valid_time,
            "as_of_transaction_time": self.as_of_transaction_time,
            "candidates": [c.to_dict() for c in self.candidates],
            "excluded_records": self.excluded_records,
            "trace_hash": self.trace_hash,
        }


def build_defeasible_proof_trace(
    query: str,
    all_tenant_records: list[MemoryRecord],
    active_records: list[MemoryRecord],
    attention_weights: list[float] | None,
    top_local_idx: int | None,
    abstained: bool,
    abstention_reason: str | None,
    at_valid_time: datetime,
    as_of_transaction_time: datetime | None = None,
) -> DefeasibleProofTrace:
    """
    Construct a `DefeasibleProofTrace` explaining candidate selection, canon-based
    defeat relationships, and bi-temporal / jurisdictional exclusions.
    """
    valid_iso = (
        at_valid_time.replace(tzinfo=timezone.utc).isoformat()
        if at_valid_time.tzinfo is None
        else at_valid_time.isoformat()
    )
    tx_iso = (
        (
            as_of_transaction_time.replace(tzinfo=timezone.utc).isoformat()
            if as_of_transaction_time.tzinfo is None
            else as_of_transaction_time.isoformat()
        )
        if as_of_transaction_time is not None
        else None
    )

    active_ids = {rec.record_id or rec.text for rec in active_records}
    excluded: list[dict[str, Any]] = []
    for rec in all_tenant_records:
        rid = rec.record_id or rec.text[:32]
        if (rec.record_id or rec.text) in active_ids:
            continue
        if not rec.is_known_as_of(as_of_transaction_time):
            excluded.append({
                "record_id": rid,
                "rule_text": rec.text,
                "status": "EXCLUDED_TRANSACTION_TIME",
                "rationale": (
                    f"Record ingested at {rec.transaction_time.isoformat()} after "
                    f"as_of_transaction_time cutoff ({tx_iso})."
                ),
            })
        elif not rec.is_temporally_valid(at_valid_time):
            excluded.append({
                "record_id": rid,
                "rule_text": rec.text,
                "status": "EXCLUDED_TEMPORAL_VALIDITY",
                "rationale": (
                    f"Record outside valid-time window [{rec.valid_from.isoformat()}, "
                    f"{rec.valid_to.isoformat() if rec.valid_to else 'inf'}) at {valid_iso}."
                ),
            })
        else:
            excluded.append({
                "record_id": rid,
                "rule_text": rec.text,
                "status": "EXCLUDED_JURISDICTION",
                "rationale": (
                    f"Record jurisdiction '{rec.jurisdiction}' outside applicable "
                    f"choice-of-law scope."
                ),
            })

    superseded_ids = _collect_superseded_identifiers(active_records)
    # Map superseded target -> superseding record_id
    superseder_map: dict[str, str] = {}
    for rec in active_records:
        src_id = str(rec.record_id or rec.text[:32])
        meta = rec.metadata if isinstance(rec.metadata, dict) else {}
        sup = meta.get("supersedes")
        if isinstance(sup, str):
            superseder_map[sup] = src_id
        elif isinstance(sup, (list, tuple, set)):
            for item in sup:
                superseder_map[str(item)] = src_id
        rels = meta.get("relations")
        if isinstance(rels, list):
            for rel in rels:
                if isinstance(rel, dict):
                    rtype = str(rel.get("relation_type", "")).lower()
                    target = rel.get("target_id")
                    if target and ("supersede" in rtype or "override" in rtype):
                        superseder_map[str(target)] = src_id

    winner_rec = (
        active_records[top_local_idx]
        if (not abstained and top_local_idx is not None and 0 <= top_local_idx < len(active_records))
        else None
    )
    winner_id = str(winner_rec.record_id or winner_rec.text[:32]) if winner_rec is not None else None
    winner_spec = _compute_specificity(winner_rec) if winner_rec is not None else 0.5
    winner_col_bonus = (
        float(winner_rec.metadata.get("_choice_of_law_bonus", 0.0))
        if (winner_rec is not None and isinstance(winner_rec.metadata, dict))
        else 0.0
    )

    candidate_nodes: list[CandidateProofNode] = []
    for idx, rec in enumerate(active_records):
        rid = str(rec.record_id or rec.text[:32])
        weight = (
            float(attention_weights[idx])
            if (attention_weights is not None and idx < len(attention_weights))
            else 0.0
        )
        modality = _extract_deontic_modality(rec)
        spec = _compute_specificity(rec)
        rec_ids = _record_identifiers(rec)
        rec_col_bonus = (
            float(rec.metadata.get("_choice_of_law_bonus", 0.0))
            if isinstance(rec.metadata, dict)
            else 0.0
        )

        if abstained:
            status = "ABSTAINED"
            defeated_by = None
            rationale = f"Epistemic or conformal abstention triggered: {abstention_reason}."
        elif winner_rec is not None and idx == top_local_idx:
            status = "PREVAILED"
            defeated_by = None
            rationale = (
                f"Prevailed with highest composite deontic precedence and semantic alignment "
                f"(authority_rank={rec.authority_rank}, specificity={spec:.2f}, weight={weight:.4f})."
            )
        elif bool(rec_ids & superseded_ids):
            matched_sup = next(iter(rec_ids & superseded_ids))
            superseder_id = superseder_map.get(matched_sup, winner_id)
            status = "DEFEATED_BY_SUPERSEDES"
            defeated_by = superseder_id
            rationale = f"Explicitly superseded or overruled by active norm '{superseder_id}'."
        elif winner_rec is not None and winner_col_bonus > rec_col_bonus + 1e-6:
            status = "DEFEATED_BY_LOIS_DE_POLICE"
            defeated_by = winner_id
            rationale = (
                f"Overridden by mandatory choice-of-law / lois de police preference of '{winner_id}' "
                f"(jurisdiction='{winner_rec.jurisdiction}')."
            )
        elif winner_rec is not None and winner_rec.authority_rank > rec.authority_rank:
            status = "DEFEATED_BY_LEX_SUPERIOR"
            defeated_by = winner_id
            rationale = (
                f"Defeated under lex superior by higher-authority norm '{winner_id}' "
                f"(rank {winner_rec.authority_rank} > rank {rec.authority_rank})."
            )
        elif (
            winner_rec is not None
            and winner_rec.authority_rank == rec.authority_rank
            and winner_spec > spec + 0.05
        ):
            status = "DEFEATED_BY_LEX_SPECIALIS"
            defeated_by = winner_id
            rationale = (
                f"Defeated under lex specialis by more specific co-equal norm '{winner_id}' "
                f"(specificity {winner_spec:.2f} > {spec:.2f})."
            )
        elif (
            winner_rec is not None
            and winner_rec.authority_rank == rec.authority_rank
            and winner_rec.valid_from > rec.valid_from
        ):
            status = "DEFEATED_BY_LEX_POSTERIOR"
            defeated_by = winner_id
            rationale = (
                f"Defeated under lex posterior by more recent co-equal norm '{winner_id}' "
                f"(valid_from {winner_rec.valid_from.isoformat()} > {rec.valid_from.isoformat()})."
            )
        else:
            status = "DEFEATED_BY_SEMANTIC_RELEVANCE"
            defeated_by = winner_id
            rationale = (
                f"Subordinated to '{winner_id}' due to lower query-to-norm alignment "
                f"(attention_weight={weight:.4f})."
            )

        candidate_nodes.append(
            CandidateProofNode(
                record_id=rid,
                rule_text=rec.text,
                status=status,
                defeated_by_record_id=defeated_by,
                authority_rank=rec.authority_rank,
                jurisdiction=rec.jurisdiction,
                deontic_modality=modality,
                specificity=spec,
                attention_weight=weight,
                rationale=rationale,
            )
        )

    defeated_statuses = {c.status for c in candidate_nodes if c.status != "PREVAILED"}
    if abstained:
        winning_canon = "ABSTAINED"
    elif winner_id is None:
        winning_canon = "NONE"
    elif "DEFEATED_BY_SUPERSEDES" in defeated_statuses:
        winning_canon = "EXPLICIT_SUPERSEDES"
    elif "DEFEATED_BY_LOIS_DE_POLICE" in defeated_statuses:
        winning_canon = "LOIS_DE_POLICE"
    elif "DEFEATED_BY_LEX_SUPERIOR" in defeated_statuses:
        winning_canon = "LEX_SUPERIOR"
    elif "DEFEATED_BY_LEX_SPECIALIS" in defeated_statuses:
        winning_canon = "LEX_SPECIALIS"
    elif "DEFEATED_BY_LEX_POSTERIOR" in defeated_statuses:
        winning_canon = "LEX_POSTERIOR"
    elif "DEFEATED_BY_SEMANTIC_RELEVANCE" in defeated_statuses:
        winning_canon = "SEMANTIC_RELEVANCE"
    else:
        winning_canon = "UNCONTESTED_PRIMARY_NORM"

    trace = DefeasibleProofTrace(
        query=query,
        prevailing_record_id=winner_id if not abstained else None,
        abstained=abstained,
        abstention_reason=abstention_reason,
        at_valid_time=valid_iso,
        as_of_transaction_time=tx_iso,
        candidates=candidate_nodes,
        excluded_records=excluded,
        winning_canon=winning_canon,
    )
    trace.compute_trace_hash()
    return trace
