"""
Automated Statutory Diff and Downstream Amendment Impact Analyzer.

Performs clause-level legislative amendment diffing between an existing statutory
rule and a proposed or enacted amendment, and traces the transitive impact across
downstream corporate policies, contractual clauses, and dependent norms in the
Semantic Knowledge Graph.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.domain.models import MemoryRecord, RelationType
from continuous_legal_memory.reasoning.modal_verifier import (
    infer_deontic_modality,
)
from continuous_legal_memory.retrieval.hybrid_lexical import extract_legal_citations

_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.;])\s+|\n+")
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_\u00C0-\u017F]{3,}")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "under",
        "shall",
        "must",
        "may",
        "not",
        "all",
        "any",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "such",
        "within",
        "upon",
        "when",
        "where",
        "which",
        "article",
        "section",
        "paragraph",
        "statute",
        "policy",
        "rule",
        "clause",
    }
)


def _split_clauses(text: str) -> list[str]:
    clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(text.strip()) if c.strip()]
    return clauses if clauses else [text.strip()]


def _extract_subject_tokens(text: str) -> set[str]:
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    return {t for t in tokens if t not in _STOPWORDS}


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 1.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


@dataclass(frozen=True)
class ClauseDiffSpan:
    """
    Fine-grained clause-level diff between original and amended statutory text.
    """

    operation: str  # "ADDED", "REMOVED", "MODIFIED", "UNCHANGED"
    old_clause: str | None
    new_clause: str | None
    old_modality: str
    new_modality: str
    modality_shifted: bool
    added_citations: list[str] = field(default_factory=list)
    removed_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "old_clause": self.old_clause,
            "new_clause": self.new_clause,
            "old_modality": self.old_modality,
            "new_modality": self.new_modality,
            "modality_shifted": self.modality_shifted,
            "added_citations": self.added_citations,
            "removed_citations": self.removed_citations,
        }


@dataclass(frozen=True)
class DownstreamObligationImpact:
    """
    Describes the impact of a statutory amendment on a subordinate contract,
    corporate policy, or dependent norm.
    """

    dependent_rule_id: str
    dependent_rule_text: str
    dependent_authority_rank: int
    dependency_path: str
    impact_severity: str
    requires_redrafting: bool
    action_vector_alignment: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependent_rule_id": self.dependent_rule_id,
            "dependent_rule_text": self.dependent_rule_text,
            "dependent_authority_rank": self.dependent_authority_rank,
            "dependency_path": self.dependency_path,
            "impact_severity": self.impact_severity,
            "requires_redrafting": self.requires_redrafting,
            "action_vector_alignment": round(self.action_vector_alignment, 6),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AmendmentImpactReport:
    """
    Complete statutory amendment diff and downstream compliance blast-radius report.
    """

    target_rule_id: str
    tenant_id: str
    old_modality: str
    new_modality: str
    modality_reversed: bool
    old_authority_rank: int
    new_authority_rank: int
    action_vector_cosine_similarity: float
    clause_diffs: list[ClauseDiffSpan]
    impacted_obligations: list[DownstreamObligationImpact]
    total_impacted_count: int
    critical_impact_count: int
    report_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_rule_id": self.target_rule_id,
            "tenant_id": self.tenant_id,
            "old_modality": self.old_modality,
            "new_modality": self.new_modality,
            "modality_reversed": self.modality_reversed,
            "old_authority_rank": self.old_authority_rank,
            "new_authority_rank": self.new_authority_rank,
            "action_vector_cosine_similarity": round(self.action_vector_cosine_similarity, 6),
            "clause_diffs": [c.to_dict() for c in self.clause_diffs],
            "impacted_obligations": [o.to_dict() for o in self.impacted_obligations],
            "total_impacted_count": self.total_impacted_count,
            "critical_impact_count": self.critical_impact_count,
            "report_hash": self.report_hash,
        }


class StatutoryAmendmentAnalyzer:
    """
    Computes clause-level statutory diffs and traces transitive downstream impacts
    across subordinate policies, contracts, and Semantic Knowledge Graph dependencies.
    """

    def diff_clauses(self, old_text: str, new_text: str) -> list[ClauseDiffSpan]:
        """
        Compute a structured sequence of ClauseDiffSpan entries between old_text and new_text.
        """
        old_clauses = _split_clauses(old_text)
        new_clauses = _split_clauses(new_text)
        matcher = difflib.SequenceMatcher(None, old_clauses, new_clauses)
        spans: list[ClauseDiffSpan] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for idx in range(i1, i2):
                    clause = old_clauses[idx]
                    mod = infer_deontic_modality(clause).name
                    spans.append(
                        ClauseDiffSpan(
                            operation="UNCHANGED",
                            old_clause=clause,
                            new_clause=clause,
                            old_modality=mod,
                            new_modality=mod,
                            modality_shifted=False,
                        )
                    )
            elif tag == "replace":
                old_slice = old_clauses[i1:i2]
                new_slice = new_clauses[j1:j2]
                pair_len = max(len(old_slice), len(new_slice))
                for k in range(pair_len):
                    oc = old_slice[k] if k < len(old_slice) else None
                    nc = new_slice[k] if k < len(new_slice) else None
                    old_mod = infer_deontic_modality(oc).name if oc else "NEUTRAL"
                    new_mod = infer_deontic_modality(nc).name if nc else "NEUTRAL"
                    old_cites_set = set(extract_legal_citations(oc)) if oc else set()
                    new_cites_set = set(extract_legal_citations(nc)) if nc else set()
                    op = (
                        "MODIFIED"
                        if (oc is not None and nc is not None)
                        else ("ADDED" if nc else "REMOVED")
                    )
                    spans.append(
                        ClauseDiffSpan(
                            operation=op,
                            old_clause=oc,
                            new_clause=nc,
                            old_modality=old_mod,
                            new_modality=new_mod,
                            modality_shifted=(old_mod != new_mod),
                            added_citations=sorted(new_cites_set - old_cites_set),
                            removed_citations=sorted(old_cites_set - new_cites_set),
                        )
                    )
            elif tag == "delete":
                for idx in range(i1, i2):
                    oc = old_clauses[idx]
                    old_mod = infer_deontic_modality(oc).name
                    removed_list = sorted(extract_legal_citations(oc))
                    spans.append(
                        ClauseDiffSpan(
                            operation="REMOVED",
                            old_clause=oc,
                            new_clause=None,
                            old_modality=old_mod,
                            new_modality="NEUTRAL",
                            modality_shifted=(old_mod != "NEUTRAL"),
                            removed_citations=removed_list,
                        )
                    )
            elif tag == "insert":
                for idx in range(j1, j2):
                    nc = new_clauses[idx]
                    new_mod = infer_deontic_modality(nc).name
                    added_list = sorted(extract_legal_citations(nc))
                    spans.append(
                        ClauseDiffSpan(
                            operation="ADDED",
                            old_clause=None,
                            new_clause=nc,
                            old_modality="NEUTRAL",
                            new_modality=new_mod,
                            modality_shifted=(new_mod != "NEUTRAL"),
                            added_citations=added_list,
                        )
                    )
        return spans

    @staticmethod
    def _find_graph_dependents(
        target_rule_id: str,
        graph: SemanticKnowledgeGraph,
        tenant_id: str,
    ) -> dict[str, str]:
        """
        Traverse the SemanticKnowledgeGraph to find all rules that depend on or derive
        from `target_rule_id` (both direct and 2-hop transitive).
        """
        dependents: dict[str, str] = {}
        frontier: list[tuple[str, int]] = [(target_rule_id, 0)]
        visited: set[str] = {target_rule_id}
        tenant_edges = graph.get_edges(tenant_id=tenant_id)

        while frontier:
            current_id, depth = frontier.pop(0)
            if depth >= 3:
                continue
            for edge in tenant_edges:
                if edge.relation_type in (
                    RelationType.DEPENDS_ON,
                    RelationType.DERIVES_FROM,
                    RelationType.SUPERSEDES,
                ):
                    if edge.target_id == current_id and edge.source_id not in visited:
                        visited.add(edge.source_id)
                        dependents[edge.source_id] = f"GRAPH_{edge.relation_type.value.upper()}"
                        frontier.append((edge.source_id, depth + 1))
                    elif edge.source_id == current_id and edge.target_id not in visited:
                        visited.add(edge.target_id)
                        dependents[edge.target_id] = f"GRAPH_{edge.relation_type.value.upper()}"
                        frontier.append((edge.target_id, depth + 1))
        return dependents

    def analyze_amendment_impact(
        self,
        target_record: MemoryRecord,
        amended_text: str,
        amended_action_vector: list[float] | None,
        active_records: list[MemoryRecord],
        graph: SemanticKnowledgeGraph,
        tenant_id: str = "default",
        amended_authority_rank: int | None = None,
    ) -> AmendmentImpactReport:
        """
        Analyze the clause-level diff of amending `target_record` and compute the
        downstream impact across all subordinate/dependent norms in `active_records`.
        """
        target_id = str(target_record.record_id or "unindexed")
        old_vec = [float(x) for x in target_record.value_vector.flatten().tolist()]
        new_vec = (
            [float(x) for x in amended_action_vector]
            if amended_action_vector is not None
            else old_vec
        )
        old_rank = target_record.authority_rank
        new_rank = amended_authority_rank if amended_authority_rank is not None else old_rank

        old_mod = infer_deontic_modality(target_record.text).name
        new_mod = infer_deontic_modality(amended_text).name
        modality_reversed = (
            (old_mod == "OBLIGATION" and new_mod == "PROHIBITION")
            or (old_mod == "PROHIBITION" and new_mod in ("OBLIGATION", "PERMISSION"))
            or (old_mod == "PERMISSION" and new_mod == "PROHIBITION")
        )
        vec_sim = _cosine_similarity(old_vec, new_vec)
        clause_diffs = self.diff_clauses(target_record.text, amended_text)

        target_cites = set(extract_legal_citations(target_record.text)) | set(
            extract_legal_citations(amended_text)
        )
        target_tokens = _extract_subject_tokens(target_record.text) | _extract_subject_tokens(
            amended_text
        )
        graph_deps = self._find_graph_dependents(target_id, graph, tenant_id=tenant_id)

        impacts: list[DownstreamObligationImpact] = []
        for rec in active_records:
            rec_id = str(rec.record_id or "")
            if not rec_id or rec_id == target_id:
                continue

            dep_path: str | None = graph_deps.get(rec_id)
            rec_meta_parent = str(
                rec.metadata.get("depends_on", "") or rec.metadata.get("derives_from", "")
            )
            if dep_path is None and rec_meta_parent == target_id:
                dep_path = "METADATA_DEPENDS_ON"

            if dep_path is None and target_cites:
                rec_cites = set(extract_legal_citations(rec.text))
                if target_cites & rec_cites:
                    dep_path = "SHARED_STATUTORY_CITATION"

            if dep_path is None:
                rec_tokens = _extract_subject_tokens(rec.text)
                overlap = len(target_tokens & rec_tokens)
                union = max(1, len(target_tokens | rec_tokens))
                if overlap >= 2 and (overlap / union) >= 0.18 and rec.authority_rank <= new_rank:
                    dep_path = "SUBJECT_MATTER_SUBORDINATE"

            if dep_path is None:
                continue

            rec_vec = [float(x) for x in rec.value_vector.flatten().tolist()]
            rec_alignment = _cosine_similarity(new_vec, rec_vec)
            rec_mod = infer_deontic_modality(rec.text).name

            if (
                (new_mod == "PROHIBITION" and rec_mod in ("OBLIGATION", "PERMISSION"))
                or (new_mod == "OBLIGATION" and rec_mod == "PROHIBITION")
                or (modality_reversed and rec_alignment < 0.5)
            ):
                severity = "CRITICAL_MODALITY_REVERSAL"
                requires_redrafting = True
                rationale = (
                    f"Amended statute shifts modality to {new_mod} while dependent rule "
                    f"({rec_id}, rank {rec.authority_rank}) asserts {rec_mod} "
                    f"(action alignment={rec_alignment:.2f})."
                )
            elif rec_alignment < 0.25:
                severity = "ACTION_VECTOR_INVALIDATED"
                requires_redrafting = True
                rationale = (
                    f"Amended statutory action vector diverges from dependent rule "
                    f"({rec_id}) with cosine alignment {rec_alignment:.2f} < 0.25."
                )
            elif new_rank > old_rank and rec.authority_rank < new_rank and rec_alignment < 0.85:
                severity = "AUTHORITY_ESCALATION"
                requires_redrafting = True
                rationale = (
                    f"Amended rule escalated from authority_rank {old_rank} to {new_rank}, "
                    f"strictly overriding subordinate policy ({rec_id}, rank {rec.authority_rank})."
                )
            elif any(
                c.modality_shifted or c.added_citations or c.removed_citations for c in clause_diffs
            ):
                severity = "CITATION_DEPENDENCY_UPDATE"
                requires_redrafting = rec_alignment < 0.75
                rationale = (
                    f"Clause-level statutory text modified along dependency path {dep_path}; "
                    f"review cross-references in {rec_id}."
                )
            else:
                severity = "COMPATIBLE"
                requires_redrafting = False
                rationale = f"Dependent rule ({rec_id}) remains aligned with amended statute."

            impacts.append(
                DownstreamObligationImpact(
                    dependent_rule_id=rec_id,
                    dependent_rule_text=rec.text,
                    dependent_authority_rank=rec.authority_rank,
                    dependency_path=dep_path,
                    impact_severity=severity,
                    requires_redrafting=requires_redrafting,
                    action_vector_alignment=rec_alignment,
                    rationale=rationale,
                )
            )

        impacts.sort(
            key=lambda item: (
                0 if item.requires_redrafting else 1,
                item.action_vector_alignment,
                item.dependent_rule_id,
            )
        )
        critical_count = sum(1 for item in impacts if item.requires_redrafting)

        hash_payload = {
            "action_vector_cosine_similarity": round(vec_sim, 6),
            "clause_diffs": [c.to_dict() for c in clause_diffs],
            "critical_impact_count": critical_count,
            "impacted_obligations": [o.to_dict() for o in impacts],
            "modality_reversed": modality_reversed,
            "new_authority_rank": new_rank,
            "new_modality": new_mod,
            "old_authority_rank": old_rank,
            "old_modality": old_mod,
            "target_rule_id": target_id,
            "tenant_id": tenant_id,
        }
        report_hash = hashlib.sha256(
            json.dumps(hash_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return AmendmentImpactReport(
            target_rule_id=target_id,
            tenant_id=tenant_id,
            old_modality=old_mod,
            new_modality=new_mod,
            modality_reversed=modality_reversed,
            old_authority_rank=old_rank,
            new_authority_rank=new_rank,
            action_vector_cosine_similarity=vec_sim,
            clause_diffs=clause_diffs,
            impacted_obligations=impacts,
            total_impacted_count=len(impacts),
            critical_impact_count=critical_count,
            report_hash=report_hash,
        )
