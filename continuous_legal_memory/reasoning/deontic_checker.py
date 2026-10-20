"""
Formal Deontic Consistency Checker and Antinomy Detector.

Grounded in Defeasible Deontic Logic (Sartor, 2005; Governatori et al., AI & Law):
1. Detects directed normative cycles (`SUPERSEDES` and `DEPENDS_ON` / `DERIVES_FROM`)
   in `SemanticKnowledgeGraph` using Tarjan's Strongly Connected Components algorithm in O(|V| + |E|).
2. Identifies unresolved horizontal antinomies (`DeonticAntinomy`) between active rules
   that are *in pari materia*, share overlapping temporal validity windows and jurisdiction,
   prescribe opposing deontic outcomes, and cannot be disambiguated by explicit `SUPERSEDES`
   links, *lex superior* (`authority_rank`), *lex specialis* (`specificity`), or *lex posterior*
   (`valid_from` timestamp).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch
import torch.nn.functional as F

from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.domain.models import MemoryRecord, RelationType
from continuous_legal_memory.retrieval.precedence import (
    _are_in_pari_materia,
    _extract_specificity,
    _extract_valid_timestamp,
)


@dataclass
class DeonticAntinomy:
    """
    Represents an unresolved normative contradiction (antinomy) between two active legal rules.
    """

    rule_a_id: str
    rule_b_id: str
    rule_a_text: str
    rule_b_text: str
    similarity: float
    action_alignment: float
    authority_rank: int
    specificity: float
    jurisdiction: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_a_id": self.rule_a_id,
            "rule_b_id": self.rule_b_id,
            "rule_a_text": self.rule_a_text,
            "rule_b_text": self.rule_b_text,
            "similarity": self.similarity,
            "action_alignment": self.action_alignment,
            "authority_rank": self.authority_rank,
            "specificity": self.specificity,
            "jurisdiction": self.jurisdiction,
            "reason": self.reason,
        }


@dataclass
class DeonticConsistencyReport:
    """
    Comprehensive audit report produced by `DeonticConsistencyChecker`.
    """

    is_consistent: bool
    antinomies: list[DeonticAntinomy] = field(default_factory=list)
    supersession_cycles: list[list[str]] = field(default_factory=list)
    dependency_cycles: list[list[str]] = field(default_factory=list)
    total_active_rules: int = 0
    tenant_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "antinomies": [a.to_dict() for a in self.antinomies],
            "supersession_cycles": self.supersession_cycles,
            "dependency_cycles": self.dependency_cycles,
            "total_active_rules": self.total_active_rules,
            "tenant_id": self.tenant_id,
        }


def _intervals_overlap(rec_a: MemoryRecord, rec_b: MemoryRecord) -> bool:
    """Return True if the bi-temporal validity intervals of `rec_a` and `rec_b` overlap."""
    vf_a = rec_a.valid_from.replace(tzinfo=timezone.utc) if rec_a.valid_from.tzinfo is None else rec_a.valid_from
    vf_b = rec_b.valid_from.replace(tzinfo=timezone.utc) if rec_b.valid_from.tzinfo is None else rec_b.valid_from
    start_max = max(vf_a, vf_b)

    vt_a = (
        rec_a.valid_to.replace(tzinfo=timezone.utc)
        if (rec_a.valid_to is not None and rec_a.valid_to.tzinfo is None)
        else rec_a.valid_to
    )
    vt_b = (
        rec_b.valid_to.replace(tzinfo=timezone.utc)
        if (rec_b.valid_to is not None and rec_b.valid_to.tzinfo is None)
        else rec_b.valid_to
    )

    if vt_a is not None and start_max > vt_a:
        return False
    return not (vt_b is not None and start_max > vt_b)


def _outcomes_conflict(rec_a: MemoryRecord, rec_b: MemoryRecord) -> tuple[bool, float]:
    """
    Determine whether two records prescribe opposing normative action vectors.
    Returns `(is_conflicting, cosine_alignment)`.
    """
    va = rec_a.value_vector.detach().flatten().float()
    vb = rec_b.value_vector.detach().flatten().float()
    if va.numel() == 0 or vb.numel() == 0 or va.shape != vb.shape:
        return False, 1.0
    norm_a = float(torch.norm(va).item())
    norm_b = float(torch.norm(vb).item())
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return False, 1.0
    cos_sim = float(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())
    argmax_diff = int(torch.argmax(va).item()) != int(torch.argmax(vb).item())
    return (argmax_diff or cos_sim < 0.25), cos_sim


class DeonticConsistencyChecker:
    """
    Symbolic and geometric verifier for normative graph consistency and antinomy detection.
    """

    def __init__(
        self,
        pari_materia_threshold: float = 0.35,
        require_explicit_supersession_for_amendments: bool = False,
    ) -> None:
        self.pari_materia_threshold = float(pari_materia_threshold)
        self.require_explicit_supersession_for_amendments = bool(
            require_explicit_supersession_for_amendments
        )

    @staticmethod
    def detect_cycles(
        graph: SemanticKnowledgeGraph,
        relation_types: set[RelationType],
        tenant_id: str | None = None,
    ) -> list[list[str]]:
        """
        Find all strongly connected cycle components (and self-loops) among edges of
        `relation_types` using Tarjan's SCC algorithm in O(|V| + |E|).
        """
        adj: dict[str, list[str]] = {}
        nodes_set: set[str] = set()
        for edge in graph.edges:
            if edge.relation_type not in relation_types:
                continue
            if tenant_id is not None and edge.tenant_id != tenant_id:
                continue
            adj.setdefault(edge.source_id, []).append(edge.target_id)
            nodes_set.add(edge.source_id)
            nodes_set.add(edge.target_id)

        index_counter = 0
        indices: dict[str, int] = {}
        lowlink: dict[str, int] = {}
        stack: list[str] = []
        on_stack: set[str] = set()
        cycles: list[list[str]] = []

        def strongconnect(node: str) -> None:
            nonlocal index_counter
            indices[node] = index_counter
            lowlink[node] = index_counter
            index_counter += 1
            stack.append(node)
            on_stack.add(node)

            for neighbor in adj.get(node, []):
                if neighbor not in indices:
                    strongconnect(neighbor)
                    lowlink[node] = min(lowlink[node], lowlink[neighbor])
                elif neighbor in on_stack:
                    lowlink[node] = min(lowlink[node], indices[neighbor])

            if lowlink[node] == indices[node]:
                scc: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == node:
                        break
                if len(scc) > 1:
                    scc.reverse()
                    cycles.append(scc)
                elif len(scc) == 1:
                    single = scc[0]
                    if single in adj.get(single, []):
                        cycles.append([single])

        for n in sorted(nodes_set):
            if n not in indices:
                strongconnect(n)

        return cycles

    def audit(
        self,
        records: list[MemoryRecord],
        graph: SemanticKnowledgeGraph,
        tenant_id: str = "default",
        at_time: datetime | None = None,
        register_contradiction_edges: bool = True,
    ) -> DeonticConsistencyReport:
        """
        Audit active legal memory records and semantic knowledge graph for deontic cycles
        and unresolved normative antinomies.
        """
        eval_time = at_time or datetime.now(timezone.utc)
        active_records = [
            r
            for r in records
            if r.tenant_id == tenant_id and r.is_temporally_valid(eval_time)
        ]

        supersession_cycles = self.detect_cycles(
            graph, {RelationType.SUPERSEDES}, tenant_id=tenant_id
        )
        dependency_cycles = self.detect_cycles(
            graph, {RelationType.DEPENDS_ON, RelationType.DERIVES_FROM}, tenant_id=tenant_id
        )

        # Build set of explicit supersession pairs from graph and metadata
        explicit_supersedes: set[tuple[str, str]] = set()
        for edge in graph.edges:
            if edge.tenant_id == tenant_id and edge.relation_type == RelationType.SUPERSEDES:
                explicit_supersedes.add((edge.source_id, edge.target_id))

        for rec in active_records:
            rid = rec.record_id or str(rec.metadata.get("node_id", ""))
            sup_raw = rec.metadata.get("supersedes") if isinstance(rec.metadata, dict) else None
            if rid and sup_raw:
                if isinstance(sup_raw, str):
                    explicit_supersedes.add((rid, sup_raw))
                elif isinstance(sup_raw, (list, tuple, set)):
                    for t in sup_raw:
                        explicit_supersedes.add((rid, str(t)))

        antinomies: list[DeonticAntinomy] = []
        n = len(active_records)
        for i in range(n):
            rec_a = active_records[i]
            id_a = rec_a.record_id or str(rec_a.metadata.get("node_id", f"idx_{i}"))
            for j in range(i + 1, n):
                rec_b = active_records[j]
                id_b = rec_b.record_id or str(rec_b.metadata.get("node_id", f"idx_{j}"))

                # 1. Must share jurisdiction (or both be general/unspecified)
                if rec_a.jurisdiction != rec_b.jurisdiction:
                    continue

                # 2. Must have overlapping temporal validity intervals
                if not _intervals_overlap(rec_a, rec_b):
                    continue

                # 3. Skip if an explicit SUPERSEDES link exists between them
                if (id_a, id_b) in explicit_supersedes or (id_b, id_a) in explicit_supersedes:
                    continue

                # 4. Must prescribe conflicting deontic action vectors
                conflicts, alignment = _outcomes_conflict(rec_a, rec_b)
                if not conflicts:
                    continue

                # 5. Must be in pari materia (addressing the same legal subject matter)
                if not _are_in_pari_materia(rec_a, rec_b, self.pari_materia_threshold):
                    continue

                # 6. Check whether lex superior or lex specialis resolves the conflict
                if int(rec_a.authority_rank) != int(rec_b.authority_rank):
                    continue

                spec_a = _extract_specificity(rec_a)
                spec_b = _extract_specificity(rec_b)
                if abs(spec_a - spec_b) > 1e-6:
                    continue

                # 7. Equal authority & equal specificity: check lex posterior timestamp
                ts_a = _extract_valid_timestamp(rec_a)
                ts_b = _extract_valid_timestamp(rec_b)
                if abs(ts_a - ts_b) > 1e-6 and not self.require_explicit_supersession_for_amendments:
                    continue

                va_key = F.normalize(rec_a.key_vector.detach().flatten().float(), p=2, dim=0)
                vb_key = F.normalize(rec_b.key_vector.detach().flatten().float(), p=2, dim=0)
                sim = float(torch.dot(va_key, vb_key).item())

                antinomy = DeonticAntinomy(
                    rule_a_id=id_a,
                    rule_b_id=id_b,
                    rule_a_text=rec_a.text,
                    rule_b_text=rec_b.text,
                    similarity=sim,
                    action_alignment=alignment,
                    authority_rank=int(rec_a.authority_rank),
                    specificity=spec_a,
                    jurisdiction=rec_a.jurisdiction,
                    reason=(
                        "Simultaneous horizontal antinomy: identical authority_rank, "
                        "specificity, and valid_from timestamp with opposing deontic outcomes."
                    ),
                )
                antinomies.append(antinomy)

                if (
                    register_contradiction_edges
                    and id_a in graph.nodes
                    and id_b in graph.nodes
                ):
                    existing = {
                        (e.source_id, e.target_id)
                        for e in graph.edges
                        if e.relation_type == RelationType.CONTRADICTS and e.tenant_id == tenant_id
                    }
                    if (id_a, id_b) not in existing and (id_b, id_a) not in existing:
                        graph.add_edge(
                            id_a,
                            id_b,
                            RelationType.CONTRADICTS,
                            weight=1.0,
                            tenant_id=tenant_id,
                        )

        is_consistent = (
            len(antinomies) == 0
            and len(supersession_cycles) == 0
            and len(dependency_cycles) == 0
        )
        return DeonticConsistencyReport(
            is_consistent=is_consistent,
            antinomies=antinomies,
            supersession_cycles=supersession_cycles,
            dependency_cycles=dependency_cycles,
            total_active_rules=len(active_records),
            tenant_id=tenant_id,
        )
