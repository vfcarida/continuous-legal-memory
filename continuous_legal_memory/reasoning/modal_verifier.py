"""
Formal Deontic Modal Satisfiability (KD) & Pre-Flight Rule Ingestion Simulator.

Implements a decision procedure over Standard Deontic Logic (SDL / system KD, von Wright 1951;
Governatori & Sartor, Defeasible Deontic Logic) combined with Kelsenian hierarchical precedence
and bi-temporal interval intersection.

Enables legal AI agents and legislative/contract drafting workflows to run a non-mutating
pre-flight simulation (`simulate_rule_ingestion`) before committing a candidate norm to
`EpisodicMemory`, `SemanticKnowledgeGraph`, or `SparseMerkleTree`:
1. KD Modal Axioms:
   - Axiom D (Consistency of Obligations): O(phi) -> P(phi), i.e., ~(O(phi) & F(phi)) and ~(O(phi) & O(~phi)).
   - Modal Duality: F(phi) <-> ~P(phi), so F(phi) & P(phi) is contradictory on the same scope.
2. Defeasibility & Shadowing Analysis:
   - Detects whether the candidate norm is immediately shadowed by an existing higher-authority
     statute (*lex superior*), overrides an existing norm (*lex posterior* / *lex specialis* / `SUPERSEDES`),
     or creates an unresolvable horizontal antinomy.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F

from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.domain.models import EntityType, MemoryRecord, RelationType
from continuous_legal_memory.reasoning.deontic_checker import (
    _intervals_overlap,
    _outcomes_conflict,
)
from continuous_legal_memory.retrieval.precedence import (
    _extract_specificity,
    _extract_valid_timestamp,
)


class DeonticModality(str, Enum):
    """Standard Deontic Logic (KD) modal operators."""

    OBLIGATION = "O"
    PROHIBITION = "F"
    PERMISSION = "P"


_PROHIBITION_RE = re.compile(
    r"\b(?:shall\s+not|must\s+not|may\s+not|prohibited|forbidden|unlawful|banned|never|"
    r"proibido|vedado|n[aã]o\s+poder[aá]|n[aã]o\s+deve|"
    r"interdit|ne\s+peut\s+pas|ne\s+doit\s+pas|"
    r"verboten|unzul[aä]ssig|darf\s+nicht)\b",
    re.IGNORECASE,
)
_PERMISSION_RE = re.compile(
    r"\b(?:may|permitted|allowed|entitled|optional|discretionary|"
    r"permitido|poder[aá]|facultado|"
    r"peut|autoris[eé]|"
    r"darf|zul[aä]ssig)\b",
    re.IGNORECASE,
)
_OBLIGATION_RE = re.compile(
    r"\b(?:shall|must|mandatory|required|obligated|compulsory|duty|"
    r"deve|dever[aá]|obrigat[oó]rio|"
    r"doit|obligatoire|"
    r"muss|verpflichtet)\b",
    re.IGNORECASE,
)


def infer_deontic_modality(
    text: str,
    action_vector: list[float] | torch.Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> DeonticModality:
    """
    Infer the primary Standard Deontic Logic operator (`O`, `F`, or `P`) for a legal rule.
    """
    if metadata:
        raw_mod = str(metadata.get("modality", "")).upper().strip()
        if raw_mod in ("O", "OBLIGATION"):
            return DeonticModality.OBLIGATION
        if raw_mod in ("F", "PROHIBITION", "FORBIDDEN"):
            return DeonticModality.PROHIBITION
        if raw_mod in ("P", "PERMISSION", "PERMITTED"):
            return DeonticModality.PERMISSION

        raw_et = str(metadata.get("entity_type", "")).lower().strip()
        if raw_et == EntityType.PROHIBITION.value:
            return DeonticModality.PROHIBITION
        if raw_et == EntityType.PERMISSION.value:
            return DeonticModality.PERMISSION
        if raw_et == EntityType.OBLIGATION.value:
            return DeonticModality.OBLIGATION

    if _PROHIBITION_RE.search(text):
        return DeonticModality.PROHIBITION
    if _PERMISSION_RE.search(text) and not _OBLIGATION_RE.search(text):
        return DeonticModality.PERMISSION
    if _OBLIGATION_RE.search(text):
        return DeonticModality.OBLIGATION

    if action_vector is not None:
        vec = (
            action_vector.detach().cpu().flatten().tolist()
            if isinstance(action_vector, torch.Tensor)
            else list(action_vector)
        )
        if len(vec) >= 2 and float(vec[0]) > float(vec[1]):
            return DeonticModality.PROHIBITION

    return DeonticModality.OBLIGATION


def check_kd_modal_conflict(
    mod_a: DeonticModality,
    mod_b: DeonticModality,
    outcomes_oppose: bool = False,
    action_alignment: float = 0.0,
) -> tuple[bool, str | None]:
    """
    Evaluate Standard Deontic Logic (KD) satisfiability between two co-applicable norms.

    Returns:
        Tuple `(has_modal_conflict, kd_axiom_violation_code)`.
    """
    if {mod_a, mod_b} == {DeonticModality.OBLIGATION, DeonticModality.PROHIBITION}:
        return (
            True,
            "KD_AXIOM_D_CONTRARIETY (O(phi) & F(phi) -> False)",
        )
    if {mod_a, mod_b} == {DeonticModality.PROHIBITION, DeonticModality.PERMISSION}:
        return (
            True,
            "KD_DUALITY_CONTRADICTION (F(phi) & P(phi) -> False)",
        )
    if outcomes_oppose:
        if mod_a == DeonticModality.OBLIGATION and mod_b == DeonticModality.OBLIGATION:
            return (
                True,
                f"KD_OPPOSING_OBLIGATIONS (O(phi) & O(~phi) -> False, align={action_alignment:.3f})",
            )
        return (
            True,
            f"KD_ACTION_POLARITY_CONFLICT ({mod_a.value} vs {mod_b.value}, align={action_alignment:.3f})",
        )
    return False, None


@dataclass(frozen=True)
class ModalConflictFinding:
    """
    Detailed finding for a pair of norms evaluated by `DeonticModalVerifier`.
    """

    existing_record_id: str
    existing_text: str
    existing_modality: str
    candidate_modality: str
    kd_violation: str
    similarity: float
    action_alignment: float
    resolution_status: str
    governing_canon: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreFlightSimulationReport:
    """
    Result of running a non-mutating pre-flight ingestion simulation for a candidate legal rule.
    """

    is_satisfiable: bool
    candidate_modality: str
    would_create_cycle: bool = False
    unresolved_conflicts: list[ModalConflictFinding] = field(default_factory=list)
    resolved_overrides: list[ModalConflictFinding] = field(default_factory=list)
    shadowed_by: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    tenant_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_satisfiable": self.is_satisfiable,
            "candidate_modality": self.candidate_modality,
            "would_create_cycle": self.would_create_cycle,
            "unresolved_conflicts": [c.to_dict() for c in self.unresolved_conflicts],
            "resolved_overrides": [c.to_dict() for c in self.resolved_overrides],
            "shadowed_by": list(self.shadowed_by),
            "recommendations": list(self.recommendations),
            "tenant_id": self.tenant_id,
        }


class DeonticModalVerifier:
    """
    Pre-flight Standard Deontic Logic (KD) verifier and defeasible shadowing simulator.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        timestamp_tie_tolerance_sec: float = 1.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.timestamp_tie_tolerance_sec = timestamp_tie_tolerance_sec

    def simulate_candidate(
        self,
        candidate: MemoryRecord,
        existing_records: list[MemoryRecord],
        graph: SemanticKnowledgeGraph | None = None,
    ) -> PreFlightSimulationReport:
        """
        Simulate ingesting `candidate` against `existing_records` and `graph` without mutating state.
        """
        cand_mod = infer_deontic_modality(
            candidate.text,
            candidate.value_vector,
            candidate.metadata,
        )
        tenant_id = candidate.tenant_id

        # 1. Check if proposed graph relations would introduce a cycle
        would_cycle = False
        recommendations: list[str] = []
        cand_node_id = (
            str(candidate.metadata.get("node_id"))
            if (candidate.metadata and "node_id" in candidate.metadata)
            else (candidate.record_id or "candidate_node")
        )
        explicit_supersedes: set[str] = set()
        if candidate.metadata:
            sup = candidate.metadata.get("supersedes")
            if isinstance(sup, str):
                explicit_supersedes.add(sup)
            elif isinstance(sup, (list, set, tuple)):
                explicit_supersedes.update(str(x) for x in sup)
            for rel in candidate.metadata.get("relations", []):
                if isinstance(rel, dict):
                    rel_t = str(rel.get("relation_type", "")).lower()
                    tgt = str(rel.get("target_id", ""))
                    if rel_t == RelationType.SUPERSEDES.value and tgt:
                        explicit_supersedes.add(tgt)
                    if graph is not None and tgt and rel_t in (
                        RelationType.SUPERSEDES.value,
                        RelationType.DEPENDS_ON.value,
                        RelationType.DERIVES_FROM.value,
                    ):
                        if tgt == cand_node_id:
                            would_cycle = True
                        else:
                            # Check if cand_node_id is reachable from tgt
                            visited: set[str] = set()
                            frontier = [tgt]
                            while frontier:
                                curr = frontier.pop()
                                if curr == cand_node_id:
                                    would_cycle = True
                                    break
                                if curr not in visited:
                                    visited.add(curr)
                                    for edge in graph.edges:
                                        if (
                                            edge.source_id == curr
                                            and edge.relation_type.value == rel_t
                                            and edge.tenant_id == tenant_id
                                        ):
                                            frontier.append(edge.target_id)

        if would_cycle:
            recommendations.append(
                f"Remove cyclic relation on '{cand_node_id}' before ingestion to preserve DAG hierarchy."
            )

        unresolved: list[ModalConflictFinding] = []
        resolved: list[ModalConflictFinding] = []
        shadowed_by: list[str] = []

        cand_vec = F.normalize(candidate.key_vector.detach().flatten().float(), p=2, dim=0)
        cand_spec = _extract_specificity(candidate)
        cand_ts = _extract_valid_timestamp(candidate)

        for existing in existing_records:
            if existing.tenant_id != tenant_id:
                continue
            if not _intervals_overlap(candidate, existing):
                continue
            if (
                candidate.jurisdiction is not None
                and existing.jurisdiction is not None
                and candidate.jurisdiction != existing.jurisdiction
            ):
                continue

            ex_vec = F.normalize(existing.key_vector.detach().flatten().float(), p=2, dim=0)
            sim = float(torch.dot(cand_vec, ex_vec).item()) if cand_vec.shape == ex_vec.shape else 0.0
            if sim < self.similarity_threshold:
                continue

            ex_mod = infer_deontic_modality(existing.text, existing.value_vector, existing.metadata)
            oppose, align = _outcomes_conflict(candidate, existing)
            has_conflict, kd_violation = check_kd_modal_conflict(cand_mod, ex_mod, oppose, align)
            if not has_conflict or kd_violation is None:
                continue

            ex_id = existing.record_id or "unknown"

            # Determine if hierarchy / specificity / supersession / posteriority resolves the conflict
            if ex_id in explicit_supersedes:
                resolved.append(
                    ModalConflictFinding(
                        existing_record_id=ex_id,
                        existing_text=existing.text,
                        existing_modality=ex_mod.value,
                        candidate_modality=cand_mod.value,
                        kd_violation=kd_violation,
                        similarity=sim,
                        action_alignment=align,
                        resolution_status="CANDIDATE_OVERRIDES_EXISTING",
                        governing_canon="EXPLICIT_SUPERSEDES",
                    )
                )
                continue

            if int(candidate.authority_rank) > int(existing.authority_rank):
                resolved.append(
                    ModalConflictFinding(
                        existing_record_id=ex_id,
                        existing_text=existing.text,
                        existing_modality=ex_mod.value,
                        candidate_modality=cand_mod.value,
                        kd_violation=kd_violation,
                        similarity=sim,
                        action_alignment=align,
                        resolution_status="CANDIDATE_OVERRIDES_EXISTING",
                        governing_canon="LEX_SUPERIOR",
                    )
                )
                continue

            if int(candidate.authority_rank) < int(existing.authority_rank):
                shadowed_by.append(ex_id)
                resolved.append(
                    ModalConflictFinding(
                        existing_record_id=ex_id,
                        existing_text=existing.text,
                        existing_modality=ex_mod.value,
                        candidate_modality=cand_mod.value,
                        kd_violation=kd_violation,
                        similarity=sim,
                        action_alignment=align,
                        resolution_status="SHADOWED_BY_EXISTING",
                        governing_canon="LEX_SUPERIOR",
                    )
                )
                recommendations.append(
                    f"Candidate norm (authority_rank={candidate.authority_rank}) will be shadowed by "
                    f"higher-authority rule '{ex_id}' (authority_rank={existing.authority_rank})."
                )
                continue

            ex_spec = _extract_specificity(existing)
            if abs(cand_spec - ex_spec) > 1e-6:
                if cand_spec > ex_spec:
                    resolved.append(
                        ModalConflictFinding(
                            existing_record_id=ex_id,
                            existing_text=existing.text,
                            existing_modality=ex_mod.value,
                            candidate_modality=cand_mod.value,
                            kd_violation=kd_violation,
                            similarity=sim,
                            action_alignment=align,
                            resolution_status="CANDIDATE_OVERRIDES_EXISTING",
                            governing_canon="LEX_SPECIALIS",
                        )
                    )
                else:
                    shadowed_by.append(ex_id)
                    resolved.append(
                        ModalConflictFinding(
                            existing_record_id=ex_id,
                            existing_text=existing.text,
                            existing_modality=ex_mod.value,
                            candidate_modality=cand_mod.value,
                            kd_violation=kd_violation,
                            similarity=sim,
                            action_alignment=align,
                            resolution_status="SHADOWED_BY_EXISTING",
                            governing_canon="LEX_SPECIALIS",
                        )
                    )
                continue

            ex_ts = _extract_valid_timestamp(existing)
            if abs(cand_ts - ex_ts) > self.timestamp_tie_tolerance_sec:
                if cand_ts > ex_ts:
                    resolved.append(
                        ModalConflictFinding(
                            existing_record_id=ex_id,
                            existing_text=existing.text,
                            existing_modality=ex_mod.value,
                            candidate_modality=cand_mod.value,
                            kd_violation=kd_violation,
                            similarity=sim,
                            action_alignment=align,
                            resolution_status="CANDIDATE_OVERRIDES_EXISTING",
                            governing_canon="LEX_POSTERIOR",
                        )
                    )
                else:
                    shadowed_by.append(ex_id)
                    resolved.append(
                        ModalConflictFinding(
                            existing_record_id=ex_id,
                            existing_text=existing.text,
                            existing_modality=ex_mod.value,
                            candidate_modality=cand_mod.value,
                            kd_violation=kd_violation,
                            similarity=sim,
                            action_alignment=align,
                            resolution_status="SHADOWED_BY_EXISTING",
                            governing_canon="LEX_POSTERIOR",
                        )
                    )
                continue

            # Unresolved horizontal antinomy!
            unresolved.append(
                ModalConflictFinding(
                    existing_record_id=ex_id,
                    existing_text=existing.text,
                    existing_modality=ex_mod.value,
                    candidate_modality=cand_mod.value,
                    kd_violation=kd_violation,
                    similarity=sim,
                    action_alignment=align,
                    resolution_status="UNRESOLVED_HORIZONTAL_ANTINOMY",
                    governing_canon="NONE",
                )
            )
            recommendations.append(
                f"Resolve KD conflict ({kd_violation}) with '{ex_id}' by specifying "
                f"metadata={{'supersedes': '{ex_id}'}}, increasing authority_rank > {existing.authority_rank}, "
                f"or setting metadata['specificity'] > {ex_spec}."
            )

        is_satisfiable = (not would_cycle) and (len(unresolved) == 0)
        return PreFlightSimulationReport(
            is_satisfiable=is_satisfiable,
            candidate_modality=cand_mod.value,
            would_create_cycle=would_cycle,
            unresolved_conflicts=unresolved,
            resolved_overrides=resolved,
            shadowed_by=shadowed_by,
            recommendations=recommendations,
            tenant_id=tenant_id,
        )
