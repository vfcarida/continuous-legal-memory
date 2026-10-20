"""
Formal Legislative & Contract Normative Drafting Co-Pilot (`NormativeDraftingCopilot`).

Synthesizes mathematically verified normative repair proposals when a candidate rule
triggers a Standard Deontic Logic (KD) conflict (`O(phi) & F(phi)`, `F(phi) & P(phi)`),
is shadowed on arrival by an existing statute (*lex superior* / *lex specialis* / *lex posterior*),
or introduces a cycle into the `SemanticKnowledgeGraph` DAG.

Every synthesized repair proposal (`DraftingRepairOption`) is automatically re-verified
through `DeonticModalVerifier.simulate_candidate` to guarantee post-repair KD satisfiability
and zero unintended shadowing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.reasoning.modal_verifier import (
    DeonticModalVerifier,
)
from continuous_legal_memory.retrieval.precedence import (
    _extract_specificity,
    _extract_valid_timestamp,
)


class DraftingRepairStrategy(str, Enum):
    """Formal legislative/contractual repair strategies for resolving deontic conflicts."""

    EXPLICIT_SUPERSESSION = "EXPLICIT_SUPERSESSION"
    LEX_SPECIALIS_CARVEOUT = "LEX_SPECIALIS_CARVEOUT"
    TEMPORAL_PARTITION = "TEMPORAL_PARTITION"
    DAG_CYCLE_PRUNING = "DAG_CYCLE_PRUNING"


@dataclass
class DraftingRepairOption:
    """
    A concrete, pre-verified normative repair proposal for a conflicting or shadowed rule.
    """

    strategy: str
    description: str
    repaired_text: str
    repaired_authority_rank: int
    repaired_valid_from: str | None
    repaired_valid_to: str | None
    repaired_metadata: dict[str, Any]
    post_repair_satisfiable: bool
    post_repair_shadowed_by: list[str] = field(default_factory=list)
    target_conflict_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DraftingCopilotReport:
    """
    Complete analysis and verified repair synthesis produced by `NormativeDraftingCopilot`.
    """

    original_simulation: dict[str, Any]
    needs_repair: bool
    repair_options: list[DraftingRepairOption] = field(default_factory=list)
    recommended_option: dict[str, Any] | None = None
    tenant_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_simulation": self.original_simulation,
            "needs_repair": self.needs_repair,
            "repair_options": [opt.to_dict() for opt in self.repair_options],
            "recommended_option": self.recommended_option,
            "tenant_id": self.tenant_id,
        }


class NormativeDraftingCopilot:
    """
    Synthesizes and verifies formal legislative and contractual drafting repairs
    over `DeonticModalVerifier`.
    """

    def __init__(self, modal_verifier: DeonticModalVerifier | None = None) -> None:
        self.verifier = modal_verifier or DeonticModalVerifier()

    def propose_repairs(
        self,
        candidate: MemoryRecord,
        existing_records: list[MemoryRecord],
        graph: SemanticKnowledgeGraph | None = None,
    ) -> DraftingCopilotReport:
        """
        Evaluate `candidate` via `DeonticModalVerifier` and synthesize verified
        `DraftingRepairOption` proposals if any KD conflict, normative shadowing,
        or DAG cycle is detected.
        """
        initial_sim = self.verifier.simulate_candidate(
            candidate=candidate,
            existing_records=existing_records,
            graph=graph,
        )
        needs_repair = (
            (not initial_sim.is_satisfiable)
            or (len(initial_sim.shadowed_by) > 0)
            or initial_sim.would_create_cycle
        )

        if not needs_repair:
            return DraftingCopilotReport(
                original_simulation=initial_sim.to_dict(),
                needs_repair=False,
                repair_options=[],
                recommended_option=None,
                tenant_id=candidate.tenant_id,
            )

        # Map existing records by record_id for conflict inspection
        by_id: dict[str, MemoryRecord] = {
            (r.record_id or ""): r for r in existing_records if r.record_id
        }

        conflict_ids: list[str] = []
        for c in initial_sim.unresolved_conflicts:
            if c.existing_record_id not in conflict_ids:
                conflict_ids.append(c.existing_record_id)
        for sid in initial_sim.shadowed_by:
            if sid not in conflict_ids:
                conflict_ids.append(sid)

        conflicting_records = [by_id[cid] for cid in conflict_ids if cid in by_id]
        base_meta = dict(candidate.metadata or {})

        # If a DAG cycle was detected, strip cyclic relations first in base_meta
        options: list[DraftingRepairOption] = []
        if initial_sim.would_create_cycle:
            pruned_meta = dict(base_meta)
            pruned_meta.pop("relations", None)
            pruned_meta.pop("supersedes", None)
            pruned_record = self._clone_candidate(
                candidate=candidate,
                text=candidate.text,
                authority_rank=candidate.authority_rank,
                valid_from=candidate.valid_from,
                valid_to=candidate.valid_to,
                metadata=pruned_meta,
            )
            pruned_sim = self.verifier.simulate_candidate(
                candidate=pruned_record,
                existing_records=existing_records,
                graph=graph,
            )
            options.append(
                DraftingRepairOption(
                    strategy=DraftingRepairStrategy.DAG_CYCLE_PRUNING.value,
                    description=(
                        "Remove cyclic graph relations from metadata to preserve strict DAG "
                        "hierarchy in SemanticKnowledgeGraph."
                    ),
                    repaired_text=pruned_record.text,
                    repaired_authority_rank=pruned_record.authority_rank,
                    repaired_valid_from=(
                        pruned_record.valid_from.isoformat() if pruned_record.valid_from else None
                    ),
                    repaired_valid_to=(
                        pruned_record.valid_to.isoformat() if pruned_record.valid_to else None
                    ),
                    repaired_metadata=pruned_meta,
                    post_repair_satisfiable=pruned_sim.is_satisfiable,
                    post_repair_shadowed_by=list(pruned_sim.shadowed_by),
                    target_conflict_ids=list(conflict_ids),
                )
            )
            base_meta = pruned_meta

        if conflicting_records:
            max_auth = max(int(r.authority_rank) for r in conflicting_records)
            max_spec = max(_extract_specificity(r) for r in conflicting_records)
            max_ts = max(_extract_valid_timestamp(r) for r in conflicting_records)
            ids_str = ", ".join(conflict_ids)

            # Strategy 1: Explicit Supersession Clause (SUPERSEDES)
            sup_meta = dict(base_meta)
            existing_sup = sup_meta.get("supersedes", [])
            sup_list = (
                [existing_sup]
                if isinstance(existing_sup, str)
                else list(existing_sup)
                if isinstance(existing_sup, (list, tuple, set))
                else []
            )
            for cid in conflict_ids:
                if cid not in sup_list:
                    sup_list.append(cid)
            sup_meta["supersedes"] = sup_list
            sup_auth = max(int(candidate.authority_rank), max_auth)
            sup_text = f"Repeals and supersedes {ids_str}. {candidate.text}"

            sup_record = self._clone_candidate(
                candidate=candidate,
                text=sup_text,
                authority_rank=sup_auth,
                valid_from=candidate.valid_from,
                valid_to=candidate.valid_to,
                metadata=sup_meta,
            )
            sup_sim = self.verifier.simulate_candidate(
                candidate=sup_record,
                existing_records=existing_records,
                graph=graph,
            )
            options.append(
                DraftingRepairOption(
                    strategy=DraftingRepairStrategy.EXPLICIT_SUPERSESSION.value,
                    description=(
                        f"Add explicit repeal/supersession clause over [{ids_str}] with "
                        f"authority_rank={sup_auth} and SUPERSEDES graph binding."
                    ),
                    repaired_text=sup_text,
                    repaired_authority_rank=sup_auth,
                    repaired_valid_from=(
                        sup_record.valid_from.isoformat() if sup_record.valid_from else None
                    ),
                    repaired_valid_to=(
                        sup_record.valid_to.isoformat() if sup_record.valid_to else None
                    ),
                    repaired_metadata=sup_meta,
                    post_repair_satisfiable=sup_sim.is_satisfiable,
                    post_repair_shadowed_by=list(sup_sim.shadowed_by),
                    target_conflict_ids=list(conflict_ids),
                )
            )

            # Strategy 2: Lex Specialis Exception Carve-Out
            spec_meta = dict(base_meta)
            new_spec = round(max( _extract_specificity(candidate), max_spec) + 1.0, 2)
            spec_meta["specificity"] = new_spec
            spec_auth = max(int(candidate.authority_rank), max_auth)
            spec_text = (
                f"Notwithstanding {ids_str}, in the specific context of this provision: "
                f"{candidate.text}"
            )
            spec_record = self._clone_candidate(
                candidate=candidate,
                text=spec_text,
                authority_rank=spec_auth,
                valid_from=candidate.valid_from,
                valid_to=candidate.valid_to,
                metadata=spec_meta,
            )
            spec_sim = self.verifier.simulate_candidate(
                candidate=spec_record,
                existing_records=existing_records,
                graph=graph,
            )
            options.append(
                DraftingRepairOption(
                    strategy=DraftingRepairStrategy.LEX_SPECIALIS_CARVEOUT.value,
                    description=(
                        f"Draft as a lex specialis exception carve-out (specificity={new_spec}, "
                        f"authority_rank={spec_auth}) coexisting with general norm(s) [{ids_str}]."
                    ),
                    repaired_text=spec_text,
                    repaired_authority_rank=spec_auth,
                    repaired_valid_from=(
                        spec_record.valid_from.isoformat() if spec_record.valid_from else None
                    ),
                    repaired_valid_to=(
                        spec_record.valid_to.isoformat() if spec_record.valid_to else None
                    ),
                    repaired_metadata=spec_meta,
                    post_repair_satisfiable=spec_sim.is_satisfiable,
                    post_repair_shadowed_by=list(spec_sim.shadowed_by),
                    target_conflict_ids=list(conflict_ids),
                )
            )

            # Strategy 3: Temporal Partitioning / Lex Posterior Effective Date
            temp_meta = dict(base_meta)
            finite_valid_tos = [r.valid_to for r in conflicting_records if r.valid_to is not None]
            if len(finite_valid_tos) == len(conflicting_records):
                latest_end = max(finite_valid_tos)
                if latest_end.tzinfo is None:
                    latest_end = latest_end.replace(tzinfo=timezone.utc)
                new_valid_from = latest_end + timedelta(seconds=1)
                temp_desc = (
                    f"Partition effective date (valid_from={new_valid_from.isoformat()}) "
                    f"to commence immediately after the sunset of [{ids_str}]."
                )
                temp_auth = int(candidate.authority_rank)
            else:
                base_dt = datetime.fromtimestamp(max(max_ts, 0.0), tz=timezone.utc)
                new_valid_from = base_dt + timedelta(days=1)
                temp_auth = max(int(candidate.authority_rank), max_auth)
                temp_desc = (
                    f"Advance effective date (valid_from={new_valid_from.isoformat()}, "
                    f"authority_rank={temp_auth}) so lex posterior deterministically governs over [{ids_str}]."
                )

            temp_record = self._clone_candidate(
                candidate=candidate,
                text=candidate.text,
                authority_rank=temp_auth,
                valid_from=new_valid_from,
                valid_to=candidate.valid_to,
                metadata=temp_meta,
            )
            temp_sim = self.verifier.simulate_candidate(
                candidate=temp_record,
                existing_records=existing_records,
                graph=graph,
            )
            options.append(
                DraftingRepairOption(
                    strategy=DraftingRepairStrategy.TEMPORAL_PARTITION.value,
                    description=temp_desc,
                    repaired_text=candidate.text,
                    repaired_authority_rank=temp_auth,
                    repaired_valid_from=new_valid_from.isoformat(),
                    repaired_valid_to=(
                        temp_record.valid_to.isoformat() if temp_record.valid_to else None
                    ),
                    repaired_metadata=temp_meta,
                    post_repair_satisfiable=temp_sim.is_satisfiable,
                    post_repair_shadowed_by=list(temp_sim.shadowed_by),
                    target_conflict_ids=list(conflict_ids),
                )
            )

        recommended: dict[str, Any] | None = None
        for opt in options:
            if opt.post_repair_satisfiable and len(opt.post_repair_shadowed_by) == 0:
                recommended = opt.to_dict()
                break
        if recommended is None and options:
            recommended = options[0].to_dict()

        return DraftingCopilotReport(
            original_simulation=initial_sim.to_dict(),
            needs_repair=True,
            repair_options=options,
            recommended_option=recommended,
            tenant_id=candidate.tenant_id,
        )

    @staticmethod
    def _clone_candidate(
        candidate: MemoryRecord,
        text: str,
        authority_rank: int,
        valid_from: datetime | None,
        valid_to: datetime | None,
        metadata: dict[str, Any],
    ) -> MemoryRecord:
        return MemoryRecord(
            text=text,
            key_vector=candidate.key_vector,
            value_vector=candidate.value_vector,
            authority_rank=authority_rank,
            valid_from=valid_from or candidate.valid_from,
            valid_to=valid_to,
            transaction_time=candidate.transaction_time,
            jurisdiction=candidate.jurisdiction,
            metadata=metadata,
            record_id=candidate.record_id,
            tenant_id=candidate.tenant_id,
        )
