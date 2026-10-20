"""
Bi-Temporal Regulatory Horizon Simulator ("What-If" Regulatory Timeline & Sunset Sweep).

Simulates how legal compliance outcomes, prevailing statutes, deontic modalities
(OBLIGATORY, FORBIDDEN, PERMITTED), and horizontal antinomies evolve across time
as phased regulatory mandates (e.g., EU AI Act 2025-2027 staggered entry-into-force)
take effect (`valid_from`) or expire via sunset clauses (`valid_to`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.domain.models import MemoryRecord, PredictionResult
from continuous_legal_memory.reasoning.deontic_checker import DeonticConsistencyChecker
from continuous_legal_memory.reasoning.modal_verifier import infer_deontic_modality


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class TimelineEpochTransition:
    """
    Snapshot of normative state at a specific temporal transition boundary `epoch_time`.
    """

    epoch_index: int
    epoch_time: str
    active_rule_ids: list[str]
    prevailing_record_id: str | None
    prevailing_modality: str | None
    winning_canon: str | None
    predicted_action_vector: list[float]
    unresolved_antinomy_count: int
    is_consistent: bool
    triggered_by_rule_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegulatoryTimelineReport:
    """
    Complete multi-epoch regulatory horizon forecast across `[start_time, end_time]`.
    """

    tenant_id: str
    query_text: str | None
    start_time: str
    end_time: str
    total_epochs: int
    modality_shifts_count: int
    future_antinomy_epochs_count: int
    epochs: list[TimelineEpochTransition] = field(default_factory=list)
    regime_changes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "query_text": self.query_text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_epochs": self.total_epochs,
            "modality_shifts_count": self.modality_shifts_count,
            "future_antinomy_epochs_count": self.future_antinomy_epochs_count,
            "epochs": [e.to_dict() for e in self.epochs],
            "regime_changes": list(self.regime_changes),
        }


class TemporalHorizonSimulator:
    """
    Sweeps across all normative phase-in (`valid_from`) and sunset (`valid_to`) boundaries
    within a target horizon to forecast regime changes and future antinomies.
    """

    def __init__(self, consistency_checker: DeonticConsistencyChecker | None = None) -> None:
        self.consistency_checker = consistency_checker or DeonticConsistencyChecker()

    def simulate_horizon(
        self,
        records: list[MemoryRecord],
        graph: SemanticKnowledgeGraph,
        query_text: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        tenant_id: str = "default",
        predict_fn: Callable[[str, datetime, str], PredictionResult] | None = None,
    ) -> RegulatoryTimelineReport:
        """
        Compute the normative timeline across all transition points in `[start_time, end_time]`.
        """
        tenant_records = [r for r in records if r.tenant_id == tenant_id]
        now = datetime.now(timezone.utc)

        # Collect candidate boundary timestamps from records
        raw_points: set[datetime] = set()
        for r in tenant_records:
            vf = _ensure_utc(r.valid_from)
            raw_points.add(vf)
            if r.valid_to is not None:
                vt = _ensure_utc(r.valid_to)
                raw_points.add(vt)
                # Also sample 1 second after valid_to to capture post-sunset state
                raw_points.add(vt + timedelta(seconds=1))

        if start_time is not None:
            eff_start = _ensure_utc(start_time)
        elif raw_points:
            eff_start = min(raw_points)
        else:
            eff_start = now

        if end_time is not None:
            eff_end = _ensure_utc(end_time)
        elif raw_points:
            eff_end = max(raw_points) + timedelta(days=1)
        else:
            eff_end = eff_start + timedelta(days=365)

        if eff_end < eff_start:
            eff_start, eff_end = eff_end, eff_start

        horizon_points: set[datetime] = {eff_start, eff_end}
        for pt in raw_points:
            if eff_start <= pt <= eff_end:
                horizon_points.add(pt)

        sorted_points = sorted(horizon_points)
        rec_by_id = {str(r.record_id): r for r in tenant_records if r.record_id is not None}

        epochs: list[TimelineEpochTransition] = []
        regime_changes: list[dict[str, Any]] = []
        modality_shifts = 0
        antinomy_epochs = 0

        prev_record_id: str | None = None
        prev_modality: str | None = None

        for idx, pt in enumerate(sorted_points):
            active_recs = [r for r in tenant_records if r.is_temporally_valid(at_time=pt)]
            active_ids = [str(r.record_id) for r in active_recs if r.record_id is not None]

            # Identify which rules triggered this boundary timestamp
            triggered: list[str] = []
            for r in tenant_records:
                rid = str(r.record_id or "")
                vf = _ensure_utc(r.valid_from)
                vt_opt = _ensure_utc(r.valid_to) if r.valid_to is not None else None
                if vf == pt or (
                    vt_opt is not None and (vt_opt == pt or vt_opt + timedelta(seconds=1) == pt)
                ):
                    triggered.append(rid)

            audit_report = self.consistency_checker.audit(
                records=tenant_records,
                graph=graph,
                tenant_id=tenant_id,
                at_time=pt,
                register_contradiction_edges=False,
            )
            if not audit_report.is_consistent:
                antinomy_epochs += 1

            prevailing_id: str | None = None
            prevailing_mod: str | None = None
            winning_canon: str | None = None
            action_vec: list[float] = []

            if query_text and query_text.strip() and predict_fn is not None:
                pred = predict_fn(query_text, pt, tenant_id)
                action_vec = list(pred.predicted_action_vector)
                if pred.proof_trace is not None:
                    prevailing_id = pred.proof_trace.get("prevailing_record_id")
                    winning_canon = pred.proof_trace.get("winning_canon")
                if prevailing_id and prevailing_id in rec_by_id:
                    prevailing_mod = infer_deontic_modality(rec_by_id[prevailing_id].text).value
            elif active_recs:
                # Deterministic Tri-Canon ranking across active records at `pt`
                ranked = sorted(
                    active_recs,
                    key=lambda r: (
                        int(r.authority_rank),
                        float(r.metadata.get("specificity", 1.0))
                        if isinstance(r.metadata, dict)
                        else 1.0,
                        _ensure_utc(r.valid_from).timestamp(),
                    ),
                    reverse=True,
                )
                top = ranked[0]
                prevailing_id = str(top.record_id)
                prevailing_mod = infer_deontic_modality(top.text).value
                action_vec = [float(x) for x in top.value_vector.detach().cpu().flatten().tolist()]
                if len(ranked) == 1:
                    winning_canon = "UNCONTESTED_PRIMARY_NORM"
                else:
                    runner = ranked[1]
                    if int(top.authority_rank) > int(runner.authority_rank):
                        winning_canon = "LEX_SUPERIOR"
                    else:
                        s_top = (
                            float(top.metadata.get("specificity", 1.0))
                            if isinstance(top.metadata, dict)
                            else 1.0
                        )
                        s_run = (
                            float(runner.metadata.get("specificity", 1.0))
                            if isinstance(runner.metadata, dict)
                            else 1.0
                        )
                        winning_canon = "LEX_SPECIALIS" if s_top > s_run else "LEX_POSTERIOR"

            epoch_entry = TimelineEpochTransition(
                epoch_index=idx,
                epoch_time=pt.isoformat(),
                active_rule_ids=active_ids,
                prevailing_record_id=prevailing_id,
                prevailing_modality=prevailing_mod,
                winning_canon=winning_canon,
                predicted_action_vector=action_vec,
                unresolved_antinomy_count=len(audit_report.antinomies),
                is_consistent=audit_report.is_consistent,
                triggered_by_rule_ids=triggered,
            )
            epochs.append(epoch_entry)

            if idx > 0 and (prevailing_id != prev_record_id or prevailing_mod != prev_modality):
                if prevailing_mod != prev_modality and prev_modality is not None and prevailing_mod is not None:
                    modality_shifts += 1
                regime_changes.append(
                    {
                        "epoch_index": idx,
                        "epoch_time": pt.isoformat(),
                        "from_record_id": prev_record_id,
                        "to_record_id": prevailing_id,
                        "from_modality": prev_modality,
                        "to_modality": prevailing_mod,
                        "winning_canon": winning_canon,
                        "triggered_by_rule_ids": triggered,
                    }
                )

            prev_record_id = prevailing_id
            prev_modality = prevailing_mod

        return RegulatoryTimelineReport(
            tenant_id=tenant_id,
            query_text=query_text,
            start_time=eff_start.isoformat(),
            end_time=eff_end.isoformat(),
            total_epochs=len(epochs),
            modality_shifts_count=modality_shifts,
            future_antinomy_epochs_count=antinomy_epochs,
            epochs=epochs,
            regime_changes=regime_changes,
        )
