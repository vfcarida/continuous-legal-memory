"""
Formal Deontic Reasoning and Consistency Verification Subsystem.
"""

from continuous_legal_memory.reasoning.amendment_impact import (
    AmendmentImpactReport,
    ClauseDiffSpan,
    DownstreamObligationImpact,
    StatutoryAmendmentAnalyzer,
)
from continuous_legal_memory.reasoning.deontic_checker import (
    DeonticAntinomy,
    DeonticConsistencyChecker,
    DeonticConsistencyReport,
)
from continuous_legal_memory.reasoning.drafting_copilot import (
    DraftingCopilotReport,
    DraftingRepairOption,
    DraftingRepairStrategy,
    NormativeDraftingCopilot,
)
from continuous_legal_memory.reasoning.modal_verifier import (
    DeonticModality,
    DeonticModalVerifier,
    ModalConflictFinding,
    PreFlightSimulationReport,
    check_kd_modal_conflict,
    infer_deontic_modality,
)
from continuous_legal_memory.reasoning.norm_extractor import (
    ExtractedNormStructure,
    NormativeGraphExtractor,
)
from continuous_legal_memory.reasoning.proof_trace import (
    CandidateProofNode,
    DefeasibleProofTrace,
    build_defeasible_proof_trace,
)
from continuous_legal_memory.reasoning.smtlib_exporter import (
    SMTLibDeonticExporter,
    SMTLibExportBundle,
)
from continuous_legal_memory.reasoning.temporal_simulator import (
    RegulatoryTimelineReport,
    TemporalHorizonSimulator,
    TimelineEpochTransition,
)

__all__ = [
    "AmendmentImpactReport",
    "CandidateProofNode",
    "ClauseDiffSpan",
    "DefeasibleProofTrace",
    "DeonticAntinomy",
    "DeonticConsistencyChecker",
    "DeonticConsistencyReport",
    "DeonticModalVerifier",
    "DeonticModality",
    "DownstreamObligationImpact",
    "DraftingCopilotReport",
    "DraftingRepairOption",
    "DraftingRepairStrategy",
    "ExtractedNormStructure",
    "ModalConflictFinding",
    "NormativeDraftingCopilot",
    "NormativeGraphExtractor",
    "PreFlightSimulationReport",
    "RegulatoryTimelineReport",
    "SMTLibDeonticExporter",
    "SMTLibExportBundle",
    "StatutoryAmendmentAnalyzer",
    "TemporalHorizonSimulator",
    "TimelineEpochTransition",
    "build_defeasible_proof_trace",
    "check_kd_modal_conflict",
    "infer_deontic_modality",
]
