"""
Continuous Legal Memory (CLM).

Bi-temporal, deontic precedence-aware cognitive memory framework for legal and
regulated-enterprise AI agents.
"""

from continuous_legal_memory.domain.interfaces import BaseRerankerPort
from continuous_legal_memory.domain.models import (
    EntityType,
    GraphEdge,
    GraphNode,
    MemoryRecord,
    MemoryTier,
    PredictionResult,
    RelationType,
)
from continuous_legal_memory.evaluation.multijurisdictional import (
    evaluate_multijurisdictional_suite,
    generate_multijurisdictional_benchmark,
    to_deepeval_test_cases,
    to_ragas_records,
)
from continuous_legal_memory.mcp import MCPServer
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
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
from continuous_legal_memory.retrieval.choice_of_law import (
    SUPPORTED_JURISDICTION_REGIMES,
    ChoiceOfLawResolution,
    ChoiceOfLawRouter,
    ConnectingFactors,
)
from continuous_legal_memory.retrieval.conformal import (
    ConformalPredictionOutput,
    SplitConformalCalibrator,
)
from continuous_legal_memory.retrieval.hybrid_lexical import (
    HybridLexicalCitationScorer,
    extract_legal_citations,
)
from continuous_legal_memory.retrieval.precedence import (
    PrecedenceConfig,
    apply_legal_precedence,
)
from continuous_legal_memory.retrieval.reranker import (
    DeonticCrossEncoderReranker,
    HuggingFaceCrossEncoderReranker,
)
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
    MemoryAttestationToken,
)
from continuous_legal_memory.security.federated_sync import (
    CrossPartyAlignmentReport,
    FederatedComplianceVerifier,
    MerkleStateDiff,
)
from continuous_legal_memory.security.keyring import (
    CryptographicKeyRing,
    KeyMetadata,
)
from continuous_legal_memory.security.merkle import (
    AppendOnlyMerkleLog,
    SparseMerkleProof,
    SparseMerkleTree,
    verify_sparse_merkle_proof,
)
from continuous_legal_memory.security.pq_hybrid import (
    PQ_HYBRID_ALGORITHM,
    PostQuantumHybridAttestationModule,
    pq_hybrid_sign,
    pq_hybrid_verify,
)
from continuous_legal_memory.security.selective_disclosure import (
    SelectiveDisclosureBundle,
    SelectiveDisclosureComplianceProof,
    SelectiveDisclosureProver,
    compute_salted_commitment,
)
from continuous_legal_memory.security.standalone_verifier import (
    StandaloneAuditVerifierExporter,
    StandaloneVerifierArtifact,
)
from continuous_legal_memory.security.tee_attestation import (
    TEEAttestationProvider,
    TEEAttestationQuote,
    TEEPlatform,
    TEEQuoteVerificationReport,
)
from continuous_legal_memory.security.timestamp_gossip import (
    GossipCheckpoint,
    SplitViewAuditReport,
    TimestampToken,
    TransparencyGossipAuditor,
)
from continuous_legal_memory.storage.portable_bundle import (
    DistributedVectorGraphExporter,
    PortableBundleManager,
    SignedComplianceBundle,
)

__version__ = "1.2.0"
__author__ = "Vinicius Caridá"

__all__ = [
    "PQ_HYBRID_ALGORITHM",
    "SUPPORTED_JURISDICTION_REGIMES",
    "AmendmentImpactReport",
    "AppendOnlyMerkleLog",
    "AsymmetricAttestationModule",
    "BaseRerankerPort",
    "CandidateProofNode",
    "ChoiceOfLawResolution",
    "ChoiceOfLawRouter",
    "ClauseDiffSpan",
    "ConformalPredictionOutput",
    "ConnectingFactors",
    "CrossPartyAlignmentReport",
    "CryptographicKeyRing",
    "DefeasibleProofTrace",
    "DeonticAntinomy",
    "DeonticConsistencyChecker",
    "DeonticConsistencyReport",
    "DeonticCrossEncoderReranker",
    "DeonticModalVerifier",
    "DeonticModality",
    "DistributedVectorGraphExporter",
    "DownstreamObligationImpact",
    "DraftingCopilotReport",
    "DraftingRepairOption",
    "DraftingRepairStrategy",
    "EntityType",
    "ExtractedNormStructure",
    "FederatedComplianceVerifier",
    "GossipCheckpoint",
    "GraphEdge",
    "GraphNode",
    "HuggingFaceCrossEncoderReranker",
    "HybridLexicalCitationScorer",
    "KeyMetadata",
    "KeyedHashAttestationModule",
    "LegalMemoryOrchestrator",
    "MCPServer",
    "MemoryAttestationToken",
    "MemoryRecord",
    "MemoryTier",
    "MerkleStateDiff",
    "ModalConflictFinding",
    "NormativeDraftingCopilot",
    "NormativeGraphExtractor",
    "PortableBundleManager",
    "PostQuantumHybridAttestationModule",
    "PreFlightSimulationReport",
    "PrecedenceConfig",
    "PredictionResult",
    "RegulatoryTimelineReport",
    "RelationType",
    "SMTLibDeonticExporter",
    "SMTLibExportBundle",
    "SelectiveDisclosureBundle",
    "SelectiveDisclosureComplianceProof",
    "SelectiveDisclosureProver",
    "SignedComplianceBundle",
    "SparseMerkleProof",
    "SparseMerkleTree",
    "SplitConformalCalibrator",
    "SplitViewAuditReport",
    "StandaloneAuditVerifierExporter",
    "StandaloneVerifierArtifact",
    "StatutoryAmendmentAnalyzer",
    "TEEAttestationProvider",
    "TEEAttestationQuote",
    "TEEPlatform",
    "TEEQuoteVerificationReport",
    "TemporalHorizonSimulator",
    "TimelineEpochTransition",
    "TimestampToken",
    "TransparencyGossipAuditor",
    "__author__",
    "__version__",
    "apply_legal_precedence",
    "build_defeasible_proof_trace",
    "check_kd_modal_conflict",
    "compute_salted_commitment",
    "evaluate_multijurisdictional_suite",
    "extract_legal_citations",
    "generate_multijurisdictional_benchmark",
    "infer_deontic_modality",
    "pq_hybrid_sign",
    "pq_hybrid_verify",
    "to_deepeval_test_cases",
    "to_ragas_records",
    "verify_sparse_merkle_proof",
]
