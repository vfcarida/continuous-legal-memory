"""
Cross-Tenant & Multi-Party Federated Normative Verification (`FederatedComplianceVerifier`).

Implements zero-knowledge multi-party compliance alignment verification and 256-bit Sparse Merkle
state reconciliation across sovereign regulatory nodes or cross-border data controllers/processors
(GDPR Chapter V, LGPD Art. 33, EU AI Act Art. 12 & 78):

1. `MerkleStateDiff`:
   Computes exact leaf-level cryptographic differences (`added_rule_ids`, `removed_rule_ids`,
   `modified_rule_ids`, `unchanged_rule_ids`) between two `SparseMerkleTree` instances or
   `SignedComplianceBundle` snapshots while verifying bundle signatures and Merkle roots.
2. `CrossPartyAlignmentReport` & `FederatedComplianceVerifier.verify_cross_party_alignment()`:
   Verifies a set of `SelectiveDisclosureComplianceProof` certificates from multiple independent
   parties without accessing plaintext contract/statute text, proving that:
   - Every party's cryptographic proof signature and 256-bit Sparse Merkle inclusion path are valid.
   - Every party satisfies `authority_rank >= min_authority_rank` and `temporally_valid == True`.
   - No party abstained (`abstained == False`).
   - Pairwise cosine alignment between all parties' `predicted_action_vector` claims meets
     `min_action_alignment` (ensuring no cross-party deontic outcome contradiction).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
)
from continuous_legal_memory.security.merkle import SparseMerkleTree
from continuous_legal_memory.security.pq_hybrid import PostQuantumHybridAttestationModule
from continuous_legal_memory.security.selective_disclosure import (
    SelectiveDisclosureComplianceProof,
    SelectiveDisclosureProver,
)
from continuous_legal_memory.storage.portable_bundle import (
    PortableBundleManager,
    SignedComplianceBundle,
)


@dataclass(frozen=True)
class MerkleStateDiff:
    """
    Exact cryptographic leaf-level state reconciliation report between two Sparse Merkle states.
    """

    root_a: str
    root_b: str
    roots_equal: bool
    added_rule_ids: list[str]
    removed_rule_ids: list[str]
    modified_rule_ids: list[str]
    unchanged_rule_ids: list[str]
    bundle_a_valid: bool = True
    bundle_b_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossPartyAlignmentReport:
    """
    Multi-party zero-knowledge normative alignment verification report.
    """

    is_aligned: bool
    parties_count: int
    all_proofs_valid: bool
    all_temporally_valid: bool
    all_authority_satisfied: bool
    min_pairwise_action_alignment: float
    conflicting_pairs: list[dict[str, Any]] = field(default_factory=list)
    party_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return float(dot / (norm_a * norm_b))


class FederatedComplianceVerifier:
    """
    Verifies cross-party selective-disclosure proofs and reconciles Sparse Merkle Tree states.
    """

    def __init__(
        self,
        attestation_module: (
            AsymmetricAttestationModule
            | KeyedHashAttestationModule
            | PostQuantumHybridAttestationModule
            | None
        ) = None,
    ) -> None:
        self.attestation_module = (
            attestation_module if attestation_module is not None else AsymmetricAttestationModule()
        )
        self._prover = SelectiveDisclosureProver(attestation_module=self.attestation_module)
        self._bundle_mgr = PortableBundleManager(attestation_module=self.attestation_module)

    def _extract_leaves_and_root(
        self,
        state: SparseMerkleTree | SignedComplianceBundle | dict[str, Any],
    ) -> tuple[str, dict[str, str], bool]:
        if isinstance(state, SparseMerkleTree):
            leaves = {rid: lbytes.hex() for rid, lbytes in state._active_rules.items()}
            return state.root_hex, leaves, True

        bundle = SignedComplianceBundle.from_dict(state) if isinstance(state, dict) else state
        is_valid = self._bundle_mgr.verify_bundle(bundle)
        return bundle.merkle_root, dict(bundle.merkle_leaves), is_valid

    def diff_merkle_states(
        self,
        state_a: SparseMerkleTree | SignedComplianceBundle | dict[str, Any],
        state_b: SparseMerkleTree | SignedComplianceBundle | dict[str, Any],
    ) -> MerkleStateDiff:
        """
        Compute the exact cryptographic leaf-level diff between `state_a` (baseline)
        and `state_b` (target).
        """
        root_a, leaves_a, valid_a = self._extract_leaves_and_root(state_a)
        root_b, leaves_b, valid_b = self._extract_leaves_and_root(state_b)

        keys_a = set(leaves_a.keys())
        keys_b = set(leaves_b.keys())

        added = sorted(keys_b - keys_a)
        removed = sorted(keys_a - keys_b)
        common = sorted(keys_a & keys_b)

        modified: list[str] = []
        unchanged: list[str] = []
        for k in common:
            if leaves_a[k] != leaves_b[k]:
                modified.append(k)
            else:
                unchanged.append(k)

        return MerkleStateDiff(
            root_a=root_a,
            root_b=root_b,
            roots_equal=(root_a == root_b),
            added_rule_ids=added,
            removed_rule_ids=removed,
            modified_rule_ids=modified,
            unchanged_rule_ids=unchanged,
            bundle_a_valid=valid_a,
            bundle_b_valid=valid_b,
        )

    def verify_cross_party_alignment(
        self,
        proofs: list[SelectiveDisclosureComplianceProof | dict[str, Any]],
        min_authority_rank: int = 1,
        min_action_alignment: float = 0.0,
        require_temporal_validity: bool = True,
    ) -> CrossPartyAlignmentReport:
        """
        Verify that multiple parties' `SelectiveDisclosureComplianceProof` certificates
        are cryptographically valid, satisfy minimum authority and temporal constraints,
        and exhibit mutually aligned normative action vectors (`cosine >= min_action_alignment`).
        """
        if not proofs:
            return CrossPartyAlignmentReport(
                is_aligned=False,
                parties_count=0,
                all_proofs_valid=False,
                all_temporally_valid=False,
                all_authority_satisfied=False,
                min_pairwise_action_alignment=0.0,
            )

        parsed_proofs = [
            SelectiveDisclosureComplianceProof.from_dict(p) if isinstance(p, dict) else p
            for p in proofs
        ]

        all_valid = True
        all_temp = True
        all_auth = True
        party_summaries: list[dict[str, Any]] = []
        vectors: list[list[float]] = []

        for p in parsed_proofs:
            p_valid = self._prover.verify_proof(
                p,
                min_authority_rank=min_authority_rank,
                require_temporal_validity=require_temporal_validity,
            )
            claims = p.disclosed_claims
            temp_ok = bool(claims.get("temporally_valid", False))
            auth_rank = int(claims.get("authority_rank", 0))
            auth_ok = auth_rank >= min_authority_rank
            abstained = bool(claims.get("abstained", False))
            vec = [float(x) for x in claims.get("predicted_action_vector", [])]
            vectors.append(vec)

            if not p_valid or abstained:
                all_valid = False
            if not temp_ok:
                all_temp = False
            if not auth_ok:
                all_auth = False

            party_summaries.append(
                {
                    "proof_id": p.proof_id,
                    "tenant_id": p.tenant_id,
                    "record_id": p.record_id,
                    "jurisdiction": claims.get("jurisdiction"),
                    "authority_rank": auth_rank,
                    "temporally_valid": temp_ok,
                    "proof_verified": p_valid,
                    "merkle_root": p.merkle_root,
                }
            )

        conflicting_pairs: list[dict[str, Any]] = []
        min_align = 1.0 if len(parsed_proofs) == 1 else 1.0

        for i in range(len(parsed_proofs)):
            for j in range(i + 1, len(parsed_proofs)):
                sim = _cosine_similarity(vectors[i], vectors[j])
                if sim < min_align:
                    min_align = sim
                if sim < min_action_alignment:
                    conflicting_pairs.append(
                        {
                            "party_a_proof_id": parsed_proofs[i].proof_id,
                            "party_a_tenant_id": parsed_proofs[i].tenant_id,
                            "party_b_proof_id": parsed_proofs[j].proof_id,
                            "party_b_tenant_id": parsed_proofs[j].tenant_id,
                            "action_alignment": round(sim, 6),
                        }
                    )

        is_aligned = (
            all_valid
            and (all_temp if require_temporal_validity else True)
            and all_auth
            and len(conflicting_pairs) == 0
        )

        return CrossPartyAlignmentReport(
            is_aligned=is_aligned,
            parties_count=len(parsed_proofs),
            all_proofs_valid=all_valid,
            all_temporally_valid=all_temp,
            all_authority_satisfied=all_auth,
            min_pairwise_action_alignment=round(min_align, 6),
            conflicting_pairs=conflicting_pairs,
            party_summaries=party_summaries,
        )
