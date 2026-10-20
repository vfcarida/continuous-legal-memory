"""
Formal SMT-LIB 2.6 Export Bridge for Standard Deontic Logic (KD) & Defeasible Precedence.

Translates a tenant's active normative rules and optional `DefeasibleProofTrace` into
a self-contained SMT-LIB 2.6 specification (`(set-logic QF_UFLIA)`) suitable for external
formal verification in automated theorem provers and SMT solvers such as Z3 and cvc5.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.reasoning.modal_verifier import (
    DeonticModality,
    infer_deontic_modality,
)
from continuous_legal_memory.retrieval.precedence import (
    _extract_specificity,
    _extract_valid_timestamp,
)


def _sanitize_smt_comment(text: str) -> str:
    """Strip newlines and semicolons to safely embed rule text in SMT-LIB `;` comments."""
    return text.replace("\r", " ").replace("\n", " ").replace(";", ",")[:120]


@dataclass(frozen=True)
class SMTLibExportBundle:
    """
    Container for a generated SMT-LIB 2.6 script and its cryptographic audit metadata.
    """

    tenant_id: str
    logic: str
    num_rules: int
    prevailing_record_id: str | None
    winning_canon: str | None
    smtlib_script: str
    script_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SMTLibDeonticExporter:
    """
    Compiles `MemoryRecord` ledgers and `DefeasibleProofTrace` derivations into
    standard SMT-LIB 2.6 scripts encoding Standard Deontic Logic (KD) and Tri-Canon precedence.
    """

    LOGIC = "QF_UFLIA"

    def export_tenant_theory(
        self,
        records: list[MemoryRecord],
        proof_trace: dict[str, Any] | None = None,
        tenant_id: str = "default",
    ) -> SMTLibExportBundle:
        """
        Generate a complete SMT-LIB 2.6 script asserting KD modal axioms (`~(O & F)`, `~(F & P)`),
        Kelsenian `authority_rank`, `specificity`, `valid_from` timestamps, and defeasible
        precedence relations over `records`.
        """
        lines: list[str] = [
            "; ==========================================================================",
            "; Continuous Legal Memory (CLM) — Formal SMT-LIB 2.6 Deontic Specification",
            f"; Tenant: {tenant_id} | Rules: {len(records)} | Logic: {self.LOGIC}",
            "; ==========================================================================",
            f"(set-logic {self.LOGIC})",
            "(set-option :produce-models true)",
            "",
            "; --- Uninterpreted Functions & Deontic Modal Predicates ---",
            "(declare-fun authority_rank (Int) Int)",
            "(declare-fun specificity_millis (Int) Int)",
            "(declare-fun valid_from_epoch (Int) Int)",
            "(declare-fun is_active (Int) Bool)",
            "(declare-fun modality_O (Int) Bool)",
            "(declare-fun modality_F (Int) Bool)",
            "(declare-fun modality_P (Int) Bool)",
            "(declare-fun defeats (Int Int) Bool)",
            "(declare-fun prevails (Int) Bool)",
            "",
        ]

        id_to_idx: dict[str, int] = {}
        for idx, rec in enumerate(records):
            rid = str(rec.record_id or f"rule_{idx}")
            id_to_idx[rid] = idx
            mod = infer_deontic_modality(rec.text, rec.value_vector, rec.metadata)
            spec_millis = round(_extract_specificity(rec) * 1000.0)
            epoch_sec = round(_extract_valid_timestamp(rec))

            is_o = "true" if mod == DeonticModality.OBLIGATION else "false"
            is_f = "true" if mod == DeonticModality.PROHIBITION else "false"
            is_p = "true" if mod == DeonticModality.PERMISSION else "false"

            lines.extend([
                f"; Rule [{idx}] ID={rid} : {_sanitize_smt_comment(rec.text)}",
                f"(assert (= (is_active {idx}) true))",
                f"(assert (= (authority_rank {idx}) {int(rec.authority_rank)}))",
                f"(assert (= (specificity_millis {idx}) {spec_millis}))",
                f"(assert (= (valid_from_epoch {idx}) {epoch_sec}))",
                f"(assert (= (modality_O {idx}) {is_o}))",
                f"(assert (= (modality_F {idx}) {is_f}))",
                f"(assert (= (modality_P {idx}) {is_p}))",
                "; KD Axiom D (Consistency of Obligations) & Modal Duality on Rule",
                f"(assert (=> (prevails {idx}) (not (and (modality_O {idx}) (modality_F {idx})))))",
                f"(assert (=> (prevails {idx}) (not (and (modality_F {idx}) (modality_P {idx})))))",
                "",
            ])

        # Encode pairwise Tri-Canon precedence assertions
        if len(records) > 1:
            lines.append("; --- Pairwise Tri-Canon Defeasible Precedence (Lex Superior / Specialis / Posterior) ---")
            for i in range(len(records)):
                for j in range(len(records)):
                    if i == j:
                        continue
                    lines.append(
                        f"(assert (=> (or "
                        f"(> (authority_rank {i}) (authority_rank {j})) "
                        f"(and (= (authority_rank {i}) (authority_rank {j})) (> (specificity_millis {i}) (specificity_millis {j}))) "
                        f"(and (= (authority_rank {i}) (authority_rank {j})) (= (specificity_millis {i}) (specificity_millis {j})) (> (valid_from_epoch {i}) (valid_from_epoch {j})))"
                        f") (defeats {i} {j})))"
                    )
            lines.append("")

        prevailing_id: str | None = None
        winning_canon: str | None = None
        if proof_trace is not None:
            prevailing_id = proof_trace.get("prevailing_record_id")
            winning_canon = proof_trace.get("winning_canon")
            if prevailing_id and prevailing_id in id_to_idx:
                win_idx = id_to_idx[prevailing_id]
                lines.append(
                    f"; --- Defeasible Proof Trace Assertion (Prevailing={prevailing_id}, Canon={winning_canon}) ---"
                )
                lines.append(f"(assert (= (prevails {win_idx}) true))")
                for _other_rid, other_idx in id_to_idx.items():
                    if other_idx != win_idx:
                        lines.append(f"(assert (= (prevails {other_idx}) false))")
                lines.append("")

        lines.append("(check-sat)")
        if records:
            prevails_terms = " ".join(f"(prevails {i})" for i in range(len(records)))
            lines.append(f"(get-value ({prevails_terms}))")
        lines.append("")

        script = "\n".join(lines)
        script_sha256 = hashlib.sha256(script.encode("utf-8")).hexdigest()
        return SMTLibExportBundle(
            tenant_id=tenant_id,
            logic=self.LOGIC,
            num_rules=len(records),
            prevailing_record_id=prevailing_id,
            winning_canon=winning_canon,
            smtlib_script=script,
            script_sha256=script_sha256,
        )
