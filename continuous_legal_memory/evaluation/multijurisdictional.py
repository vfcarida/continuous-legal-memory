"""
Multi-Jurisdictional Statutory Graph Benchmark (540 Cases) & RAGAS / DeepEval Adapters.

Provides a deterministic, seeded 540-case evaluation suite spanning three regulatory
jurisdictions (`EU`, `US`, `BR`) and six normative reasoning categories:
1. `lex_superior` (Authority hierarchy preemption)
2. `lex_specialis` (Specificity preemption within equal authority tiers)
3. `lex_posterior` (Chronological amendment supersession in pari materia)
4. `temporal_expiry` (Bi-temporal validity window expiration)
5. `gdpr_erasure` (Right-to-Erasure physical slot removal + Sparse Merkle non-membership verification)
6. `epistemic_abstention` (Out-of-domain query abstention via similarity and conformal gates)

Also provides zero-dependency export and scoring adapters compatible with **RAGAS**
(`SingleTurnSample` schema) and **DeepEval** (`LLMTestCase` schema).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.evaluation.harness import (
    DeterministicTokenEncoder,
    PlainRAGBaseline,
)
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.retrieval.conformal import SplitConformalCalibrator
from continuous_legal_memory.retrieval.reranker import DeonticCrossEncoderReranker
from continuous_legal_memory.security.merkle import verify_sparse_merkle_proof

JURISDICTION_REGIMES: dict[str, dict[str, str]] = {
    "EU": {
        "supreme_statute": "EU Regulation 2016/679 (GDPR) and EU AI Act Regulation 2024/1689",
        "sectoral_statute": "EU DORA Regulation 2022/2554 Financial ICT Risk Framework",
        "general_statute": "EU General Commercial Record-Keeping Directive",
    },
    "US": {
        "supreme_statute": "US Federal Statute 45 CFR Part 164 (HIPAA) and 17 CFR 240.17a-4 (SEC)",
        "sectoral_statute": "US GLBA 16 CFR Part 314 Financial Safeguards Rule",
        "general_statute": "US Uniform State Corporate Filing Guidelines",
    },
    "BR": {
        "supreme_statute": "Brazil Federal Law No. 13.709/2018 (LGPD) and Law No. 12.965/2014 (Marco Civil)",
        "sectoral_statute": "Brazil Central Bank BACEN Resolution CMN No. 4.893/2021 Cybersecurity Rule",
        "general_statute": "Brazil General Commercial Archiving Normative Instruction",
    },
}

REGULATORY_DOMAINS: list[tuple[str, str]] = [
    ("biometric_authentication", "biometric facial recognition authentication templates"),
    ("wire_transfer_audit", "cross-border wire transfer anti-money-laundering ledger logs"),
    ("clinical_telemetry", "patient wearable cardiac telemetry diagnostic records"),
    ("algorithmic_credit_scoring", "automated creditworthiness scoring model training features"),
    ("cloud_key_escrow", "hardware security module cryptographic key rotation logs"),
    ("employee_payroll_tax", "withholding payroll tax social security declarations"),
    ("autonomous_vehicle_lidar", "autonomous fleet cabin video and sensor telemetry"),
    ("securities_order_book", "broker-dealer high-frequency institutional order book audit trails"),
    ("pediatric_genomic_data", "pediatric whole-genome sequencing clinical consent records"),
    ("smart_grid_metering", "residential smart meter high-frequency power consumption profiles"),
    ("insurance_actuarial_claims", "health insurance pre-existing condition actuarial claim files"),
    ("CrossBorder_Swift_Settlement", "interbank clearing house settlement reconciliation messages"),
    ("pharmaceutical_adverse_events", "post-market pharmacovigilance adverse drug reaction reports"),
    ("biobank_tissue_samples", "cryogenic biobank donor genetic lineage metadata"),
    ("telecommunications_cdr", "mobile subscriber call detail record location tower logs"),
    ("aviation_flight_recorder", "commercial cockpit voice and flight data recorder telemetry"),
    ("critical_infrastructure_scada", "nuclear and hydro utility SCADA intrusion detection logs"),
    ("digital_identity_kyc", "remote video onboarding sovereign identity verification dossiers"),
    ("sovereign_debt_custody", "central securities depository treasury bond ownership ledgers"),
    ("maritime_cargo_manifest", "customs hazardous material bill of lading declarations"),
    ("pension_fiduciary_ledger", "occupational pension fund beneficiary contribution schedules"),
    ("clinical_trial_randomization", "double-blind phase III clinical trial unblinding keys"),
    ("derivatives_margin_calls", "central counterparty clearing house intraday margin logs"),
    ("sanctions_screening_hits", "OFAC and UN consolidated sanctions watchlist match alerts"),
    ("electronic_health_prescriptions", "controlled substance electronic prescription dispensation logs"),
    ("consumer_behavioral_adtech", "real-time bidding cross-site browser tracking identifiers"),
    ("voice_biometric_callcenter", "customer support acoustic voiceprint authentication embeddings"),
    ("corporate_whistleblower_vault", "encrypted internal ethics hotline retaliation investigation files"),
    ("supply_chain_cobalt_trace", "conflict mineral smelter due diligence provenance certificates"),
    ("Automated_Hiring_Psychometrics", "candidate video interview psychometric assessment scores"),
]

OUT_OF_DOMAIN_QUERIES: list[str] = [
    "Calculate orbital eccentricity parameters around Jupiter moon Europa.",
    "Bake artisan sourdough ciabatta bread using fermented wheat starter.",
    "Estimate Schwarzschild horizon diameter around supermassive galactic pulsar.",
    "Analyze baroque impasto brushwork pigments inside Dutch canvas portraits.",
    "Cultivate highland Arabica espresso beans across volcanic soil terraces.",
    "Transpose chromatic saxophone bebop chord progressions across twelve keys.",
]


@dataclass
class MultiJurisdictionalCase:
    """Single self-contained normative evaluation scenario with rules and ground-truth query."""

    case_id: str
    jurisdiction: str
    category: str
    domain_slug: str
    rules: list[dict[str, Any]]
    query_text: str
    eval_time: str
    expected_action: list[float]
    expected_rule_id: str | None
    expect_abstention: bool = False
    erased_rule_id: str | None = None


def generate_multijurisdictional_benchmark(
    domains_per_category: int = 30,
) -> list[MultiJurisdictionalCase]:
    """
    Generate the deterministic multi-jurisdictional benchmark suite.

    With `domains_per_category=30` (default), generates exact:
        3 jurisdictions (EU, US, BR) * 6 categories * 30 domains = 540 cases.
    """
    selected_domains = REGULATORY_DOMAINS[: max(1, min(domains_per_category, len(REGULATORY_DOMAINS)))]
    cases: list[MultiJurisdictionalCase] = []
    categories = [
        "lex_superior",
        "lex_specialis",
        "lex_posterior",
        "temporal_expiry",
        "gdpr_erasure",
        "epistemic_abstention",
    ]

    for juris, statutes in JURISDICTION_REGIMES.items():
        for cat in categories:
            for idx, (slug, phrase) in enumerate(selected_domains):
                cid = f"{juris}_{cat}_{idx:02d}"
                r_base = f"{cid}_r1"
                r_comp = f"{cid}_r2"
                eval_ts = "2026-06-15T12:00:00Z"

                if cat == "lex_superior":
                    rules = [
                        {
                            "rule_id": r_base,
                            "text": (
                                f"[{juris}] Internal corporate guideline on {phrase}: "
                                f"informal local deletion permitted without audit."
                            ),
                            "action": [1.0, 0.0],
                            "authority_rank": 1,
                            "valid_from": "2024-01-01T00:00:00Z",
                            "jurisdiction": juris,
                            "metadata": {"specificity": 1.0},
                        },
                        {
                            "rule_id": r_comp,
                            "text": (
                                f"[{juris}] {statutes['supreme_statute']} mandatory requirement on {phrase}: "
                                f"strictly retain encrypted compliance records."
                            ),
                            "action": [0.0, 1.0],
                            "authority_rank": 5,
                            "valid_from": "2022-01-01T00:00:00Z",
                            "jurisdiction": juris,
                            "metadata": {"specificity": 1.0},
                        },
                    ]
                    cases.append(
                        MultiJurisdictionalCase(
                            case_id=cid,
                            jurisdiction=juris,
                            category=cat,
                            domain_slug=slug,
                            rules=rules,
                            query_text=f"Under {juris} law, what is the compliance requirement for {phrase}?",
                            eval_time=eval_ts,
                            expected_action=[0.0, 1.0],
                            expected_rule_id=r_comp,
                        )
                    )

                elif cat == "lex_specialis":
                    rules = [
                        {
                            "rule_id": r_base,
                            "text": (
                                f"[{juris}] {statutes['general_statute']} general rule covering {phrase}: "
                                f"standard commercial archive policy applies."
                            ),
                            "action": [1.0, 0.0],
                            "authority_rank": 3,
                            "valid_from": "2023-01-01T00:00:00Z",
                            "jurisdiction": juris,
                            "metadata": {"specificity": 1.0},
                        },
                        {
                            "rule_id": r_comp,
                            "text": (
                                f"[{juris}] {statutes['sectoral_statute']} special sectoral clause for {phrase}: "
                                f"mandatory cryptographic retention control applies."
                            ),
                            "action": [0.0, 1.0],
                            "authority_rank": 3,
                            "valid_from": "2023-01-01T00:00:00Z",
                            "jurisdiction": juris,
                            "metadata": {"specificity": 10.0},
                        },
                    ]
                    cases.append(
                        MultiJurisdictionalCase(
                            case_id=cid,
                            jurisdiction=juris,
                            category=cat,
                            domain_slug=slug,
                            rules=rules,
                            query_text=f"Which {juris} rule governs sectoral handling of {phrase}?",
                            eval_time=eval_ts,
                            expected_action=[0.0, 1.0],
                            expected_rule_id=r_comp,
                        )
                    )

                elif cat == "lex_posterior":
                    rules = [
                        {
                            "rule_id": r_base,
                            "text": (
                                f"[{juris}] 2020 Statutory Provision governing {phrase}: "
                                f"legacy paper filing schedule."
                            ),
                            "action": [1.0, 0.0],
                            "authority_rank": 4,
                            "valid_from": "2020-01-01T00:00:00Z",
                            "jurisdiction": juris,
                            "metadata": {"specificity": 2.0},
                        },
                        {
                            "rule_id": r_comp,
                            "text": (
                                f"[{juris}] 2025 Amended Statutory Provision governing {phrase}: "
                                f"updated digital cryptographic ledger mandate."
                            ),
                            "action": [0.0, 1.0],
                            "authority_rank": 4,
                            "valid_from": "2025-06-01T00:00:00Z",
                            "jurisdiction": juris,
                            "metadata": {"specificity": 2.0},
                        },
                    ]
                    cases.append(
                        MultiJurisdictionalCase(
                            case_id=cid,
                            jurisdiction=juris,
                            category=cat,
                            domain_slug=slug,
                            rules=rules,
                            query_text=f"What is the current amended {juris} statutory provision governing {phrase}?",
                            eval_time=eval_ts,
                            expected_action=[0.0, 1.0],
                            expected_rule_id=r_comp,
                        )
                    )

                elif cat == "temporal_expiry":
                    rules = [
                        {
                            "rule_id": r_base,
                            "text": (
                                f"[{juris}] Temporary emergency waiver for {phrase}: "
                                f"waiver expired in 2024."
                            ),
                            "action": [1.0, 0.0],
                            "authority_rank": 4,
                            "valid_from": "2021-01-01T00:00:00Z",
                            "valid_to": "2024-12-31T23:59:59Z",
                            "jurisdiction": juris,
                            "metadata": {"specificity": 5.0},
                        },
                        {
                            "rule_id": r_comp,
                            "text": (
                                f"[{juris}] Permanent baseline statute for {phrase}: "
                                f"mandatory continuous compliance monitoring."
                            ),
                            "action": [0.0, 1.0],
                            "authority_rank": 4,
                            "valid_from": "2021-01-01T00:00:00Z",
                            "jurisdiction": juris,
                            "metadata": {"specificity": 2.0},
                        },
                    ]
                    cases.append(
                        MultiJurisdictionalCase(
                            case_id=cid,
                            jurisdiction=juris,
                            category=cat,
                            domain_slug=slug,
                            rules=rules,
                            query_text=f"In 2026 under {juris} law, what active statute governs {phrase}?",
                            eval_time=eval_ts,
                            expected_action=[0.0, 1.0],
                            expected_rule_id=r_comp,
                        )
                    )

                elif cat == "gdpr_erasure":
                    rules = [
                        {
                            "rule_id": r_base,
                            "text": (
                                f"[{juris}] Data Subject Personal Consent Override for {phrase}: "
                                f"temporary opt-in profile."
                            ),
                            "action": [1.0, 0.0],
                            "authority_rank": 5,
                            "valid_from": "2025-01-01T00:00:00Z",
                            "personal_data": True,
                            "jurisdiction": juris,
                            "metadata": {"specificity": 10.0, "node_id": r_base},
                        },
                        {
                            "rule_id": r_comp,
                            "text": (
                                f"[{juris}] {statutes['supreme_statute']} default statutory baseline for {phrase}: "
                                f"prohibit processing after consent revocation."
                            ),
                            "action": [0.0, 1.0],
                            "authority_rank": 4,
                            "valid_from": "2023-01-01T00:00:00Z",
                            "personal_data": False,
                            "jurisdiction": juris,
                            "metadata": {"specificity": 2.0, "node_id": r_comp},
                        },
                    ]
                    cases.append(
                        MultiJurisdictionalCase(
                            case_id=cid,
                            jurisdiction=juris,
                            category=cat,
                            domain_slug=slug,
                            rules=rules,
                            query_text=f"After data subject erasure under {juris} law, what governs {phrase}?",
                            eval_time=eval_ts,
                            expected_action=[0.0, 1.0],
                            expected_rule_id=r_comp,
                            erased_rule_id=r_base,
                        )
                    )

                elif cat == "epistemic_abstention":
                    rules = [
                        {
                            "rule_id": r_comp,
                            "text": (
                                f"[{juris}] {statutes['supreme_statute']} statutory mandate on {phrase}: "
                                f"enforce regulatory audit retention."
                            ),
                            "action": [0.0, 1.0],
                            "authority_rank": 5,
                            "valid_from": "2023-01-01T00:00:00Z",
                            "jurisdiction": juris,
                            "metadata": {"specificity": 2.0},
                        },
                    ]
                    ood_query = OUT_OF_DOMAIN_QUERIES[idx % len(OUT_OF_DOMAIN_QUERIES)]
                    cases.append(
                        MultiJurisdictionalCase(
                            case_id=cid,
                            jurisdiction=juris,
                            category=cat,
                            domain_slug=slug,
                            rules=rules,
                            query_text=ood_query,
                            eval_time=eval_ts,
                            expected_action=[0.0, 0.0],
                            expected_rule_id=None,
                            expect_abstention=True,
                        )
                    )

    return cases


def to_ragas_records(eval_traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Export CLM evaluation traces into standard RAGAS v0.2+ `SingleTurnSample` dictionaries.

    Schema:
        - `user_input`: Query text submitted to the legal memory orchestrator.
        - `retrieved_contexts`: List of retrieved statutory rule snippets.
        - `response`: Formatted decision vector and abstention status.
        - `reference`: Expected statutory rule text and decision vector.
        - `metadata`: Jurisdiction, conflict category, and conformal/Merkle audit telemetry.
    """
    samples: list[dict[str, Any]] = []
    for trace in eval_traces:
        samples.append(
            {
                "user_input": trace["query_text"],
                "retrieved_contexts": trace.get("retrieved_contexts", []),
                "response": trace["actual_response"],
                "reference": trace["expected_reference"],
                "metadata": {
                    "case_id": trace["case_id"],
                    "jurisdiction": trace["jurisdiction"],
                    "category": trace["category"],
                    "abstained": trace.get("abstained", False),
                    "conformal_set": trace.get("conformal_set"),
                    "merkle_verified": trace.get("merkle_verified"),
                },
            }
        )
    return samples


def to_deepeval_test_cases(eval_traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Export CLM evaluation traces into standard DeepEval `LLMTestCase` dictionaries.

    Schema:
        - `input`: Query text string.
        - `actual_output`: Formatted predicted decision and cited rule.
        - `expected_output`: Ground-truth decision and expected rule.
        - `retrieval_context`: List of retrieved statutory contexts.
        - `additional_metadata`: Jurisdiction, category, and cryptographic verification flags.
    """
    test_cases: list[dict[str, Any]] = []
    for trace in eval_traces:
        test_cases.append(
            {
                "input": trace["query_text"],
                "actual_output": trace["actual_response"],
                "expected_output": trace["expected_reference"],
                "retrieval_context": trace.get("retrieved_contexts", []),
                "additional_metadata": {
                    "case_id": trace["case_id"],
                    "jurisdiction": trace["jurisdiction"],
                    "category": trace["category"],
                    "abstained": trace.get("abstained", False),
                    "conformal_set": trace.get("conformal_set"),
                    "merkle_verified": trace.get("merkle_verified"),
                },
            }
        )
    return test_cases


def evaluate_multijurisdictional_suite(
    domains_per_category: int = 30,
    encoder: BaseEncoderPort | None = None,
) -> dict[str, Any]:
    """
    Execute the multi-jurisdictional statutory benchmark comparing `PlainRAGBaseline`
    against `LegalMemoryOrchestrator` (equipped with `DeonticCrossEncoderReranker`,
    `SplitConformalCalibrator`, and `SparseMerkleTree` erasure verification).
    """
    enc = encoder or DeterministicTokenEncoder(embedding_dim=64)
    cases = generate_multijurisdictional_benchmark(domains_per_category=domains_per_category)

    reranker = DeonticCrossEncoderReranker(blend_weight=0.35)
    calibrator = SplitConformalCalibrator(
        alpha=0.10,
        min_similarity_gate=0.22,
        abstain_on_empty=True,
        default_quantile=0.45,
    )

    clm_correct = 0
    rag_correct = 0
    context_precision_hits = 0
    non_abstention_count = 0
    abstention_tp = 0
    abstention_fp = 0
    abstention_fn = 0
    smt_erasure_proofs_verified = 0
    smt_erasure_proofs_total = 0

    by_jurisdiction: dict[str, dict[str, int]] = {
        j: {"total": 0, "clm_correct": 0, "rag_correct": 0} for j in ("EU", "US", "BR")
    }
    by_category: dict[str, dict[str, int]] = {}
    eval_traces: list[dict[str, Any]] = []

    for case in cases:
        by_jurisdiction[case.jurisdiction]["total"] += 1
        cat_stats = by_category.setdefault(
            case.category, {"total": 0, "clm_correct": 0, "rag_correct": 0}
        )
        cat_stats["total"] += 1

        # 1. PlainRAG Baseline
        rag = PlainRAGBaseline(value_dim=2, temperature=0.05)
        rag.fit(case.rules, enc)
        eval_dt = datetime.fromisoformat(case.eval_time.replace("Z", "+00:00"))
        rag_pred = rag.predict(case.query_text, at_time=eval_dt)
        if case.expect_abstention:
            rag_ok = False
        else:
            rag_winner = 0 if rag_pred[0] >= rag_pred[1] else 1
            exp_winner = 0 if case.expected_action[0] >= case.expected_action[1] else 1
            rag_ok = rag_winner == exp_winner
        if rag_ok:
            rag_correct += 1
            by_jurisdiction[case.jurisdiction]["rag_correct"] += 1
            cat_stats["rag_correct"] += 1

        # 2. Continuous Legal Memory Orchestrator
        orch = LegalMemoryOrchestrator(
            encoder=enc,
            value_dim=2,
            temperature=0.05,
            seed=42,
            engine_mode="structured",
            min_similarity_threshold=0.22,
            reranker=reranker,
            conformal_calibrator=calibrator,
            enable_attestation=True,
        )

        expected_rule_text = ""
        for r in case.rules:
            vf = datetime.fromisoformat(r["valid_from"].replace("Z", "+00:00"))
            vt = datetime.fromisoformat(r["valid_to"].replace("Z", "+00:00")) if r.get("valid_to") else None
            meta = dict(r.get("metadata", {}))
            meta["node_id"] = r["rule_id"]
            rec = orch.update_memory(
                rule_text=r["text"],
                action_vector=r["action"],
                valid_from=vf,
                valid_to=vt,
                metadata=meta,
                authority_rank=r["authority_rank"],
                jurisdiction=r["jurisdiction"],
                personal_data=r.get("personal_data", False),
            )
            rec.record_id = r["rule_id"]
            if r["rule_id"] == case.expected_rule_id:
                expected_rule_text = r["text"]

        merkle_ok = True
        if case.erased_rule_id is not None:
            smt_erasure_proofs_total += 1
            receipt = orch.delete_rule(record_id=case.erased_rule_id)
            proof_dict = receipt["merkle_non_membership_proof"]
            merkle_ok = verify_sparse_merkle_proof(
                proof_dict,
                expected_root_hash=receipt["merkle_root"],
                attestor=orch.attestor,
            ) and (not proof_dict["is_member"])
            if merkle_ok:
                smt_erasure_proofs_verified += 1

        res = orch.predict(case.query_text, at_time=eval_dt)

        if case.expect_abstention:
            clm_ok = res.abstained
            if res.abstained:
                abstention_tp += 1
            else:
                abstention_fn += 1
        else:
            non_abstention_count += 1
            if res.abstained:
                clm_ok = False
                abstention_fp += 1
            else:
                clm_winner = 0 if res.predicted_action_vector[0] >= res.predicted_action_vector[1] else 1
                exp_winner = 0 if case.expected_action[0] >= case.expected_action[1] else 1
                clm_ok = (clm_winner == exp_winner) and merkle_ok
                if res.most_relevant_rule == expected_rule_text:
                    context_precision_hits += 1

        if clm_ok:
            clm_correct += 1
            by_jurisdiction[case.jurisdiction]["clm_correct"] += 1
            cat_stats["clm_correct"] += 1

        eval_traces.append(
            {
                "case_id": case.case_id,
                "jurisdiction": case.jurisdiction,
                "category": case.category,
                "query_text": case.query_text,
                "retrieved_contexts": [res.most_relevant_rule] if res.most_relevant_rule else [],
                "actual_response": (
                    "ABSTAINED"
                    if res.abstained
                    else f"ACTION={res.predicted_action_vector}|RULE={res.most_relevant_rule}"
                ),
                "expected_reference": (
                    "ABSTAINED"
                    if case.expect_abstention
                    else f"ACTION={case.expected_action}|RULE={expected_rule_text}"
                ),
                "abstained": res.abstained,
                "conformal_set": res.conformal_set,
                "merkle_verified": merkle_ok,
            }
        )

    total_cases = len(cases)
    precision_abstain = (
        abstention_tp / (abstention_tp + abstention_fp)
        if (abstention_tp + abstention_fp) > 0
        else 1.0
    )
    recall_abstain = (
        abstention_tp / (abstention_tp + abstention_fn)
        if (abstention_tp + abstention_fn) > 0
        else 1.0
    )
    abstention_f1 = (
        2.0 * precision_abstain * recall_abstain / (precision_abstain + recall_abstain)
        if (precision_abstain + recall_abstain) > 0
        else 0.0
    )

    return {
        "total_cases": total_cases,
        "clm_accuracy": clm_correct / total_cases if total_cases else 0.0,
        "plain_rag_accuracy": rag_correct / total_cases if total_cases else 0.0,
        "context_precision": (
            context_precision_hits / non_abstention_count if non_abstention_count else 1.0
        ),
        "abstention_f1": abstention_f1,
        "smt_erasure_verification_rate": (
            smt_erasure_proofs_verified / smt_erasure_proofs_total
            if smt_erasure_proofs_total
            else 1.0
        ),
        "by_jurisdiction": {
            j: {
                "total": d["total"],
                "clm_accuracy": d["clm_correct"] / d["total"] if d["total"] else 0.0,
                "plain_rag_accuracy": d["rag_correct"] / d["total"] if d["total"] else 0.0,
            }
            for j, d in by_jurisdiction.items()
        },
        "by_category": {
            c: {
                "total": d["total"],
                "clm_accuracy": d["clm_correct"] / d["total"] if d["total"] else 0.0,
                "plain_rag_accuracy": d["rag_correct"] / d["total"] if d["total"] else 0.0,
            }
            for c, d in by_category.items()
        },
        "ragas_samples": to_ragas_records(eval_traces[:12]),
        "deepeval_test_cases": to_deepeval_test_cases(eval_traces[:12]),
    }
