"""
Bi-Temporal Replay, Hybrid Lexical-Dense Retrieval, and Defeasible Proof-Trace Verification Suite:
1. Strict Bi-Temporal 'As-Of' Auditing (`at_time` x `as_of_transaction_time`) & Historical Merkle Root Replay.
2. Hybrid Lexical-Dense Normative Retrieval (`BM25+` & Statutory Citation Pinpointing).
3. Defeasible Proof-Tree Explanation & Auditable Reasoning Trace (`DefeasibleProofTrace`).
4. Full REST Microservice (`server.py`) & CLI (`cli.py`) Parity.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from continuous_legal_memory import __version__
from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.cli import main as cli_main
from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.retrieval.hybrid_lexical import (
    HybridLexicalCitationScorer,
    extract_legal_citations,
)
from continuous_legal_memory.retrieval.temporal_structured import TemporalStructuredRetriever
from continuous_legal_memory.security.attestation import AsymmetricAttestationModule
from continuous_legal_memory.server import create_server


def _http_json(
    url: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_bitemporal_as_of_transaction_time_and_merkle_root_replay() -> None:
    """
    Verify orthogonal bi-temporal querying (`at_time` x `as_of_transaction_time`):
    - A retroactive amendment ingested at `tx = 2026-06-01` with `valid_from = 2024-01-01`
      must NOT be visible when auditing `as_of_transaction_time = 2025-02-01`, and MUST
      be visible when auditing `as_of_transaction_time = 2026-07-01`.
    - Historical Sparse Merkle Tree roots must match the exact point-in-time state.
    """
    orch = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), value_dim=2)

    t_valid_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tx_initial = datetime(2024, 1, 10, tzinfo=timezone.utc)
    tx_retroactive = datetime(2026, 6, 1, tzinfo=timezone.utc)

    orch.update_memory(
        rule_text="Data Retention Rule: Financial logs shall be retained for 3 years.",
        action_vector=[1.0, 0.0],
        valid_from=t_valid_start,
        transaction_time=tx_initial,
        authority_rank=5,
        metadata={"record_id": "rule_2024_initial"},
    )
    root_after_first = orch.merkle_tree.root_hex

    orch.update_memory(
        rule_text="Data Retention Rule: Financial logs shall be retained for 10 years retroactively.",
        action_vector=[0.0, 1.0],
        valid_from=t_valid_start,
        transaction_time=tx_retroactive,
        authority_rank=8,
        metadata={"record_id": "rule_2026_retro", "supersedes": "rule_2024_initial"},
    )
    root_after_second = orch.merkle_tree.root_hex
    assert root_after_first != root_after_second

    eval_valid_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    # 1. Audit as of 2025-02-01 (before retroactive amendment was ingested)
    audit_cutoff_early = datetime(2025, 2, 1, tzinfo=timezone.utc)
    res_early = orch.predict(
        query_text="Data Retention Rule: Financial logs retention period",
        at_time=eval_valid_time,
        as_of_transaction_time=audit_cutoff_early,
    )
    assert res_early.predicted_action_vector[0] > res_early.predicted_action_vector[1]
    assert res_early.merkle_root == root_after_first
    assert res_early.proof_trace is not None
    assert res_early.proof_trace["prevailing_record_id"] == "rule_2024_initial"
    excluded_early = res_early.proof_trace["excluded_records"]
    assert len(excluded_early) == 1
    assert excluded_early[0]["record_id"] == "rule_2026_retro"
    assert excluded_early[0]["status"] == "EXCLUDED_TRANSACTION_TIME"

    # 2. Audit as of 2026-07-01 (after retroactive amendment was ingested)
    audit_cutoff_late = datetime(2026, 7, 1, tzinfo=timezone.utc)
    res_late = orch.predict(
        query_text="Data Retention Rule: Financial logs retention period",
        at_time=eval_valid_time,
        as_of_transaction_time=audit_cutoff_late,
    )
    assert res_late.predicted_action_vector[1] > res_late.predicted_action_vector[0]
    assert res_late.merkle_root == root_after_second
    assert res_late.proof_trace is not None
    assert res_late.proof_trace["prevailing_record_id"] == "rule_2026_retro"
    status_by_id = {c["record_id"]: c["status"] for c in res_late.proof_trace["candidates"]}
    assert status_by_id["rule_2026_retro"] == "PREVAILED"
    assert status_by_id["rule_2024_initial"] == "DEFEATED_BY_SUPERSEDES"


def test_hybrid_lexical_citation_pinpointing_and_bm25_plus() -> None:
    """
    Verify multi-jurisdictional statutory citation extraction (`EU`, `US`, `BR`)
    and exact pin-cite disambiguation in `TemporalStructuredRetriever`.
    """
    eu_cites = extract_legal_citations("Pursuant to Art. 17(3)(b) GDPR and Regulation (EU) 2024/1689")
    assert "art:17(3)(b)" in eu_cites
    assert "art:17" in eu_cites
    assert "inst:gdpr" in eu_cites
    assert "eu_reg:2024/1689" in eu_cites

    us_cites = extract_legal_citations("Jurisdiction under 28 U.S.C. § 1332 and SEC Rule 10b-5")
    assert "usc:28:1332" in us_cites
    assert "sec:1332" in us_cites
    assert "rule:10b-5" in us_cites

    br_cites = extract_legal_citations("Nos termos da Lei 13.709/2018 Artigo 7º da LGPD")
    assert "lei:13709/2018" in br_cites
    assert "art:7" in br_cites
    assert "inst:lgpd" in br_cites

    # Verify exact pin-cite tie-breaking when dense embeddings are identical
    identical_vec = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    keys = torch.cat([identical_vec, identical_vec], dim=0)
    importance = torch.ones(2, dtype=torch.float32)

    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rec_17_1 = MemoryRecord(
        text="GDPR Art. 17(1): The data subject shall have the right to obtain erasure of personal data.",
        key_vector=identical_vec,
        value_vector=torch.tensor([[1.0, 0.0]]),
        record_id="gdpr_17_1",
        valid_from=t0,
        authority_rank=8,
    )
    rec_17_3_b = MemoryRecord(
        text="GDPR Art. 17(3)(b): Erasure shall not apply where processing is necessary for a legal obligation.",
        key_vector=identical_vec,
        value_vector=torch.tensor([[0.0, 1.0]]),
        record_id="gdpr_17_3_b",
        valid_from=t0,
        authority_rank=8,
    )

    retriever = TemporalStructuredRetriever(
        lexical_scorer=HybridLexicalCitationScorer(),
        lexical_blend=0.15,
        citation_boost=0.10,
    )
    adj_scores, weights, snippets = retriever.retrieve(
        query_embed=identical_vec,
        keys=keys,
        rule_importance=importance,
        query_text="Does GDPR Art. 17(3)(b) exempt legal obligation compliance from erasure?",
        records=[rec_17_1, rec_17_3_b],
    )
    assert float(weights[0, 1].item()) > float(weights[0, 0].item())
    assert snippets == [rec_17_3_b.text]


def test_defeasible_proof_trace_canons_and_deterministic_hash() -> None:
    """
    Verify that `DefeasibleProofTrace` records canon-specific defeat reasons
    (`DEFEATED_BY_LEX_SUPERIOR`, `DEFEATED_BY_LOIS_DE_POLICE`, `EXCLUDED_TEMPORAL_VALIDITY`)
    and computes a reproducible SHA-256 `trace_hash`.
    """
    orch = LegalMemoryOrchestrator(encoder=SemanticMockEncoder(), value_dim=2)
    t_active = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t_expired = datetime(2023, 1, 1, tzinfo=timezone.utc)

    # 1. Expired norm
    orch.update_memory(
        rule_text="Cross-border transfer rule: Legacy 2022 policy.",
        action_vector=[1.0, 0.0],
        valid_from=datetime(2022, 1, 1, tzinfo=timezone.utc),
        valid_to=t_expired,
        authority_rank=3,
        jurisdiction="US",
        metadata={"record_id": "expired_policy"},
    )
    # 2. Subordinate corporate policy
    orch.update_memory(
        rule_text="Cross-border transfer rule: Corporate policy permits unrestricted transfer.",
        action_vector=[1.0, 0.0],
        valid_from=t_active,
        authority_rank=2,
        jurisdiction="US",
        metadata={"record_id": "corp_policy_us"},
    )
    # 3. Higher-authority overriding mandatory statute
    orch.update_memory(
        rule_text="Cross-border transfer rule: EU GDPR Art. 44 prohibits transfer without adequacy.",
        action_vector=[0.0, 1.0],
        valid_from=t_active,
        authority_rank=9,
        jurisdiction="EU",
        metadata={"record_id": "gdpr_art_44", "overriding_mandatory": True},
    )

    res1 = orch.predict(
        query_text="Cross-border transfer rule compliance",
        at_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        connecting_factors={
            "data_subject_domicile": "EU",
            "choice_of_forum": "US",
        },
    )
    res2 = orch.predict(
        query_text="Cross-border transfer rule compliance",
        at_time=datetime(2025, 6, 1, tzinfo=timezone.utc),
        connecting_factors={
            "data_subject_domicile": "EU",
            "choice_of_forum": "US",
        },
    )

    assert res1.proof_trace is not None
    assert res2.proof_trace is not None
    assert res1.proof_trace["trace_hash"] == res2.proof_trace["trace_hash"]
    assert len(res1.proof_trace["trace_hash"]) == 64

    cand_map = {c["record_id"]: c for c in res1.proof_trace["candidates"]}
    assert cand_map["gdpr_art_44"]["status"] == "PREVAILED"
    assert cand_map["corp_policy_us"]["status"] in ("DEFEATED_BY_LOIS_DE_POLICE", "DEFEATED_BY_LEX_SUPERIOR")
    assert cand_map["corp_policy_us"]["defeated_by_record_id"] == "gdpr_art_44"

    excl_map = {e["record_id"]: e["status"] for e in res1.proof_trace["excluded_records"]}
    assert excl_map["expired_policy"] == "EXCLUDED_TEMPORAL_VALIDITY"


def test_rest_server_and_cli_parity(tmp_path: Path) -> None:
    """
    Verify full REST API (`server.py`) and CLI (`cli.py`) parity for Merkle, audit, and conformal endpoints:
    - `GET /health` version == `__version__` and `merkle_root`
    - `POST /v1/rules` with `record_id` and `transaction_time`
    - `GET /v1/merkle/root`, `GET /v1/merkle/proof/<id>`, `POST /v1/merkle/verify`
    - `POST /v1/audit/consistency`
    - `POST /v1/conformal/calibrate` and `POST /v1/predict`
    - `clm audit` and `clm merkle-proof` CLI commands
    """
    orch = LegalMemoryOrchestrator(
        encoder=SemanticMockEncoder(),
        attestor=AsymmetricAttestationModule(),
        value_dim=2,
    )
    server = create_server(orch, host="127.0.0.1", port=0)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        status, health = _http_json(f"{base_url}/health")
        assert status == 200
        assert health["version"] == __version__
        assert len(health["merkle_root"]) == 64

        # Ingest two horizontally conflicting norms to test /v1/audit/consistency
        status, r1 = _http_json(
            f"{base_url}/v1/rules",
            method="POST",
            payload={
                "rule_text": "Biometric retention: Controllers must retain biometric logs for 24 months.",
                "action_vector": [0.0, 1.0],
                "authority_rank": 7,
                "jurisdiction": "EU",
                "record_id": "bio_must_retain",
                "valid_from": "2025-01-01T00:00:00+00:00",
                "transaction_time": "2025-01-10T00:00:00+00:00",
            },
        )
        assert status == 201
        assert r1["record_id"] == "bio_must_retain"

        status, r2 = _http_json(
            f"{base_url}/v1/rules",
            method="POST",
            payload={
                "rule_text": "Biometric retention: Controllers are prohibited from retaining biometric logs.",
                "action_vector": [1.0, 0.0],
                "authority_rank": 7,
                "jurisdiction": "EU",
                "record_id": "bio_forbid_retain",
                "valid_from": "2025-01-01T00:00:00+00:00",
                "transaction_time": "2025-01-11T00:00:00+00:00",
            },
        )
        assert status == 201

        # 1. Test GET /v1/merkle/root and GET /v1/merkle/proof/<id> + POST /v1/merkle/verify
        status, root_info = _http_json(f"{base_url}/v1/merkle/root")
        assert status == 200
        assert root_info["active_leaves"] == 2
        assert root_info["chain_verified"] is True

        status, proof_inc = _http_json(f"{base_url}/v1/merkle/proof/bio_must_retain")
        assert status == 200
        assert proof_inc["is_member"] is True

        status, ver_inc = _http_json(
            f"{base_url}/v1/merkle/verify",
            method="POST",
            payload={"proof": proof_inc, "expected_root_hash": root_info["merkle_root"]},
        )
        assert status == 200
        assert ver_inc["valid"] is True

        # 2. Test POST /v1/audit/consistency
        status, audit_rep = _http_json(
            f"{base_url}/v1/audit/consistency",
            method="POST",
            payload={"tenant_id": "default"},
        )
        assert status == 200
        assert audit_rep["is_consistent"] is False
        assert len(audit_rep["antinomies"]) >= 1

        # 3. Test POST /v1/conformal/calibrate and POST /v1/predict
        status, cal_resp = _http_json(
            f"{base_url}/v1/conformal/calibrate",
            method="POST",
            payload={
                "predicted_vectors": [[0.1, 0.9], [0.85, 0.15], [0.2, 0.8], [0.9, 0.1]],
                "true_labels": [1, 0, 1, 0],
                "alpha": 0.10,
            },
        )
        assert status == 200
        assert cal_resp["status"] == "CALIBRATED"

        status, pred_resp = _http_json(
            f"{base_url}/v1/predict",
            method="POST",
            payload={
                "query_text": "Biometric retention: Controllers biometric logs",
                "jurisdiction": "EU",
            },
        )
        assert status == 200
        assert pred_resp["conformal_set"] is not None
        assert pred_resp["proof_trace"] is not None
        assert pred_resp["merkle_root"] == root_info["merkle_root"]

        # 4. Persist to SQLite and test CLI subcommands `audit` and `merkle-proof`
        db_file = str(tmp_path / "cli_round5.db")
        orch.save_to_disk(db_file)

        rc_audit = cli_main(["audit", "--db-path", db_file, "--mock-encoder"])
        assert rc_audit == 0

        rc_proof = cli_main(["merkle-proof", "bio_must_retain", "--db-path", db_file, "--mock-encoder"])
        assert rc_proof == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
