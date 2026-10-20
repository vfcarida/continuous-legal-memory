"""
Zero-Knowledge Selective Disclosure, Deontic Polarity PPR, and Portable Bundle Verification Suite:
1. Selective-Disclosure / Zero-Knowledge Normative Compliance Proofs (EU AI Act Art. 12 & 78).
2. Deontic Polarity-Aware Graph Diffusion (`SUPERSEDES` redirection & `CONTRADICTS` suppression)
   and bi-temporal `as_of_transaction_time` filtering in `SemanticKnowledgeGraph.personalized_pagerank()`.
3. Signed Portable Compliance Bundles (`.clm.bundle.json`) & Distributed Vector/Graph Exporters
   (`Qdrant`, `pgvector`, `Neo4j` Cypher).
4. `HybridLegalRetriever` Precedence-Scaled Score Return & Statutory Citation/Keyword-Density Snippet Extraction.
5. REST (`server.py`) & MCP (`mcp.py`) Protocol Parity.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from datetime import datetime, timezone

import torch

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import EntityType, RelationType
from continuous_legal_memory.mcp import MCPServer
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.retrieval.hybrid_retriever import HybridLegalRetriever
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    KeyedHashAttestationModule,
)
from continuous_legal_memory.security.selective_disclosure import SelectiveDisclosureProver
from continuous_legal_memory.server import create_server


class _DeterministicEncoder(BaseEncoderPort):
    """Lightweight deterministic encoder for offline unit testing."""

    @property
    def embedding_dim(self) -> int:
        return 8

    def get_embedding(self, texts: list[str]) -> torch.Tensor:
        rows: list[list[float]] = []
        for t in texts:
            low = t.lower()
            vec = [
                1.0 if "gdpr" in low or "art. 17" in low else 0.1,
                1.0 if "erasure" in low or "delete" in low else 0.1,
                1.0 if "ai act" in low or "art. 12" in low else 0.1,
                1.0 if "confidential" in low or "secret" in low else 0.1,
                1.0 if "repealed" in low or "old" in low else 0.1,
                1.0 if "active" in low or "new" in low else 0.1,
                float(len(t) % 11) / 11.0,
                0.5,
            ]
            rows.append(vec)
        tensor = torch.tensor(rows, dtype=torch.float32)
        return torch.nn.functional.normalize(tensor, p=2, dim=-1)


def test_selective_disclosure_compliance_proof_and_unblinding() -> None:
    """
    Verify that a SelectiveDisclosureComplianceProof proves Sparse Merkle Tree inclusion,
    minimum authority rank, temporal validity, and jurisdiction WITHOUT exposing plaintext rule or query,
    and supports selective unblinding via salt verification.
    """
    ed_attestor = AsymmetricAttestationModule()
    orch = LegalMemoryOrchestrator(
        encoder=_DeterministicEncoder(),
        value_dim=2,
        attestor=ed_attestor,
    )
    confidential_clause = (
        "CONFIDENTIAL M&A CLAUSE 9.4: Acquirer shall retain GDPR Art. 17 erasure logs "
        "for 10 years under EU AI Act Art. 12 supervision."
    )
    rec = orch.update_memory(
        rule_text=confidential_clause,
        action_vector=[0.0, 1.0],
        authority_rank=5,
        jurisdiction="EU",
        metadata={"record_id": "clause_9_4_secret"},
        tenant_id="deal_room_alpha",
    )
    assert rec.record_id == "clause_9_4_secret"

    secret_query = "What is Acquirer's retention duty under M&A Clause 9.4?"
    bundle = orch.generate_compliance_proof(
        record_id="clause_9_4_secret",
        query=secret_query,
        min_required_authority_rank=4,
        tenant_id="deal_room_alpha",
    )

    proof_dict = bundle.proof.to_dict()
    serialized = json.dumps(proof_dict)

    # 1. Ensure confidential plaintext is NEVER leaked in the serialized proof
    assert "CONFIDENTIAL M&A CLAUSE" not in serialized
    assert "Acquirer" not in serialized
    assert secret_query not in serialized

    # 2. Verify cryptographic proof, Merkle inclusion, authority_rank >= 4, and jurisdiction == 'EU'
    assert orch.verify_compliance_proof(
        bundle.proof,
        expected_merkle_root=orch.merkle_tree.root_hex,
        min_authority_rank=4,
        required_jurisdiction="EU",
    )
    # Policy check fails if verifier demands higher authority rank than the record possesses (5 < 8)
    assert not orch.verify_compliance_proof(
        bundle.proof,
        min_authority_rank=8,
    )
    # Policy check fails if verifier demands a different jurisdiction
    assert not orch.verify_compliance_proof(
        bundle.proof,
        required_jurisdiction="US-NY",
    )

    # 3. Selective unblinding under court/audit order
    assert SelectiveDisclosureProver.verify_disclosed_text(
        bundle.proof, confidential_clause, bundle.text_salt
    )
    assert not SelectiveDisclosureProver.verify_disclosed_text(
        bundle.proof, "Fabricated clause text", bundle.text_salt
    )
    assert SelectiveDisclosureProver.verify_disclosed_query(
        bundle.proof, secret_query, bundle.query_salt
    )

    # 4. Also verify HMAC-SHA256 symmetric mode
    hmac_attestor = KeyedHashAttestationModule(secret_key=b"super-secret-enterprise-key-32b!")
    prover_hmac = SelectiveDisclosureProver(attestation_module=hmac_attestor)
    hmac_bundle = prover_hmac.generate_proof(
        record=rec,
        merkle_tree=orch.merkle_tree,
        query=secret_query,
        min_required_authority_rank=3,
    )
    assert hmac_bundle.proof.algorithm == "HMAC-SHA256"
    assert prover_hmac.verify_proof(hmac_bundle.proof, min_authority_rank=3)


def test_deontic_polarity_pagerank_supersedes_and_contradicts() -> None:
    """
    Verify that `SemanticKnowledgeGraph.personalized_pagerank()` with `deontic_polarity=True`:
    1. Redirects activation mass from a superseded statute (`old_statute`) to its active superseding
       statute (`new_statute`),
    2. Blocks positive activation walk leakage across `CONTRADICTS` edges,
    3. Preserves the exact stochastic normalization invariant (`sum(ppr.values()) == 1.0`),
    4. Filters out nodes ingested after `as_of_transaction_time`.
    """
    orch = LegalMemoryOrchestrator(encoder=_DeterministicEncoder(), value_dim=2)
    graph = orch.semantic_graph

    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, tzinfo=timezone.utc)

    graph.add_node(
        node_id="directive_95_46_ec",
        entity_type=EntityType.STATUTE,
        label="Data Protection Directive 95/46/EC",
        description="Repealed EU data protection directive.",
        transaction_time=t1,
    )
    graph.add_node(
        node_id="gdpr_2016_679",
        entity_type=EntityType.STATUTE,
        label="GDPR Regulation (EU) 2016/679",
        description="Active EU General Data Protection Regulation superseding 95/46/EC.",
        transaction_time=t1,
    )
    graph.add_node(
        node_id="gdpr_enf_guideline",
        entity_type=EntityType.CLAUSE,
        label="EDPB Enforcement Guideline",
        description="Enforcement guideline dependent on GDPR.",
        transaction_time=t1,
    )
    graph.add_node(
        node_id="contradicting_local_memo",
        entity_type=EntityType.CLIENT_PREFERENCE,
        label="Unlawful Local Memo",
        description="Contradicts GDPR erasure obligations.",
        transaction_time=t1,
    )
    graph.add_node(
        node_id="future_2026_amendment",
        entity_type=EntityType.STATUTE,
        label="2026 AI Amendment",
        description="Ingested in 2026.",
        transaction_time=t2,
    )

    # GDPR supersedes Directive 95/46/EC (edge: gdpr_2016_679 -> directive_95_46_ec)
    graph.add_edge("gdpr_2016_679", "directive_95_46_ec", RelationType.SUPERSEDES, weight=1.0)
    # GDPR links to enforcement guideline
    graph.add_edge("gdpr_2016_679", "gdpr_enf_guideline", RelationType.APPLIES_TO, weight=1.0)
    # Contradicting memo has CONTRADICTS edge from GDPR
    graph.add_edge("gdpr_2016_679", "contradicting_local_memo", RelationType.CONTRADICTS, weight=1.0)
    # Future node linked from guideline
    graph.add_edge("gdpr_enf_guideline", "future_2026_amendment", RelationType.APPLIES_TO, weight=1.0)

    # Seed ONLY the outdated/superseded Directive 95/46/EC as of 2025-06-01
    as_of_2025 = datetime(2025, 6, 1, tzinfo=timezone.utc)
    ppr = graph.personalized_pagerank(
        seed_weights=["directive_95_46_ec"],
        as_of_transaction_time=as_of_2025,
        deontic_polarity=True,
    )

    # 1. Future node (transaction_time = 2026) is excluded by as_of_transaction_time = 2025-06-01
    assert "future_2026_amendment" not in ppr

    # 2. Exact probability distribution normalization invariant
    assert abs(sum(ppr.values()) - 1.0) < 1e-5

    # 3. Superseding norm (gdpr_2016_679) receives redirected activation mass exceeding the superseded norm!
    assert ppr["gdpr_2016_679"] > ppr["directive_95_46_ec"]
    assert ppr["gdpr_enf_guideline"] > 0.0

    # 4. Contradicting node receives ZERO positive walk mass
    assert ppr["contradicting_local_memo"] == 0.0


def test_signed_compliance_bundle_and_distributed_exporters() -> None:
    """
    Verify SignedComplianceBundle export, tamper detection, PII redaction, state
    restoration, and Qdrant / pgvector / Neo4j Cypher exporters.
    """
    orch = LegalMemoryOrchestrator(
        encoder=_DeterministicEncoder(),
        value_dim=2,
        attestor=AsymmetricAttestationModule(),
    )
    orch.update_memory(
        rule_text="GDPR Art. 17 right to erasure mandatory rule.",
        action_vector=[0.0, 1.0],
        authority_rank=5,
        jurisdiction="EU",
        metadata={"record_id": "rec_gdpr_17", "node_id": "rec_gdpr_17"},
        tenant_id="tenant_eu",
    )
    orch.update_memory(
        rule_text="Data Subject John Doe personal preference.",
        action_vector=[1.0, 0.0],
        authority_rank=1,
        jurisdiction="EU",
        personal_data=True,
        metadata={
            "record_id": "rec_pii_1",
            "node_id": "rec_pii_1",
            "relations": [{"target_id": "rec_gdpr_17", "relation_type": "depends_on"}],
        },
        tenant_id="tenant_eu",
    )

    # 1. Export signed bundle (with PII redaction enabled)
    redacted_bundle = orch.export_compliance_bundle(
        tenant_id="tenant_eu",
        redact_personal_data=True,
    )
    assert orch.verify_compliance_bundle(redacted_bundle)
    pii_entry = next(r for r in redacted_bundle.records if r["record_id"] == "rec_pii_1")
    assert pii_entry["text"] == "[REDACTED_PERSONAL_DATA]"

    # 2. Export full signed bundle and restore into a fresh orchestrator
    full_bundle = orch.export_compliance_bundle(tenant_id="tenant_eu", redact_personal_data=False)
    raw_json = full_bundle.to_json()
    reloaded_bundle = full_bundle.from_json(raw_json)
    assert orch.verify_compliance_bundle(reloaded_bundle)

    # Tamper check: modifying any record invalidates signature verification
    tampered_dict = reloaded_bundle.to_dict()
    tampered_dict["records"][0]["authority_rank"] = 99
    assert not orch.verify_compliance_bundle(tampered_dict)

    fresh_orch = LegalMemoryOrchestrator(encoder=_DeterministicEncoder(), value_dim=2)
    fresh_orch.restore_compliance_bundle(reloaded_bundle)
    assert fresh_orch.merkle_tree.root_hex == orch.merkle_tree.root_hex
    assert len(fresh_orch.episodic_memory.get_records(tenant_id="tenant_eu")) == 2

    # 3. Distributed Vector & Graph Exporters (Qdrant, pgvector, Neo4j Cypher)
    qdrant_pts = orch.export_to_qdrant(tenant_id="tenant_eu")
    assert len(qdrant_pts) == 2
    assert qdrant_pts[0]["payload"]["merkle_leaf_hash"] is not None
    assert "key_vector" in qdrant_pts[0]["vector"]

    pg_rows = orch.export_to_pgvector(tenant_id="tenant_eu")
    assert len(pg_rows) == 2
    assert pg_rows[0]["key_embedding"].startswith("[")
    assert pg_rows[0]["merkle_leaf_hash"] is not None

    cypher_stmts = orch.export_to_neo4j(tenant_id="tenant_eu")
    assert len(cypher_stmts) == 3  # 2 nodes + 1 DEPENDS_ON edge
    assert any("MERGE (src)-[r:DEPENDS_ON]->(tgt)" in s["statement"] for s in cypher_stmts)


def test_hybrid_legal_retriever_scaled_scores_and_citation_snippet() -> None:
    """
    Verify that `HybridLegalRetriever.retrieve()` returns precedence-scaled scores
    and `extract_character_snippet()` centers on statutory citations or highest keyword density.
    """
    enc = _DeterministicEncoder()
    retriever = HybridLegalRetriever(encoder=enc, bm25_weight=0.5, dense_weight=0.5)

    long_doc = (
        "Preamble: The general framework applies to commercial entities in the Union. "
        + ("Boilerplate background provision text. " * 8)
        + "Pursuant to Art. 17(1)(a), the data subject shall have the right to obtain from the "
        "controller the erasure of personal data without undue delay."
    )
    start_char, end_char, snippet = retriever.extract_character_snippet(
        long_doc,
        "What does the law state under Art. 17(1)(a) regarding erasure?",
        max_chars=120,
    )
    assert "Art. 17(1)(a)" in snippet
    assert start_char > 50
    assert end_char <= len(long_doc)

    # Verify precedence scaling is reflected in returned `scaled_scores`
    orch = LegalMemoryOrchestrator(encoder=enc, retriever=retriever, value_dim=2)
    r_low = orch.update_memory(
        "Under Art. 17(1)(a), general policy note.",
        [1.0, 0.0],
        authority_rank=1,
    )
    r_high = orch.update_memory(
        "Under Art. 17(1)(a), binding statutory mandate.",
        [0.0, 1.0],
        authority_rank=5,
    )
    q_emb = enc.get_embedding(["Art. 17(1)(a)"])
    keys = torch.cat([r_low.key_vector, r_high.key_vector], dim=0)
    imp = torch.tensor([1.0, 1.0], dtype=torch.float32)
    scaled_scores, _, snippets = retriever.retrieve(
        q_emb,
        keys,
        imp,
        query_text="Art. 17(1)(a)",
        records=[r_low, r_high],
    )
    assert snippets is not None and len(snippets) == 2
    # Higher authority_rank (5 vs 1) receives precedence multiplier boost in scaled_scores
    assert float(scaled_scores[0, 1].item()) > float(scaled_scores[0, 0].item())


def test_rest_and_mcp_parity() -> None:
    """
    Verify REST endpoints (`/v1/compliance/prove`, `/v1/compliance/verify`,
    `/v1/bundle/export`, `/v1/bundle/restore`) and MCP tools (`legal_memory_compliance_prove`,
    `legal_memory_compliance_verify`, `legal_memory_bundle_export`).
    """
    orch = LegalMemoryOrchestrator(
        encoder=_DeterministicEncoder(),
        value_dim=2,
        attestor=AsymmetricAttestationModule(),
    )
    orch.update_memory(
        rule_text="EU AI Act Art. 12 record-keeping obligation.",
        action_vector=[0.0, 1.0],
        authority_rank=4,
        jurisdiction="EU",
        metadata={"record_id": "ai_act_art_12"},
    )

    # 1. MCP tool verification
    mcp = MCPServer(orch)
    tool_names = {t["name"] for t in mcp.list_tools()}
    assert "legal_memory_compliance_prove" in tool_names
    assert "legal_memory_compliance_verify" in tool_names
    assert "legal_memory_bundle_export" in tool_names

    prove_res = mcp.call_tool(
        "legal_memory_compliance_prove",
        {"record_id": "ai_act_art_12", "query": "Art. 12 logging", "min_required_authority_rank": 3},
    )
    prove_payload = json.loads(prove_res["content"][0]["text"])
    proof_obj = prove_payload["proof"]

    verify_res = mcp.call_tool(
        "legal_memory_compliance_verify",
        {"proof": proof_obj, "min_authority_rank": 3, "required_jurisdiction": "EU"},
    )
    assert json.loads(verify_res["content"][0]["text"])["valid"] is True

    # 2. HTTP REST server verification
    server = create_server(orch, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"

        req_prove = urllib.request.Request(
            f"{base_url}/v1/compliance/prove",
            data=json.dumps({"record_id": "ai_act_art_12", "min_required_authority_rank": 4}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_prove) as resp:
            rest_prove = json.loads(resp.read().decode())
        assert "proof" in rest_prove

        req_verify = urllib.request.Request(
            f"{base_url}/v1/compliance/verify",
            data=json.dumps({"proof": rest_prove["proof"], "min_authority_rank": 4}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_verify) as resp:
            rest_verify = json.loads(resp.read().decode())
        assert rest_verify["valid"] is True

        req_export = urllib.request.Request(
            f"{base_url}/v1/bundle/export",
            data=json.dumps({"tenant_id": "default"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_export) as resp:
            exported_bundle = json.loads(resp.read().decode())
        assert exported_bundle["merkle_root"] == orch.merkle_tree.root_hex

        req_restore = urllib.request.Request(
            f"{base_url}/v1/bundle/restore",
            data=json.dumps({"bundle": exported_bundle}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_restore) as resp:
            restored_resp = json.loads(resp.read().decode())
        assert restored_resp["status"] == "RESTORED"
        assert restored_resp["records_restored"] == 1
    finally:
        server.shutdown()
        server.server_close()
