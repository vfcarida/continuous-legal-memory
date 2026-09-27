"""
Multi-Tenant Legal Compliance and Cryptographic Attestation Example.

This script demonstrates:
1. Multi-tenant memory partitioning using the orchestrator.tenant() context manager.
2. Verified isolation preventing cross-tenant information leakage.
3. Asymmetric Ed25519 cryptographic decision attestation.
4. GDPR Article 17 Right-to-be-Forgotten multi-tier erasure.
"""

from __future__ import annotations

from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.security.attestation import AsymmetricAttestationModule


def run_compliance_demo() -> None:
    print("=" * 70)
    print("1. Initializing Multi-Tenant Orchestrator with Ed25519 Attestation")
    print("=" * 70)

    encoder = SemanticMockEncoder(embedding_dim=64)
    attestor = AsymmetricAttestationModule()

    orch = LegalMemoryOrchestrator(
        encoder=encoder,
        value_dim=2,
        attestor=attestor,
        enable_attestation=True,
    )

    print(f"Public Key Hex: {attestor.public_key_hex[:32]}...")

    print("=" * 70)
    print("2. Ingesting Partitioned Tenant Rules")
    print("=" * 70)

    # Ingest for Client Alpha
    with orch.tenant("client_alpha"):
        r_alpha = orch.ingest_rule(
            rule_text="Alpha Policy: Confidential M&A merger negotiations with Partner Corp.",
            action_vector=[1.0, 0.0],
            authority_rank=5,
            personal_data=True,
        )
        print(f"Client Alpha Rule Ingested: ID={r_alpha.record_id}")

    # Ingest for Client Beta
    with orch.tenant("client_beta"):
        r_beta = orch.ingest_rule(
            rule_text="Beta Policy: Public ESG sustainability reporting standards.",
            action_vector=[0.0, 1.0],
            authority_rank=3,
            personal_data=False,
        )
        print(f"Client Beta Rule Ingested:  ID={r_beta.record_id}")

    print("=" * 70)
    print("3. Verifying Cross-Tenant Isolation")
    print("=" * 70)

    query = "What are the confidential M&A merger terms?"

    # Query within Client Beta context
    with orch.tenant("client_beta"):
        res_beta = orch.predict(query)
        print(f"Client Beta Query: '{query}'")
        print(f"Result Rule:       {res_beta.most_relevant_rule}")
        assert res_beta.most_relevant_rule != r_alpha.text
        print("PASS: Client Beta cannot see Client Alpha's confidential M&A data.")

    # Query within Client Alpha context
    with orch.tenant("client_alpha"):
        res_alpha = orch.predict(query)
        print(f"\nClient Alpha Query: '{query}'")
        print(f"Result Rule:        {res_alpha.most_relevant_rule}")
        assert res_alpha.attestation_token is not None
        print(f"Attestation Signature: {res_alpha.attestation_token.integrity_tag[:32]}...")

        # Cryptographically verify the decision token
        valid = attestor.verify_attestation(res_alpha.attestation_token, res_alpha)
        print(f"Ed25519 Cryptographic Signature Verification: {'VALID' if valid else 'INVALID'}")
        assert valid is True

    print("=" * 70)
    print("4. Executing Right-to-Erasure (GDPR Art. 17)")
    print("=" * 70)

    erasure_audit = orch.delete_rule(r_alpha.record_id, tenant_id="client_alpha")
    print(f"Erasure Audit: {erasure_audit}")

    # Verify rule is completely purged from Client Alpha
    with orch.tenant("client_alpha"):
        res_post_erasure = orch.predict(query)
        print(f"Post-Erasure Result: {res_post_erasure.most_relevant_rule}")
        assert res_post_erasure.most_relevant_rule is None

    print("=" * 70)
    print("Multi-Tenant Compliance Demo completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_compliance_demo()
