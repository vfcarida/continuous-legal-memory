"""
Unit and integration tests for Multi-Tenancy Scoping & Isolation (ROAD-01 / SEC-01).

Verifies strict tenant isolation across:
1. Domain models (MemoryRecord, PredictionResult)
2. Episodic Memory ledger queries and index validity
3. LegalMemoryOrchestrator ingestion, prediction, and right-to-erasure
4. SQLite persistence schema and tenant filtering
"""

import tempfile
import unittest
from pathlib import Path

import torch

from continuous_legal_memory.adapters.ollama import OllamaGemmaAdapter
from continuous_legal_memory.core.memory_tiers.episodic_memory import EpisodicMemory
from continuous_legal_memory.domain.models import MemoryRecord, PredictionResult
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.storage.sqlite_store import SqliteMemoryStore


class TestMultiTenancyIsolation(unittest.TestCase):
    """Test suite verifying cross-tenant memory isolation and zero data leakage."""

    def setUp(self) -> None:
        self.encoder = OllamaGemmaAdapter(strict_privacy_mode=True, allow_pseudo_embeddings=True)
        self.orchestrator = LegalMemoryOrchestrator(
            encoder=self.encoder,
            value_dim=2,
            seed=42,
        )

    def test_01_domain_models_tenant_id_support(self) -> None:
        """Verify default and custom tenant_id in MemoryRecord and PredictionResult."""
        rec_default = MemoryRecord(
            text="General Rule",
            key_vector=torch.zeros(1, 768),
            value_vector=torch.zeros(1, 2),
        )
        self.assertEqual(rec_default.tenant_id, "default")

        rec_custom = MemoryRecord(
            text="Confidential Tenant Rule",
            key_vector=torch.zeros(1, 768),
            value_vector=torch.zeros(1, 2),
            tenant_id="tenant_enterprise_xyz",
        )
        self.assertEqual(rec_custom.tenant_id, "tenant_enterprise_xyz")

        pred_res = PredictionResult(
            query="Query",
            predicted_action_vector=[0.0, 1.0],
            tenant_id="tenant_alpha",
        )
        self.assertEqual(pred_res.tenant_id, "tenant_alpha")

    def test_02_episodic_memory_tenant_filtering(self) -> None:
        """Verify EpisodicMemory filters valid records and indices strictly by tenant."""
        mem = EpisodicMemory()
        t1 = mem.append(
            text="Tenant Alpha Rule A",
            key_vector=torch.randn(1, 768),
            value_vector=torch.tensor([[1.0, 0.0]]),
            tenant_id="tenant_alpha",
        )
        t2 = mem.append(
            text="Tenant Beta Rule B",
            key_vector=torch.randn(1, 768),
            value_vector=torch.tensor([[0.0, 1.0]]),
            tenant_id="tenant_beta",
        )
        t3 = mem.append(
            text="Tenant Alpha Rule C",
            key_vector=torch.randn(1, 768),
            value_vector=torch.tensor([[1.0, 0.0]]),
            tenant_id="tenant_alpha",
        )

        # Filter by tenant_alpha
        alpha_records = mem.get_valid_records(tenant_id="tenant_alpha")
        alpha_indices = mem.get_valid_indices(tenant_id="tenant_alpha")
        self.assertEqual(len(alpha_records), 2)
        self.assertEqual(alpha_records[0].record_id, t1.record_id)
        self.assertEqual(alpha_records[1].record_id, t3.record_id)
        self.assertEqual(alpha_indices, [0, 2])

        # Filter by tenant_beta
        beta_records = mem.get_valid_records(tenant_id="tenant_beta")
        beta_indices = mem.get_valid_indices(tenant_id="tenant_beta")
        self.assertEqual(len(beta_records), 1)
        self.assertEqual(beta_records[0].record_id, t2.record_id)
        self.assertEqual(beta_indices, [1])

        # Unfiltered returns all
        all_records = mem.get_valid_records(tenant_id=None)
        self.assertEqual(len(all_records), 3)

    def test_03_orchestrator_multi_tenant_query_isolation(self) -> None:
        """
        Verify that queries scoped to tenant_alpha NEVER retrieve rules or actions
        belonging to tenant_beta, preventing cross-tenant leakage.
        """
        # Ingest confidential directive for Tenant Alpha
        rec_alpha = self.orchestrator.update_memory(
            rule_text="Tenant Alpha Secret: Authorize patent licensing payment immediately.",
            action_vector=[1.0, 0.0],
            tenant_id="tenant_alpha",
        )

        # Ingest confidential directive for Tenant Beta
        rec_beta = self.orchestrator.update_memory(
            rule_text="Tenant Beta Secret: Deny all patent licensing claims indefinitely.",
            action_vector=[0.0, 1.0],
            tenant_id="tenant_beta",
        )

        self.assertEqual(rec_alpha.tenant_id, "tenant_alpha")
        self.assertEqual(rec_beta.tenant_id, "tenant_beta")

        # Query 1: Tenant Alpha queries for patent licensing
        res_alpha = self.orchestrator.predict(
            "Patent licensing claims payment authorization",
            tenant_id="tenant_alpha",
        )
        self.assertEqual(res_alpha.tenant_id, "tenant_alpha")
        self.assertEqual(res_alpha.most_relevant_rule, rec_alpha.text)
        self.assertGreater(res_alpha.predicted_action_vector[0], res_alpha.predicted_action_vector[1])

        # Query 2: Tenant Beta queries for the exact same text -> MUST retrieve Tenant Beta rule
        res_beta = self.orchestrator.predict(
            "Patent licensing claims payment authorization",
            tenant_id="tenant_beta",
        )
        self.assertEqual(res_beta.tenant_id, "tenant_beta")
        self.assertEqual(res_beta.most_relevant_rule, rec_beta.text)
        self.assertGreater(res_beta.predicted_action_vector[1], res_beta.predicted_action_vector[0])

        # Query 3: Non-existent Tenant Gamma queries -> Must return zero vector and None rule
        res_gamma = self.orchestrator.predict(
            "Patent licensing claims payment authorization",
            tenant_id="tenant_gamma",
        )
        self.assertEqual(res_gamma.predicted_action_vector, [0.0, 0.0])
        self.assertIsNone(res_gamma.most_relevant_rule)

    def test_04_delete_rule_tenant_authorization(self) -> None:
        """Verify right-to-erasure validates tenant ownership when tenant_id is provided."""
        rec = self.orchestrator.update_memory(
            rule_text="Tenant X confidential directive.",
            action_vector=[1.0, 0.0],
            tenant_id="tenant_x",
        )

        # Unauthorized attempt: Tenant Y attempts to delete Tenant X's rule
        with self.assertRaises(KeyError) as ctx:
            self.orchestrator.delete_rule(rec.record_id, tenant_id="tenant_y")
        self.assertIn("does not belong to tenant 'tenant_y'", str(ctx.exception))

        # Authorized deletion by Tenant X succeeds
        audit = self.orchestrator.delete_rule(rec.record_id, tenant_id="tenant_x")
        self.assertEqual(audit["status"], "ERASED")
        self.assertEqual(audit["record_id"], rec.record_id)

    def test_05_sqlite_store_multi_tenant_roundtrip(self) -> None:
        """Verify SqliteMemoryStore correctly persists, indexes, and reloads tenant_id."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "multi_tenant.db"
            store = SqliteMemoryStore(db_path)

            orch1 = LegalMemoryOrchestrator(
                encoder=self.encoder,
                value_dim=2,
                seed=42,
            )
            orch1.update_memory("Rule for tenant 1", [1.0, 0.0], tenant_id="tenant_1")
            orch1.update_memory("Rule for tenant 2", [0.0, 1.0], tenant_id="tenant_2")

            store.save_orchestrator(orch1)

            # Check store-level filtering
            t1_records = store.get_records(tenant_id="tenant_1")
            t2_records = store.get_records(tenant_id="tenant_2")
            self.assertEqual(len(t1_records), 1)
            self.assertEqual(len(t2_records), 1)
            self.assertEqual(t1_records[0].tenant_id, "tenant_1")
            self.assertEqual(t2_records[0].tenant_id, "tenant_2")

            # Load into new orchestrator and test query isolation
            orch2 = LegalMemoryOrchestrator(
                encoder=self.encoder,
                value_dim=2,
            )
            store.load_into_orchestrator(orch2)

            res1 = orch2.predict("Rule for tenant 1", tenant_id="tenant_1")
            self.assertEqual(res1.most_relevant_rule, "Rule for tenant 1")

            res2_leak = orch2.predict("Rule for tenant 1", tenant_id="tenant_2")
            # Tenant 2 should NOT retrieve Tenant 1's rule!
            self.assertNotEqual(res2_leak.most_relevant_rule, "Rule for tenant 1")


if __name__ == "__main__":
    unittest.main()
