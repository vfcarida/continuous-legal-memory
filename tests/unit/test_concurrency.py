"""
Unit and stress tests for Multi-Threading and Concurrency Safety (Testing Gap 3).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.storage.sqlite_store import SqliteMemoryStore


def test_concurrent_multi_tenant_queries(offline_encoder: BaseEncoderPort) -> None:
    """Verify thread-safety of concurrent predictions across multiple isolated tenants."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    # Ingest distinct rules per tenant
    tenants = [f"tenant_{i}" for i in range(5)]
    for i, tenant in enumerate(tenants):
        orch.update_memory(
            f"Statutory rule for {tenant}: Mandatory procedure code {i * 100}.",
            [1.0 if i % 2 == 0 else 0.0, 0.0 if i % 2 == 0 else 1.0],
            authority_rank=i + 1,
            tenant_id=tenant,
        )

    def query_tenant(t_id: str) -> tuple[str, str | None]:
        res = orch.predict(f"What is the procedure code for {t_id}?", tenant_id=t_id)
        return t_id, res.most_relevant_rule

    # Execute concurrent predictions across 10 threads
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(query_tenant, tenants[i % len(tenants)]) for i in range(30)]
        for f in as_completed(futures):
            t_id, rule = f.result()
            assert rule is not None
            assert t_id in rule


def test_concurrent_ingestion_and_queries(offline_encoder: BaseEncoderPort) -> None:
    """Verify concurrent writers and readers operate without deadlocks or corrupt memory buffers."""
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    # Seed baseline rule
    orch.update_memory("Foundational statute: Commercial integrity must be maintained.", [1.0, 0.0])

    def write_task(idx: int) -> None:
        orch.update_memory(
            f"Amendment {idx}: Additional governance requirements for branch {idx}.",
            [0.0, 1.0],
            authority_rank=2,
            tenant_id=f"branch_{idx % 3}",
        )

    def read_task(idx: int) -> str | None:
        res = orch.predict("Commercial integrity governance", tenant_id=f"branch_{idx % 3}")
        return res.most_relevant_rule

    with ThreadPoolExecutor(max_workers=8) as executor:
        write_futures = [executor.submit(write_task, i) for i in range(15)]
        read_futures = [executor.submit(read_task, i) for i in range(15)]

        for f in as_completed(write_futures + read_futures):
            # All concurrent tasks must complete cleanly without throwing exceptions
            f.result()

    assert len(orch.episodic_memory.get_records()) == 16


def test_concurrent_sqlite_persistence_reads(offline_encoder: BaseEncoderPort, tmp_path: Path) -> None:
    """Verify concurrent reads from SQLite persistence store across multiple threads."""
    db_file = tmp_path / "concurrent_store.db"
    orch = LegalMemoryOrchestrator(encoder=offline_encoder, value_dim=2, engine_mode="structured")

    for i in range(10):
        orch.update_memory(f"Statute {i}: Legal clause content for testing.", [1.0, 0.0], tenant_id=f"tenant_{i % 2}")

    orch.save_to_disk(db_file)

    def load_and_query(worker_id: int) -> int:
        store = SqliteMemoryStore(db_file)
        tenant = f"tenant_{worker_id % 2}"
        records = store.get_records(tenant_id=tenant)
        return len(records)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(load_and_query, i) for i in range(18)]
        for f in as_completed(futures):
            count = f.result()
            assert count == 5  # Each tenant has 5 records
