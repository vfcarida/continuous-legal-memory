"""
Command-Line Interface (CLI) for Continuous Legal Memory (CLM).

Provides administrative utilities for running the REST API server, ingesting legal rules,
querying memory state, executing GDPR right-to-erasure, exploring semantic graph associations,
and running evaluation benchmarks.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from continuous_legal_memory.adapters.mock_encoder import SemanticMockEncoder
from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.mcp import MCPServer
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.security.attestation import (
    AsymmetricAttestationModule,
    MemoryAttestationToken,
)
from continuous_legal_memory.server import run_server

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clm",
        description="Continuous Legal Memory (CLM) - Cognitive Memory Framework for Legal AI Agents.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute.")

    # 1. clm serve
    serve_parser = subparsers.add_parser("serve", help="Launch the REST API HTTP server and Prometheus exporter.")
    serve_parser.add_argument("--host", default="0.0.0.0", help="HTTP server host address (default: 0.0.0.0).")
    serve_parser.add_argument("--port", type=int, default=8000, help="HTTP server port (default: 8000).")
    serve_parser.add_argument("--db-path", type=str, default=None, help="Path to persistent SQLite database.")
    serve_parser.add_argument("--engine-mode", choices=["structured", "hybrid"], default="structured", help="Core retrieval engine mode.")
    serve_parser.add_argument("--strict-privacy", action="store_true", help="Enforce local-only offline privacy mode.")
    serve_parser.add_argument("--enable-attestation", action="store_true", help="Enable cryptographic attestation tokens.")
    serve_parser.add_argument("--enable-telemetry", action="store_true", help="Enable operational telemetry logging.")
    serve_parser.add_argument("--mock-encoder", action="store_true", help="Use lightweight SemanticMockEncoder for testing/offline environments.")

    # 2. clm ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a legal rule or a JSONL file of rules into memory.")
    ingest_parser.add_argument("--text", type=str, help="Text content of the legal rule or statute.")
    ingest_parser.add_argument("--action", type=float, nargs="+", help="Target decision vector (e.g. 1.0 0.0).")
    ingest_parser.add_argument("--file", type=str, help="Path to JSONL file containing rule records.")
    ingest_parser.add_argument("--tenant", default="default", help="Tenant identifier (default: default).")
    ingest_parser.add_argument("--authority-rank", type=int, default=1, help="Hierarchical authority rank (1-10).")
    ingest_parser.add_argument("--jurisdiction", type=str, default=None, help="Jurisdiction label.")
    ingest_parser.add_argument("--personal-data", action="store_true", help="Flag indicating personal data requiring right-to-erasure.")
    ingest_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    ingest_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder for fast testing.")

    # 3. clm query
    query_parser = subparsers.add_parser("query", help="Query the cognitive memory state for decision vectors.")
    query_parser.add_argument("query_text", type=str, help="Inquiry or legal scenario text to query.")
    query_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    query_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    query_parser.add_argument("--at-time", type=str, default=None, help="ISO 8601 timestamp for valid-time evaluation.")
    query_parser.add_argument("--as-of-tx", type=str, default=None, help="ISO 8601 transaction-time cutoff for bi-temporal 'as-of' replay.")
    query_parser.add_argument("--jurisdiction", type=str, default=None, help="Optional jurisdiction filter.")
    query_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder for fast testing.")

    # 4. clm erase
    erase_parser = subparsers.add_parser("erase", help="Execute GDPR Art. 17 / Right-to-Erasure across all memory tiers.")
    erase_parser.add_argument("record_id", type=str, help="Record ID of the directive to erase.")
    erase_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    erase_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    erase_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 5. clm associate
    assoc_parser = subparsers.add_parser("associate", help="Discover multi-hop associative statutes via HippoRAG Personalized PageRank.")
    assoc_parser.add_argument("seed_nodes", nargs="+", help="Seed node IDs to start random walks from.")
    assoc_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    assoc_parser.add_argument("--damping", type=float, default=0.85, help="Teleportation damping factor (default: 0.85).")
    assoc_parser.add_argument("--max-results", type=int, default=10, help="Maximum related entities to return.")
    assoc_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    assoc_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 6. clm verify
    verify_parser = subparsers.add_parser("verify", help="Verify cryptographic attestation token integrity and state proof.")
    verify_parser.add_argument("--token-file", type=str, required=True, help="Path to JSON file containing attestation token.")
    verify_parser.add_argument("--query", type=str, required=True, help="Canonical query string.")
    verify_parser.add_argument("--action", type=float, nargs="+", required=True, help="Canonical decision vector.")
    verify_parser.add_argument("--rule", type=str, default=None, help="Most relevant retrieved rule text.")
    verify_parser.add_argument("--public-key", type=str, default=None, help="Optional Ed25519 public key hex string.")

    # 7. clm evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Execute LegalBench-RAG evaluation benchmark suite.")
    eval_parser.add_argument("--benchmark", type=str, default="tests/fixtures/synthetic_legal_benchmark.jsonl", help="Benchmark JSONL dataset.")
    eval_parser.add_argument("--engine-mode", choices=["structured", "hybrid"], default="structured", help="Retrieval engine mode.")
    eval_parser.add_argument("--output", type=str, default=None, help="Optional JSON output filepath for benchmark metrics.")

    # 8. clm mcp
    mcp_parser = subparsers.add_parser("mcp", help="Launch the Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio.")
    mcp_parser.add_argument("--db-path", type=str, default=None, help="Optional path to persistent SQLite database.")
    mcp_parser.add_argument("--engine-mode", choices=["structured", "hybrid"], default="structured", help="Core retrieval engine mode.")
    mcp_parser.add_argument("--enable-attestation", action="store_true", help="Enable cryptographic attestation tokens.")
    mcp_parser.add_argument("--mock-encoder", action="store_true", help="Use lightweight SemanticMockEncoder for offline execution.")

    # 9. clm audit
    audit_parser = subparsers.add_parser("audit", help="Audit active norms and semantic graph for deontic cycles and antinomies.")
    audit_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    audit_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    audit_parser.add_argument("--at-time", type=str, default=None, help="Optional ISO 8601 timestamp.")
    audit_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 10. clm merkle-proof
    merkle_parser = subparsers.add_parser("merkle-proof", help="Generate and verify a 256-bit Sparse Merkle Tree inclusion or erasure proof.")
    merkle_parser.add_argument("record_id", type=str, help="Rule or record ID to prove.")
    merkle_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    merkle_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 11. clm compliance-proof
    zk_parser = subparsers.add_parser("compliance-proof", help="Generate a Selective-Disclosure / Zero-Knowledge Normative Compliance Proof.")
    zk_parser.add_argument("record_id", type=str, help="Rule or record ID to prove.")
    zk_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    zk_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    zk_parser.add_argument("--min-authority-rank", type=int, default=1, help="Minimum required authority rank.")
    zk_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 12. clm export-bundle
    export_parser = subparsers.add_parser("export-bundle", help="Export a cryptographically signed portable compliance bundle (.clm.bundle.json).")
    export_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    export_parser.add_argument("--output", type=str, required=True, help="Output file path for .clm.bundle.json.")
    export_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    export_parser.add_argument("--redact-personal-data", action="store_true", help="Redact personal data plaintext in bundle.")
    export_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 13. clm restore-bundle
    restore_parser = subparsers.add_parser("restore-bundle", help="Verify and restore a signed portable compliance bundle into SQLite storage.")
    restore_parser.add_argument("--bundle-file", type=str, required=True, help="Input path to .clm.bundle.json file.")
    restore_parser.add_argument("--db-path", type=str, required=True, help="Target path to SQLite memory database.")
    restore_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 14. clm simulate-rule
    sim_parser = subparsers.add_parser("simulate-rule", help="Run a non-mutating Pre-Flight Standard Deontic Logic (KD) simulation.")
    sim_parser.add_argument("--text", type=str, required=True, help="Candidate legal rule text.")
    sim_parser.add_argument("--action", type=float, nargs="+", required=True, help="Candidate target decision vector.")
    sim_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    sim_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    sim_parser.add_argument("--authority-rank", type=int, default=1, help="Candidate authority rank.")
    sim_parser.add_argument("--jurisdiction", type=str, default=None, help="Candidate jurisdiction.")
    sim_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 15. clm key-list
    klist_parser = subparsers.add_parser("key-list", help="List all cryptographic signing keys and lifecycle statuses in CryptographicKeyRing.")
    klist_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    klist_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 16. clm key-rotate
    krot_parser = subparsers.add_parser("key-rotate", help="Atomically rotate the active signing key in CryptographicKeyRing.")
    krot_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    krot_parser.add_argument("--key-id", type=str, default=None, help="Optional new key identifier.")
    krot_parser.add_argument("--algorithm", choices=["Ed25519", "HMAC-SHA256", "PQ-Hybrid-Ed25519-SHA256"], default=None, help="Signing algorithm.")
    krot_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 17. clm key-revoke
    krev_parser = subparsers.add_parser("key-revoke", help="Permanently revoke a compromised signing key ID in CryptographicKeyRing.")
    krev_parser.add_argument("key_id", type=str, help="Key ID to revoke.")
    krev_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    krev_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 18. clm repair-rule
    rep_parser = subparsers.add_parser("repair-rule", help="Synthesize and verify legislative/contractual drafting repair proposals for a candidate rule.")
    rep_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    rep_parser.add_argument("--text", type=str, required=True, help="Candidate legal rule text.")
    rep_parser.add_argument("--action", type=float, nargs="+", default=None, help="Candidate target action vector.")
    rep_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    rep_parser.add_argument("--authority-rank", type=int, default=1, help="Candidate authority rank.")
    rep_parser.add_argument("--jurisdiction", type=str, default=None, help="Candidate jurisdiction.")
    rep_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 19. clm tee-quote
    tee_parser = subparsers.add_parser("tee-quote", help="Generate a hardware-anchored TEE Remote Attestation Quote (AWS_NITRO or INTEL_TDX).")
    tee_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    tee_parser.add_argument("--platform", choices=["AWS_NITRO", "INTEL_TDX"], default="AWS_NITRO", help="TEE hardware platform.")
    tee_parser.add_argument("--nonce", type=str, default=None, help="Optional anti-replay challenge nonce.")
    tee_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    tee_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 20. clm export-smtlib
    smt_parser = subparsers.add_parser("export-smtlib", help="Export active tenant norms as a formal SMT-LIB 2.6 specification for Z3 / cvc5.")
    smt_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    smt_parser.add_argument("--query", type=str, default=None, help="Optional query to bind DefeasibleProofTrace into SMT-LIB.")
    smt_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    smt_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 21. clm simulate-timeline
    tl_parser = subparsers.add_parser("simulate-timeline", help="Simulate bi-temporal regulatory horizon transitions across time.")
    tl_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    tl_parser.add_argument("--query", type=str, default=None, help="Optional query to evaluate at each epoch.")
    tl_parser.add_argument("--start-time", type=str, default=None, help="ISO-8601 start timestamp.")
    tl_parser.add_argument("--end-time", type=str, default=None, help="ISO-8601 end timestamp.")
    tl_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    tl_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 22. clm export-verifier
    ver_parser = subparsers.add_parser("export-verifier", help="Export a self-contained HTML5 + WebCrypto standalone audit verifier.")
    ver_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    ver_parser.add_argument("--output", type=str, default=None, help="Optional path to write standalone HTML file.")
    ver_parser.add_argument("--query", type=str, default=None, help="Optional query for embedded SMT-LIB specification.")
    ver_parser.add_argument("--redact-personal-data", action="store_true", help="Redact personal data in embedded bundle.")
    ver_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    ver_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 23. clm timestamp-root
    tst_parser = subparsers.add_parser("timestamp-root", help="Issue an RFC 3161 cryptographic TimestampToken over the tenant Merkle root.")
    tst_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    tst_parser.add_argument("--nonce", type=str, default=None, help="Optional anti-replay nonce.")
    tst_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    tst_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 24. clm gossip-audit
    gos_parser = subparsers.add_parser("gossip-audit", help="Audit transparency log witness checkpoints for split-view equivocation.")
    gos_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    gos_parser.add_argument("--checkpoints-file", type=str, required=True, help="Path to JSON file containing a list of GossipCheckpoint dicts.")
    gos_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    gos_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    # 25. clm amendment-impact
    amd_parser = subparsers.add_parser("amendment-impact", help="Analyze clause-level statutory amendment diffs and downstream obligation impacts.")
    amd_parser.add_argument("rule_id", type=str, help="Target rule ID being amended.")
    amd_parser.add_argument("--db-path", type=str, required=True, help="Path to SQLite memory database.")
    amd_parser.add_argument("--text", type=str, required=True, help="Proposed or enacted amended statutory text.")
    amd_parser.add_argument("--action", type=float, nargs="+", default=None, help="Optional amended action vector.")
    amd_parser.add_argument("--authority-rank", type=int, default=None, help="Optional amended authority rank.")
    amd_parser.add_argument("--tenant", default="default", help="Tenant scope.")
    amd_parser.add_argument("--mock-encoder", action="store_true", help="Use mock encoder.")

    return parser


def _init_orchestrator(
    db_path: str | None = None,
    engine_mode: str = "structured",
    strict_privacy: bool = False,
    enable_attestation: bool = False,
    enable_telemetry: bool = False,
    mock_encoder: bool = False,
) -> LegalMemoryOrchestrator:
    encoder = SemanticMockEncoder() if mock_encoder else None
    orch = LegalMemoryOrchestrator(
        encoder=encoder,
        engine_mode=engine_mode,
        strict_privacy_mode=strict_privacy,
        enable_attestation=enable_attestation,
        enable_telemetry=enable_telemetry,
    )
    if db_path and Path(db_path).exists():
        orch.load_from_disk(db_path)
    return orch


def _cmd_serve(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(
        db_path=args.db_path,
        engine_mode=args.engine_mode,
        strict_privacy=args.strict_privacy,
        enable_attestation=args.enable_attestation,
        enable_telemetry=args.enable_telemetry,
        mock_encoder=args.mock_encoder,
    )
    print(f"Starting Continuous Legal Memory REST server on http://{args.host}:{args.port}")
    print(f"Engine mode: {args.engine_mode}, Persistent DB: {args.db_path or 'In-Memory'}")
    run_server(orch, host=args.host, port=args.port)
    if args.db_path:
        orch.save_to_disk(args.db_path)
        print(f"State saved to database: {args.db_path}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: JSONL file not found at {args.file}", file=sys.stderr)
            return 1

        ingested = 0
        with file_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("type") == "rule":
                    orch.update_memory(
                        rule_text=item["text"],
                        action_vector=item["action"],
                        authority_rank=item.get("authority_rank", 1),
                        jurisdiction=item.get("jurisdiction"),
                        personal_data=item.get("personal_data", False),
                        tenant_id=item.get("tenant", "default"),
                        metadata={"node_id": item.get("rule_id")},
                    )
                    ingested += 1
        orch.save_to_disk(args.db_path)
        print(f"Successfully ingested {ingested} rules from {args.file} into {args.db_path}")
        return 0

    if not args.text or not args.action:
        print("Error: Either --file or both --text and --action must be specified.", file=sys.stderr)
        return 1

    rec = orch.update_memory(
        rule_text=args.text,
        action_vector=args.action,
        authority_rank=args.authority_rank,
        jurisdiction=args.jurisdiction,
        personal_data=args.personal_data,
        tenant_id=args.tenant,
    )
    orch.save_to_disk(args.db_path)
    print(json.dumps({
        "status": "INGESTED",
        "record_id": rec.record_id,
        "tenant_id": rec.tenant_id,
        "authority_rank": rec.authority_rank,
        "db_path": args.db_path,
    }, indent=2))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    at_time = datetime.fromisoformat(args.at_time) if args.at_time else None
    as_of_tx = datetime.fromisoformat(args.as_of_tx) if getattr(args, "as_of_tx", None) else None
    jurisdiction = getattr(args, "jurisdiction", None)
    result = orch.predict(
        query_text=args.query_text,
        at_time=at_time,
        tenant_id=args.tenant,
        jurisdiction=jurisdiction,
        as_of_transaction_time=as_of_tx,
    )

    output: dict[str, Any] = {
        "query": result.query,
        "predicted_action_vector": result.predicted_action_vector,
        "source_tier": result.source_tier.value if (result.source_tier is not None and hasattr(result.source_tier, "value")) else str(result.source_tier),
        "most_relevant_rule": result.most_relevant_rule,
        "confidence": result.confidence,
        "tenant_id": result.tenant_id,
        "merkle_root": result.merkle_root,
        "proof_trace": result.proof_trace,
    }
    if result.attestation_token:
        output["attestation_token"] = {
            "state_hash": result.attestation_token.state_hash,
            "integrity_tag": result.attestation_token.integrity_tag,
            "algorithm": result.attestation_token.algorithm,
        }
    print(json.dumps(output, indent=2))
    return 0


def _cmd_erase(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    try:
        audit = orch.delete_rule(record_id=args.record_id, tenant_id=args.tenant)
        orch.save_to_disk(args.db_path)
        print(json.dumps(audit, indent=2))
        return 0
    except KeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _cmd_associate(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    results = orch.associate_statutes(
        seed_nodes=args.seed_nodes,
        damping=args.damping,
        max_results=args.max_results,
        tenant_id=args.tenant,
    )
    formatted = [
        {
            "node_id": node.node_id,
            "label": node.label,
            "entity_type": node.entity_type.value if hasattr(node.entity_type, "value") else str(node.entity_type),
            "score": score,
        }
        for node, score in results
    ]
    print(json.dumps({"results": formatted, "total": len(formatted)}, indent=2))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    token_path = Path(args.token_file)
    if not token_path.exists():
        print(f"Error: Token file not found at {args.token_file}", file=sys.stderr)
        return 1

    with token_path.open(encoding="utf-8") as f:
        tok_data = json.load(f)

    tok = MemoryAttestationToken(
        timestamp=datetime.fromisoformat(tok_data["timestamp"]),
        state_hash=tok_data["state_hash"],
        integrity_tag=tok_data["integrity_tag"],
        key_id=tok_data.get("key_id", "default"),
        algorithm=tok_data.get("algorithm", "Ed25519"),
        public_key=tok_data.get("public_key") or args.public_key,
    )

    pred = PredictionResult(
        query=args.query,
        predicted_action_vector=args.action,
        most_relevant_rule=args.rule,
    )

    verifier = AsymmetricAttestationModule(public_key=args.public_key or tok.public_key)
    valid = verifier.verify_attestation(tok, pred, retrieved_text=args.rule)

    print(json.dumps({
        "valid": valid,
        "algorithm": tok.algorithm,
        "state_hash": tok.state_hash,
    }, indent=2))
    return 0 if valid else 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from continuous_legal_memory.evaluation.legalbench_eval import run_legalbench_evaluation

    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        print(f"Error: Benchmark dataset not found at {args.benchmark}", file=sys.stderr)
        return 1

    print(f"Executing LegalBench-RAG evaluation using engine_mode='{args.engine_mode}'...")
    report = run_legalbench_evaluation(
        benchmark_path=bench_path,
        engine_mode=args.engine_mode,
        use_mock_encoder=True,
    )

    print("\n--- LegalBench-RAG Evaluation Results ---")
    print(f"Total Scenarios Evaluated: {report.total_scenarios}")
    print(f"Overall Decision Accuracy: {report.overall_accuracy:.2%}")
    print(f"Mean Temporal Precision:   {report.mean_temporal_precision:.2%}")
    print(f"Attestation Coverage:      {report.attestation_coverage:.2%}")
    print(f"Erasure Sub-10ms Ratio:    {report.erasure_latency_p99_ms}ms (P99)")

    print("\nPer-Scenario Accuracy Breakdown:")
    for scn, acc in sorted(report.scenario_accuracies.items()):
        print(f"  - {scn:<25}: {acc:.2%}")

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({
                "total_scenarios": report.total_scenarios,
                "overall_accuracy": report.overall_accuracy,
                "mean_temporal_precision": report.mean_temporal_precision,
                "attestation_coverage": report.attestation_coverage,
                "erasure_latency_p99_ms": report.erasure_latency_p99_ms,
                "scenario_accuracies": report.scenario_accuracies,
            }, f, indent=2)
        print(f"\nSaved full benchmark report to {args.output}")

    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(
        db_path=args.db_path,
        engine_mode=args.engine_mode,
        enable_attestation=args.enable_attestation,
        mock_encoder=args.mock_encoder,
    )
    server = MCPServer(orch)
    server.serve_stdio()
    if args.db_path:
        orch.save_to_disk(args.db_path)
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    at_time = datetime.fromisoformat(args.at_time) if args.at_time else None
    report = orch.audit_deontic_consistency(tenant_id=args.tenant, at_time=at_time)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_merkle_proof(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    proof = orch.merkle_tree.generate_proof(args.record_id, attestor=orch.attestor)
    verified = orch.verify_merkle_proof(proof, expected_root_hash=orch.merkle_tree.root_hex)
    out = proof.to_dict()
    out["verified"] = verified
    print(json.dumps(out, indent=2))
    return 0 if verified else 1


def _cmd_compliance_proof(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    bundle = orch.generate_compliance_proof(
        record_id=args.record_id,
        min_required_authority_rank=args.min_authority_rank,
        tenant_id=args.tenant,
    )
    verified = orch.verify_compliance_proof(bundle.proof)
    out = bundle.proof.to_dict()
    out["verified"] = verified
    print(json.dumps(out, indent=2))
    return 0 if verified else 1


def _cmd_export_bundle(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    bundle = orch.export_compliance_bundle(
        tenant_id=args.tenant,
        redact_personal_data=args.redact_personal_data,
    )
    out_path = Path(args.output)
    out_path.write_text(bundle.to_json(), encoding="utf-8")
    print(json.dumps({
        "status": "EXPORTED",
        "output": str(out_path),
        "tenant_id": bundle.tenant_id,
        "merkle_root": bundle.merkle_root,
        "records_count": len(bundle.records),
    }, indent=2))
    return 0


def _cmd_restore_bundle(args: argparse.Namespace) -> int:
    bundle_path = Path(args.bundle_file)
    if not bundle_path.exists():
        print(f"Error: Bundle file not found at {args.bundle_file}", file=sys.stderr)
        return 1
    bundle_data = json.loads(bundle_path.read_text(encoding="utf-8"))
    orch = _init_orchestrator(db_path=None, mock_encoder=args.mock_encoder)
    parsed = orch.restore_compliance_bundle(bundle_data)
    orch.save_to_disk(args.db_path)
    print(json.dumps({
        "status": "RESTORED",
        "db_path": args.db_path,
        "tenant_id": parsed.tenant_id,
        "merkle_root": parsed.merkle_root,
        "records_restored": len(parsed.records),
    }, indent=2))
    return 0


def _cmd_simulate_rule(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    report = orch.simulate_rule_ingestion(
        rule_text=args.text,
        action_vector=args.action,
        authority_rank=args.authority_rank,
        jurisdiction=args.jurisdiction,
        tenant_id=args.tenant,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_key_list(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    print(json.dumps({
        "active_key_id": orch.keyring.active_key_id,
        "keys": orch.list_signing_keys(),
    }, indent=2))
    return 0


def _cmd_key_rotate(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    meta = orch.rotate_signing_key(new_key_id=args.key_id, algorithm=args.algorithm)
    orch.save_to_disk(args.db_path)
    print(json.dumps(meta.to_dict(), indent=2))
    return 0


def _cmd_key_revoke(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    meta = orch.revoke_signing_key(args.key_id)
    orch.save_to_disk(args.db_path)
    print(json.dumps(meta.to_dict(), indent=2))
    return 0


def _cmd_repair_rule(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    report = orch.propose_norm_repairs(
        rule_text=args.text,
        action_vector=args.action,
        authority_rank=args.authority_rank,
        jurisdiction=args.jurisdiction,
        tenant_id=args.tenant,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_tee_quote(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    quote = orch.generate_tee_quote(
        nonce=args.nonce,
        platform=args.platform,
        tenant_id=args.tenant,
    )
    print(json.dumps(quote.to_dict(), indent=2))
    return 0


def _cmd_export_smtlib(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    bundle = orch.export_smtlib(
        query_text=args.query,
        tenant_id=args.tenant,
    )
    print(json.dumps(bundle.to_dict(), indent=2))
    return 0


def _cmd_simulate_timeline(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    st_dt = datetime.fromisoformat(args.start_time) if args.start_time else None
    et_dt = datetime.fromisoformat(args.end_time) if args.end_time else None
    report = orch.simulate_temporal_horizon(
        query_text=args.query,
        start_time=st_dt,
        end_time=et_dt,
        tenant_id=args.tenant,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_export_verifier(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    artifact = orch.export_standalone_verifier(
        tenant_id=args.tenant,
        redact_personal_data=args.redact_personal_data,
        query_text=args.query,
    )
    if args.output:
        Path(args.output).write_text(artifact.html_content, encoding="utf-8")
    print(json.dumps(artifact.to_dict(), indent=2))
    return 0


def _cmd_timestamp_root(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    tst = orch.issue_timestamp_token(
        tenant_id=args.tenant,
        nonce=args.nonce,
    )
    print(json.dumps(tst.to_dict(), indent=2))
    return 0


def _cmd_gossip_audit(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    raw = json.loads(Path(args.checkpoints_file).read_text(encoding="utf-8"))
    checkpoints = raw if isinstance(raw, list) else raw.get("checkpoints", [])
    report = orch.audit_transparency_gossip(
        checkpoints=checkpoints,
        tenant_id=args.tenant,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_amendment_impact(args: argparse.Namespace) -> int:
    orch = _init_orchestrator(db_path=args.db_path, mock_encoder=args.mock_encoder)
    report = orch.analyze_statutory_amendment(
        target_rule_id=args.rule_id,
        amended_text=args.text,
        amended_action_vector=args.action,
        amended_authority_rank=args.authority_rank,
        tenant_id=args.tenant,
    )
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    handlers = {
        "serve": _cmd_serve,
        "ingest": _cmd_ingest,
        "query": _cmd_query,
        "erase": _cmd_erase,
        "associate": _cmd_associate,
        "verify": _cmd_verify,
        "evaluate": _cmd_evaluate,
        "mcp": _cmd_mcp,
        "audit": _cmd_audit,
        "merkle-proof": _cmd_merkle_proof,
        "compliance-proof": _cmd_compliance_proof,
        "export-bundle": _cmd_export_bundle,
        "restore-bundle": _cmd_restore_bundle,
        "simulate-rule": _cmd_simulate_rule,
        "key-list": _cmd_key_list,
        "key-rotate": _cmd_key_rotate,
        "key-revoke": _cmd_key_revoke,
        "repair-rule": _cmd_repair_rule,
        "tee-quote": _cmd_tee_quote,
        "export-smtlib": _cmd_export_smtlib,
        "simulate-timeline": _cmd_simulate_timeline,
        "export-verifier": _cmd_export_verifier,
        "timestamp-root": _cmd_timestamp_root,
        "gossip-audit": _cmd_gossip_audit,
        "amendment-impact": _cmd_amendment_impact,
    }

    handler = handlers.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except Exception as exc:
        print(f"Execution error: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
