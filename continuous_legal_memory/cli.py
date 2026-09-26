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
    query_parser.add_argument("--at-time", type=str, default=None, help="ISO 8601 timestamp for temporal evaluation.")
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
    result = orch.predict(query_text=args.query_text, at_time=at_time, tenant_id=args.tenant)

    output: dict[str, Any] = {
        "query": result.query,
        "predicted_action_vector": result.predicted_action_vector,
        "source_tier": result.source_tier.value if (result.source_tier is not None and hasattr(result.source_tier, "value")) else str(result.source_tier),
        "most_relevant_rule": result.most_relevant_rule,
        "confidence": result.confidence,
        "tenant_id": result.tenant_id,
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
