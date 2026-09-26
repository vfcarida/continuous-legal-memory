"""
Unit tests for Continuous Legal Memory CLI utility.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from continuous_legal_memory.cli import main
from continuous_legal_memory.domain.models import PredictionResult
from continuous_legal_memory.security.attestation import AsymmetricAttestationModule


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Continuous Legal Memory (CLM)" in captured.out


def test_cli_ingest_query_erase_lifecycle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = str(tmp_path / "cli_test.db")

    # 1. Ingest single rule
    ret = main([
        "ingest",
        "--text", "Federal Statute 404: Consumer credit history retention rules.",
        "--action", "1.0", "0.0",
        "--authority-rank", "7",
        "--jurisdiction", "FEDERAL",
        "--db-path", db_path,
        "--mock-encoder",
    ])
    assert ret == 0
    out = capsys.readouterr().out
    ingest_json = json.loads(out)
    assert ingest_json["status"] == "INGESTED"
    record_id = ingest_json["record_id"]

    # 2. Query
    ret = main([
        "query",
        "Consumer credit history guidelines",
        "--db-path", db_path,
        "--mock-encoder",
    ])
    assert ret == 0
    q_out = capsys.readouterr().out
    query_json = json.loads(q_out)
    assert "predicted_action_vector" in query_json
    assert query_json["most_relevant_rule"] == "Federal Statute 404: Consumer credit history retention rules."

    # 3. Associate
    ret = main([
        "associate",
        record_id,
        "--db-path", db_path,
        "--mock-encoder",
    ])
    assert ret == 0
    assoc_out = capsys.readouterr().out
    assoc_json = json.loads(assoc_out)
    assert "results" in assoc_json

    # 4. Erase
    ret = main([
        "erase",
        record_id,
        "--db-path", db_path,
        "--mock-encoder",
    ])
    assert ret == 0
    erase_out = capsys.readouterr().out
    erase_json = json.loads(erase_out)
    assert erase_json["status"] == "ERASED"
    assert erase_json["record_id"] == record_id


def test_cli_ingest_jsonl_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = str(tmp_path / "cli_batch.db")
    jsonl_path = tmp_path / "rules.jsonl"

    rules = [
        {"type": "rule", "text": "Statute X1: Rule one text.", "action": [1.0, 0.0], "authority_rank": 3},
        {"type": "rule", "text": "Statute X2: Rule two text.", "action": [0.0, 1.0], "authority_rank": 5},
    ]
    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rules:
            f.write(json.dumps(r) + "\n")

    ret = main([
        "ingest",
        "--file", str(jsonl_path),
        "--db-path", db_path,
        "--mock-encoder",
    ])
    assert ret == 0
    out = capsys.readouterr().out
    assert "Successfully ingested 2 rules" in out


def test_cli_verify_attestation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    attestor = AsymmetricAttestationModule()
    pred = PredictionResult(
        query="Legal query",
        predicted_action_vector=[1.0, 0.0],
        most_relevant_rule="Statute 1",
    )
    tok = attestor.sign_attestation(pred, retrieved_text="Statute 1")

    tok_file = tmp_path / "token.json"
    with tok_file.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": tok.timestamp.isoformat(),
            "state_hash": tok.state_hash,
            "integrity_tag": tok.integrity_tag,
            "key_id": tok.key_id,
            "algorithm": tok.algorithm,
            "public_key": tok.public_key,
        }, f)

    ret = main([
        "verify",
        "--token-file", str(tok_file),
        "--query", "Legal query",
        "--action", "1.0", "0.0",
        "--rule", "Statute 1",
        "--public-key", attestor.public_key_hex,
    ])
    assert ret == 0
    out = capsys.readouterr().out
    res_json = json.loads(out)
    assert res_json["valid"] is True


def test_cli_evaluate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bench_file = Path("tests/fixtures/synthetic_legal_benchmark.jsonl")
    if not bench_file.exists():
        pytest.skip("Benchmark fixture not found.")

    out_file = str(tmp_path / "eval_out.json")
    ret = main([
        "evaluate",
        "--benchmark", str(bench_file),
        "--output", out_file,
    ])
    assert ret == 0
    out = capsys.readouterr().out
    assert "LegalBench-RAG Evaluation Results" in out
    assert Path(out_file).exists()
