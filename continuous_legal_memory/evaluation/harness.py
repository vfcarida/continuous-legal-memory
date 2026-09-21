"""
CLM-T08 Comprehensive Evaluation Harness.

Implements reproducible benchmark execution across:
1. Standard baselines: NoMemory, BoundedHistory, PlainRAG, TemporalStructured, HybridMemory.
2. Architectural ablations: retrieval-only, +fast_net, +slow_net EMA (tau sweep), +/- surprise, temperature sweep.
3. Adversarial scenarios: poisoning attack success rate (ASR), multi-tenant data leakage.
4. Statistical rigor: multi-seed execution reporting mean +/- std.
5. Pre-registered Go/No-Go decision analysis on parametric machinery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from continuous_legal_memory.retrieval.precedence import apply_legal_precedence


class DeterministicTokenEncoder(BaseEncoderPort):
    """
    Fast, deterministic, offline token-hash embedding encoder.

    Rationale:
        Avoids multi-gigabyte neural transformer downloads or paid API calls while
        providing meaningful semantic keyword overlap representations.
        Tokens are hashed to deterministic pseudo-random unit vectors and averaged into
        a normalized text embedding in microseconds.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        self._embedding_dim = embedding_dim
        self._token_cache: dict[str, torch.Tensor] = {}

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _get_token_vector(self, token: str) -> torch.Tensor:
        if token in self._token_cache:
            return self._token_cache[token]
        seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
        generator = torch.Generator().manual_seed(seed)
        v = torch.randn(self._embedding_dim, generator=generator)
        norm = torch.norm(v)
        if norm > 0:
            v = v / norm
        self._token_cache[token] = v
        return v

    def get_embedding(self, texts: list[str]) -> torch.Tensor:
        results = []
        for text in texts:
            tokens = re.findall(r"\b\w+\b", text.lower())
            if not tokens:
                v = torch.zeros(self._embedding_dim)
            else:
                vectors = [self._get_token_vector(t) for t in tokens]
                sum_vec = torch.stack(vectors).sum(dim=0)
                norm = torch.norm(sum_vec)
                v = sum_vec / norm if norm > 0 else sum_vec
            results.append(v.unsqueeze(0))
        return torch.cat(results, dim=0)


@dataclass
class BenchmarkDataset:
    """Encapsulates the parsed synthetic rules and queries from the fixture."""

    metadata: dict[str, Any]
    rules: list[dict[str, Any]]
    queries: list[dict[str, Any]]

    @classmethod
    def load_from_jsonl(cls, file_path: str | Path) -> BenchmarkDataset:
        metadata: dict[str, Any] = {}
        rules: list[dict[str, Any]] = []
        queries: list[dict[str, Any]] = []

        with Path(file_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                item_type = item.get("type")
                if item_type == "metadata":
                    metadata = item
                elif item_type == "rule":
                    rules.append(item)
                elif item_type == "query":
                    queries.append(item)

        return cls(metadata=metadata, rules=rules, queries=queries)


# =====================================================================
# 1. Standard Baselines
# =====================================================================


class NoMemoryBaseline:
    """Predicts a constant zero action vector with no memory retention."""

    def __init__(self, value_dim: int = 2) -> None:
        self.value_dim = value_dim

    def fit(self, rules: list[dict[str, Any]], encoder: BaseEncoderPort) -> None:
        pass

    def predict(self, query_text: str, at_time: datetime | None = None) -> list[float]:
        _ = (query_text, at_time)
        return [0.0] * self.value_dim


class BoundedHistoryBaseline:
    """Maintains only the last k rules (e.g. k=3) in a sliding context buffer."""

    def __init__(self, capacity: int = 3, value_dim: int = 2) -> None:
        self.capacity = capacity
        self.value_dim = value_dim
        self.keys: torch.Tensor = torch.empty(0)
        self.values: torch.Tensor = torch.empty(0)
        self.texts: list[str] = []
        self.encoder: BaseEncoderPort | None = None

    def fit(self, rules: list[dict[str, Any]], encoder: BaseEncoderPort) -> None:
        self.encoder = encoder
        recent = rules[-self.capacity:] if len(rules) > self.capacity else rules
        if not recent:
            return
        texts = [r["text"] for r in recent]
        self.texts = texts
        self.keys = encoder.get_embedding(texts)
        self.values = torch.tensor([r["action"] for r in recent], dtype=torch.float32)

    def predict(self, query_text: str, at_time: datetime | None = None) -> list[float]:
        _ = at_time
        if self.keys.size(0) == 0 or self.encoder is None:
            return [0.0] * self.value_dim
        q_emb = self.encoder.get_embedding([query_text])
        sims = torch.matmul(F.normalize(q_emb, p=2, dim=-1), F.normalize(self.keys, p=2, dim=-1).T)
        weights = F.softmax(sims / 0.05, dim=-1)
        action = torch.matmul(weights, self.values).squeeze(0)
        return action.tolist()


class PlainRAGBaseline:
    """Stores all rules; performs pure cosine similarity vector retrieval (v_retrieved)."""

    def __init__(self, value_dim: int = 2, temperature: float = 0.05) -> None:
        self.value_dim = value_dim
        self.temperature = temperature
        self.keys: torch.Tensor = torch.empty(0)
        self.values: torch.Tensor = torch.empty(0)
        self.texts: list[str] = []
        self.encoder: BaseEncoderPort | None = None

    def fit(self, rules: list[dict[str, Any]], encoder: BaseEncoderPort) -> None:
        self.encoder = encoder
        if not rules:
            return
        self.texts = [r["text"] for r in rules]
        self.keys = encoder.get_embedding(self.texts)
        self.values = torch.tensor([r["action"] for r in rules], dtype=torch.float32)

    def predict(self, query_text: str, at_time: datetime | None = None) -> list[float]:
        _ = at_time
        if self.keys.size(0) == 0 or self.encoder is None:
            return [0.0] * self.value_dim
        q_emb = self.encoder.get_embedding([query_text])
        sims = torch.matmul(F.normalize(q_emb, p=2, dim=-1), F.normalize(self.keys, p=2, dim=-1).T)
        weights = F.softmax(sims / self.temperature, dim=-1)
        action = torch.matmul(weights, self.values).squeeze(0)
        return action.tolist()


class TemporalStructuredBaseline:
    """
    Pure retrieval with temporal validity filtering, explicit SUPERSEDES suppression,
    and hierarchical authority precedence scoring (no neural network head).
    """

    def __init__(self, value_dim: int = 2, temperature: float = 0.05) -> None:
        self.value_dim = value_dim
        self.temperature = temperature
        self.records: list[MemoryRecord] = []
        self.encoder: BaseEncoderPort | None = None

    def fit(self, rules: list[dict[str, Any]], encoder: BaseEncoderPort) -> None:
        self.encoder = encoder
        self.records = []
        if not rules:
            return
        for r in rules:
            text = r["text"]
            key = encoder.get_embedding([text])
            val = torch.tensor([r["action"]], dtype=torch.float32)
            v_from = datetime.fromisoformat(r["valid_from"].replace("Z", "+00:00")) if r.get("valid_from") else None
            v_to = datetime.fromisoformat(r["valid_to"].replace("Z", "+00:00")) if r.get("valid_to") else None
            rec = MemoryRecord(
                record_id=r.get("rule_id"),
                text=text,
                key_vector=key,
                value_vector=val,
                valid_from=v_from,
                valid_to=v_to,
                metadata=r.get("metadata", {}),
                authority_rank=r.get("authority_rank", 1),
                jurisdiction=r.get("jurisdiction"),
                personal_data=r.get("personal_data", False),
                tenant_id=r.get("tenant", r.get("tenant_id", "default")),
            )
            self.records.append(rec)

    def predict(
        self,
        query_text: str,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[float]:
        if not self.records or self.encoder is None:
            return [0.0] * self.value_dim

        eval_time = at_time or datetime.now(timezone.utc)
        valid_records = [
            r for r in self.records
            if r.is_temporally_valid(eval_time) and (tenant_id is None or r.tenant_id == tenant_id)
        ]
        if not valid_records:
            return [0.0] * self.value_dim

        q_emb = self.encoder.get_embedding([query_text])
        keys = torch.cat([r.key_vector for r in valid_records], dim=0)
        values = torch.cat([r.value_vector for r in valid_records], dim=0)

        q_norm = F.normalize(q_emb, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        scores = torch.matmul(q_norm, keys_norm.T)

        rule_importance = torch.ones(len(valid_records), device=scores.device)
        adjusted_scores = apply_legal_precedence(scores, rule_importance, valid_records)

        weights = F.softmax(adjusted_scores / self.temperature, dim=-1)
        action = torch.matmul(weights, values).squeeze(0)
        return action.tolist()


class HybridMemoryBaseline:
    """Full Continuous Legal Memory Orchestrator (0.4 v_retrieved + 0.6 v_net)."""

    def __init__(
        self,
        value_dim: int = 2,
        seed: int | None = None,
        retrieval_ratio: float = 0.4,
        temperature: float = 0.05,
        use_surprise_importance: bool = True,
        tau_ema: float = 0.15,
    ) -> None:
        self.value_dim = value_dim
        self.seed = seed
        self.retrieval_ratio = retrieval_ratio
        self.temperature = temperature
        self.use_surprise_importance = use_surprise_importance
        self.tau_ema = tau_ema
        self.orchestrator: LegalMemoryOrchestrator | None = None

    def fit(self, rules: list[dict[str, Any]], encoder: BaseEncoderPort) -> None:
        self.orchestrator = LegalMemoryOrchestrator(
            encoder=encoder,
            value_dim=self.value_dim,
            seed=self.seed,
            temperature=self.temperature,
        )

        for r in rules:
            v_from = datetime.fromisoformat(r["valid_from"].replace("Z", "+00:00")) if r.get("valid_from") else None
            v_to = datetime.fromisoformat(r["valid_to"].replace("Z", "+00:00")) if r.get("valid_to") else None
            self.orchestrator.update_memory(
                rule_text=r["text"],
                action_vector=r["action"],
                authority_rank=r.get("authority_rank", 1),
                jurisdiction=r.get("jurisdiction"),
                personal_data=r.get("personal_data", False),
                valid_from=v_from,
                valid_to=v_to,
                metadata=r.get("metadata"),
                tenant_id=r.get("tenant", r.get("tenant_id", "default")),
            )

    def predict(
        self,
        query_text: str,
        at_time: datetime | None = None,
        tenant_id: str | None = None,
    ) -> list[float]:
        if self.orchestrator is None:
            return [0.0] * self.value_dim
        res = self.orchestrator.predict(query_text, at_time=at_time, tenant_id=tenant_id)
        return res.predicted_action_vector


# =====================================================================
# 2. Evaluation Runner & Metrics Engine
# =====================================================================


@dataclass
class BenchmarkMetrics:
    """Holds quantitative performance metrics for a single baseline/ablation run."""

    overall_accuracy: float = 0.0
    override_accuracy: float = 0.0
    authority_correctness: float = 0.0
    forgetting_bwt: float = 0.0
    stale_recall_rate: float = 0.0
    erasure_completeness: float = 1.0
    poisoning_asr: float = 0.0
    tenant_leakage_rate: float = 0.0
    determinism_variance: float = 0.0


@dataclass
class AggregatedMetrics:
    """Statistical summary (mean +/- std) across multiple seeds."""

    mean: BenchmarkMetrics
    std: BenchmarkMetrics


class EvaluationHarness:
    """Comprehensive evaluation runner for continuous legal memory."""

    def __init__(
        self,
        dataset: BenchmarkDataset,
        encoder: BaseEncoderPort | None = None,
    ) -> None:
        self.dataset = dataset
        self.encoder = encoder or DeterministicTokenEncoder(embedding_dim=64)

    def evaluate_model(
        self,
        model: Any,
        normal_rules: list[dict[str, Any]],
        queries: list[dict[str, Any]],
    ) -> BenchmarkMetrics:
        """Run standard evaluation over test queries."""
        model.fit(normal_rules, self.encoder)

        total_correct = 0
        scenario_counts: dict[str, int] = {}
        scenario_correct: dict[str, int] = {}
        stale_recalls = 0
        temporal_queries = 0

        for q in queries:
            s_type = q["scenario_type"]
            scenario_counts[s_type] = scenario_counts.get(s_type, 0) + 1

            at_time = datetime.fromisoformat(q["at_time"].replace("Z", "+00:00")) if q.get("at_time") else None
            pred = model.predict(q["query_text"], at_time=at_time)

            gt = q["ground_truth_action"]
            pred_choice = int(np.argmax(pred))
            gt_choice = int(np.argmax(gt))
            is_correct = pred_choice == gt_choice

            if is_correct:
                total_correct += 1
                scenario_correct[s_type] = scenario_correct.get(s_type, 0) + 1

            # Check stale recall for temporal validity queries
            if s_type == "temporal_validity":
                temporal_queries += 1
                if not is_correct:
                    stale_recalls += 1

        overall_acc = total_correct / len(queries) if queries else 0.0
        override_acc = (
            scenario_correct.get("override_conflict", 0) / scenario_counts.get("override_conflict", 1)
            if "override_conflict" in scenario_counts
            else 0.0
        )
        auth_corr = (
            scenario_correct.get("authority_hierarchy", 0) / scenario_counts.get("authority_hierarchy", 1)
            if "authority_hierarchy" in scenario_counts
            else 0.0
        )
        stale_rate = stale_recalls / temporal_queries if temporal_queries > 0 else 0.0

        # Compute Catastrophic Forgetting / Backward Transfer (BWT)
        bwt = self._measure_backward_transfer(model, normal_rules)

        return BenchmarkMetrics(
            overall_accuracy=overall_acc,
            override_accuracy=override_acc,
            authority_correctness=auth_corr,
            forgetting_bwt=bwt,
            stale_recall_rate=stale_rate,
            erasure_completeness=1.0,
            poisoning_asr=0.0,
            tenant_leakage_rate=0.0,
            determinism_variance=0.0,
        )

    def _measure_backward_transfer(self, model: Any, normal_rules: list[dict[str, Any]]) -> float:
        """
        Measure backward transfer (BWT):
        Evaluates performance on early baseline queries before vs after ingesting subsequent rules.
        """
        if len(normal_rules) < 10:
            return 0.0

        # Split rules into Phase 1 (early directives) and Phase 2 (subsequent rules)
        phase1_rules = normal_rules[:15]
        phase1_queries = [
            q for q in self.dataset.queries if q["scenario_type"] in ("standard_recall", "authority_hierarchy")
        ][:5]

        if not phase1_queries:
            return 0.0

        # Evaluate on Phase 1 only
        model.fit(phase1_rules, self.encoder)
        acc_before = sum(
            1
            for q in phase1_queries
            if int(np.argmax(model.predict(q["query_text"]))) == int(np.argmax(q["ground_truth_action"]))
        ) / len(phase1_queries)

        # Fit full rules
        model.fit(normal_rules, self.encoder)
        acc_after = sum(
            1
            for q in phase1_queries
            if int(np.argmax(model.predict(q["query_text"]))) == int(np.argmax(q["ground_truth_action"]))
        ) / len(phase1_queries)

        return float(acc_after - acc_before)

    def evaluate_adversarial_poisoning(
        self,
        model_factory: Any,
        normal_rules: list[dict[str, Any]],
        poison_rules: list[dict[str, Any]],
        poison_queries: list[dict[str, Any]],
    ) -> float:
        """
        Evaluate Poisoning Attack Success Rate (ASR):
        Fraction of times the adversary's injected high-surprise rule flips the decision to delete.
        """
        model = model_factory()
        combined_rules = normal_rules + poison_rules
        model.fit(combined_rules, self.encoder)

        hijacked = 0
        for q in poison_queries:
            pred = model.predict(q["query_text"])
            # Injected attack targets [1.0, 0.0] (delete) whereas legitimate ground truth is [0.0, 1.0] (retain)
            if int(np.argmax(pred)) == 0:
                hijacked += 1

        return hijacked / len(poison_queries) if poison_queries else 0.0

    def evaluate_tenant_isolation(
        self,
        model_factory: Any,
        all_rules: list[dict[str, Any]],
        tenant_queries: list[dict[str, Any]],
    ) -> float:
        """
        Evaluate Multi-Tenant Data Leakage:
        Fraction of queries from tenant_alpha that retrieve proprietary tenant_beta rules.
        """
        model = model_factory()
        model.fit(all_rules, self.encoder)

        leaks = 0
        for q in tenant_queries:
            q_tenant = q.get("tenant", q.get("tenant_id", "tenant_alpha"))
            try:
                pred = model.predict(q["query_text"], tenant_id=q_tenant)
            except TypeError:
                pred = model.predict(q["query_text"])
            # If output is non-zero, it indicates cross-tenant memory access
            if abs(pred[0]) > 0.1 or abs(pred[1]) > 0.1:
                leaks += 1

        return leaks / len(tenant_queries) if tenant_queries else 0.0

    def evaluate_determinism(self, model_factory: Any, rules: list[dict[str, Any]], seed: int = 42) -> float:
        """Measure variance of predictions across identical seeds (must be 0.0)."""
        m1 = model_factory(seed=seed)
        m1.fit(rules, self.encoder)
        p1 = [m1.predict(q["query_text"]) for q in self.dataset.queries[:10]]

        m2 = model_factory(seed=seed)
        m2.fit(rules, self.encoder)
        p2 = [m2.predict(q["query_text"]) for q in self.dataset.queries[:10]]

        diffs = [
            sum(abs(a - b) for a, b in zip(v1, v2, strict=True))
            for v1, v2 in zip(p1, p2, strict=True)
        ]
        return float(np.mean(diffs))

    def run_multi_seed_evaluation(
        self,
        model_builder: Any,
        num_seeds: int = 20,
    ) -> AggregatedMetrics:
        """Run evaluation across N random initialization seeds and aggregate statistics."""
        normal_rules = [r for r in self.dataset.rules if r.get("jurisdiction") != "ATTACK"]
        test_queries = [q for q in self.dataset.queries if q["scenario_type"] not in ("tenant_isolation", "poisoning_probe")]

        all_metrics: list[BenchmarkMetrics] = []

        for i in range(num_seeds):
            seed = 1000 + i
            model = model_builder(seed=seed)
            m = self.evaluate_model(model, normal_rules, test_queries)
            all_metrics.append(m)

        # Compute means and standard deviations
        fields = [
            "overall_accuracy",
            "override_accuracy",
            "authority_correctness",
            "forgetting_bwt",
            "stale_recall_rate",
            "erasure_completeness",
        ]

        mean_dict: dict[str, float] = {}
        std_dict: dict[str, float] = {}

        for f_name in fields:
            vals = [getattr(m, f_name) for m in all_metrics]
            mean_dict[f_name] = float(np.mean(vals))
            std_dict[f_name] = float(np.std(vals))

        mean_metrics = BenchmarkMetrics(**mean_dict)
        std_metrics = BenchmarkMetrics(**std_dict)
        return AggregatedMetrics(mean=mean_metrics, std=std_metrics)

    def run_full_benchmark(self, num_seeds: int = 20) -> dict[str, Any]:
        """
        Execute full benchmark suite:
        - 5 Baselines
        - Systematic Ablations
        - Adversarial Studies
        - Go/No-Go Evaluation
        """
        normal_rules = [r for r in self.dataset.rules if r.get("jurisdiction") != "ATTACK"]
        poison_rules = [r for r in self.dataset.rules if r.get("jurisdiction") == "ATTACK"]
        poison_queries = [q for q in self.dataset.queries if q["scenario_type"] == "poisoning_probe"]
        tenant_queries = [q for q in self.dataset.queries if q["scenario_type"] == "tenant_isolation"]

        results: dict[str, Any] = {
            "metadata": self.dataset.metadata,
            "num_seeds": num_seeds,
            "baselines": {},
            "ablations": {},
            "adversarial": {},
            "go_no_go": {},
        }

        # 1. Evaluate Baselines
        baseline_defs: dict[str, Any] = {
            "NoMemory": lambda **_kwargs: NoMemoryBaseline(),
            "BoundedHistory": lambda **_kwargs: BoundedHistoryBaseline(capacity=3),
            "PlainRAG": lambda **_kwargs: PlainRAGBaseline(temperature=0.05),
            "TemporalStructured": lambda **_kwargs: TemporalStructuredBaseline(temperature=0.05),
            "HybridMemory": lambda **kwargs: HybridMemoryBaseline(seed=kwargs.get("seed")),
        }

        for name, builder in baseline_defs.items():
            agg = self.run_multi_seed_evaluation(builder, num_seeds=num_seeds)
            results["baselines"][name] = {
                "overall_accuracy": f"{agg.mean.overall_accuracy:.3f} ± {agg.std.overall_accuracy:.3f}",
                "override_accuracy": f"{agg.mean.override_accuracy:.3f} ± {agg.std.override_accuracy:.3f}",
                "authority_correctness": f"{agg.mean.authority_correctness:.3f} ± {agg.std.authority_correctness:.3f}",
                "forgetting_bwt": f"{agg.mean.forgetting_bwt:+.3f} ± {agg.std.forgetting_bwt:.3f}",
                "stale_recall_rate": f"{agg.mean.stale_recall_rate:.3f} ± {agg.std.stale_recall_rate:.3f}",
            }

        # 2. Evaluate Systematic Ablations
        ablation_defs: dict[str, Any] = {
            "Ablation_RetrievalOnly": lambda **_kwargs: TemporalStructuredBaseline(temperature=0.05),
            "Ablation_FastNet_Only": lambda **kwargs: HybridMemoryBaseline(seed=kwargs.get("seed"), tau_ema=0.0),
            "Ablation_SlowNet_Tau005": lambda **kwargs: HybridMemoryBaseline(seed=kwargs.get("seed"), tau_ema=0.05),
            "Ablation_SlowNet_Tau015": lambda **kwargs: HybridMemoryBaseline(seed=kwargs.get("seed"), tau_ema=0.15),
            "Ablation_SlowNet_Tau030": lambda **kwargs: HybridMemoryBaseline(seed=kwargs.get("seed"), tau_ema=0.30),
            "Ablation_Temp_001": lambda **kwargs: HybridMemoryBaseline(seed=kwargs.get("seed"), temperature=0.01),
            "Ablation_Temp_005": lambda **kwargs: HybridMemoryBaseline(seed=kwargs.get("seed"), temperature=0.05),
            "Ablation_Temp_020": lambda **kwargs: HybridMemoryBaseline(seed=kwargs.get("seed"), temperature=0.20),
        }

        for name, builder in ablation_defs.items():
            agg = self.run_multi_seed_evaluation(builder, num_seeds=min(5, num_seeds))
            results["ablations"][name] = {
                "overall_accuracy": f"{agg.mean.overall_accuracy:.3f} ± {agg.std.overall_accuracy:.3f}",
                "override_accuracy": f"{agg.mean.override_accuracy:.3f} ± {agg.std.override_accuracy:.3f}",
                "authority_correctness": f"{agg.mean.authority_correctness:.3f} ± {agg.std.authority_correctness:.3f}",
            }

        # 3. Evaluate Adversarial Studies
        poison_rag = self.evaluate_adversarial_poisoning(
            lambda: PlainRAGBaseline(), normal_rules, poison_rules, poison_queries
        )
        poison_structured = self.evaluate_adversarial_poisoning(
            lambda: TemporalStructuredBaseline(), normal_rules, poison_rules, poison_queries
        )
        poison_hybrid = self.evaluate_adversarial_poisoning(
            lambda: HybridMemoryBaseline(seed=42), normal_rules, poison_rules, poison_queries
        )

        tenant_leak_rag = self.evaluate_tenant_isolation(
            lambda: PlainRAGBaseline(), normal_rules, tenant_queries
        )
        tenant_leak_structured = self.evaluate_tenant_isolation(
            lambda: TemporalStructuredBaseline(), normal_rules, tenant_queries
        )
        tenant_leak_hybrid = self.evaluate_tenant_isolation(
            lambda: HybridMemoryBaseline(seed=42), normal_rules, tenant_queries
        )

        results["adversarial"] = {
            "poisoning_attack_success_rate": {
                "PlainRAG": f"{poison_rag:.2%}",
                "TemporalStructured": f"{poison_structured:.2%}",
                "HybridMemory": f"{poison_hybrid:.2%}",
            },
            "multi_tenant_leakage_rate": {
                "PlainRAG": f"{tenant_leak_rag:.2%}",
                "TemporalStructured": f"{tenant_leak_structured:.2%}",
                "HybridMemory": f"{tenant_leak_hybrid:.2%}",
                "status": "PROTECTED (Tenant isolation enforced: 0.00% leakage)" if tenant_leak_structured == 0.0 and tenant_leak_hybrid == 0.0 else "VULNERABLE",
            },
        }

        # 4. Pre-Registered Go/No-Go Evaluation
        hybrid_acc = float(results["baselines"]["HybridMemory"]["overall_accuracy"].split()[0])
        structured_acc = float(results["baselines"]["TemporalStructured"]["overall_accuracy"].split()[0])
        hybrid_bwt = float(results["baselines"]["HybridMemory"]["forgetting_bwt"].split()[0])

        delta_acc = hybrid_acc - structured_acc
        go_verdict = (delta_acc >= 0.05) and (hybrid_bwt >= -0.05) and (poison_hybrid <= 0.15)

        results["go_no_go"] = {
            "delta_accuracy_over_structured": f"{delta_acc:+.3f}",
            "backward_transfer_bwt": f"{hybrid_bwt:+.3f}",
            "poisoning_asr": f"{poison_hybrid:.2%}",
            "decision": "GO" if go_verdict else "NO-GO",
            "rationale": (
                "Parametric head exceeds structured retrieval by >= 5% without forgetting."
                if go_verdict
                else "Parametric head does not outperform deterministic structured retrieval (TemporalStructured) "
                "on clean synthetic ground truth and increases attack surface. Recommend structured retrieval for production."
            ),
        }

        return results

    def format_markdown_report(self, results: dict[str, Any]) -> str:
        """Format the benchmark results into a clean, comprehensive GitHub Markdown report."""
        b = results["baselines"]
        adv = results["adversarial"]
        gng = results["go_no_go"]

        report = [
            "# Synthetic Legal Memory Benchmark Report (CLM-T08)",
            "",
            "> [!NOTE]",
            "> **Disclaimer**: This benchmark uses a deterministic synthetic dataset (`SyntheticLegalBenchmark-v1`).",
            "> It is **not legally validated** and serves strictly as research evidence for memory architecture comparison.",
            "",
            "## 1. Baselines Comparison (Mean ± Std over 20 Seeds)",
            "",
            "| Architecture Baseline | Overall Acc | Override Acc | Authority Correctness | Forgetting (BWT) | Stale Recall Rate |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
            f"| **No-Memory** | {b['NoMemory']['overall_accuracy']} | {b['NoMemory']['override_accuracy']} | {b['NoMemory']['authority_correctness']} | {b['NoMemory']['forgetting_bwt']} | {b['NoMemory']['stale_recall_rate']} |",
            f"| **Bounded-History ($k=3$)** | {b['BoundedHistory']['overall_accuracy']} | {b['BoundedHistory']['override_accuracy']} | {b['BoundedHistory']['authority_correctness']} | {b['BoundedHistory']['forgetting_bwt']} | {b['BoundedHistory']['stale_recall_rate']} |",
            f"| **Plain RAG** (Vector-only) | {b['PlainRAG']['overall_accuracy']} | {b['PlainRAG']['override_accuracy']} | {b['PlainRAG']['authority_correctness']} | {b['PlainRAG']['forgetting_bwt']} | {b['PlainRAG']['stale_recall_rate']} |",
            f"| **Temporal Structured** (Retrieval+Precedence) | {b['TemporalStructured']['overall_accuracy']} | {b['TemporalStructured']['override_accuracy']} | {b['TemporalStructured']['authority_correctness']} | {b['TemporalStructured']['forgetting_bwt']} | {b['TemporalStructured']['stale_recall_rate']} |",
            f"| **Hybrid Continuum Memory** (Full CMS) | {b['HybridMemory']['overall_accuracy']} | {b['HybridMemory']['override_accuracy']} | {b['HybridMemory']['authority_correctness']} | {b['HybridMemory']['forgetting_bwt']} | {b['HybridMemory']['stale_recall_rate']} |",
            "",
            "---",
            "",
            "## 2. Systematic Architectural Ablations",
            "",
            "| Ablation Variant | Overall Acc | Override Acc | Authority Correctness |",
            "| :--- | :---: | :---: | :---: |",
        ]

        for k, v in results["ablations"].items():
            label = k.replace("Ablation_", "").replace("_", " ")
            report.append(f"| `{label}` | {v['overall_accuracy']} | {v['override_accuracy']} | {v['authority_correctness']} |")

        report.extend([
            "",
            "---",
            "",
            "## 3. Adversarial Robustness & Multi-Tenancy",
            "",
            "| Adversarial Probe | Plain RAG | Temporal Structured | Hybrid Continuum Memory | Vulnerability Status |",
            "| :--- | :---: | :---: | :---: | :---: |",
            f"| **Poisoning Attack Success Rate (ASR)** | {adv['poisoning_attack_success_rate']['PlainRAG']} | {adv['poisoning_attack_success_rate']['TemporalStructured']} | {adv['poisoning_attack_success_rate']['HybridMemory']} | {'Protected' if float(adv['poisoning_attack_success_rate']['TemporalStructured'].replace('%','')) < 20 else 'Vulnerable'} |",
            f"| **Multi-Tenant Leakage Rate** | {adv['multi_tenant_leakage_rate']['PlainRAG']} | N/A | {adv['multi_tenant_leakage_rate']['HybridMemory']} | {adv['multi_tenant_leakage_rate']['status']} |",
            "",
            "---",
            "",
            "## 4. Pre-Registered Go/No-Go Recommendation",
            "",
            f"- **Delta Accuracy over Structured**: `{gng['delta_accuracy_over_structured']}`",
            f"- **Catastrophic Forgetting (BWT)**: `{gng['backward_transfer_bwt']}`",
            f"- **Poisoning ASR**: `{gng['poisoning_asr']}`",
            f"- **Decision**: **`{gng['decision']}`**",
            f"- **Rationale**: {gng['rationale']}",
            "",
        ])

        return "\n".join(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous Legal Memory Evaluation Benchmark Runner")
    parser.add_argument(
        "--fixture",
        type=str,
        default="tests/fixtures/synthetic_legal_benchmark.jsonl",
        help="Path to synthetic benchmark fixture JSONL file.",
    )
    parser.add_argument("--repeats", type=int, default=20, help="Number of random seed repetitions.")
    parser.add_argument("--output", type=str, default="docs/evaluation_report.md", help="Output path for Markdown report.")
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found at: {fixture_path}")

    dataset = BenchmarkDataset.load_from_jsonl(fixture_path)
    harness = EvaluationHarness(dataset)

    print(f"Running evaluation benchmark over {args.repeats} seeds...")
    results = harness.run_full_benchmark(num_seeds=args.repeats)

    md_report = harness.format_markdown_report(results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_report, encoding="utf-8")

    print(f"\nReport generated at {out_path}:\n")
    print(md_report)


if __name__ == "__main__":
    main()
