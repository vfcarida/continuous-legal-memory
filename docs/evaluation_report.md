# Synthetic Legal Memory Benchmark Report (CLM-T08)

> [!NOTE]
> **Disclaimer**: This benchmark uses a deterministic synthetic dataset (`SyntheticLegalBenchmark-v1`).
> It is **not legally validated** and serves strictly as research evidence for memory architecture comparison.

## 1. Baselines Comparison (Mean ± Std over 20 Seeds)

| Architecture Baseline | Overall Acc | Override Acc | Authority Correctness | Forgetting (BWT) | Stale Recall Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **No-Memory** | 0.235 ± 0.000 | 0.333 ± 0.000 | 0.200 ± 0.000 | +0.000 ± 0.000 | 0.600 ± 0.000 |
| **Bounded-History ($k=3$)** | 0.765 ± 0.000 | 0.667 ± 0.000 | 0.800 ± 0.000 | +0.000 ± 0.000 | 0.400 ± 0.000 |
| **Plain RAG** (Vector-only) | 0.794 ± 0.000 | 0.667 ± 0.000 | 0.600 ± 0.000 | +0.000 ± 0.000 | 0.600 ± 0.000 |
| **Temporal Structured** (Retrieval+Precedence) | 0.853 ± 0.000 | 0.667 ± 0.000 | 0.600 ± 0.000 | +0.000 ± 0.000 | 0.200 ± 0.000 |
| **Hybrid Continuum Memory** (Full CMS) | 0.760 ± 0.045 | 0.433 ± 0.213 | 0.540 ± 0.092 | +0.110 ± 0.118 | 0.250 ± 0.087 |

---

## 2. Systematic Architectural Ablations

| Ablation Variant | Overall Acc | Override Acc | Authority Correctness |
| :--- | :---: | :---: | :---: |
| `RetrievalOnly` | 0.853 ± 0.000 | 0.667 ± 0.000 | 0.600 ± 0.000 |
| `FastNet Only` | 0.729 ± 0.051 | 0.400 ± 0.249 | 0.480 ± 0.098 |
| `SlowNet Tau005` | 0.729 ± 0.051 | 0.400 ± 0.249 | 0.480 ± 0.098 |
| `SlowNet Tau015` | 0.729 ± 0.051 | 0.400 ± 0.249 | 0.480 ± 0.098 |
| `SlowNet Tau030` | 0.729 ± 0.051 | 0.400 ± 0.249 | 0.480 ± 0.098 |
| `Temp 001` | 0.706 ± 0.067 | 0.333 ± 0.298 | 0.440 ± 0.080 |
| `Temp 005` | 0.729 ± 0.051 | 0.400 ± 0.249 | 0.480 ± 0.098 |
| `Temp 020` | 0.812 ± 0.035 | 0.733 ± 0.249 | 0.600 ± 0.000 |

---

## 3. Adversarial Robustness & Multi-Tenancy

| Adversarial Probe | Plain RAG | Temporal Structured | Hybrid Continuum Memory | Vulnerability Status |
| :--- | :---: | :---: | :---: | :---: |
| **Poisoning Attack Success Rate (ASR)** | 33.33% | 0.00% | 0.00% | Protected |
| **Multi-Tenant Leakage Rate** | 100.00% | N/A | 100.00% | VULNERABLE (Multi-tenancy isolation not yet enforced) |

---

## 4. Pre-Registered Go/No-Go Recommendation

- **Delta Accuracy over Structured**: `-0.093`
- **Catastrophic Forgetting (BWT)**: `+0.110`
- **Poisoning ASR**: `0.00%`
- **Decision**: **`NO-GO`**
- **Rationale**: Parametric head does not outperform deterministic structured retrieval (TemporalStructured) on clean synthetic ground truth and increases attack surface. Recommend structured retrieval for production.
