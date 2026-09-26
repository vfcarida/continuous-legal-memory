# Benchmark Evaluation Methodology

The Continuous Legal Memory evaluation harness (`continuous_legal_memory.evaluation.harness`) provides a rigorous, multi-seed statistical framework for testing continual learning, precedence resolution, and memory stability in legal reasoning.

---

## Pre-Registered Statistical Criteria

The benchmark evaluates architectures across 10 evaluation seeds against strict Go/No-Go criteria:

1. **Precedence Override Accuracy**:
   Higher statutory rules must override conflicting earlier policies with $> 95\%$ statistical reliability.
2. **Backward Transfer & Catastrophic Forgetting**:
   Updating memory on new statutory amendments must not degrade recall accuracy on unrelated legal provisions ($BWT \ge -0.02$).
3. **Temporal Invalidation Integrity**:
   Rules that have reached their `valid_to` expiration date must have $0\%$ recall at current transaction time.
4. **Latency & Throughput Bounds**:
   Single-rule ingestion latency must remain under 10ms in production structured retrieval mode.
