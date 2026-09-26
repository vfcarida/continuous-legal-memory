# LegalBench-RAG Evaluation Suite

Continuous Legal Memory provides a real-world legal evaluation benchmark inspired by **LegalBench** (Guha et al., NeurIPS 2023) and **LegalBench-RAG** (Pipitone et al., 2024).

---

## 1. Motivation & Benchmark Philosophy

Standard general-purpose RAG benchmarks (such as TriviaQA or MS-MARCO) measure loose semantic retrieval over generic prose. In contrast, legal compliance systems require:

1. **Character-Level Snippet Precision**: Legal practitioners and judicial auditors require exact statutory phrases and contractual clauses rather than vague chunk summaries.
2. **Hierarchical Authority Overrides (*Lex Superior*)**: When a federal statute conflicts with an enterprise contract term, the system must deterministically prioritize the higher-ranking law.
3. **Zero-Hallucination Obligation Recall**: Failure to extract a notice window (e.g. "not later than 48 hours") constitutes an intolerable compliance breach.

The `continuous_legal_memory.evaluation.legalbench_eval` module bridges this gap with an offline-contained benchmark evaluating character-level precision and legal precedence over real-world commercial contracts and regulatory statutes.

---

## 2. Core Metrics

| Metric | Target | Description |
| :--- | :---: | :--- |
| **Precision@1** | $\ge 0.80$ | Fraction of queries where the top-1 retrieved rule matches the ground truth governing clause. |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.80$ | Reciprocal rank of the first relevant governing statute. |
| **Character Snippet Precision** | $\ge 0.50$ | Proportion of retrieved characters that directly support the ground-truth legal obligation. |
| **Character Snippet Recall** | $\ge 0.80$ | Proportion of ground-truth obligation text successfully extracted in retrieved snippets. |
| **Action Vector Cosine Sim** | $\ge 0.70$ | Cosine similarity between predicted action vectors and target legal decisions. |

---

## 3. Running the Benchmark

The benchmark can be executed programmatically or integrated into automated regression pipelines:

```python
from continuous_legal_memory.adapters.ollama import OllamaGemmaAdapter
from continuous_legal_memory.evaluation.legalbench_eval import LegalBenchEvaluator

# Initialize offline-first encoder
encoder = OllamaGemmaAdapter(host_url="http://localhost:11434")

# Instantiate evaluator
evaluator = LegalBenchEvaluator(
    encoder=encoder,
    tenant_id="legalbench_evaluation_run",
)

# Run benchmark
metrics = evaluator.run_evaluation()

# Generate Markdown audit report
report = evaluator.generate_markdown_report(metrics)
print(report)
```

---

## 4. Evaluated Legal Categories

The offline corpus covers key commercial and regulatory categories:

- **Limitation of Liability**: Aggregated liability caps, carve-outs for confidentiality breaches and gross negligence.
- **Data Privacy & GDPR**: Security incident notification deadlines (Art. 33), controller notifications, remediation disclosures.
- **Intellectual Property Indemnification**: Third-party patent, trademark, and copyright infringement defense obligations.
- **Termination for Convenience**: Written notice periods, fee accruals, and surviving terms.
- **Mandatory Banking Recordkeeping**: Federal statutory record retention rules overriding conflicting contractual erasure requests.
