"""
Agentic Legal Memory Evaluation (LLMOps) Heuristics Suite.

Implements heuristic metric evaluators measuring Context Precision, Plan Adherence,
and Decision Vector Recall for continuous legal memory agents.

Note:
    These metric evaluators are standalone, rule- and token-based heuristic implementations.
    Integration with formal DeepEval / Ragas evaluation frameworks is scheduled for CLM-T08.
"""



class AgenticLegalEvaluator:
    """
    Continuous Agentic Evaluation heuristic framework.

    Rationale:
        Provides lightweight heuristic metrics across three core quality dimensions:
        1. Context Precision: Heuristic token overlap between retrieved snippet and reference ground truth.
        2. Plan Adherence: Substring-matching verification across ordered procedural workflow steps.
        3. Decision Vector Recall: L1 distance verification between predicted and expected decision vectors.
        (Formal DeepEval / Ragas suite integration planned in CLM-T08).
    """

    def evaluate_context_precision(self, query: str, retrieved_snippet: str, ground_truth: str) -> float:
        """
        Compute Context Precision score (0.0 to 1.0) evaluating snippet signal-to-noise ratio.

        Args:
            query: Input legal query.
            retrieved_snippet: Extracted memory text snippet.
            ground_truth: Ground truth legal reference clause.

        Returns:
            Context Precision score float between 0.0 and 1.0.
        """
        if not retrieved_snippet or not ground_truth:
            return 0.0

        q_tokens = set(query.lower().split()) if query else set()
        gt_tokens = set(ground_truth.lower().split()).union(q_tokens)
        snippet_tokens = retrieved_snippet.lower().split()
        if not snippet_tokens:
            return 0.0

        overlap = sum(1 for token in snippet_tokens if token in gt_tokens)
        precision = overlap / len(snippet_tokens)
        return min(1.0, max(0.0, precision * 1.5))

    def evaluate_plan_adherence(self, executed_steps: list[str], expected_procedure: list[str]) -> float:
        """
        Evaluate Plan Adherence score (0.0 to 1.0) verifying procedural step compliance.

        Args:
            executed_steps: Sequence of action steps executed by the agent.
            expected_procedure: Required legal protocol workflow steps.

        Returns:
            Plan Adherence score float.
        """
        if not expected_procedure:
            return 1.0
        if not executed_steps:
            return 0.0

        matches = 0
        for i, step in enumerate(expected_procedure):
            if i < len(executed_steps) and step.lower() in executed_steps[i].lower():
                matches += 1

        return matches / len(expected_procedure)

    def evaluate_cross_session_recall(self, predicted_action: list[float], expected_action: list[float], tolerance: float = 0.25) -> float:
        """
        Evaluate Cross-Session Recall score (0.0 to 1.0) verifying accurate decision vector recall.

        Args:
            predicted_action: Action decision vector output by the memory network.
            expected_action: Expected target decision vector.
            tolerance: L1 error tolerance bound.

        Returns:
            Recall score float (1.0 if within tolerance, scaled down otherwise).
        """
        if len(predicted_action) != len(expected_action):
            return 0.0

        l1_error = sum(abs(p - e) for p, e in zip(predicted_action, expected_action, strict=True)) / len(predicted_action)
        if l1_error <= tolerance:
            return 1.0
        return max(0.0, 1.0 - (l1_error - tolerance))
