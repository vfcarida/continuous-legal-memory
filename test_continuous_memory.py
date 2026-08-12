"""
Test suite for auditing and verifying the refactored Continuum Memory System
operating on top of frozen base encoder models.
"""

import unittest
from continuous_legal_memory.domain.exceptions import InvalidMemoryVectorError
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


class TestContinuousLegalMemory(unittest.TestCase):
    """
    Test suite for auditing and verifying the refactored Continuum Memory System.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize shared orchestrator instance."""
        cls.orchestrator = LegalMemoryOrchestrator(value_dim=2)
        cls.ACTION_DELETE = [1.0, 0.0]
        cls.ACTION_RETAIN = [0.0, 1.0]

    def test_01_model_weights_are_frozen(self) -> None:
        """
        Verify that all parameters inside the encoder model have requires_grad set to False,
        strictly complying with the zero-backpropagation constraint.
        """
        for name, param in self.orchestrator.encoder.model.named_parameters():
            self.assertFalse(param.requires_grad, f"Parameter {name} is not frozen!")

    def test_02_empty_memory_edge_case(self) -> None:
        """
        Verify that requesting predictions when memory is empty returns a clean, zeroed vector.
        """
        empty_orchestrator = LegalMemoryOrchestrator(value_dim=2)
        res = empty_orchestrator.predict("Query with no stored rules")
        self.assertEqual(res.predicted_action_vector, [0.0, 0.0])
        self.assertIsNone(res.most_relevant_rule)
        self.assertIsNone(res.confidence)

    def test_03_input_validation_and_safety(self) -> None:
        """
        Verify strict domain exception throwing on invalid query or action vector inputs.
        """
        with self.assertRaises(InvalidMemoryVectorError):
            self.orchestrator.predict("")
        with self.assertRaises(InvalidMemoryVectorError):
            self.orchestrator.predict("   ")

        with self.assertRaises(InvalidMemoryVectorError):
            self.orchestrator.update_memory("Valid rule", [1.0])
        with self.assertRaises(InvalidMemoryVectorError):
            self.orchestrator.update_memory("", [1.0, 0.0])

    def test_04_surprise_momentum_tracking(self) -> None:
        """
        Validate that consecutive consistent rules produce low surprise values,
        while conflicting inputs trigger a spike in the surprise tracker.
        """
        track_orchestrator = LegalMemoryOrchestrator(value_dim=2)
        track_orchestrator.update_memory("Article 1: Customers may request data deletion.", self.ACTION_DELETE)
        track_orchestrator.update_memory("Article 2: Deletion requests must be executed quickly.", self.ACTION_DELETE)
        initial_surprise = track_orchestrator.hope_module.memory.surprise_momentum.item()

        track_orchestrator.update_memory("New Anti-Fraud Rule: Deletion of active audit data is forbidden.", self.ACTION_RETAIN)
        updated_surprise = track_orchestrator.hope_module.memory.surprise_momentum.item()

        self.assertGreater(updated_surprise, initial_surprise, "Surprise momentum did not increase on conflicting rule!")

    def test_05_decision_override_and_catastrophic_forgetting(self) -> None:
        """
        Verify full legal override workflow and catastrophic forgetting prevention.
        """
        eval_orchestrator = LegalMemoryOrchestrator(value_dim=2)

        base_knowledge = [
            "Article 1: Every client has the right to request deletion of personal data.",
            "Article 2: Data deletion must be completed within 15 business days.",
            "Article 3: Credit evaluation depends on historical transaction data.",
        ]

        for rule in base_knowledge:
            eval_orchestrator.update_memory(rule, self.ACTION_DELETE)

        credit_query = "Client John paid off a loan last month and wants his financial credit history deleted."
        general_query = "Client requested deletion of his marketing email address."

        res_before = eval_orchestrator.predict(credit_query)
        self.assertGreater(res_before.predicted_action_vector[0], res_before.predicted_action_vector[1])

        override_rule = "New Directive: Deletion of credit operation records active in last 5 years is strictly prohibited."
        eval_orchestrator.update_memory(override_rule, self.ACTION_RETAIN)

        res_after_credit = eval_orchestrator.predict(credit_query)
        self.assertGreater(res_after_credit.predicted_action_vector[1], res_after_credit.predicted_action_vector[0])
        self.assertEqual(res_after_credit.most_relevant_rule, override_rule)

        res_after_general = eval_orchestrator.predict(general_query)
        self.assertGreater(res_after_general.predicted_action_vector[0], res_after_general.predicted_action_vector[1])


if __name__ == "__main__":
    unittest.main()
