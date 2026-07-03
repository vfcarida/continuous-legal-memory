import unittest
import torch
from poc_continuous_memory import LegalIncrementalModel, MemoryMLP, ContinuousMemory

class TestContinuousLegalMemory(unittest.TestCase):
    """
    Test suite for auditing and verifying the refactored Continuum Memory System
    operating on top of the frozen base models.
    """
    
    @classmethod
    def setUpClass(cls):
        # Initialize a single shared instance to save HF Hub caching/loading times
        cls.model = LegalIncrementalModel(model_name='neuralmind/bert-base-portuguese-cased')
        cls.ACTION_EXCLUIR = [1.0, 0.0]
        cls.ACTION_RETER = [0.0, 1.0]

    def test_01_model_weights_are_frozen(self):
        """
        Verify that all parameters inside the encoder/BERT model have requires_grad set to False,
        strictly complying with the zero-backpropagation constraint on the base model.
        """
        for name, param in self.model.encoder.bert.named_parameters():
            self.assertFalse(param.requires_grad, f"Parameter {name} is not frozen!")

    def test_02_empty_memory_edge_case(self):
        """
        Verify that requesting predictions when the memory is empty returns a clean,
        zeroed vector rather than raising division by zero or indexing errors.
        """
        empty_model = LegalIncrementalModel(model_name='neuralmind/bert-base-portuguese-cased')
        # Prediction with empty memory
        res = empty_model.predict("Query with no stored rules")
        self.assertEqual(res["predicted_action_vector"], [0.0, 0.0])
        self.assertNotIn("most_relevant_rule", res)
        self.assertNotIn("confidence", res)

    def test_03_input_validation_and_safety(self):
        """
        Verify strict parameter type and dimension assertions on inputs to protect state stability.
        """
        # Invalid query texts
        with self.assertRaises(ValueError):
            self.model.predict("")
        with self.assertRaises(ValueError):
            self.model.predict("   ")
        
        # Invalid rule actions
        with self.assertRaises(ValueError):
            self.model.update_memory("Valid rule", [1.0])  # Length must be 2
        with self.assertRaises(ValueError):
            self.model.update_memory("", [1.0, 0.0])  # Empty rule text

    def test_04_surprise_momentum_tracking(self):
        """
        Validate that consecutive, consistent rules produce low surprise values,
        while conflicting inputs trigger a spike in the surprise tracker.
        """
        track_model = LegalIncrementalModel(model_name='neuralmind/bert-base-portuguese-cased')
        
        # First memory injection initializes the network
        track_model.update_memory("Artigo 1: Todo cliente tem o direito de solicitar a exclusão de dados.", self.ACTION_EXCLUIR)
        
        # Second injection is consistent (same action EXCLUIR). Surprise should be low.
        track_model.update_memory("Artigo 2: O processo de exclusão deve ser rápido.", self.ACTION_EXCLUIR)
        initial_surprise = track_model.hope_module.memory.surprise_momentum.item()
        
        # Third injection is contradictory (maps to RETER [0.0, 1.0]). Surprise should increase.
        track_model.update_memory("Nova Diretriz: É proibida a exclusão de dados em auditoria.", self.ACTION_RETER)
        updated_surprise = track_model.hope_module.memory.surprise_momentum.item()
        
        self.assertGreater(updated_surprise, initial_surprise, "Surprise momentum did not increase on conflicting rule!")

    def test_05_decision_override_and_catastrophic_forgetting(self):
        """
        Verifies the full legal override workflow:
        1. Base rules mandate EXCLUIR.
        2. Credit-related query predicts EXCLUIR.
        3. Anti-fraud rule overrides this specifically for credit cases to RETER.
        4. Same query now predicts RETER.
        5. General (non-credit) query still predicts EXCLUIR (no catastrophic forgetting).
        """
        eval_model = LegalIncrementalModel(model_name='neuralmind/bert-base-portuguese-cased')
        
        base_knowledge = [
            "Artigo 1: Todo cliente tem o direito inalienável de solicitar a exclusão completa de seus dados pessoais.",
            "Artigo 2: O processo de exclusão de dados deve ser concluído em até 15 dias úteis.",
            "Artigo 3: A análise de crédito depende do histórico de transações."
        ]
        
        # Phase 1: Inject base legislation (excluir)
        for rule in base_knowledge:
            eval_model.update_memory(rule, self.ACTION_EXCLUIR)
            
        credit_query = "O cliente João quitou um empréstimo mês passado e quer excluir o histórico de crédito do banco."
        general_query = "O cliente pediu para apagar o e-mail cadastrado e o número de telefone das comunicações."
        
        # Phase 2: Verify prediction before update
        res_before = eval_model.predict(credit_query)
        self.assertGreater(res_before["predicted_action_vector"][0], res_before["predicted_action_vector"][1],
                           "Model did not default to EXCLUIR before update.")
        
        # Phase 3: Inject the overriding anti-fraud directive (reter)
        new_normative_rule = "Nova Diretriz Antifraude: É estritamente proibida a exclusão de dados vinculados a históricos de operações de crédito ativas ou liquidadas nos últimos 5 anos."
        eval_model.update_memory(new_normative_rule, self.ACTION_RETER)
        
        # Phase 4: Verify that credit query is overridden to RETER
        res_after_credit = eval_model.predict(credit_query)
        self.assertGreater(res_after_credit["predicted_action_vector"][1], res_after_credit["predicted_action_vector"][0],
                           "Model failed to override decision to RETER for credit history query!")
        self.assertEqual(res_after_credit["most_relevant_rule"], new_normative_rule,
                         "The most relevant rule for credit query was not updated to the new directive.")
        
        # Phase 5: Verify that general query still predicts EXCLUIR (catastrophic forgetting guard)
        res_after_general = eval_model.predict(general_query)
        self.assertGreater(res_after_general["predicted_action_vector"][0], res_after_general["predicted_action_vector"][1],
                           "Model suffered from catastrophic forgetting and altered non-credit prediction!")

if __name__ == '__main__':
    unittest.main()
