"""
Unit tests for LegalMemoryOrchestrator and core Continuum Memory System operations.
"""

import pytest

from continuous_legal_memory.domain.exceptions import InvalidMemoryVectorError
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator


@pytest.fixture(scope="module")
def shared_orchestrator() -> LegalMemoryOrchestrator:
    """Initialize a single shared orchestrator instance to optimize HuggingFace model loading."""
    return LegalMemoryOrchestrator(value_dim=2)


def test_01_model_weights_are_frozen(shared_orchestrator: LegalMemoryOrchestrator) -> None:
    """
    Verify that all parameters inside the base HuggingFace BERT model have requires_grad set to False,
    strictly complying with zero-backpropagation constraints on the base encoder.
    """
    adapter = shared_orchestrator.encoder
    for name, param in adapter.model.named_parameters():
        assert not param.requires_grad, f"Parameter {name} is not frozen!"


def test_02_empty_memory_edge_case() -> None:
    """
    Verify that requesting predictions when memory is empty returns a clean, zeroed vector.
    """
    empty_orchestrator = LegalMemoryOrchestrator(value_dim=2)
    res = empty_orchestrator.predict("Query with no stored rules")
    assert res.predicted_action_vector == [0.0, 0.0]
    assert res.most_relevant_rule is None
    assert res.confidence is None


def test_03_input_validation_and_safety(shared_orchestrator: LegalMemoryOrchestrator) -> None:
    """
    Verify strict domain exception throwing on invalid query or action vector inputs.
    """
    with pytest.raises(InvalidMemoryVectorError):
        shared_orchestrator.predict("")

    with pytest.raises(InvalidMemoryVectorError):
        shared_orchestrator.predict("   ")

    with pytest.raises(InvalidMemoryVectorError):
        shared_orchestrator.update_memory("Valid rule", [1.0])  # Length must be 2

    with pytest.raises(InvalidMemoryVectorError):
        shared_orchestrator.update_memory("", [1.0, 0.0])  # Empty rule text


def test_04_surprise_momentum_tracking() -> None:
    """
    Validate that consecutive consistent rules produce low surprise values,
    while conflicting inputs trigger a spike in the surprise tracker.
    """
    track_orchestrator = LegalMemoryOrchestrator(value_dim=2)
    action_delete = [1.0, 0.0]
    action_retain = [0.0, 1.0]

    # Ingest baseline rule
    track_orchestrator.update_memory("Article 1: Customers may request full data deletion.", action_delete)

    # Ingest consistent rule (Surprise should remain low)
    track_orchestrator.update_memory("Article 2: Deletion requests must be executed quickly.", action_delete)
    initial_surprise = track_orchestrator.hope_module.memory.surprise_momentum.item()

    # Ingest contradictory rule (Surprise should spike)
    track_orchestrator.update_memory("New Anti-Fraud Rule: Deletion of active audit data is forbidden.", action_retain)
    updated_surprise = track_orchestrator.hope_module.memory.surprise_momentum.item()

    assert updated_surprise > initial_surprise, "Surprise momentum did not increase on conflicting rule!"


def test_05_decision_override_and_catastrophic_forgetting() -> None:
    """
    Verify full legal override workflow:
    1. Base rules mandate data DELETION ([1.0, 0.0]).
    2. Credit query predicts DELETION.
    3. Overriding anti-fraud rule mandates RETAIN ([0.0, 1.0]) for credit data.
    4. Credit query now predicts RETAIN.
    5. Non-credit general query still predicts DELETION (no catastrophic forgetting).
    """
    eval_orchestrator = LegalMemoryOrchestrator(value_dim=2)
    action_delete = [1.0, 0.0]
    action_retain = [0.0, 1.0]

    base_knowledge = [
        "Article 1: Every client has the right to request deletion of personal data.",
        "Article 2: Data deletion must be completed within 15 business days.",
        "Article 3: Credit evaluation depends on historical transaction data.",
    ]

    for rule in base_knowledge:
        eval_orchestrator.update_memory(rule, action_delete)

    credit_query = "Client John paid off a loan last month and wants his financial credit history deleted."
    general_query = "Client requested deletion of his marketing email address."

    # Prediction before new directive
    res_before = eval_orchestrator.predict(credit_query)
    assert res_before.predicted_action_vector[0] > res_before.predicted_action_vector[1]

    # Inject anti-fraud directive override
    override_rule = "New Directive: Deletion of credit operation records active in last 5 years is strictly prohibited."
    eval_orchestrator.update_memory(override_rule, action_retain)

    # Verify credit query is overridden to RETAIN
    res_after_credit = eval_orchestrator.predict(credit_query)
    assert res_after_credit.predicted_action_vector[1] > res_after_credit.predicted_action_vector[0]
    assert res_after_credit.most_relevant_rule == override_rule

    # Verify general query still predicts DELETION (guarding against catastrophic forgetting)
    res_after_general = eval_orchestrator.predict(general_query)
    assert res_after_general.predicted_action_vector[0] > res_after_general.predicted_action_vector[1]
