"""
Unit tests for LegalMemoryOrchestrator and core Continuum Memory System operations.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from continuous_legal_memory.adapters.encoders import HuggingFaceEncoderAdapter
from continuous_legal_memory.domain.exceptions import InvalidMemoryVectorError
from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.orchestrator import LegalMemoryOrchestrator
from tests.conftest import SemanticMockEncoder


@pytest.fixture
def shared_orchestrator(offline_encoder: BaseEncoderPort) -> LegalMemoryOrchestrator:
    """Initialize an orchestrator instance using the fast offline mock encoder."""
    return LegalMemoryOrchestrator(value_dim=2, encoder=offline_encoder)


def test_01_model_weights_are_frozen() -> None:
    """
    Verify that all parameters inside the base HuggingFace transformer model have requires_grad set to False,
    strictly complying with zero-backpropagation constraints on the base encoder.
    Uses a mocked HuggingFace transformer to ensure zero-network, sub-second test execution.
    """
    class DummyHFModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(32, 32)
            self.config = MagicMock()
            self.config.hidden_size = 32

    dummy_model = DummyHFModel()
    assert any(p.requires_grad for p in dummy_model.parameters()), "Initial mock parameters should require grad."

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("transformers.AutoModel.from_pretrained", return_value=dummy_model),
    ):
        adapter = HuggingFaceEncoderAdapter("dummy-model-test")
        for name, param in adapter.model.named_parameters():
            assert not param.requires_grad, f"Parameter {name} is not frozen!"


def test_02_empty_memory_edge_case(offline_encoder: BaseEncoderPort) -> None:
    """
    Verify that requesting predictions when memory is empty returns a clean, zeroed vector.
    """
    empty_orchestrator = LegalMemoryOrchestrator(value_dim=2, encoder=offline_encoder)
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


def test_04_surprise_momentum_tracking(semantic_mock_encoder: SemanticMockEncoder) -> None:
    """
    Validate that consecutive consistent rules produce low surprise values,
    while conflicting inputs trigger a spike in the surprise tracker.
    """
    track_orchestrator = LegalMemoryOrchestrator(
        value_dim=2,
        encoder=semantic_mock_encoder,
        seed=42,
        engine_mode="hybrid",
    )
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


def test_05_decision_override_and_catastrophic_forgetting(semantic_mock_encoder: SemanticMockEncoder) -> None:
    """
    Verify full legal override workflow:
    1. Base rules mandate data DELETION ([1.0, 0.0]).
    2. Credit query predicts DELETION.
    3. Overriding anti-fraud rule mandates RETAIN ([0.0, 1.0]) for credit data.
    4. Credit query now predicts RETAIN.
    5. Non-credit general query still predicts DELETION (no catastrophic forgetting).
    """
    eval_orchestrator = LegalMemoryOrchestrator(
        value_dim=2,
        encoder=semantic_mock_encoder,
        seed=42,
        engine_mode="hybrid",
    )
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

