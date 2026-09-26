"""
Shared Pytest Fixtures for Continuous Legal Memory Test Suite.
"""

import pytest
import torch
import torch.nn.functional as F

from continuous_legal_memory.domain.interfaces import BaseEncoderPort
from continuous_legal_memory.evaluation.harness import DeterministicTokenEncoder


class SemanticMockEncoder(BaseEncoderPort):
    """
    Deterministic, offline-safe mock encoder for unit tests.

    Maps semantic clusters (e.g. credit/loan vs general/marketing) to distinct
    orthogonal subspaces, enabling fast, zero-network associative testing.
    """

    def __init__(self, embedding_dim: int = 64) -> None:
        self._embedding_dim = embedding_dim
        g_gen = torch.Generator().manual_seed(101)
        self.v_general = F.normalize(torch.randn(embedding_dim, generator=g_gen), dim=-1)
        g_cred = torch.Generator().manual_seed(202)
        self.v_credit = F.normalize(torch.randn(embedding_dim, generator=g_cred), dim=-1)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def get_embedding(self, texts: list[str]) -> torch.Tensor:
        res = []
        for t in texts:
            tl = t.lower()
            if any(k in tl for k in ("credit", "loan", "fraud")):
                base = self.v_credit
            else:
                base = self.v_general
            vec = base + 0.02 * torch.randn(self._embedding_dim)
            res.append(F.normalize(vec, dim=-1))
        return torch.stack(res, dim=0)


@pytest.fixture
def offline_encoder() -> BaseEncoderPort:
    """Fast, offline-safe mock embedding adapter for unit tests."""
    return SemanticMockEncoder(embedding_dim=64)


@pytest.fixture
def semantic_mock_encoder() -> SemanticMockEncoder:
    """Fixture returning a SemanticMockEncoder instance."""
    return SemanticMockEncoder(embedding_dim=64)


@pytest.fixture
def token_encoder() -> DeterministicTokenEncoder:
    """Fast deterministic token-hash encoder for unit tests."""
    return DeterministicTokenEncoder(embedding_dim=64)

