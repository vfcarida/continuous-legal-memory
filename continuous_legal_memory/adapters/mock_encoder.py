"""
Deterministic Offline Semantic Mock Encoder.

Provides a fast, zero-network mock embedding adapter mapping legal concepts
into deterministic, orthogonal subspaces for air-gapped unit testing,
CLI demonstrations, and offline verification without requiring PyTorch HuggingFace downloads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from continuous_legal_memory.domain.interfaces import BaseEncoderPort


class SemanticMockEncoder(BaseEncoderPort):
    """
    Deterministic, offline-safe mock encoder for unit tests and local CLI evaluation.

    Maps semantic clusters (e.g. credit/loan, privacy/erasure, health/medical, banking)
    to distinct orthogonal subspaces, enabling deterministic associative testing.
    """

    def __init__(self, embedding_dim: int = 64, seed: int = 42) -> None:
        self._embedding_dim = embedding_dim
        self._seed = seed

        # Seeded orthogonal concept prototypes
        g1 = torch.Generator().manual_seed(seed + 1)
        self.v_general = F.normalize(torch.randn(embedding_dim, generator=g1), dim=-1)

        g2 = torch.Generator().manual_seed(seed + 2)
        self.v_credit = F.normalize(torch.randn(embedding_dim, generator=g2), dim=-1)

        g3 = torch.Generator().manual_seed(seed + 3)
        self.v_privacy = F.normalize(torch.randn(embedding_dim, generator=g3), dim=-1)

        g4 = torch.Generator().manual_seed(seed + 4)
        self.v_health = F.normalize(torch.randn(embedding_dim, generator=g4), dim=-1)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def get_embedding(self, texts: list[str]) -> torch.Tensor:
        res: list[torch.Tensor] = []
        for t in texts:
            tl = t.lower()
            if any(k in tl for k in ("credit", "loan", "fraud", "interest", "bank")):
                base = self.v_credit
            elif any(k in tl for k in ("privacy", "gdpr", "erase", "delete", "personal", "esquecimento")):
                base = self.v_privacy
            elif any(k in tl for k in ("health", "medical", "patient", "clinical", "hospital")):
                base = self.v_health
            else:
                base = self.v_general

            # Deterministic pseudo-noise derived from text length to preserve cosine similarity clusters
            text_hash_factor = (len(tl) % 10) * 0.001
            vec = base + text_hash_factor * torch.ones(self._embedding_dim)
            res.append(F.normalize(vec, dim=-1))
        return torch.stack(res, dim=0)
