"""
Hexagonal Ports (Abstract Interfaces) for Continuous Legal Memory.

Following clean Hexagonal Architecture (Ports & Adapters), these interfaces decouple core legal reasoning
and neural associative memory operations from external infrastructure dependencies such as HuggingFace models,
local Ollama edge runtimes, vector database providers, or local storage layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from continuous_legal_memory.domain.models import MemoryRecord


class BaseEncoderPort(ABC):
    """
    Abstract Port for semantic text embedding model adapters.

    Rationale:
        Decouples PyTorch neural memory networks from concrete model implementations (e.g. HuggingFace BERT,
        SentenceTransformers, Ollama Gemma 4 E2B). Allows seamless offline, privacy-first edge execution.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of output embeddings produced by this encoder."""
        pass

    @abstractmethod
    def get_embedding(self, texts: list[str]) -> torch.Tensor:
        """
        Generate semantic key embedding vectors for a list of text strings.

        Args:
            texts: List of text inputs to convert into dense vector embeddings.

        Returns:
            A 2D PyTorch Tensor of shape (len(texts), embedding_dim).

        Raises:
            EncoderInferenceError: If model inference fails.
        """
        pass


class BaseMemoryStorePort(ABC):
    """
    Abstract Port for memory record storage and associative indexing.

    Rationale:
        Abstracts lower-level tensor buffers and graph structures, enabling pluggable persistence backends.
    """

    @abstractmethod
    def add_record(self, record: MemoryRecord) -> None:
        """Store a new legal memory record in the persistent index."""
        pass

    @abstractmethod
    def get_records(self, tenant_id: str | None = None) -> list[MemoryRecord]:
        """Retrieve all currently registered memory records, optionally filtered by tenant."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Purge all stored memory records."""
        pass


class BaseRetrieverPort(ABC):
    """
    Abstract Port for legal memory retrieval engines.

    Rationale:
        Decouples attention calculation, keyword searching (BM25), and semantic vector scoring
        from the neural memory module (HopeModule) and orchestrator, enabling pluggable retrieval engines
        and baseline comparisons.
    """

    @abstractmethod
    def retrieve(
        self,
        query_embed: torch.Tensor,
        keys: torch.Tensor,
        rule_importance: torch.Tensor,
        temperature: float = 0.05,
        query_text: str | None = None,
        records: list[MemoryRecord] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
        """
        Compute retrieval alignment scores, attention distribution, and optional text snippets.

        Args:
            query_embed: 2D PyTorch Tensor of shape (batch_size, embed_dim).
            keys: 2D PyTorch Tensor of shape (num_keys, embed_dim).
            rule_importance: 1D or 2D Tensor containing rule importance weights.
            temperature: Softmax scaling temperature for attention sharpness.
            query_text: Optional query text string for lexical/keyword scoring.
            records: Optional list of MemoryRecord objects corresponding to keys.

        Returns:
            Tuple containing:
            - raw_scores: 2D Tensor (batch_size, num_keys) unscaled alignment/similarity scores.
            - attention_weights: 2D Tensor (batch_size, num_keys) normalized softmax probability distribution.
            - snippets: Optional list of extracted text snippet strings, or None.
        """
        pass


# Backward-compatibility / ergonomic alias
RetrieverPort = BaseRetrieverPort
