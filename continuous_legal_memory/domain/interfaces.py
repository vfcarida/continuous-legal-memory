"""
Hexagonal Ports (Abstract Interfaces) for Continuous Legal Memory.

Following clean Hexagonal Architecture (Ports & Adapters), these interfaces decouple core legal reasoning
and neural associative memory operations from external infrastructure dependencies such as HuggingFace models,
local Ollama edge runtimes, vector database providers, or local storage layers.
"""

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
    def get_records(self) -> list[MemoryRecord]:
        """Retrieve all currently registered memory records."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Purge all stored memory records."""
        pass
