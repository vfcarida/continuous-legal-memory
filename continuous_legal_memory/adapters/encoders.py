"""
Infrastructure Adapters for Semantic Text Encoders.

This module provides concrete implementations of the `BaseEncoderPort` interface, enabling
interoperability with pre-trained HuggingFace Transformers, SentenceTransformers, and local edge runtimes.
"""

import torch
from transformers import AutoModel, AutoTokenizer

from continuous_legal_memory.domain.exceptions import EncoderInferenceError
from continuous_legal_memory.domain.interfaces import BaseEncoderPort


class HuggingFaceEncoderAdapter(BaseEncoderPort):
    """
    Adapter bridging HuggingFace AutoModel transformers into the `BaseEncoderPort` interface.

    Rationale:
        Encapsulates pre-trained transformer model initialization, tokenization, mean pooling,
        and parameter freezing. Strictly disables gradient computation (`requires_grad = False`)
        to guarantee zero backpropagation on the frozen base encoder.
    """

    def __init__(self, model_name: str = "neuralmind/bert-base-portuguese-cased", max_length: int = 256) -> None:
        """
        Initialize the HuggingFace encoder adapter.

        Args:
            model_name: HuggingFace hub model string or local file path.
            max_length: Maximum sequence token truncation limit for tokenization.

        Raises:
            EncoderInferenceError: If model or tokenizer fails to load.
        """
        self.model_name = model_name
        self.max_length = max_length

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()

            # Strictly freeze encoder parameters to satisfy zero-backprop base model requirements
            for param in self.model.parameters():
                param.requires_grad = False

        except Exception as e:
            raise EncoderInferenceError(
                f"Failed to initialize HuggingFace encoder model '{model_name}': {e}",
                payload={"model_name": model_name},
            ) from e

        self._embedding_dim = self.model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        """Return the hidden vector dimension produced by the underlying transformer model."""
        return self._embedding_dim

    def get_embedding(self, texts: list[str]) -> torch.Tensor:
        """
        Generate semantic text embeddings using masked mean pooling over hidden states.

        Args:
            texts: List of non-empty text strings to embed.

        Returns:
            A 2D PyTorch Tensor of shape (len(texts), hidden_size).

        Raises:
            EncoderInferenceError: If inputs are invalid or tokenizer/model execution throws an error.
        """
        if not texts or any(not isinstance(t, str) or len(t.strip()) == 0 for t in texts):
            raise EncoderInferenceError("Input texts list must contain non-empty string elements.")

        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state

            # Compute attention-masked mean pooling over sentence tokens
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

            return sum_embeddings / sum_mask

        except Exception as e:
            raise EncoderInferenceError(
                f"Encoder inference failure during embedding generation: {e}",
                payload={"texts_count": len(texts)},
            ) from e
