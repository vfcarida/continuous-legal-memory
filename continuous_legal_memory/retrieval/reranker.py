"""
Calibrated Cross-Encoder Legal Rerankers.

Provides second-stage pairwise query-statute interaction scoring (`BaseRerankerPort`)
to resolve fine-grained statutory pinpoint citations, deontic modal alignment, and
jurisdictional scope before legal precedence resolution.

Includes:
1. `DeonticCrossEncoderReranker`: Deterministic, zero-network cross-attention and
   lexical-deontic reranker that executes in microseconds on CPU.
2. `HuggingFaceCrossEncoderReranker`: Optional neural cross-encoder adapter wrapping
   `sentence_transformers.CrossEncoder` when the `[nlp]` optional extra is installed.
"""

from __future__ import annotations

import re
from typing import Any

import torch

from continuous_legal_memory.domain.interfaces import BaseRerankerPort
from continuous_legal_memory.domain.models import MemoryRecord

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "Can", "can", "do", "does",
    "for", "from", "has", "have", "how", "if", "in", "is", "it", "its", "of",
    "on", "or", "our", "should", "that", "the", "their", "this", "to", "under",
    "was", "we", "what", "when", "where", "which", "who", "with",
})

_CITATION_PATTERN = re.compile(
    r"\b(?:art(?:icle)?\.?\s*\d+|sec(?:tion)?\.?\s*\d+|cfr\s*\d+|lgpd|gdpr|hipaa|glba|ccpa|cpra|ai\s+act)\b",
    re.IGNORECASE,
)

_DEONTIC_MODALS: dict[str, frozenset[str]] = {
    "prohibition": frozenset({"prohibited", "forbidden", "unlawful", "ban", "banned", "never", "cannot"}),
    "obligation": frozenset({"must", "shall", "mandatory", "required", "obligated", "compulsory", "retain"}),
    "erasure": frozenset({"erase", "erasure", "delete", "deletion", "purge", "remove", "forget"}),
    "permission": frozenset({"may", "permitted", "allowed", "exempt", "exception", "authorized"}),
}


def _extract_content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-z0-9]{2,}\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _extract_citations(text: str) -> set[str]:
    return {re.sub(r"\s+", " ", m.group(0).lower().strip()) for m in _CITATION_PATTERN.finditer(text)}


class DeonticCrossEncoderReranker(BaseRerankerPort):
    """
    Deterministic, zero-network legal cross-encoder reranker.

    Computes pairwise interaction scores in `[0.0, 1.0]` between a legal query and candidate
    `MemoryRecord` objects by combining:
    - Fine-grained content-token recall and F1 overlap (`w_lexical = 0.55`)
    - Statutory citation / article number exact pinpoint matching (`w_citation = 0.25`)
    - Deontic modal category alignment (`w_deontic = 0.20`)
    """

    def __init__(
        self,
        blend_weight: float = 0.35,
        w_lexical: float = 0.55,
        w_citation: float = 0.25,
        w_deontic: float = 0.20,
    ) -> None:
        self.blend_weight = float(blend_weight)
        self.w_lexical = float(w_lexical)
        self.w_citation = float(w_citation)
        self.w_deontic = float(w_deontic)

    def score_pairs(
        self,
        query_text: str,
        records: list[MemoryRecord],
    ) -> torch.Tensor:
        """
        Compute pairwise interaction scores in `[0.0, 1.0]` for `(query_text, record)`.
        """
        if not records:
            return torch.empty(0, dtype=torch.float32)

        q_tokens = _extract_content_tokens(query_text)
        q_citations = _extract_citations(query_text)
        q_deontic = {
            cat for cat, vocab in _DEONTIC_MODALS.items() if q_tokens.intersection(vocab)
        }

        scores: list[float] = []
        for rec in records:
            r_tokens = _extract_content_tokens(rec.text)
            if not q_tokens or not r_tokens:
                lex_score = 0.0
            else:
                overlap = len(q_tokens.intersection(r_tokens))
                recall_q = overlap / len(q_tokens)
                precision_r = overlap / len(r_tokens)
                f1 = (
                    2.0 * recall_q * precision_r / (recall_q + precision_r)
                    if (recall_q + precision_r) > 0
                    else 0.0
                )
                lex_score = 0.65 * recall_q + 0.35 * f1

            r_citations = _extract_citations(rec.text)
            if q_citations and r_citations:
                cit_score = len(q_citations.intersection(r_citations)) / len(q_citations)
            elif not q_citations:
                cit_score = lex_score
            else:
                cit_score = 0.0

            r_deontic = {
                cat for cat, vocab in _DEONTIC_MODALS.items() if r_tokens.intersection(vocab)
            }
            if q_deontic and r_deontic:
                deontic_score = (
                    1.0 if q_deontic.intersection(r_deontic) else 0.25
                )
            else:
                deontic_score = lex_score

            combined = (
                self.w_lexical * lex_score
                + self.w_citation * cit_score
                + self.w_deontic * deontic_score
            )
            scores.append(float(max(0.0, min(1.0, combined))))

        return torch.tensor(scores, dtype=torch.float32)

    def blend_scores(
        self,
        bi_encoder_scores: torch.Tensor,
        query_text: str,
        records: list[MemoryRecord],
    ) -> torch.Tensor:
        """
        Interpolate bi-encoder cosine similarities `(batch_size, num_keys)` with cross-encoder
        pairwise interaction scores `(num_keys,)`.
        """
        if not records or bi_encoder_scores.numel() == 0:
            return bi_encoder_scores
        cross_scores = self.score_pairs(query_text, records).to(bi_encoder_scores.device)
        return (1.0 - self.blend_weight) * bi_encoder_scores + self.blend_weight * cross_scores.unsqueeze(0)


class HuggingFaceCrossEncoderReranker(BaseRerankerPort):
    """
    Optional neural Cross-Encoder reranker wrapping `sentence_transformers.CrossEncoder`.

    Falls back cleanly to `DeonticCrossEncoderReranker` when `allow_fallback=True` and
    `sentence-transformers` is not installed or offline mode is active.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        blend_weight: float = 0.35,
        allow_fallback: bool = True,
    ) -> None:
        self.model_name = model_name
        self.blend_weight = float(blend_weight)
        self.allow_fallback = allow_fallback
        self._fallback = DeonticCrossEncoderReranker(blend_weight=blend_weight)
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            return self._model
        except Exception as exc:
            if self.allow_fallback:
                return None
            raise RuntimeError(
                f"Failed to load HuggingFace CrossEncoder '{self.model_name}'. "
                "Install `continuous-legal-memory[nlp]` or set allow_fallback=True."
            ) from exc

    def score_pairs(
        self,
        query_text: str,
        records: list[MemoryRecord],
    ) -> torch.Tensor:
        if not records:
            return torch.empty(0, dtype=torch.float32)
        model = self._get_model()
        if model is None:
            return self._fallback.score_pairs(query_text, records)
        pairs = [(query_text, r.text) for r in records]
        raw_logits = model.predict(pairs)
        logits_tensor = torch.tensor(raw_logits, dtype=torch.float32)
        return torch.sigmoid(logits_tensor)
