"""
Legal-Grade Precision Hybrid Retrieval Engine.

Combines BM25 keyword search, dense vector embeddings, and cross-encoder re-ranking with
exact character-level snippet extraction (LegalBench-RAG benchmark standard).
"""

import math
import re

import torch

from continuous_legal_memory.domain.interfaces import BaseEncoderPort, BaseRetrieverPort
from continuous_legal_memory.domain.models import MemoryRecord
from continuous_legal_memory.retrieval.hybrid_lexical import (
    compute_lexical_citation_scores,
    extract_legal_citations,
)
from continuous_legal_memory.retrieval.precedence import apply_legal_precedence


class BM25Okapi:
    """Simple, lightweight BM25 Okapi implementation for exact legal terminology keyword scoring."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_tokens = [self._tokenize(doc) for doc in corpus]
        self.avgdl = sum(len(d) for d in self.doc_tokens) / max(1, self.corpus_size)
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self._calc_idf()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _calc_idf(self) -> None:
        df: dict[str, int] = {}
        for doc in self.doc_tokens:
            frequencies: dict[str, int] = {}
            for token in doc:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                df[token] = df.get(token, 0) + 1

        for token, freq in df.items():
            # BM25 IDF formula
            self.idf[token] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query: str) -> list[float]:
        q_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for i, doc in enumerate(self.doc_tokens):
            doc_len = len(doc)
            for token in q_tokens:
                if token not in self.doc_freqs[i]:
                    continue
                freq = self.doc_freqs[i][token]
                idf = self.idf.get(token, 0.0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(1.0, self.avgdl)))
                scores[i] += idf * (numerator / denominator)

        return scores


class HybridLegalRetriever(BaseRetrieverPort):
    """
    Precision Hybrid Legal Retrieval Engine.

    Rationale:
        Legal text analysis requires both exact keyword precision (e.g. specific article numbers or latin phrases)
        and deep semantic vector retrieval. This engine:
        1. BM25 + Statutory Citation Pinpointing + Dense Hybrid Scoring: Combines keyword BM25 scores,
           canonical statutory citation matching, and vector cosine similarity.
        2. Exact Character Snippet Extraction: Extracts exact start and end character indices (`start_char`, `end_char`)
           centered on the highest-density statutory citation or multi-keyword span (LegalBench-RAG standard).
    """

    _STOPWORDS = frozenset(
        {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "with"}
    )

    def __init__(
        self,
        encoder: BaseEncoderPort,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
    ) -> None:
        """
        Initialize HybridLegalRetriever.

        Args:
            encoder: BaseEncoderPort instance.
            bm25_weight: Weight assigned to BM25 + citation lexical score.
            dense_weight: Weight assigned to dense vector cosine similarity score.
        """
        self.encoder = encoder
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

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
        Compute hybrid retrieval scores, attention weights, and optional snippets (BaseRetrieverPort).

        Combines BM25 keyword matching and statutory citation pinpointing (if records and query_text
        are provided) with dense vector cosine similarity, then applies hierarchical legal precedence.
        """
        query_norm = torch.nn.functional.normalize(query_embed, p=2, dim=-1)
        keys_norm = torch.nn.functional.normalize(keys, p=2, dim=-1)
        dense_scores = torch.matmul(query_norm, keys_norm.T)  # Shape: (batch, num_keys)

        snippets: list[str] | None = None
        if records and query_text:
            lex_scores = compute_lexical_citation_scores(query_text, records)
            bm25_tensor = torch.tensor([lex_scores], dtype=query_embed.dtype, device=query_embed.device)
            scores = self.bm25_weight * bm25_tensor + self.dense_weight * dense_scores
            snippets = [self.extract_character_snippet(r.text, query_text)[2] for r in records]
        else:
            scores = dense_scores

        scaled_scores = apply_legal_precedence(scores, rule_importance, records)
        attention_weights = torch.nn.functional.softmax(scaled_scores / temperature, dim=-1)

        return scaled_scores, attention_weights, snippets

    def retrieve_with_snippets(
        self,
        query: str,
        records: list[MemoryRecord],
        top_k: int = 3,
        snippet_length: int = 200,
    ) -> list[tuple[MemoryRecord, float, tuple[int, int], str]]:
        """
        Perform hybrid retrieval and extract minimal exact-character text snippets.

        Args:
            query: Query text.
            records: Stored memory records to retrieve from.
            top_k: Number of top records to return.
            snippet_length: Target character length for extracted snippet.

        Returns:
            List of tuples: (MemoryRecord, hybrid_score, (start_char_idx, end_char_idx), snippet_text).
        """
        if not records:
            return []

        norm_bm25 = compute_lexical_citation_scores(query, records)

        # Compute Dense Cosine Similarities
        query_vec = self.encoder.get_embedding([query])
        query_norm = torch.nn.functional.normalize(query_vec, p=2, dim=-1)

        key_tensors = torch.cat([r.key_vector for r in records], dim=0)
        keys_norm = torch.nn.functional.normalize(key_tensors, p=2, dim=-1)

        dense_sims = torch.matmul(query_norm, keys_norm.T).squeeze(0).tolist()
        if isinstance(dense_sims, float):
            dense_sims = [dense_sims]

        # Combine Hybrid Scores
        results = []
        for i, rec in enumerate(records):
            h_score = (self.bm25_weight * norm_bm25[i]) + (self.dense_weight * dense_sims[i])
            start_idx, end_idx, snippet = self.extract_character_snippet(rec.text, query, snippet_length)
            results.append((rec, h_score, (start_idx, end_idx), snippet))

        # Sort descending by hybrid score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def extract_character_snippet(self, text: str, query: str, max_chars: int = 200) -> tuple[int, int, str]:
        """
        Extract exact character start/end bounds surrounding the highest-density statutory citation
        or multi-keyword match window.

        Args:
            text: Full text document string.
            query: Query string.
            max_chars: Maximum window size in characters.

        Returns:
            Tuple of (start_char_index, end_char_index, extracted_snippet_str).
        """
        if not text:
            return 0, 0, ""

        lower_text = text.lower()
        q_citations = extract_legal_citations(query)

        # 1. Check if a statutory citation from the query appears in the text
        best_pos = -1
        if q_citations:
            for m in re.finditer(r"(?:art(?:icle|igo)?\.?|§+|sec(?:tion)?\.?|s[úu]mula)\s*\d+", lower_text):
                span_cites = extract_legal_citations(text[m.start() : min(len(text), m.end() + 24)])
                if span_cites & q_citations:
                    best_pos = m.start()
                    break

        # 2. Otherwise find the highest-density keyword window across non-stopword tokens
        if best_pos == -1:
            raw_words = re.findall(r"\w+", query.lower())
            content_words = [w for w in raw_words if w not in self._STOPWORDS and len(w) >= 2]
            target_words = content_words if content_words else raw_words

            candidate_positions: list[int] = []
            for w in target_words:
                pos = lower_text.find(w)
                while pos != -1:
                    candidate_positions.append(pos)
                    pos = lower_text.find(w, pos + len(w))

            if candidate_positions:
                best_density = -1.0
                best_pos = candidate_positions[0]
                for cand in candidate_positions:
                    win_start = max(0, cand - 20)
                    win_end = min(len(text), win_start + max_chars)
                    window_slice = lower_text[win_start:win_end]
                    density = sum(len(w) for w in set(target_words) if w in window_slice)
                    if density > best_density:
                        best_density = float(density)
                        best_pos = cand
            else:
                best_pos = 0

        start_idx = max(0, best_pos - 20)
        end_idx = min(len(text), start_idx + max_chars)
        snippet = text[start_idx:end_idx]

        return start_idx, end_idx, snippet

