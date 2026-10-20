"""
Hybrid Lexical-Dense Normative Retrieval & Statutory Citation Pinpointing.

Implements Okapi BM25+ (Lv & Zhai, 2011) and canonical multi-jurisdictional legal
citation extraction (EU Articles/Regulations, US Code/CFR/Rules, BR Artigos/Leis)
to eliminate dense-embedding dilution on exact statutory pin-cites (Pipitone & Alami,
LegalBench-RAG, 2024).
"""

from __future__ import annotations

import math
import re
from collections import Counter

import torch

from continuous_legal_memory.domain.models import MemoryRecord

_TOKEN_RE = re.compile(r"[a-z0-9§º°]+(?:-[a-z0-9]+)*", re.IGNORECASE)

# Canonical patterns for statutory articles, sections, paragraphs, and numbered laws
_ARTICLE_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "art",
        re.compile(
            r"\b(?:art(?:icle|igo|ikel)?\.?\s*)(\d+[a-z]?(?:º|°)?(?:\(\d+\))*(?:\([a-z0-9]+\))*)",
            re.IGNORECASE,
        ),
    ),
    (
        "sec",
        re.compile(
            r"(?:§+\s*|(?:\b(?:sec(?:tion)?|ss?)\.?\s*))(\d+[a-z]?(?:-\d+)?(?:\([a-z0-9]+\))*)",
            re.IGNORECASE,
        ),
    ),
    (
        "usc",
        re.compile(
            r"\b(\d+)\s+u\.?s\.?c\.?\s*(?:§+\s*)?(\d+[a-z]?(?:-\d+)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "lei",
        re.compile(
            r"\blei\s+(?:n[º°.]?\s*)?(\d+(?:\.\d+)?(?:/\d{2,4})?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "eu_reg",
        re.compile(
            r"\b(?:regulation|directive)\s*(?:\([a-z]{2,4}\)\s*)?(?:no\.?\s*)?(\d{4}/\d+|\d+/\d{4})\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ch_sr",
        re.compile(
            r"\b(?:sr|rs)\s+(\d{3}(?:\.\d+)+)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rule",
        re.compile(
            r"\brule\s+(\d+[a-z]?-\d+|\d+(?:\.\d+)?)\b",
            re.IGNORECASE,
        ),
    ),
)

_INSTRUMENT_ALIASES: dict[str, tuple[str, ...]] = {
    "gdpr": ("gdpr", "2016/679", "rgpd"),
    "eu_ai_act": ("eu ai act", "2024/1689", "ai act"),
    "rome_i": ("rome i", "593/2008", "roma i"),
    "lgpd": ("lgpd", "13.709", "13709"),
    "ccpa": ("ccpa", "cpra", "1798."),
    "hipaa": ("hipaa", "164.5"),
    "uk_gdpr": ("uk gdpr", "uk-gdpr", "data protection act 2018", "dpa 2018"),
    "fca": ("fca handbook", "sysc"),
    "pdpa_sg": ("pdpa", "personal data protection act 2012"),
    "mas_trm": ("mas trm", "technology risk management"),
    "appi_jp": ("appi", "act on the protection of personal information"),
    "fadp_ch": ("fadp", "revdsg", "sr 235.1", "lpd"),
    "finma": ("finma", "finma circular"),
}


def extract_legal_citations(text: str) -> set[str]:
    """
    Extract canonicalized statutory citations and instrument identifiers from legal text.

    Examples:
        - `"Art. 17(3)(b) GDPR"` -> `{"art:17(3)(b)", "art:17", "inst:gdpr"}`
        - `"28 U.S.C. § 1332"` -> `{"usc:28:1332", "sec:1332"}`
        - `"Lei 13.709/2018 Artigo 7"` -> `{"lei:13709/2018", "art:7", "inst:lgpd"}`
    """
    if not text:
        return set()

    lowered = text.lower()
    citations: set[str] = set()

    for prefix, pattern in _ARTICLE_SECTION_PATTERNS:
        for match in pattern.finditer(lowered):
            if prefix == "usc":
                title, sec = match.group(1), match.group(2)
                citations.add(f"usc:{title}:{sec}")
                citations.add(f"sec:{sec}")
            elif prefix == "lei":
                raw_num = match.group(1).replace(".", "")
                citations.add(f"lei:{raw_num}")
                base_num = raw_num.split("/")[0]
                citations.add(f"lei:{base_num}")
            elif prefix == "art":
                raw_art = match.group(1).rstrip("º°")
                citations.add(f"art:{raw_art}")
                base_art = raw_art.split("(")[0]
                citations.add(f"art:{base_art}")
            else:
                norm_val = match.group(1)
                citations.add(f"{prefix}:{norm_val}")

    for inst_key, aliases in _INSTRUMENT_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            citations.add(f"inst:{inst_key}")

    return citations


def _tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) > 1]


class HybridLexicalCitationScorer:
    """
    Zero-dependency BM25+ and exact statutory citation matcher for legal retrieval.

    Computes a hybrid lexical relevance score in `[0.0, 1.0]` combining:
    1. Okapi BM25+ lexical term-frequency / inverse-document-frequency overlap.
    2. Exact statutory pin-cite overlap (`Art.`, `§`, `U.S.C.`, `Lei`, `Regulation`).
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        citation_weight: float = 0.60,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)
        self.citation_weight = float(max(0.0, min(1.0, citation_weight)))

    def score_citation_overlap(
        self,
        query_text: str,
        records: list[MemoryRecord],
    ) -> tuple[torch.Tensor, bool]:
        """
        Compute exact statutory citation match scores in `[0.0, 1.0]` for each record.

        Returns:
            Tuple of `(citation_scores_tensor, has_specific_citation)` where
            `has_specific_citation` is True if the query contains at least one concrete
            article/section/statute number (`art:`, `sec:`, `usc:`, `lei:`, `eu_reg:`, `rule:`).
        """
        n = len(records)
        if n == 0:
            return torch.zeros(0, dtype=torch.float32), False

        q_cites = extract_legal_citations(query_text)
        specific_q_cites = {c for c in q_cites if not c.startswith("inst:")}
        has_specific = len(specific_q_cites) > 0

        if not q_cites:
            return torch.zeros(n, dtype=torch.float32), False

        scores: list[float] = []
        for rec in records:
            meta_cite_str = str(rec.metadata.get("citation", "")) if isinstance(rec.metadata, dict) else ""
            r_cites = extract_legal_citations(f"{rec.text} {meta_cite_str}")
            if not r_cites:
                scores.append(0.0)
                continue

            overlap = q_cites & r_cites
            specific_overlap = specific_q_cites & r_cites

            if specific_q_cites:
                spec_ratio = len(specific_overlap) / float(len(specific_q_cites))
                inst_q = q_cites - specific_q_cites
                inst_ratio = (len(inst_q & r_cites) / float(len(inst_q))) if inst_q else 1.0
                # High score only when specific article/section matches; partial bonus if instrument also matches
                score = spec_ratio * (0.75 + 0.25 * inst_ratio) if spec_ratio > 0 else 0.0
            else:
                score = len(overlap) / float(len(q_cites))

            scores.append(float(min(1.0, score)))

        return torch.tensor(scores, dtype=torch.float32), has_specific

    def score_bm25_plus(
        self,
        query_text: str,
        records: list[MemoryRecord],
    ) -> torch.Tensor:
        """
        Compute normalized Okapi BM25+ lexical similarity scores in `[0.0, 1.0]`.
        """
        n = len(records)
        if n == 0:
            return torch.zeros(0, dtype=torch.float32)

        q_tokens = _tokenize(query_text)
        if not q_tokens:
            return torch.zeros(n, dtype=torch.float32)

        doc_tokens_list = [_tokenize(rec.text) for rec in records]
        doc_lens = [len(tokens) for tokens in doc_tokens_list]
        avgdl = max(1.0, sum(doc_lens) / float(n))

        df: Counter[str] = Counter()
        for tokens in doc_tokens_list:
            for term in set(tokens):
                df[term] += 1

        raw_scores: list[float] = []
        for tokens, dl in zip(doc_tokens_list, doc_lens):
            tf = Counter(tokens)
            score = 0.0
            for q_term in q_tokens:
                term_tf = tf.get(q_term, 0)
                if term_tf == 0:
                    continue
                idf = math.log(1.0 + (n - df[q_term] + 0.5) / (df[q_term] + 0.5))
                norm_tf = (term_tf * (self.k1 + 1.0)) / (
                    term_tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                )
                score += idf * (norm_tf + self.delta)
            raw_scores.append(score)

        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score <= 1e-9:
            return torch.zeros(n, dtype=torch.float32)

        norm_scores = [s / max_score for s in raw_scores]
        return torch.tensor(norm_scores, dtype=torch.float32)

    def score_hybrid_lexical(
        self,
        query_text: str,
        records: list[MemoryRecord],
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Compute combined BM25+ and statutory citation overlap scores.

        Returns:
            Tuple of `(combined_lexical_scores, citation_scores, has_specific_citation)`.
        """
        bm25_scores = self.score_bm25_plus(query_text, records)
        cite_scores, has_specific = self.score_citation_overlap(query_text, records)
        if has_specific:
            w_cite = self.citation_weight
            combined = (1.0 - w_cite) * bm25_scores + w_cite * cite_scores
        else:
            combined = bm25_scores
        return combined, cite_scores, has_specific


_DEFAULT_LEXICAL_SCORER = HybridLexicalCitationScorer()


def compute_lexical_citation_scores(
    query_text: str,
    records: list[MemoryRecord],
) -> list[float]:
    """
    Compute normalized BM25+ and statutory citation match scores in `[0.0, 1.0]` for `records`.
    """
    if not records:
        return []
    combined, _, _ = _DEFAULT_LEXICAL_SCORER.score_hybrid_lexical(query_text, records)
    return [float(x) for x in combined.detach().cpu().flatten().tolist()]

