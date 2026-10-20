"""
Automated Normative Entity & Deontic Graph Extractor (`NormativeGraphExtractor`).

Infers semantic `EntityType` (`OBLIGATION`, `PROHIBITION`, `PERMISSION`, `STATUTE`) from
deontic modal operators and automatically discovers `RelationType.SUPERSEDES` and
`RelationType.DEPENDS_ON` graph edges from textual supersession clauses and shared
statutory pin-cites (`extract_legal_citations`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from continuous_legal_memory.core.memory_tiers.semantic_memory import SemanticKnowledgeGraph
from continuous_legal_memory.domain.models import EntityType, RelationType
from continuous_legal_memory.reasoning.modal_verifier import (
    _OBLIGATION_RE,
    _PERMISSION_RE,
    _PROHIBITION_RE,
)
from continuous_legal_memory.retrieval.hybrid_lexical import extract_legal_citations

_TEXTUAL_SUPERSEDES_RE = re.compile(
    r"\b(?:supersedes|repeals|replaces|overrides|revoga|substitui|abroge|remplace|ersetzt)\s+([A-Za-z0-9_.\-]+)",
    re.IGNORECASE,
)


@dataclass
class ExtractedNormStructure:
    """
    Structured output of `NormativeGraphExtractor.extract()`.
    """

    node_id: str
    entity_type: EntityType
    citations: set[str] = field(default_factory=set)
    edges: list[tuple[str, RelationType, float]] = field(default_factory=list)


class NormativeGraphExtractor:
    """
    Deterministic extractor mapping legal rule text and metadata into `EntityType`
    classifications and acyclic `SemanticKnowledgeGraph` edges.
    """

    @staticmethod
    def infer_entity_type(
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> EntityType:
        """
        Infer `EntityType` from explicit metadata or deontic modal cues in `text`.
        Defaults to `EntityType.STATUTE` when no explicit modal operator is present.
        """
        if metadata and "entity_type" in metadata:
            raw_et = metadata["entity_type"]
            if isinstance(raw_et, EntityType):
                return raw_et
            if isinstance(raw_et, str):
                try:
                    return EntityType(raw_et)
                except ValueError:
                    pass

        if _PROHIBITION_RE.search(text):
            return EntityType.PROHIBITION
        if _PERMISSION_RE.search(text) and not _OBLIGATION_RE.search(text):
            return EntityType.PERMISSION
        if _OBLIGATION_RE.search(text):
            return EntityType.OBLIGATION
        return EntityType.STATUTE

    @staticmethod
    def _would_create_cycle(
        graph: SemanticKnowledgeGraph,
        source_id: str,
        target_id: str,
        tenant_id: str,
    ) -> bool:
        if source_id == target_id:
            return True
        visited: set[str] = set()
        stack = [target_id]
        while stack:
            curr = stack.pop()
            if curr == source_id:
                return True
            if curr in visited:
                continue
            visited.add(curr)
            for edge in graph.edges:
                if (
                    edge.tenant_id == tenant_id
                    and edge.source_id == curr
                    and edge.relation_type
                    in (
                        RelationType.SUPERSEDES,
                        RelationType.DEPENDS_ON,
                        RelationType.DERIVES_FROM,
                    )
                ):
                    stack.append(edge.target_id)
        return False

    @classmethod
    def extract(
        cls,
        node_id: str,
        text: str,
        metadata: dict[str, Any] | None,
        graph: SemanticKnowledgeGraph,
        tenant_id: str = "default",
    ) -> ExtractedNormStructure:
        """
        Extract `EntityType`, canonical citations, and acyclic `SemanticKnowledgeGraph`
        edges (`SUPERSEDES` and `DEPENDS_ON`) for `node_id`.
        """
        entity_type = cls.infer_entity_type(text, metadata=metadata)
        citations = extract_legal_citations(text)
        seen_targets: set[tuple[str, RelationType]] = set()
        edges: list[tuple[str, RelationType, float]] = []

        def _try_add_edge(tgt: str, rel: RelationType, weight: float) -> None:
            if not tgt or tgt == node_id or tgt not in graph.nodes:
                return
            if getattr(graph.nodes[tgt], "tenant_id", "default") != tenant_id:
                return
            if (tgt, rel) in seen_targets:
                return
            if rel in (
                RelationType.SUPERSEDES,
                RelationType.DEPENDS_ON,
                RelationType.DERIVES_FROM,
            ) and cls._would_create_cycle(graph, node_id, tgt, tenant_id):
                return
            seen_targets.add((tgt, rel))
            edges.append((tgt, rel, weight))

        # 1. Explicit metadata relations
        if metadata:
            if "relations" in metadata and isinstance(metadata["relations"], list):
                for rel in metadata["relations"]:
                    if not isinstance(rel, dict):
                        continue
                    target = str(rel.get("target_id", ""))
                    raw_rel = rel.get("relation_type", RelationType.DEPENDS_ON)
                    rel_type = RelationType(raw_rel) if isinstance(raw_rel, str) else raw_rel
                    weight = float(rel.get("weight", 1.0))
                    if target and target in graph.nodes:
                        seen_targets.add((target, rel_type))
                        edges.append((target, rel_type, weight))

            sup = metadata.get("supersedes")
            if isinstance(sup, str):
                _try_add_edge(sup, RelationType.SUPERSEDES, 1.0)
            elif isinstance(sup, (list, tuple, set)):
                for s_id in sup:
                    _try_add_edge(str(s_id), RelationType.SUPERSEDES, 1.0)

        # 2. Textual supersession clauses ("supersedes <node_id>")
        for match in _TEXTUAL_SUPERSEDES_RE.finditer(text):
            candidate_target = match.group(1).rstrip(".,;:")
            if candidate_target in graph.nodes:
                _try_add_edge(candidate_target, RelationType.SUPERSEDES, 1.0)

        # 3. Shared statutory citation cross-reference linking
        if citations:
            for existing_id, existing_node in graph.nodes.items():
                if existing_id == node_id or getattr(existing_node, "tenant_id", "default") != tenant_id:
                    continue
                if any(t == existing_id for t, _, _ in edges):
                    continue
                existing_cites = extract_legal_citations(existing_node.description)
                if citations & existing_cites:
                    _try_add_edge(existing_id, RelationType.DEPENDS_ON, 0.8)

        return ExtractedNormStructure(
            node_id=node_id,
            entity_type=entity_type,
            citations=citations,
            edges=edges,
        )
