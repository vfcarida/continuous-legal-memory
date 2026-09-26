"""
Legal Precedence Scoring and Hierarchy Resolution.

Implements legal doctrine conflict resolution:
1. Explicit SUPERSEDES relationships (statute amendments, higher court precedents).
2. Hierarchical Authority Tiers (lex superior derogat legi inferiori): Constitutional / statutory
   mandates strictly override subordinate administrative or contractual policies.
3. Recency (lex posterior derogat legi priori) applied strictly within matching authority tiers.
"""

from dataclasses import dataclass

import torch

from continuous_legal_memory.domain.models import MemoryRecord, RelationType


@dataclass
class PrecedenceConfig:
    """
    Configuration parameters for legal precedence scoring and penalty calibration.

    Attributes:
        superseded_penalty: Logit penalty deducted for explicitly superseded statutes. Defaults to 1000.0.
        subordinate_authority_penalty_scale: Scale factor multiplied by authority rank delta. Defaults to 20.0.
        candidate_similarity_margin: Margin from max similarity to consider candidate relevant. Defaults to 0.35.
        min_candidate_similarity: Minimum similarity required for candidate evaluation. Defaults to 0.15.
        use_boolean_masking: If True, uses -inf for superseded records to avoid mixed-precision underflow. Defaults to False.
    """

    superseded_penalty: float = 1000.0
    subordinate_authority_penalty_scale: float = 20.0
    candidate_similarity_margin: float = 0.35
    min_candidate_similarity: float = 0.15
    use_boolean_masking: bool = False


def apply_legal_precedence(
    scores: torch.Tensor,
    rule_importance: torch.Tensor,
    records: list[MemoryRecord] | None,
    config: PrecedenceConfig | None = None,
) -> torch.Tensor:
    """
    Adjust retrieval scores according to explicit legal precedence and authority hierarchy.

    Args:
        scores: 2D Tensor of shape (batch_size, num_keys) unscaled alignment similarities.
        rule_importance: 1D Tensor of shape (num_keys,) surprise-weighted importance factors.
        records: List of MemoryRecord objects corresponding to memory slots.
        config: Optional PrecedenceConfig instance defining penalty constants.

    Returns:
        Adjusted scores Tensor of shape (batch_size, num_keys).
    """
    cfg = config or PrecedenceConfig()
    scaled_scores = scores * rule_importance

    if not records or len(records) <= 1:
        return scaled_scores

    ranks = [getattr(r, "authority_rank", 1) for r in records]
    has_different_ranks = max(ranks) > min(ranks)

    # Detect explicit SUPERSEDES relations
    superseded_indices: set[int] = set()
    record_id_to_idx = {r.record_id: i for i, r in enumerate(records) if r.record_id}

    for j, rec in enumerate(records):
        # 1. Direct supersedes attribute in metadata
        superseded_target = rec.metadata.get("supersedes")
        if superseded_target and superseded_target in record_id_to_idx:
            target_idx = record_id_to_idx[superseded_target]
            if ranks[j] >= ranks[target_idx]:
                superseded_indices.add(target_idx)

        # 2. Relations list in metadata
        relations = rec.metadata.get("relations", [])
        if isinstance(relations, list):
            for rel in relations:
                if isinstance(rel, dict):
                    rel_type = rel.get("relation_type")
                    if rel_type in ("supersedes", RelationType.SUPERSEDES):
                        target_id = rel.get("target_id")
                        if target_id and target_id in record_id_to_idx:
                            target_idx = record_id_to_idx[target_id]
                            if ranks[j] >= ranks[target_idx]:
                                superseded_indices.add(target_idx)

    has_superseded = len(superseded_indices) > 0

    # If all authority ranks are identical and no rules are superseded, preserve exact baseline math
    if not has_different_ranks and not has_superseded:
        return scaled_scores

    adjusted = scaled_scores.clone()

    # Suppress superseded records
    for idx in superseded_indices:
        if cfg.use_boolean_masking:
            adjusted[:, idx] = -float("inf")
        else:
            adjusted[:, idx] -= cfg.superseded_penalty

    # Apply authority hierarchy across candidate pools
    if has_different_ranks:
        batch_size = scores.size(0)
        for b in range(batch_size):
            batch_scores = scores[b]
            max_sim = batch_scores.max().item()

            # Active candidate pool: semantically relevant to query and not superseded
            relevant_indices = [
                i
                for i in range(len(records))
                if i not in superseded_indices
                and batch_scores[i].item() >= (max_sim - cfg.candidate_similarity_margin)
                and batch_scores[i].item() > cfg.min_candidate_similarity
            ]

            if relevant_indices:
                max_authority = max(ranks[i] for i in relevant_indices)
                for i in range(len(records)):
                    if i in superseded_indices:
                        continue
                    if i in relevant_indices and ranks[i] < max_authority:
                        # Subordinate candidate cannot override superior authority
                        adjusted[b, i] -= (max_authority - ranks[i]) * cfg.subordinate_authority_penalty_scale

    return adjusted
