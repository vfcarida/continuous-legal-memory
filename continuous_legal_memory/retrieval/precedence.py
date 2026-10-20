"""
Legal Precedence Scoring and Hierarchy Resolution.

Implements formal tri-canon statutory conflict resolution:
1. Explicit SUPERSEDES relationships (statute amendments, higher court precedents).
2. Hierarchical Authority Tiers (lex superior derogat legi inferiori): Constitutional / statutory
   mandates strictly override subordinate administrative or contractual policies.
3. Normative Specificity (lex specialis derogat legi generali): More specific clauses override
   general provisions within matching authority tiers.
4. Chronological Recency (lex posterior derogat legi priori): Newer provisions override older
   provisions in pari materia within matching authority and specificity tiers.
"""

from dataclasses import dataclass
from datetime import timezone

import torch

from continuous_legal_memory.domain.models import MemoryRecord, RelationType


@dataclass
class PrecedenceConfig:
    """
    Configuration parameters for legal precedence scoring and penalty calibration.

    Attributes:
        superseded_penalty: Logit penalty deducted for explicitly superseded statutes. Defaults to 1000.0.
        subordinate_authority_penalty_scale: Scale factor multiplied by authority rank delta. Defaults to 20.0.
        subordinate_specificity_penalty_scale: Scale factor multiplied by specificity rank delta. Defaults to 10.0.
        prior_temporal_penalty: Logit penalty applied to older rules in pari materia under lex posterior. Defaults to 2.0.
        candidate_similarity_margin: Margin from max similarity to consider candidate relevant. Defaults to 0.35.
        recency_tie_margin: Similarity margin within equal authority/specificity tiers for lex posterior. Defaults to 0.20.
        pari_materia_similarity: Minimum pairwise key similarity for implicit lex posterior repeal. Defaults to 0.35.
        min_candidate_similarity: Minimum similarity required for candidate evaluation. Defaults to 0.15.
        use_boolean_masking: If True, uses -inf for superseded records to avoid mixed-precision underflow. Defaults to False.
    """

    superseded_penalty: float = 1000.0
    subordinate_authority_penalty_scale: float = 20.0
    subordinate_specificity_penalty_scale: float = 10.0
    prior_temporal_penalty: float = 2.0
    candidate_similarity_margin: float = 0.35
    recency_tie_margin: float = 0.20
    pari_materia_similarity: float = 0.35
    min_candidate_similarity: float = 0.15
    use_boolean_masking: bool = False


def _extract_specificity(record: MemoryRecord) -> float:
    """Extract numeric specificity score from a MemoryRecord or its metadata."""
    attr_spec = getattr(record, "specificity", None)
    if attr_spec is not None:
        return float(attr_spec)
    if isinstance(record.metadata, dict):
        raw = record.metadata.get("specificity", record.metadata.get("specificity_rank", 0.0))
        if isinstance(raw, (int, float, str)):
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _extract_valid_timestamp(record: MemoryRecord) -> float:
    """Extract UTC POSIX timestamp from record valid_from for chronological ordering."""
    vf = getattr(record, "valid_from", None)
    if vf is None:
        return 0.0
    if vf.tzinfo is None:
        vf = vf.replace(tzinfo=timezone.utc)
    return float(vf.timestamp())


def _are_in_pari_materia(
    rec_a: MemoryRecord,
    rec_b: MemoryRecord,
    sim_a: float = 0.0,
    sim_b: float = 0.0,
    threshold: float = 0.35,
) -> bool:
    """Determine whether two norms address the same subject matter (in pari materia)."""
    ka = getattr(rec_a, "key_vector", None)
    kb = getattr(rec_b, "key_vector", None)
    if (
        isinstance(ka, torch.Tensor)
        and isinstance(kb, torch.Tensor)
        and ka.numel() > 0
        and kb.numel() > 0
    ):
        na = float(torch.norm(ka).item())
        nb = float(torch.norm(kb).item())
        if na > 1e-6 and nb > 1e-6:
            va = ka.reshape(-1).float() / na
            vb = kb.reshape(-1).float() / nb
            pair_sim = float(torch.dot(va, vb).item())
            return pair_sim >= threshold
    return abs(sim_a - sim_b) <= 0.05


def apply_legal_precedence(
    scores: torch.Tensor,
    rule_importance: torch.Tensor,
    records: list[MemoryRecord] | None,
    config: PrecedenceConfig | None = None,
) -> torch.Tensor:
    """
    Adjust retrieval scores according to explicit legal precedence and tri-canon hierarchy.

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
    specificities = [_extract_specificity(r) for r in records]
    vf_timestamps = [_extract_valid_timestamp(r) for r in records]
    has_distinct_vf = (max(vf_timestamps) - min(vf_timestamps)) > 1.0
    chrono_keys = [
        (round(vf_timestamps[i], 0) if has_distinct_vf else 0.0, i)
        for i in range(len(records))
    ]

    # Detect explicit SUPERSEDES relations
    superseded_indices: set[int] = set()
    record_id_to_idx: dict[str, int] = {}
    for i, r in enumerate(records):
        if r.record_id:
            record_id_to_idx[r.record_id] = i
        if isinstance(r.metadata, dict) and r.metadata.get("node_id"):
            record_id_to_idx[str(r.metadata["node_id"])] = i

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

    adjusted = scaled_scores.clone()

    # Suppress superseded records
    for idx in superseded_indices:
        if cfg.use_boolean_masking:
            adjusted[:, idx] = -float("inf")
        else:
            adjusted[:, idx] -= cfg.superseded_penalty

    # Apply Private International Law choice-of-law bonuses when present in record metadata
    for idx, rec in enumerate(records):
        if isinstance(rec.metadata, dict) and "_choice_of_law_bonus" in rec.metadata:
            col_bonus = float(rec.metadata["_choice_of_law_bonus"])
            if col_bonus != 0.0:
                adjusted[:, idx] += col_bonus

    # Apply tri-canon hierarchy across active candidate pools
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

        if not relevant_indices:
            continue

        # 1. Lex Superior (authority_rank hierarchy)
        max_authority = max(ranks[i] for i in relevant_indices)
        for i in relevant_indices:
            if ranks[i] < max_authority:
                adjusted[b, i] -= (max_authority - ranks[i]) * cfg.subordinate_authority_penalty_scale

        # 2. Lex Specialis (specificity hierarchy within equal highest authority tier)
        top_authority_indices = [i for i in relevant_indices if ranks[i] == max_authority]
        if len(top_authority_indices) > 1:
            max_spec = max(specificities[i] for i in top_authority_indices)
            for i in top_authority_indices:
                if specificities[i] < max_spec:
                    adjusted[b, i] -= (max_spec - specificities[i]) * cfg.subordinate_specificity_penalty_scale

            # 3. Lex Posterior (chronological recency in pari materia within equal authority & specificity)
            top_spec_indices = [i for i in top_authority_indices if specificities[i] == max_spec]
            if len(top_spec_indices) > 1 and cfg.prior_temporal_penalty > 0.0:
                max_top_sim = max(batch_scores[i].item() for i in top_spec_indices)
                near_tie_indices = [
                    i
                    for i in top_spec_indices
                    if batch_scores[i].item() >= (max_top_sim - cfg.recency_tie_margin)
                ]
                if len(near_tie_indices) > 1:
                    for i in near_tie_indices:
                        has_newer_pari_materia = any(
                            chrono_keys[j] > chrono_keys[i]
                            and _are_in_pari_materia(
                                records[i],
                                records[j],
                                batch_scores[i].item(),
                                batch_scores[j].item(),
                                cfg.pari_materia_similarity,
                            )
                            for j in near_tie_indices
                            if j != i
                        )
                        if has_newer_pari_materia:
                            adjusted[b, i] -= cfg.prior_temporal_penalty

    return adjusted
