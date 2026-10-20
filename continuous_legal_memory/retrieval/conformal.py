"""
Split-Conformal Legal Prediction Sets and Distribution-Free Epistemic Abstention.

Implements finite-sample Split Conformal Prediction (Vovk et al., 2005; Shafer & Vovk,
JMLR 2008; Angelopoulos & Bates, 2023) tailored for normative legal outcome spaces:

1. Non-conformity score:
       s(x, y) = 1.0 - g(sim(x)) * p_hat(y | x)
   where `p_hat(y | x)` is the normalized outcome probability for class `y` and `g(sim(x))`
   is a continuous retrieval-similarity gate that penalizes out-of-distribution queries.
2. Finite-sample calibration quantile at miscoverage rate `alpha` in `(0, 1)`:
       q_hat = s_{(ceil((n + 1) * (1 - alpha)))}
   guaranteeing marginal coverage P(Y_test in C_alpha(X_test)) >= 1 - alpha under exchangeability.
3. Epistemic Abstention & Ambiguity Detection (EU AI Act Art. 14 / Gartner AI TRiSM):
   - Empty prediction set `|C_alpha(x)| == 0`: Abstains when no legal outcome reaches the
     conformal threshold (e.g., out-of-domain query or low retrieval similarity).
   - Multi-label set `|C_alpha(x)| > 1`: Identifies normative ambiguity where multiple
     outcomes remain statistically plausible at confidence `1 - alpha`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class ConformalPredictionOutput:
    """
    Output of a split-conformal prediction evaluation for a single legal query.

    Attributes:
        prediction_set: Sorted list of outcome class indices included in `C_alpha(x)`.
        nonconformity_scores: Non-conformity score `s(x, y)` for each outcome index `y`.
        calibrated_probabilities: Similarity-gated probabilities `g(sim) * p_hat(y | x)`.
        quantile_threshold: Calibrated non-conformity threshold `q_hat`.
        alpha: Target miscoverage rate `alpha` (e.g., 0.10 for 90% coverage).
        is_singleton: True iff `len(prediction_set) == 1`.
        is_empty: True iff `len(prediction_set) == 0` (epistemic abstention trigger).
        is_ambiguous: True iff `len(prediction_set) > 1` (competing plausible outcomes).
    """

    prediction_set: list[int]
    nonconformity_scores: list[float]
    calibrated_probabilities: list[float]
    quantile_threshold: float
    alpha: float
    is_singleton: bool
    is_empty: bool
    is_ambiguous: bool


def _normalize_action_probabilities(action_vector: Sequence[float] | torch.Tensor) -> list[float]:
    """
    Convert a raw action vector into a valid probability simplex over outcomes.
    If all entries are non-negative and their sum is positive, normalizes by L1 sum;
    otherwise applies softmax.
    """
    if isinstance(action_vector, torch.Tensor):
        vals = [float(v) for v in action_vector.detach().cpu().flatten().tolist()]
    else:
        vals = [float(v) for v in action_vector]

    if not vals:
        return []

    min_v = min(vals)
    total = sum(vals)
    if min_v >= -1e-6 and total > 1e-6:
        clamped = [max(0.0, v) for v in vals]
        s = sum(clamped)
        return [v / s for v in clamped]

    t = torch.tensor(vals, dtype=torch.float32)
    probs = torch.softmax(t, dim=-1).tolist()
    return [float(p) for p in probs]


class SplitConformalCalibrator:
    """
    Finite-sample Split Conformal Calibrator for legal outcome vectors.

    Guarantees distribution-free marginal coverage `P(Y in C_alpha(X)) >= 1 - alpha`
    over exchangeable calibration and test queries, while gating out-of-distribution
    queries into empty sets (`C_alpha(X) = empty`) for provable epistemic abstention.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        min_similarity_gate: float = 0.25,
        abstain_on_empty: bool = True,
        abstain_on_ambiguous: bool = False,
        default_quantile: float = 0.45,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"Miscoverage rate alpha must be in (0, 1), got {alpha}.")
        self.alpha = float(alpha)
        self.min_similarity_gate = float(min_similarity_gate)
        self.abstain_on_empty = bool(abstain_on_empty)
        self.abstain_on_ambiguous = bool(abstain_on_ambiguous)
        self._quantile_threshold: float = float(default_quantile)
        self._is_calibrated: bool = False
        self._calibration_scores: list[float] = []

    @property
    def is_calibrated(self) -> bool:
        """Whether `calibrate()` has been called on a labeled calibration split."""
        return self._is_calibrated

    @property
    def quantile_threshold(self) -> float:
        """Current calibrated non-conformity threshold `q_hat`."""
        return self._quantile_threshold

    def _similarity_factor(self, max_similarity: float | None) -> float:
        """
        Compute similarity confidence multiplier in `[0.0, 1.0]`.
        When `max_similarity` is above `min_similarity_gate`, returns 1.0.
        Below `min_similarity_gate`, decays smoothly toward 0.0 so out-of-domain queries
        produce high non-conformity scores across all labels (yielding an empty set).
        """
        if max_similarity is None:
            return 1.0
        if self.min_similarity_gate <= 0.0 or max_similarity >= self.min_similarity_gate:
            return 1.0
        if max_similarity <= 0.0:
            return 0.0
        return float(max_similarity / self.min_similarity_gate)

    def calibrate(
        self,
        action_vectors: Sequence[Sequence[float]] | torch.Tensor,
        true_labels: Sequence[int],
        max_similarities: Sequence[float | None] | None = None,
    ) -> float:
        """
        Calibrate the finite-sample conformal quantile `q_hat` on a holdout calibration set.

        Args:
            action_vectors: Sequence of predicted action vectors `(n_samples, num_classes)`.
            true_labels: Sequence of ground-truth class indices `y_i in {0, ..., K-1}`.
            max_similarities: Optional sequence of top retrieval cosine similarities per calibration query.

        Returns:
            Calibrated finite-sample quantile `q_hat`.
        """
        if isinstance(action_vectors, torch.Tensor):
            vecs = action_vectors.detach().cpu().tolist()
        else:
            vecs = [list(v) for v in action_vectors]

        n = len(vecs)
        if n == 0 or len(true_labels) != n:
            raise ValueError("action_vectors and true_labels must be non-empty and have matching lengths.")

        sims: list[float | None] = (
            list(max_similarities) if max_similarities is not None else [None] * n
        )
        if len(sims) != n:
            raise ValueError("max_similarities length must match action_vectors length.")

        scores: list[float] = []
        for vec, y_true, sim in zip(vecs, true_labels, sims):
            probs = _normalize_action_probabilities(vec)
            if y_true < 0 or y_true >= len(probs):
                raise ValueError(f"Ground-truth label {y_true} out of bounds for vector of length {len(probs)}.")
            gate = self._similarity_factor(sim)
            p_true = gate * probs[y_true]
            scores.append(1.0 - p_true)

        sorted_scores = sorted(scores)
        rank = math.ceil((n + 1) * (1.0 - self.alpha))
        idx = min(max(rank - 1, 0), n - 1)
        self._quantile_threshold = float(sorted_scores[idx])
        self._calibration_scores = sorted_scores
        self._is_calibrated = True
        return self._quantile_threshold

    def predict_set(
        self,
        action_vector: Sequence[float] | torch.Tensor,
        max_similarity: float | None = None,
    ) -> ConformalPredictionOutput:
        """
        Construct the `(1 - alpha)` conformal prediction set `C_alpha(x)` for a single query.
        """
        probs = _normalize_action_probabilities(action_vector)
        gate = self._similarity_factor(max_similarity)
        gated_probs = [gate * p for p in probs]
        nonconformity = [1.0 - gp for gp in gated_probs]

        pred_set = [
            idx for idx, s in enumerate(nonconformity) if s <= self._quantile_threshold + 1e-7
        ]

        return ConformalPredictionOutput(
            prediction_set=pred_set,
            nonconformity_scores=nonconformity,
            calibrated_probabilities=gated_probs,
            quantile_threshold=self._quantile_threshold,
            alpha=self.alpha,
            is_singleton=(len(pred_set) == 1),
            is_empty=(len(pred_set) == 0),
            is_ambiguous=(len(pred_set) > 1),
        )

    def evaluate_empirical_coverage(
        self,
        action_vectors: Sequence[Sequence[float]] | torch.Tensor,
        true_labels: Sequence[int],
        max_similarities: Sequence[float | None] | None = None,
    ) -> dict[str, float]:
        """
        Evaluate empirical marginal coverage, average prediction set size, singleton rate,
        and empty-set abstention rate on a labeled evaluation set.
        """
        if isinstance(action_vectors, torch.Tensor):
            vecs = action_vectors.detach().cpu().tolist()
        else:
            vecs = [list(v) for v in action_vectors]

        n = len(vecs)
        if n == 0 or len(true_labels) != n:
            raise ValueError("action_vectors and true_labels must be non-empty and have matching lengths.")

        sims: list[float | None] = (
            list(max_similarities) if max_similarities is not None else [None] * n
        )

        covered = 0
        total_size = 0
        singletons = 0
        empty_sets = 0

        for vec, y_true, sim in zip(vecs, true_labels, sims):
            out = self.predict_set(vec, max_similarity=sim)
            if y_true in out.prediction_set:
                covered += 1
            total_size += len(out.prediction_set)
            if out.is_singleton:
                singletons += 1
            if out.is_empty:
                empty_sets += 1

        return {
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": covered / n,
            "average_set_size": total_size / n,
            "singleton_rate": singletons / n,
            "empty_set_abstention_rate": empty_sets / n,
            "quantile_threshold": self._quantile_threshold,
            "num_samples": float(n),
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize calibrated state for SQLite persistence."""
        return {
            "alpha": self.alpha,
            "min_similarity_gate": self.min_similarity_gate,
            "abstain_on_empty": self.abstain_on_empty,
            "abstain_on_ambiguous": self.abstain_on_ambiguous,
            "quantile_threshold": self._quantile_threshold,
            "is_calibrated": self._is_calibrated,
            "calibration_scores": list(self._calibration_scores),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SplitConformalCalibrator:
        """Restore a `SplitConformalCalibrator` from a serialized dictionary."""
        raw_alpha = data.get("alpha", 0.10)
        raw_gate = data.get("min_similarity_gate", 0.25)
        raw_q = data.get("quantile_threshold", 0.45)
        cal = cls(
            alpha=float(raw_alpha) if isinstance(raw_alpha, (int, float, str)) else 0.10,
            min_similarity_gate=float(raw_gate) if isinstance(raw_gate, (int, float, str)) else 0.25,
            abstain_on_empty=bool(data.get("abstain_on_empty", True)),
            abstain_on_ambiguous=bool(data.get("abstain_on_ambiguous", False)),
            default_quantile=float(raw_q) if isinstance(raw_q, (int, float, str)) else 0.45,
        )
        cal._is_calibrated = bool(data.get("is_calibrated", False))
        raw_scores = data.get("calibration_scores", [])
        if isinstance(raw_scores, list):
            cal._calibration_scores = [float(x) for x in raw_scores]
        return cal

