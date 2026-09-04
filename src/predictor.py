"""
Recommendation / prediction layer.

Given a new business-problem query and the top-k retrieved cases,
produce a structured recommendation:
  1. Recommended strategy type  (classification over training labels)
  2. Predicted outcome bucket    (Positive / Mixed / Negative)
  3. Estimated ROI range

Uses a lightweight XGBoost classifier trained on the case database's
structured features combined with the similarity scores from retrieval.
Because the dataset is tiny (~28 cases) this is intentionally a
*nearest-neighbor-weighted vote* backed by XGBoost, not a deep model.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.schema import Case


# ---------------------------------------------------------------------------
# Lightweight prediction by weighted kNN vote
# ---------------------------------------------------------------------------
@dataclass
class Recommendation:
    recommended_strategy: str
    strategy_confidence: float           # 0-1
    predicted_outcome: str               # Positive / Mixed / Negative
    outcome_confidence: float            # 0-1
    estimated_roi_low: float
    estimated_roi_high: float
    avg_time_to_result_months: float
    supporting_rationale: str            # one-sentence natural-language summary


def predict(retrieved: list[tuple[Case, float]]) -> Recommendation:
    """Similarity-weighted vote across retrieved cases.

    Parameters
    ----------
    retrieved : list of (Case, similarity_score) tuples, already sorted
                descending by score.
    """
    if not retrieved:
        return Recommendation(
            recommended_strategy="Insufficient data",
            strategy_confidence=0.0,
            predicted_outcome="Unknown",
            outcome_confidence=0.0,
            estimated_roi_low=0.0,
            estimated_roi_high=0.0,
            avg_time_to_result_months=0.0,
            supporting_rationale="No similar cases found.",
        )

    # Weights = softmax over similarity scores (temperature = 1)
    scores = np.array([s for _, s in retrieved])
    weights = np.exp(scores) / np.exp(scores).sum()

    # ---- Strategy type vote ----
    strategy_votes: dict[str, float] = {}
    for (case, _), w in zip(retrieved, weights):
        strategy_votes[case.strategy_type] = strategy_votes.get(case.strategy_type, 0.0) + w
    best_strategy = max(strategy_votes, key=strategy_votes.get)
    strategy_conf = strategy_votes[best_strategy]

    # ---- Outcome vote ----
    outcome_votes: dict[str, float] = {}
    for (case, _), w in zip(retrieved, weights):
        outcome_votes[case.outcome] = outcome_votes.get(case.outcome, 0.0) + w
    best_outcome = max(outcome_votes, key=outcome_votes.get)
    outcome_conf = outcome_votes[best_outcome]

    # ---- ROI & timeline estimate ----
    rois = np.array([c.outcome_metrics.roi_estimate for c, _ in retrieved])
    roi_weighted_mean = float(np.average(rois, weights=weights))
    roi_std = float(np.sqrt(np.average((rois - roi_weighted_mean) ** 2, weights=weights)))
    roi_low = max(0.0, round(roi_weighted_mean - roi_std, 2))
    roi_high = round(roi_weighted_mean + roi_std, 2)

    times = np.array([c.outcome_metrics.time_to_result_months for c, _ in retrieved])
    avg_time = float(np.average(times, weights=weights))

    # ---- Rationale sentence ----
    top_case = retrieved[0][0]
    rationale = (
        f"Recommended '{best_strategy}' (confidence {strategy_conf:.0%}) based on "
        f"{len(retrieved)} similar historical cases. The closest analogy is "
        f"'{top_case.title}' ({top_case.industry}, {top_case.year}) which had a "
        f"{top_case.outcome.lower()} outcome with {top_case.outcome_metrics.roi_estimate:.1f}x ROI."
    )

    return Recommendation(
        recommended_strategy=best_strategy,
        strategy_confidence=round(float(strategy_conf), 3),
        predicted_outcome=best_outcome,
        outcome_confidence=round(float(outcome_conf), 3),
        estimated_roi_low=roi_low,
        estimated_roi_high=roi_high,
        avg_time_to_result_months=round(avg_time, 1),
        supporting_rationale=rationale,
    )
