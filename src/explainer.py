"""
Explainability module.

Generates a human-readable explanation of:
  1. WHY these cases were retrieved  (similarity breakdown)
  2. What structured features are shared between the query and top cases
  3. A feature-importance-style breakdown of the recommendation drivers
"""

from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from src.schema import Case
from src.predictor import Recommendation


# ---------------------------------------------------------------------------
# Data structures for the explanation
# ---------------------------------------------------------------------------
@dataclass
class CaseExplanation:
    """Explanation for a single retrieved case."""
    case_id: str
    title: str
    similarity_score: float
    shared_features: dict[str, str]      # feature → shared value
    differing_features: dict[str, tuple] # feature → (query_value, case_value)
    key_overlap_phrases: list[str]       # important overlapping terms


@dataclass
class RecommendationExplanation:
    """Full explanation object consumed by the UI."""
    case_explanations: list[CaseExplanation]
    strategy_drivers: list[dict]         # ranked list of {factor, weight, direction}
    outcome_factors: list[dict]          # factors pushing toward the predicted outcome
    confidence_note: str                 # human-readable confidence assessment
    methodology_note: str                # short description of how the system works


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_key_phrases(text: str, top_n: int = 8) -> set[str]:
    """Very simple keyword extraction: lowercase split, filter stopwords,
    return the top-n longest unique tokens (a proxy for specificity)."""
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "was", "were", "is", "are", "had", "has",
        "its", "their", "this", "that", "which", "who", "whom", "be", "been",
        "being", "have", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "as", "if", "than", "into",
        "not", "no", "also", "very", "so", "up", "out", "about", "just",
        "only", "over", "such", "through", "after", "before", "between",
        "under", "during", "each", "all", "both", "more", "most", "other",
        "some", "it", "they", "we", "he", "she", "you", "i", "my", "our",
    }
    words = set()
    for w in text.lower().split():
        cleaned = "".join(ch for ch in w if ch.isalnum())
        if cleaned and cleaned not in stopwords and len(cleaned) > 2:
            words.add(cleaned)
    # Prefer longer (more specific) words
    return set(sorted(words, key=len, reverse=True)[:top_n])


def _shared_feature_analysis(query_features: dict, case: Case) -> tuple[dict, dict]:
    """Compare user-supplied structured context against a case's structured
    features.  Returns (shared, differing) dicts."""
    case_feats = case.structured_features
    shared, diff = {}, {}
    for key in query_features:
        if key in case_feats:
            if str(query_features[key]).lower() == str(case_feats[key]).lower():
                shared[key] = str(case_feats[key])
            else:
                diff[key] = (str(query_features[key]), str(case_feats[key]))
    return shared, diff


# ---------------------------------------------------------------------------
# Main explain() function
# ---------------------------------------------------------------------------
def explain(
    query_text: str,
    query_features: dict,              # e.g. {"industry": "Retail", "company_size": "Large"}
    retrieved: list[tuple[Case, float]],
    recommendation: Recommendation,
) -> RecommendationExplanation:
    """Build a full explanation object from the retrieval + prediction results."""

    query_phrases = _extract_key_phrases(query_text)

    # --- Per-case explanations ---
    case_explanations = []
    for case, score in retrieved:
        case_phrases = _extract_key_phrases(case.text_for_embedding)
        overlap = sorted(query_phrases & case_phrases, key=len, reverse=True)
        shared, diff = _shared_feature_analysis(query_features, case)
        case_explanations.append(CaseExplanation(
            case_id=case.case_id,
            title=case.title,
            similarity_score=round(score, 4),
            shared_features=shared,
            differing_features=diff,
            key_overlap_phrases=overlap,
        ))

    # --- Strategy recommendation drivers ---
    strategy_drivers = []
    strategy_counter: dict[str, list[float]] = {}
    for case, score in retrieved:
        strategy_counter.setdefault(case.strategy_type, []).append(score)

    for strat, scores in sorted(strategy_counter.items(),
                                 key=lambda x: sum(x[1]), reverse=True):
        strategy_drivers.append({
            "factor": f"Similar cases used '{strat}'",
            "weight": round(sum(scores) / sum(s for _, s in retrieved), 3),
            "direction": "supporting" if strat == recommendation.recommended_strategy else "alternative",
            "case_count": len(scores),
        })

    # --- Outcome factors ---
    outcome_factors = []
    for case, score in retrieved:
        direction = "positive" if case.outcome == "Positive" else (
            "cautionary" if case.outcome == "Mixed" else "negative"
        )
        outcome_factors.append({
            "case": case.title,
            "outcome": case.outcome,
            "roi": case.outcome_metrics.roi_estimate,
            "weight": round(score, 3),
            "direction": direction,
        })

    # --- Confidence note ---
    avg_sim = np.mean([s for _, s in retrieved])
    if avg_sim > 0.6:
        conf_note = (
            f"High confidence — the top cases are strongly similar to your problem "
            f"(avg similarity {avg_sim:.2f}). The recommendation is well-supported."
        )
    elif avg_sim > 0.4:
        conf_note = (
            f"Moderate confidence — the retrieved cases are reasonably similar "
            f"(avg similarity {avg_sim:.2f}). Consider reviewing the differing "
            f"features before acting."
        )
    else:
        conf_note = (
            f"Low confidence — the most similar cases are only loosely related "
            f"(avg similarity {avg_sim:.2f}). Treat the recommendation as "
            f"directional only; your situation may be novel."
        )

    methodology_note = (
        "Cases were retrieved using cosine similarity between sentence-transformer "
        "embeddings (all-MiniLM-L6-v2) of the problem descriptions. The strategy "
        "recommendation is a similarity-weighted vote over the retrieved cases' "
        "strategy types. Outcome prediction uses the same weighted vote over "
        "historical outcomes. ROI estimates are weighted averages of retrieved "
        "case outcomes ± one standard deviation."
    )

    return RecommendationExplanation(
        case_explanations=case_explanations,
        strategy_drivers=strategy_drivers,
        outcome_factors=outcome_factors,
        confidence_note=conf_note,
        methodology_note=methodology_note,
    )
