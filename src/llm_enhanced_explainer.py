"""
LLM-enhanced explanations: use Claude API for narrative explanations.

Generates human-readable explanations of recommendations using Claude.
"""

import os
from typing import Optional
import anthropic

from src.schema import Case
from src.predictor import Recommendation


def generate_llm_explanation(
    query: str,
    retrieved: list[tuple[Case, float]],
    recommendation: Recommendation,
    max_tokens: int = 500,
) -> str:
    """
    Generate a narrative explanation using Claude API.

    Parameters
    ----------
    query : str
        The user's business problem.
    retrieved : list[tuple[Case, float]]
        Retrieved similar cases.
    recommendation : Recommendation
        The system's recommendation.
    max_tokens : int
        Maximum tokens in response.

    Returns
    -------
    str
        Natural language explanation.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _fallback_explanation(query, retrieved, recommendation)

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Build context
        case_summaries = "\n".join(
            [
                f"- {case.title} ({case.industry}, {case.company_size}): "
                f"Used {case.strategy_type}, resulted in {case.outcome} outcome with "
                f"{case.outcome_metrics.roi_estimate:.1f}x ROI"
                for case, _ in retrieved
            ]
        )

        prompt = f"""You are an expert business consultant. A client has described a business problem:

"{query}"

Based on analysis of {len(retrieved)} similar historical cases, the system recommends: {recommendation.recommended_strategy} (confidence: {recommendation.strategy_confidence:.0%})

Similar cases that informed this recommendation:
{case_summaries}

Write a 2-3 paragraph narrative explanation of:
1. Why this strategy is recommended based on the similar cases
2. What outcomes they can expect (based on historical patterns)
3. Key success factors from the similar cases

Be concise, professional, and data-driven. Reference the specific cases."""

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        return message.content[0].text

    except Exception as e:
        print(f"LLM explanation failed: {e}")
        return _fallback_explanation(query, retrieved, recommendation)


def _fallback_explanation(
    query: str,
    retrieved: list[tuple[Case, float]],
    recommendation: Recommendation,
) -> str:
    """Fallback to rule-based explanation if LLM API fails."""
    case_names = ", ".join([f"'{c.title}'" for c, _ in retrieved[:3]])

    return (
        f"Based on analysis of {len(retrieved)} similar business cases including {case_names}, "
        f"we recommend the '{recommendation.recommended_strategy}' strategy with {recommendation.strategy_confidence:.0%} confidence. "
        f"Similar companies achieved an average ROI of {(recommendation.estimated_roi_low + recommendation.estimated_roi_high) / 2:.1f}x "
        f"over approximately {recommendation.avg_time_to_result_months:.0f} months. "
        f"The predicted outcome is {recommendation.predicted_outcome.lower()}."
    )
