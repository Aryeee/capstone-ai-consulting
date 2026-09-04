"""
Data schema definitions for the AI Strategy Consulting Assistant.

Each case follows a structured schema combining free-text fields
(for embedding-based retrieval) with categorical/numeric fields
(for structured prediction and explainability).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OutcomeMetrics:
    revenue_change_pct: float          # % revenue change attributed to the strategy
    time_to_result_months: int         # months until measurable results
    roi_estimate: float                # estimated return on investment (multiplier)


@dataclass
class Case:
    # --- Identity ---
    case_id: str                       # unique ID, e.g. "CASE-001"
    title: str                         # short human-readable title

    # --- Context (structured) ---
    industry: str                      # e.g. "Retail", "Healthcare"
    company_size: str                  # "Small" | "Medium" | "Large"
    region: str                        # e.g. "North America", "Europe"
    year: int                          # year the engagement occurred

    # --- Problem & solution (free-text, used for embeddings) ---
    problem_description: str           # the business problem narrative
    key_challenges: list[str]          # concise list of challenges
    actions_taken: str                 # what was actually done

    # --- Strategy & outcome ---
    strategy_type: str                 # category label, e.g. "Digital Transformation"
    outcome: str                       # "Positive" | "Mixed" | "Negative"
    outcome_description: str           # narrative of what happened
    outcome_metrics: OutcomeMetrics    # quantitative results

    # --- Metadata ---
    tags: list[str] = field(default_factory=list)

    # --- Computed at runtime ---
    embedding: Optional[list[float]] = field(default=None, repr=False)

    @property
    def text_for_embedding(self) -> str:
        """Concatenate the text fields that should be encoded into a single
        dense vector for semantic retrieval."""
        challenges = ", ".join(self.key_challenges)
        return (
            f"{self.title}. "
            f"Industry: {self.industry}. "
            f"Problem: {self.problem_description} "
            f"Key challenges: {challenges}. "
            f"Actions taken: {self.actions_taken}"
        )

    @property
    def structured_features(self) -> dict:
        """Return the categorical/numeric features used by the prediction
        model (XGBoost) alongside the embedding."""
        return {
            "industry": self.industry,
            "company_size": self.company_size,
            "region": self.region,
            "strategy_type": self.strategy_type,
            "num_challenges": len(self.key_challenges),
            "num_tags": len(self.tags),
            "year": self.year,
        }
