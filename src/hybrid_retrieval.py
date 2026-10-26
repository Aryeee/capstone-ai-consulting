"""
Hybrid retrieval module: combine semantic search with metadata filtering.

Supports filtering cases by:
  - industry
  - company_size
  - outcome
  - time_to_result_range (tuple of min/max months)
  - strategy_type

Returns filtered cases ranked by semantic similarity score.
"""

from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from src.case_db import get_model
from src.schema import Case


@dataclass
class FilterConfig:
    """Configuration for metadata filters to apply during retrieval.

    Attributes:
        industry: Filter by industry (e.g., "Retail", "Healthcare"). None = no filter.
        company_size: Filter by company size ("Small", "Medium", "Large"). None = no filter.
        outcome: Filter by outcome ("Positive", "Mixed", "Negative"). None = no filter.
        time_to_result_range: Tuple of (min_months, max_months) for time to result.
                              None = no filter.
        strategy_type: Filter by strategy type (e.g., "Digital Transformation"). None = no filter.
    """

    industry: Optional[str] = None
    company_size: Optional[str] = None
    outcome: Optional[str] = None
    time_to_result_range: Optional[tuple[int, int]] = None
    strategy_type: Optional[str] = None


def _matches_filter(case: Case, config: FilterConfig) -> bool:
    """Check if a case matches all active filters.

    Args:
        case: The Case object to check.
        config: FilterConfig with active filters.

    Returns:
        True if case matches all non-None filters, False otherwise.
    """
    if config.industry is not None and case.industry != config.industry:
        return False

    if config.company_size is not None and case.company_size != config.company_size:
        return False

    if config.outcome is not None and case.outcome != config.outcome:
        return False

    if config.strategy_type is not None and case.strategy_type != config.strategy_type:
        return False

    if config.time_to_result_range is not None:
        min_months, max_months = config.time_to_result_range
        time_to_result = case.outcome_metrics.time_to_result_months
        if not (min_months <= time_to_result <= max_months):
            return False

    return True


def retrieve_with_filters(
    query: str,
    index: faiss.IndexFlatIP,
    cases: list[Case],
    filters: Optional[FilterConfig] = None,
    top_k: int = 5,
) -> list[tuple[Case, float]]:
    """Retrieve cases via semantic search with optional metadata filtering.

    Performs a two-stage retrieval:
      1. Embed the query and search FAISS index for top_k * expansion_factor candidates
      2. Filter candidates by metadata constraints
      3. Re-rank by semantic similarity score and return top_k matches

    Args:
        query: Natural language query string.
        index: FAISS IndexFlatIP containing case embeddings.
        cases: List of Case objects (must correspond 1:1 with index entries).
        filters: Optional FilterConfig with metadata constraints.
        top_k: Number of results to return (default 5).

    Returns:
        List of (Case, score) tuples, sorted by score descending. Length <= top_k.
        Returns empty list if no cases match the filters.

    Raises:
        ValueError: If top_k <= 0.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    if filters is None:
        filters = FilterConfig()

    # Stage 1: Retrieve more candidates than needed to account for filtering
    expansion_factor = 3
    retrieval_k = min(top_k * expansion_factor, len(cases))

    model = get_model()
    q_emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    scores, indices = index.search(q_emb, retrieval_k)

    # Stage 2: Apply metadata filters and collect candidates
    filtered_results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        case = cases[idx]
        if _matches_filter(case, filters):
            filtered_results.append((case, float(score)))

    # Stage 3: Return top_k by score
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    return filtered_results[:top_k]


def retrieve_by_metadata(
    cases: list[Case], filters: FilterConfig, top_k: int = 5
) -> list[Case]:
    """Pure metadata-based retrieval (no semantic search).

    Returns all cases matching the filters, up to top_k results.

    Args:
        cases: List of Case objects to filter.
        filters: FilterConfig with metadata constraints.
        top_k: Maximum number of results to return.

    Returns:
        List of matching Case objects (length <= top_k).

    Raises:
        ValueError: If top_k <= 0.
    """
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")

    matching_cases = [c for c in cases if _matches_filter(c, filters)]
    return matching_cases[:top_k]


def get_filter_options(cases: list[Case]) -> dict:
    """Extract available filter values from the case database.

    Useful for UI/CLI to show users what filters are available.

    Args:
        cases: List of Case objects.

    Returns:
        Dictionary with keys for each filterable dimension, listing unique values:
            {
                "industries": [...],
                "company_sizes": [...],
                "outcomes": [...],
                "strategy_types": [...],
                "time_to_result_range": (min_months, max_months)
            }
    """
    industries = set()
    company_sizes = set()
    outcomes = set()
    strategy_types = set()
    min_time = float("inf")
    max_time = float("-inf")

    for case in cases:
        industries.add(case.industry)
        company_sizes.add(case.company_size)
        outcomes.add(case.outcome)
        strategy_types.add(case.strategy_type)

        time_to_result = case.outcome_metrics.time_to_result_months
        min_time = min(min_time, time_to_result)
        max_time = max(max_time, time_to_result)

    return {
        "industries": sorted(list(industries)),
        "company_sizes": sorted(list(company_sizes)),
        "outcomes": sorted(list(outcomes)),
        "strategy_types": sorted(list(strategy_types)),
        "time_to_result_range": (int(min_time), int(max_time))
        if min_time != float("inf")
        else (0, 0),
    }
