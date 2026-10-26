"""
Refined retrieval: iterative search with refinement options.

Allows users to find related cases, counter-examples, and refine searches.
"""

from typing import Optional, Literal
import numpy as np
import faiss

from src.schema import Case
from src.case_db import get_model


def retrieve_related_cases(
    case_id: str,
    cases: list[Case],
    index: faiss.IndexFlatIP,
    top_k: int = 5,
) -> list[tuple[Case, float]]:
    """
    Find cases related to a given case (similar problems/industries).

    Parameters
    ----------
    case_id : str
        The reference case ID.
    cases : list[Case]
        All available cases.
    index : faiss.IndexFlatIP
        FAISS index for similarity search.
    top_k : int
        Number of related cases to return.

    Returns
    -------
    list[tuple[Case, float]]
        Related (Case, similarity_score) pairs, excluding the reference case.
    """
    # Find the reference case
    ref_case = next((c for c in cases if c.case_id == case_id), None)
    if not ref_case:
        return []

    # Use its embedding for search
    if ref_case.embedding is None:
        model = get_model()
        embedding = model.encode(
            [ref_case.text_for_embedding],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
    else:
        embedding = np.array(ref_case.embedding, dtype=np.float32).reshape(1, -1)

    # Search, excluding the reference case
    scores, indices = index.search(embedding, top_k + 1)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if cases[idx].case_id == case_id:
            continue  # Skip the reference case itself
        results.append((cases[idx], float(score)))

    return results[:top_k]


def retrieve_counter_examples(
    strategy: str,
    cases: list[Case],
    index: faiss.IndexFlatIP,
    target_outcome: str = "Positive",
    top_k: int = 5,
) -> list[tuple[Case, float]]:
    """
    Find cases that used a DIFFERENT strategy but achieved a positive outcome.

    Useful for showing "alternative approaches that also worked."

    Parameters
    ----------
    strategy : str
        The strategy to find counter-examples for.
    cases : list[Case]
        All available cases.
    index : faiss.IndexFlatIP
        FAISS index (note: won't be used directly, filtered by metadata).
    target_outcome : str
        The outcome to filter for (default: "Positive").
    top_k : int
        Number of counter-examples to return.

    Returns
    -------
    list[tuple[Case, float]]
        Counter-example cases (different strategy, positive outcome).
    """
    counter_cases = [
        (c, 0.8)  # Dummy score for metadata-filtered results
        for c in cases
        if c.strategy_type != strategy and c.outcome == target_outcome
    ]

    # Sort by ROI to show the best counter-examples
    counter_cases.sort(key=lambda x: x[0].outcome_metrics.roi_estimate, reverse=True)

    return counter_cases[:top_k]


def retrieve_with_outcome_filter(
    query: str,
    index: faiss.IndexFlatIP,
    cases: list[Case],
    target_outcome: Optional[str] = None,
    top_k: int = 5,
) -> list[tuple[Case, float]]:
    """
    Retrieve cases filtered by outcome (Positive/Mixed/Negative).

    Useful for "Show me successful cases" or "Show me what can go wrong."

    Parameters
    ----------
    query : str
        Business problem description.
    index : faiss.IndexFlatIP
        FAISS index.
    cases : list[Case]
        All available cases.
    target_outcome : Optional[str]
        Filter by outcome. If None, don't filter.
    top_k : int
        Number of results to return.

    Returns
    -------
    list[tuple[Case, float]]
        Retrieved cases matching the outcome filter, ranked by similarity.
    """
    model = get_model()
    q_emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    # Over-retrieve to account for filtering
    retrieve_k = int(top_k * 2.5)
    scores, indices = index.search(q_emb, retrieve_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        case = cases[idx]

        # Apply outcome filter
        if target_outcome and case.outcome != target_outcome:
            continue

        results.append((case, float(score)))

    return results[:top_k]


def get_strategy_alternatives(
    strategy: str,
    cases: list[Case],
    target_outcome: str = "Positive",
) -> dict[str, list[Case]]:
    """
    For a given strategy, find alternative strategies that also work.

    Returns strategies ranked by success rate.

    Parameters
    ----------
    strategy : str
        The reference strategy.
    cases : list[Case]
        All available cases.
    target_outcome : str
        The outcome to consider as "success".

    Returns
    -------
    dict[str, list[Case]]
        Mapping of alternative strategy → successful cases using that strategy.
    """
    alternatives = {}

    for case in cases:
        if case.strategy_type == strategy:
            continue  # Not an alternative
        if case.outcome != target_outcome:
            continue  # Not successful

        if case.strategy_type not in alternatives:
            alternatives[case.strategy_type] = []

        alternatives[case.strategy_type].append(case)

    return alternatives


def retrieve_with_refinement(
    query: str,
    index: faiss.IndexFlatIP,
    cases: list[Case],
    filter_mode: Optional[Literal["better_outcomes", "contrasting_cases"]] = None,
    top_k: int = 5,
) -> list[tuple[Case, float]]:
    """
    Iterative retrieval with refinement filtering.

    Finds base cases matching the query, then allows filtering by outcome quality
    or strategy contrast for iterative search refinement.

    Parameters
    ----------
    query : str
        Business problem description or search query.
    index : faiss.IndexFlatIP
        FAISS similarity index for embeddings.
    cases : list[Case]
        All available cases in the database.
    filter_mode : Optional[Literal["better_outcomes", "contrasting_cases"]]
        Refinement filter:
        - None: return base results ranked by similarity
        - "better_outcomes": prioritize Positive > Mixed > Negative outcomes
        - "contrasting_cases": filter for cases with different strategies but positive outcomes
    top_k : int
        Number of results to return after filtering.

    Returns
    -------
    list[tuple[Case, float]]
        Filtered and ranked (Case, score) pairs. Scores reflect both semantic
        similarity and refinement criteria.
    """
    model = get_model()
    q_emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    # Over-retrieve to have candidates for filtering
    retrieve_k = int(top_k * 3.5)
    scores, indices = index.search(q_emb, retrieve_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        candidates.append((cases[idx], float(score)))

    # Apply refinement filter
    if filter_mode == "better_outcomes":
        # Rank by outcome quality, then by similarity score
        outcome_rank = {"Positive": 3, "Mixed": 2, "Negative": 1}
        candidates.sort(
            key=lambda x: (
                -outcome_rank.get(x[0].outcome, 0),  # Better outcomes first
                -x[1],  # Then by similarity
            )
        )
    elif filter_mode == "contrasting_cases":
        # Find the dominant strategy in base results, then filter for alternatives
        if candidates:
            strategy_counts = {}
            for case, _ in candidates[:5]:
                strategy_counts[case.strategy_type] = (
                    strategy_counts.get(case.strategy_type, 0) + 1
                )
            dominant_strategy = max(
                strategy_counts.items(), key=lambda x: x[1]
            )[0]

            # Filter for different strategies with positive outcomes
            filtered = [
                (c, s)
                for c, s in candidates
                if c.strategy_type != dominant_strategy and c.outcome == "Positive"
            ]
            candidates = filtered if filtered else candidates

    return candidates[:top_k]


def related_cases(
    case_id: str,
    cases: list[Case],
    index: faiss.IndexFlatIP,
    top_k: int = 5,
) -> list[tuple[Case, float]]:
    """
    Find cases similar to a given case_id.

    Uses semantic similarity on case embeddings to identify related problems,
    industries, or solution approaches.

    Parameters
    ----------
    case_id : str
        The reference case ID to find similar cases for.
    cases : list[Case]
        All available cases in the database.
    index : faiss.IndexFlatIP
        FAISS similarity index for embeddings.
    top_k : int
        Number of related cases to return.

    Returns
    -------
    list[tuple[Case, float]]
        Similar cases ranked by similarity score (highest first), excluding
        the reference case itself.
    """
    # Find the reference case
    ref_case = next((c for c in cases if c.case_id == case_id), None)
    if not ref_case:
        return []

    # Use its embedding for search
    if ref_case.embedding is None:
        model = get_model()
        embedding = model.encode(
            [ref_case.text_for_embedding],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
    else:
        embedding = np.array(ref_case.embedding, dtype=np.float32).reshape(1, -1)

    # Search, excluding the reference case
    scores, indices = index.search(embedding, top_k + 1)
    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if cases[idx].case_id == case_id:
            continue  # Skip the reference case itself
        results.append((cases[idx], float(score)))

    return results[:top_k]


def counter_examples(
    case_id: str,
    cases: list[Case],
    index: faiss.IndexFlatIP,
    top_k: int = 5,
) -> list[tuple[Case, float]]:
    """
    Find counter-examples: cases that used a different strategy but achieved
    positive outcomes.

    Useful for exploring alternative approaches and showing "what else worked."

    Parameters
    ----------
    case_id : str
        Reference case ID to find counter-examples for.
    cases : list[Case]
        All available cases in the database.
    index : faiss.IndexFlatIP
        FAISS index (used for initial similarity filtering).
    top_k : int
        Number of counter-examples to return.

    Returns
    -------
    list[tuple[Case, float]]
        Counter-example cases with different strategies but positive outcomes,
        ranked by ROI estimate (highest first).
    """
    # Find the reference case
    ref_case = next((c for c in cases if c.case_id == case_id), None)
    if not ref_case:
        return []

    ref_strategy = ref_case.strategy_type

    # Find similar cases first, then filter for different strategies with positive outcomes
    if ref_case.embedding is None:
        model = get_model()
        embedding = model.encode(
            [ref_case.text_for_embedding],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
    else:
        embedding = np.array(ref_case.embedding, dtype=np.float32).reshape(1, -1)

    # Over-retrieve to find enough counter-examples
    search_k = int(top_k * 5)
    scores, indices = index.search(embedding, search_k)

    counter_cases = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        case = cases[idx]
        if case.case_id == case_id:
            continue  # Skip reference
        if case.strategy_type == ref_strategy:
            continue  # Must use different strategy
        if case.outcome != "Positive":
            continue  # Must have positive outcome

        counter_cases.append((case, float(score)))

    # Sort by ROI to prioritize high-impact alternatives
    counter_cases.sort(
        key=lambda x: x[0].outcome_metrics.roi_estimate, reverse=True
    )

    return counter_cases[:top_k]
