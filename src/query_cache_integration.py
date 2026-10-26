"""
Integration example: using query_cache with case_db and predictor.

This module demonstrates how to integrate the query cache into a typical
retrieval + prediction workflow.
"""

import logging
from src.case_db import retrieve, load_index, get_model
from src.predictor import predict
from src.query_cache import (
    get_cached_embedding,
    cache_query_embedding,
    get_cached_recommendation,
    cache_recommendation,
    get_cache_stats,
)

logger = logging.getLogger(__name__)


def retrieve_with_cache(query: str, index, cases, top_k: int = 5):
    """Retrieve cases with query embedding caching.

    If the query embedding is cached, reuse it. Otherwise, embed and cache.

    Parameters
    ----------
    query : str
        User query
    index : faiss.IndexFlatIP
        FAISS index
    cases : list[Case]
        List of cases
    top_k : int
        Number of results to retrieve

    Returns
    -------
    list[tuple[Case, float]]
        Retrieved (Case, similarity_score) tuples
    """
    # Try to get cached embedding
    cached_embedding = get_cached_embedding(query)

    if cached_embedding is not None:
        logger.info(f"Cache hit: query embedding for '{query[:50]}...'")
        # Use the cached embedding directly
        import numpy as np
        model = get_model()
        scores, indices = index.search(cached_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((cases[idx], float(score)))
        return results
    else:
        logger.info(f"Cache miss: computing embedding for '{query[:50]}...'")
        # Use the standard retrieve function (which embeds and caches implicitly)
        results = retrieve(query, index, cases, top_k)

        # Cache the embedding for future use
        model = get_model()
        import numpy as np
        q_emb = model.encode([query], normalize_embeddings=True,
                             convert_to_numpy=True).astype(np.float32)
        cache_query_embedding(query, q_emb)

        return results


def predict_with_cache(query: str, retrieved, top_k: int = 5):
    """Predict recommendation with caching.

    If the recommendation for this query + top_k is cached, reuse it.
    Otherwise, predict and cache.

    Parameters
    ----------
    query : str
        User query
    retrieved : list[tuple[Case, float]]
        Retrieved (Case, score) tuples from retrieve_with_cache()
    top_k : int
        Number of results that were retrieved

    Returns
    -------
    Recommendation
        The recommendation object
    """
    # Try to get cached recommendation
    cached = get_cached_recommendation(query, top_k)

    if cached is not None:
        logger.info(f"Cache hit: recommendation for '{query[:50]}...' (top_k={top_k})")
        return cached["recommendation"]
    else:
        logger.info(f"Cache miss: computing recommendation for '{query[:50]}...'")
        # Predict
        recommendation = predict(retrieved)

        # Cache it
        cache_recommendation(query, top_k, recommendation, retrieved)

        return recommendation


def print_cache_stats():
    """Print cache statistics for monitoring."""
    stats = get_cache_stats()
    logger.info(
        f"Cache stats: "
        f"embedding_hits={stats['embedding_hits']}, "
        f"embedding_misses={stats['embedding_misses']}, "
        f"recommendation_hits={stats['recommendation_hits']}, "
        f"recommendation_misses={stats['recommendation_misses']}, "
        f"hit_rate_embedding={stats['hit_rate_embedding']:.1%}, "
        f"hit_rate_recommendation={stats['hit_rate_recommendation']:.1%}"
    )


if __name__ == "__main__":
    # Example usage
    from src.case_db import load_index

    # Load index
    index, cases = load_index()

    # Query with caching
    query = "We need to improve our digital capabilities in a legacy retail environment"
    retrieved = retrieve_with_cache(query, index, cases, top_k=5)
    recommendation = predict_with_cache(query, retrieved, top_k=5)

    print(f"\nRecommendation: {recommendation.recommended_strategy}")
    print(f"Confidence: {recommendation.strategy_confidence:.0%}")
    print(f"Predicted outcome: {recommendation.predicted_outcome}")

    print_cache_stats()
