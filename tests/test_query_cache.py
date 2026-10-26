"""
Tests for the query_cache module.

Tests caching of embeddings, recommendations, statistics, and cache invalidation.
"""

import numpy as np
import pytest
from pathlib import Path

from src.query_cache import (
    cache_query_embedding,
    get_cached_embedding,
    cache_recommendation,
    get_cached_recommendation,
    invalidate_cache,
    invalidate_recommendations_only,
    get_cache_stats,
    reset_cache_stats,
    close_cache,
    _hash_query,
    _hash_retrieval_params,
    CACHE_DB_PATH,
)
from src.predictor import Recommendation
from src.schema import Case, OutcomeMetrics


# Fixtures
@pytest.fixture
def sample_case():
    """Create a sample case for testing."""
    return Case(
        case_id="CASE-001",
        title="Digital Transformation",
        industry="Retail",
        company_size="Large",
        region="North America",
        year=2023,
        problem_description="Legacy systems hindering growth",
        key_challenges=["System integration", "Data silos"],
        actions_taken="Implemented cloud migration strategy",
        strategy_type="Digital Transformation",
        outcome="Positive",
        outcome_description="Successfully modernized infrastructure",
        outcome_metrics=OutcomeMetrics(
            revenue_change_pct=15.0,
            time_to_result_months=12,
            roi_estimate=2.5,
        ),
    )


@pytest.fixture
def sample_embedding():
    """Create a sample embedding."""
    return np.random.randn(1, 384).astype(np.float32)


@pytest.fixture
def sample_recommendation():
    """Create a sample recommendation."""
    return Recommendation(
        recommended_strategy="Digital Transformation",
        strategy_confidence=0.85,
        predicted_outcome="Positive",
        outcome_confidence=0.78,
        estimated_roi_low=1.5,
        estimated_roi_high=3.0,
        avg_time_to_result_months=10.5,
        supporting_rationale="Based on 3 similar historical cases.",
    )


@pytest.fixture(autouse=True)
def cleanup_cache():
    """Clean up cache before and after each test."""
    close_cache()
    if CACHE_DB_PATH.exists():
        CACHE_DB_PATH.unlink()
    yield
    close_cache()
    if CACHE_DB_PATH.exists():
        CACHE_DB_PATH.unlink()


# Tests for hashing utilities
def test_hash_query_consistency():
    """Test that query hashes are stable."""
    query = "What is the best digital strategy?"
    hash1 = _hash_query(query)
    hash2 = _hash_query(query)
    assert hash1 == hash2


def test_hash_query_differs_for_different_queries():
    """Test that different queries produce different hashes."""
    query1 = "What is the best digital strategy?"
    query2 = "What is the best AI strategy?"
    hash1 = _hash_query(query1)
    hash2 = _hash_query(query2)
    assert hash1 != hash2


def test_hash_retrieval_params_consistency():
    """Test that retrieval param hashes are stable."""
    query = "Best strategy"
    top_k = 5
    hash1 = _hash_retrieval_params(query, top_k)
    hash2 = _hash_retrieval_params(query, top_k)
    assert hash1 == hash2


def test_hash_retrieval_params_differs_for_different_top_k():
    """Test that different top_k values produce different hashes."""
    query = "Best strategy"
    hash1 = _hash_retrieval_params(query, 5)
    hash2 = _hash_retrieval_params(query, 10)
    assert hash1 != hash2


# Tests for embedding cache
def test_cache_and_retrieve_embedding(sample_embedding):
    """Test basic embedding caching and retrieval."""
    query = "What is the best strategy?"
    cache_query_embedding(query, sample_embedding)

    retrieved = get_cached_embedding(query)
    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, sample_embedding)


def test_embedding_cache_miss():
    """Test that retrieving a non-existent embedding returns None."""
    retrieved = get_cached_embedding("Non-existent query")
    assert retrieved is None


def test_embedding_cache_miss_increments_stat():
    """Test that cache misses are tracked in stats."""
    reset_cache_stats()
    _ = get_cached_embedding("Non-existent query")

    stats = get_cache_stats()
    assert stats["embedding_misses"] == 1
    assert stats["embedding_hits"] == 0


def test_embedding_cache_hit_increments_stat(sample_embedding):
    """Test that cache hits are tracked in stats."""
    reset_cache_stats()
    query = "What is the best strategy?"
    cache_query_embedding(query, sample_embedding)
    _ = get_cached_embedding(query)

    stats = get_cache_stats()
    assert stats["embedding_hits"] == 1
    assert stats["embedding_misses"] == 0


def test_embedding_hit_rate_calculation(sample_embedding):
    """Test hit rate calculation in stats."""
    reset_cache_stats()
    query = "What is the best strategy?"
    cache_query_embedding(query, sample_embedding)

    # One hit
    _ = get_cached_embedding(query)
    # One miss
    _ = get_cached_embedding("Different query")

    stats = get_cache_stats()
    assert stats["hit_rate_embedding"] == pytest.approx(0.5)


def test_embedding_cache_overwrite(sample_embedding):
    """Test that re-caching with the same query updates the entry."""
    query = "What is the best strategy?"
    cache_query_embedding(query, sample_embedding)

    # Create a different embedding
    new_embedding = np.random.randn(1, 384).astype(np.float32)
    cache_query_embedding(query, new_embedding)

    retrieved = get_cached_embedding(query)
    np.testing.assert_array_almost_equal(retrieved, new_embedding)


# Tests for recommendation cache
def test_cache_and_retrieve_recommendation(sample_case, sample_recommendation):
    """Test basic recommendation caching and retrieval."""
    query = "What strategy should we use?"
    top_k = 5
    retrieved_cases = [(sample_case, 0.95)]

    cache_recommendation(query, top_k, sample_recommendation, retrieved_cases)

    cached = get_cached_recommendation(query, top_k)
    assert cached is not None
    assert cached["recommendation"].recommended_strategy == "Digital Transformation"
    assert cached["recommendation"].strategy_confidence == 0.85
    assert len(cached["retrieved_cases"]) == 1
    assert cached["retrieved_cases"][0][0] == "CASE-001"
    assert cached["retrieved_cases"][0][1] == pytest.approx(0.95)


def test_recommendation_cache_miss():
    """Test that retrieving a non-existent recommendation returns None."""
    cached = get_cached_recommendation("Non-existent query", 5)
    assert cached is None


def test_recommendation_cache_miss_increments_stat():
    """Test that recommendation cache misses are tracked."""
    reset_cache_stats()
    _ = get_cached_recommendation("Non-existent query", 5)

    stats = get_cache_stats()
    assert stats["recommendation_misses"] == 1
    assert stats["recommendation_hits"] == 0


def test_recommendation_cache_hit_increments_stat(sample_case, sample_recommendation):
    """Test that recommendation cache hits are tracked."""
    reset_cache_stats()
    query = "What strategy should we use?"
    top_k = 5
    retrieved_cases = [(sample_case, 0.95)]

    cache_recommendation(query, top_k, sample_recommendation, retrieved_cases)
    _ = get_cached_recommendation(query, top_k)

    stats = get_cache_stats()
    assert stats["recommendation_hits"] == 1
    assert stats["recommendation_misses"] == 0


def test_recommendation_with_multiple_cases(sample_recommendation):
    """Test caching recommendations with multiple retrieved cases."""
    case1 = Case(
        case_id="CASE-001",
        title="Digital Transformation",
        industry="Retail",
        company_size="Large",
        region="North America",
        year=2023,
        problem_description="Legacy systems",
        key_challenges=["Integration"],
        actions_taken="Cloud migration",
        strategy_type="Digital Transformation",
        outcome="Positive",
        outcome_description="Success",
        outcome_metrics=OutcomeMetrics(15.0, 12, 2.5),
    )
    case2 = Case(
        case_id="CASE-002",
        title="AI Initiative",
        industry="Finance",
        company_size="Large",
        region="Europe",
        year=2022,
        problem_description="Need for automation",
        key_challenges=["ML expertise"],
        actions_taken="Built ML pipeline",
        strategy_type="AI Initiative",
        outcome="Positive",
        outcome_description="Success",
        outcome_metrics=OutcomeMetrics(20.0, 10, 3.0),
    )

    query = "What strategy should we use?"
    top_k = 5
    retrieved_cases = [(case1, 0.95), (case2, 0.88)]

    cache_recommendation(query, top_k, sample_recommendation, retrieved_cases)

    cached = get_cached_recommendation(query, top_k)
    assert len(cached["retrieved_cases"]) == 2
    assert cached["retrieved_cases"][0][0] == "CASE-001"
    assert cached["retrieved_cases"][0][1] == pytest.approx(0.95)
    assert cached["retrieved_cases"][1][0] == "CASE-002"
    assert cached["retrieved_cases"][1][1] == pytest.approx(0.88)


def test_recommendation_cache_overwrite(sample_case, sample_recommendation):
    """Test that re-caching a recommendation updates the entry."""
    query = "What strategy should we use?"
    top_k = 5
    retrieved_cases = [(sample_case, 0.95)]

    cache_recommendation(query, top_k, sample_recommendation, retrieved_cases)

    # Create a different recommendation
    new_recommendation = Recommendation(
        recommended_strategy="AI Strategy",
        strategy_confidence=0.92,
        predicted_outcome="Mixed",
        outcome_confidence=0.80,
        estimated_roi_low=1.0,
        estimated_roi_high=2.5,
        avg_time_to_result_months=8.0,
        supporting_rationale="Different rationale",
    )

    cache_recommendation(query, top_k, new_recommendation, retrieved_cases)

    cached = get_cached_recommendation(query, top_k)
    assert cached["recommendation"].recommended_strategy == "AI Strategy"
    assert cached["recommendation"].strategy_confidence == 0.92


# Tests for cache invalidation
def test_invalidate_all_cache(sample_embedding, sample_case, sample_recommendation):
    """Test that full cache invalidation clears all caches."""
    query = "What is the best strategy?"
    top_k = 5

    # Cache embedding and recommendation
    cache_query_embedding(query, sample_embedding)
    cache_recommendation(query, top_k, sample_recommendation, [(sample_case, 0.95)])

    # Verify they're cached
    assert get_cached_embedding(query) is not None
    assert get_cached_recommendation(query, top_k) is not None

    # Invalidate all
    invalidate_cache()

    # Verify they're gone
    assert get_cached_embedding(query) is None
    assert get_cached_recommendation(query, top_k) is None


def test_invalidate_cache_increments_stat(sample_embedding, sample_case, sample_recommendation):
    """Test that cache invalidation is tracked in stats."""
    reset_cache_stats()
    query = "What is the best strategy?"
    top_k = 5

    cache_query_embedding(query, sample_embedding)
    cache_recommendation(query, top_k, sample_recommendation, [(sample_case, 0.95)])

    invalidate_cache()

    stats = get_cache_stats()
    assert stats["cache_invalidations"] == 1


def test_invalidate_recommendations_only(sample_embedding, sample_case, sample_recommendation):
    """Test that selective invalidation clears only recommendations."""
    query = "What is the best strategy?"
    top_k = 5

    cache_query_embedding(query, sample_embedding)
    cache_recommendation(query, top_k, sample_recommendation, [(sample_case, 0.95)])

    # Invalidate only recommendations
    invalidate_recommendations_only()

    # Embedding should still be cached
    assert get_cached_embedding(query) is not None
    # Recommendation should be gone
    assert get_cached_recommendation(query, top_k) is None


# Tests for statistics
def test_get_cache_stats_initial():
    """Test that initial stats are zero."""
    reset_cache_stats()
    stats = get_cache_stats()

    assert stats["embedding_hits"] == 0
    assert stats["embedding_misses"] == 0
    assert stats["recommendation_hits"] == 0
    assert stats["recommendation_misses"] == 0
    assert stats["cache_invalidations"] == 0
    assert stats["hit_rate_embedding"] == 0.0
    assert stats["hit_rate_recommendation"] == 0.0
    assert stats["total_queries_processed"] == 0


def test_cache_stats_comprehensive(sample_embedding, sample_case, sample_recommendation):
    """Test comprehensive statistics tracking."""
    reset_cache_stats()

    query1 = "Strategy 1"
    query2 = "Strategy 2"
    top_k = 5

    # Cache embeddings
    cache_query_embedding(query1, sample_embedding)
    cache_query_embedding(query2, sample_embedding)

    # Multiple hits and misses
    _ = get_cached_embedding(query1)  # hit
    _ = get_cached_embedding(query1)  # hit
    _ = get_cached_embedding("new query")  # miss

    # Cache recommendations
    cache_recommendation(query1, top_k, sample_recommendation, [(sample_case, 0.95)])

    # Recommendation hits and misses
    _ = get_cached_recommendation(query1, top_k)  # hit
    _ = get_cached_recommendation(query2, top_k)  # miss

    stats = get_cache_stats()

    assert stats["embedding_hits"] == 2
    assert stats["embedding_misses"] == 1
    assert stats["recommendation_hits"] == 1
    assert stats["recommendation_misses"] == 1
    assert stats["total_queries_processed"] == 3
    assert stats["hit_rate_embedding"] == pytest.approx(2.0 / 3.0)
    assert stats["hit_rate_recommendation"] == pytest.approx(0.5)


# Tests for edge cases
def test_cache_with_special_characters_in_query():
    """Test caching queries with special characters."""
    query = "What? Why! How's it @ $100 & more?"
    embedding = np.random.randn(1, 384).astype(np.float32)

    cache_query_embedding(query, embedding)
    retrieved = get_cached_embedding(query)

    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, embedding)


def test_cache_with_unicode_query():
    """Test caching queries with unicode characters."""
    query = "What is the best stratégie für Deutschland?"
    embedding = np.random.randn(1, 384).astype(np.float32)

    cache_query_embedding(query, embedding)
    retrieved = get_cached_embedding(query)

    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, embedding)


def test_cache_with_very_long_query():
    """Test caching very long queries."""
    query = "What is the best strategy? " * 100
    embedding = np.random.randn(1, 384).astype(np.float32)

    cache_query_embedding(query, embedding)
    retrieved = get_cached_embedding(query)

    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, embedding)


def test_recommendation_top_k_variations(sample_case, sample_recommendation):
    """Test that different top_k values create separate cache entries."""
    query = "What strategy should we use?"
    retrieved_cases = [(sample_case, 0.95)]

    # Cache with top_k=5
    cache_recommendation(query, 5, sample_recommendation, retrieved_cases)

    # Create a different recommendation for top_k=10
    rec_top10 = Recommendation(
        recommended_strategy="Different Strategy",
        strategy_confidence=0.80,
        predicted_outcome="Positive",
        outcome_confidence=0.75,
        estimated_roi_low=1.0,
        estimated_roi_high=2.0,
        avg_time_to_result_months=9.0,
        supporting_rationale="Different rationale",
    )
    cache_recommendation(query, 10, rec_top10, retrieved_cases)

    # Verify both are cached separately
    cached_5 = get_cached_recommendation(query, 5)
    cached_10 = get_cached_recommendation(query, 10)

    assert cached_5["recommendation"].recommended_strategy == "Digital Transformation"
    assert cached_10["recommendation"].recommended_strategy == "Different Strategy"


def test_thread_safety_embedding_cache(sample_embedding):
    """Test that embedding cache is thread-safe."""
    import threading

    query = "Thread safety test"
    errors = []

    def cache_and_retrieve():
        try:
            cache_query_embedding(query, sample_embedding)
            result = get_cached_embedding(query)
            if result is None:
                errors.append("Failed to retrieve after caching")
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=cache_and_retrieve) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
