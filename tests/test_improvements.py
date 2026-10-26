"""
Comprehensive integration tests for all improvements.

Tests hybrid retrieval, ensemble prediction, analytics, caching, API endpoints, and more.
"""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch

from src.case_db import load_index, get_model
from src.hybrid_retrieval import RetrievalFilters, retrieve_hybrid, filter_cases_by_metadata
from src.ensemble_predictor import train_ensemble, predict_ensemble, save_ensemble, load_ensemble
from src.analytics import log_query, store_feedback, get_analytics_stats, load_query_logs
from src.query_cache import QueryCache
from src.refined_retrieval import retrieve_related_cases, retrieve_counter_examples
from src.predictor import Recommendation
from src.schema import Case, OutcomeMetrics


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_cases():
    """Create sample cases for testing."""
    return [
        Case(
            case_id="TEST-001",
            title="Test Digital Transformation",
            industry="Technology",
            company_size="Large",
            region="North America",
            year=2023,
            problem_description="Test problem description for digital transformation",
            key_challenges=["legacy systems", "change management"],
            actions_taken="Implemented cloud migration and training",
            strategy_type="Digital Transformation",
            outcome="Positive",
            outcome_description="Successful transformation",
            outcome_metrics=OutcomeMetrics(revenue_change_pct=10.0, time_to_result_months=18, roi_estimate=2.5),
            tags=["digital", "cloud", "transformation"],
        ),
        Case(
            case_id="TEST-002",
            title="Test Cost Optimization",
            industry="Manufacturing",
            company_size="Medium",
            region="Europe",
            year=2023,
            problem_description="Test problem for cost reduction initiative",
            key_challenges=["procurement", "efficiency"],
            actions_taken="Consolidated suppliers and automated processes",
            strategy_type="Cost Optimization",
            outcome="Positive",
            outcome_description="Costs reduced significantly",
            outcome_metrics=OutcomeMetrics(revenue_change_pct=0.0, time_to_result_months=12, roi_estimate=3.0),
            tags=["cost", "optimization", "procurement"],
        ),
        Case(
            case_id="TEST-003",
            title="Test Market Expansion",
            industry="Technology",
            company_size="Small",
            region="Asia",
            year=2023,
            problem_description="Test expansion into new market",
            key_challenges=["market entry", "localization"],
            actions_taken="Established local partnerships and adapted product",
            strategy_type="Market Expansion",
            outcome="Mixed",
            outcome_description="Partial success with challenges",
            outcome_metrics=OutcomeMetrics(revenue_change_pct=5.0, time_to_result_months=24, roi_estimate=1.2),
            tags=["expansion", "market entry"],
        ),
    ]


@pytest.fixture
def temp_cache_db():
    """Create temporary cache database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = Path(f.name)
    cache = QueryCache(temp_path)
    yield cache
    temp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
# Test Hybrid Retrieval
# ─────────────────────────────────────────────────────────────

def test_retrieval_filters_match(sample_cases):
    """Test metadata filtering logic."""
    filters = RetrievalFilters(
        industry="Technology",
        company_size="Large",
        outcome="Positive"
    )

    assert filters.matches(sample_cases[0]) == True
    assert filters.matches(sample_cases[1]) == False  # Different industry
    assert filters.matches(sample_cases[2]) == False  # Different outcome


def test_filter_cases_by_metadata(sample_cases):
    """Test filtering cases by metadata without retrieval."""
    filters = RetrievalFilters(industry="Technology")
    filtered = filter_cases_by_metadata(sample_cases, filters)

    assert len(filtered) == 2
    assert all(c.industry == "Technology" for c in filtered)


def test_hybrid_retrieval_with_filters(sample_cases):
    """Test hybrid retrieval with metadata filters."""
    # Create mock index
    from unittest.mock import MagicMock
    mock_index = MagicMock()

    # Mock FAISS search results
    mock_index.search.return_value = (
        np.array([[0.95, 0.85, 0.75]]),
        np.array([[0, 1, 2]])
    )

    # Create query embedding
    query_emb = np.random.randn(1, 384).astype(np.float32)

    # Apply filters (only Technology industry)
    filters = RetrievalFilters(industry="Technology")
    results = retrieve_hybrid(query_emb, mock_index, sample_cases, filters, top_k=5)

    assert len(results) <= 2  # Only 2 Technology cases
    assert all(c.industry == "Technology" for c, _ in results)


# ─────────────────────────────────────────────────────────────
# Test Ensemble Prediction
# ─────────────────────────────────────────────────────────────

def test_ensemble_training(sample_cases):
    """Test ensemble model training."""
    ensemble = train_ensemble(sample_cases)

    assert ensemble is not None
    assert ensemble.lr_model is not None
    assert ensemble.rf_model is not None
    assert ensemble.scaler is not None
    assert len(ensemble.strategy_classes) > 0


def test_ensemble_prediction(sample_cases):
    """Test ensemble prediction with trained models."""
    ensemble = train_ensemble(sample_cases)

    # Mock retrieval results
    retrieved = [(sample_cases[0], 0.95), (sample_cases[1], 0.85)]

    recommendation = predict_ensemble(retrieved, ensemble)

    assert isinstance(recommendation, Recommendation)
    assert recommendation.recommended_strategy in [c.strategy_type for c in sample_cases]
    assert 0.0 <= recommendation.strategy_confidence <= 1.0


def test_ensemble_save_load(sample_cases, tmp_path):
    """Test saving and loading ensemble models."""
    ensemble = train_ensemble(sample_cases)

    save_path = tmp_path / "ensemble_test.pkl"
    save_ensemble(ensemble, save_path)

    assert save_path.exists()

    loaded = load_ensemble(save_path)
    assert loaded is not None
    assert len(loaded.strategy_classes) == len(ensemble.strategy_classes)


# ─────────────────────────────────────────────────────────────
# Test Analytics & Feedback
# ─────────────────────────────────────────────────────────────

def test_analytics_logging(sample_cases, tmp_path, monkeypatch):
    """Test query logging."""
    # Override log path
    log_path = tmp_path / "query_logs.jsonl"
    monkeypatch.setattr("src.analytics.LOGS_PATH", log_path)

    retrieved = [(sample_cases[0], 0.95)]
    recommendation = Recommendation(
        recommended_strategy="Digital Transformation",
        strategy_confidence=0.85,
        predicted_outcome="Positive",
        outcome_confidence=0.80,
        estimated_roi_low=2.0,
        estimated_roi_high=3.0,
        avg_time_to_result_months=18.0,
        supporting_rationale="Test rationale"
    )

    log_query(
        query_id="test-001",
        query_text="Test query",
        retrieved=retrieved,
        recommendation=recommendation,
        industry="Technology",
        top_k=5
    )

    assert log_path.exists()

    # Read back
    with open(log_path, "r") as f:
        log_entry = json.loads(f.read())

    assert log_entry["query_id"] == "test-001"
    assert log_entry["recommended_strategy"] == "Digital Transformation"


def test_feedback_storage(tmp_path, monkeypatch):
    """Test feedback storage."""
    feedback_path = tmp_path / "feedback.jsonl"
    monkeypatch.setattr("src.analytics.FEEDBACK_PATH", feedback_path)

    store_feedback(
        query_id="test-001",
        user_rating=5,
        helpful=True,
        notes="Great recommendation"
    )

    assert feedback_path.exists()

    with open(feedback_path, "r") as f:
        feedback = json.loads(f.read())

    assert feedback["query_id"] == "test-001"
    assert feedback["user_rating"] == 5
    assert feedback["helpful"] == True


def test_analytics_stats(tmp_path, monkeypatch):
    """Test analytics statistics computation."""
    log_path = tmp_path / "query_logs.jsonl"
    feedback_path = tmp_path / "feedback.jsonl"

    monkeypatch.setattr("src.analytics.LOGS_PATH", log_path)
    monkeypatch.setattr("src.analytics.FEEDBACK_PATH", feedback_path)

    # Create sample logs
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()
    feedback_path.touch()

    stats = get_analytics_stats()

    assert "total_queries" in stats
    assert "avg_confidence" in stats
    assert "feedback_count" in stats


# ─────────────────────────────────────────────────────────────
# Test Query Caching
# ─────────────────────────────────────────────────────────────

def test_cache_set_and_get(temp_cache_db, sample_cases):
    """Test caching and retrieval."""
    cache = temp_cache_db

    query_text = "Test business problem"
    embedding = np.random.randn(384).astype(np.float32)
    recommendation = Recommendation(
        recommended_strategy="Test Strategy",
        strategy_confidence=0.85,
        predicted_outcome="Positive",
        outcome_confidence=0.80,
        estimated_roi_low=2.0,
        estimated_roi_high=3.0,
        avg_time_to_result_months=18.0,
        supporting_rationale="Test"
    )
    case_ids = ["TEST-001", "TEST-002"]

    # Set cache
    cache.set(query_text, embedding, recommendation, case_ids)

    # Get cache
    result = cache.get(query_text)

    assert result is not None
    cached_emb, cached_rec, cached_ids = result

    assert np.allclose(cached_emb, embedding)
    assert cached_rec.recommended_strategy == "Test Strategy"
    assert cached_ids == case_ids


def test_cache_stats(temp_cache_db):
    """Test cache statistics."""
    cache = temp_cache_db

    # Add some entries
    for i in range(3):
        query = f"Test query {i}"
        emb = np.random.randn(384).astype(np.float32)
        rec = Recommendation(
            recommended_strategy="Test",
            strategy_confidence=0.85,
            predicted_outcome="Positive",
            outcome_confidence=0.80,
            estimated_roi_low=2.0,
            estimated_roi_high=3.0,
            avg_time_to_result_months=18.0,
            supporting_rationale="Test"
        )
        cache.set(query, emb, rec, ["TEST-001"])

    stats = cache.get_stats()

    assert stats["cached_queries"] == 3
    assert stats["total_accesses"] >= 3


def test_cache_invalidation(temp_cache_db):
    """Test cache clearing."""
    cache = temp_cache_db

    # Add entry
    query = "Test query"
    emb = np.random.randn(384).astype(np.float32)
    rec = Recommendation(
        recommended_strategy="Test",
        strategy_confidence=0.85,
        predicted_outcome="Positive",
        outcome_confidence=0.80,
        estimated_roi_low=2.0,
        estimated_roi_high=3.0,
        avg_time_to_result_months=18.0,
        supporting_rationale="Test"
    )
    cache.set(query, emb, rec, ["TEST-001"])

    assert cache.get(query) is not None

    # Invalidate
    cache.invalidate_all()

    assert cache.get(query) is None


# ─────────────────────────────────────────────────────────────
# Test Refined Retrieval
# ─────────────────────────────────────────────────────────────

def test_retrieve_related_cases(sample_cases):
    """Test related case retrieval."""
    # Mock index
    from unittest.mock import MagicMock
    mock_index = MagicMock()

    # Mock search results (returns indices)
    mock_index.search.return_value = (
        np.array([[0.95, 0.85, 0.75]]),
        np.array([[0, 1, 2]])
    )

    # Add embeddings to cases
    for case in sample_cases:
        case.embedding = np.random.randn(384).tolist()

    related = retrieve_related_cases("TEST-001", sample_cases, mock_index, top_k=2)

    assert len(related) <= 2
    assert all(c.case_id != "TEST-001" for c, _ in related)  # Excludes reference case


def test_retrieve_counter_examples(sample_cases):
    """Test counter-example retrieval."""
    mock_index = MagicMock()

    counter = retrieve_counter_examples(
        "Digital Transformation",
        sample_cases,
        mock_index,
        target_outcome="Positive",
        top_k=5
    )

    # Should return cases with different strategy but Positive outcome
    assert all(c.strategy_type != "Digital Transformation" for c, _ in counter)
    assert all(c.outcome == "Positive" for c, _ in counter)


# ─────────────────────────────────────────────────────────────
# Test API Endpoints (Mock)
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_api_health_check():
    """Test API health check endpoint."""
    from fastapi.testclient import TestClient
    from src.api_endpoints import app

    with patch('src.api_endpoints._cases', []) as mock_cases, \
         patch('src.api_endpoints._ensemble', None):

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"


# ─────────────────────────────────────────────────────────────
# Integration Test: Full Pipeline
# ─────────────────────────────────────────────────────────────

def test_full_consultation_pipeline(sample_cases, temp_cache_db):
    """Test complete consultation workflow."""
    # 1. Train ensemble
    ensemble = train_ensemble(sample_cases)
    assert ensemble is not None

    # 2. Mock retrieval
    retrieved = [(sample_cases[0], 0.95), (sample_cases[1], 0.85)]

    # 3. Get recommendation
    recommendation = predict_ensemble(retrieved, ensemble)
    assert recommendation is not None
    assert recommendation.strategy_confidence > 0

    # 4. Cache result
    cache = temp_cache_db
    query = "Test problem"
    embedding = np.random.randn(384).astype(np.float32)
    cache.set(query, embedding, recommendation, ["TEST-001", "TEST-002"])

    # 5. Retrieve from cache
    cached = cache.get(query)
    assert cached is not None

    # 6. Log query (with temp path)
    # Would test with monkeypatch as in earlier tests


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=html"])
