"""
Unit tests for the case database module.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import Case, OutcomeMetrics
from src.case_db import (
    load_cases_from_json,
    embed_cases,
    build_index,
    retrieve,
    get_model,
)


@pytest.fixture
def sample_cases():
    """Create sample cases for testing."""
    return [
        Case(
            case_id="TEST-001",
            title="Digital Transformation",
            industry="Retail",
            company_size="Large",
            region="North America",
            year=2024,
            problem_description="Legacy systems hampering digital initiatives",
            key_challenges=["outdated infrastructure", "resistance to change"],
            actions_taken="Implemented cloud migration and training program",
            strategy_type="Digital Transformation",
            outcome="Positive",
            outcome_description="Successful migration completed",
            outcome_metrics=OutcomeMetrics(
                revenue_change_pct=15.0,
                time_to_result_months=12,
                roi_estimate=2.5
            ),
            tags=["cloud", "digital"]
        ),
        Case(
            case_id="TEST-002",
            title="Market Expansion",
            industry="Technology",
            company_size="Medium",
            region="Europe",
            year=2023,
            problem_description="Need to expand into new markets",
            key_challenges=["cultural differences", "regulatory compliance"],
            actions_taken="Hired local teams and legal consultants",
            strategy_type="Market Expansion",
            outcome="Mixed",
            outcome_description="Expansion successful in some regions",
            outcome_metrics=OutcomeMetrics(
                revenue_change_pct=8.0,
                time_to_result_months=18,
                roi_estimate=1.8
            ),
            tags=["expansion", "international"]
        ),
    ]


class TestCaseDatabase:
    def test_load_cases_from_json(self):
        """Test loading cases from JSON file."""
        cases = load_cases_from_json()
        assert len(cases) > 0
        assert all(isinstance(c, Case) for c in cases)
        assert all(hasattr(c, 'case_id') for c in cases)

    def test_embed_cases(self, sample_cases):
        """Test embedding generation."""
        embeddings = embed_cases(sample_cases[:1])  # Test with one case

        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension
        assert embeddings.dtype == np.float32

        # Check normalization (L2 norm should be ~1.0)
        norm = np.linalg.norm(embeddings[0])
        assert 0.99 <= norm <= 1.01

    def test_build_index(self, sample_cases):
        """Test FAISS index building."""
        embeddings = embed_cases(sample_cases)
        index = build_index(embeddings)

        assert index.ntotal == len(sample_cases)
        assert index.d == 384

    def test_retrieve(self, sample_cases):
        """Test case retrieval."""
        embeddings = embed_cases(sample_cases)
        index = build_index(embeddings)

        query = "digital transformation cloud migration"
        results = retrieve(query, index, sample_cases, top_k=2)

        assert len(results) == 2
        assert all(isinstance(r[0], Case) for r in results)
        assert all(isinstance(r[1], float) for r in results)

        # Scores should be between 0 and 1 (cosine similarity)
        assert all(0 <= score <= 1 for _, score in results)

        # Results should be sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_get_model_singleton(self):
        """Test model singleton behavior."""
        model1 = get_model()
        model2 = get_model()
        assert model1 is model2  # Should be the same instance


class TestCaseSchema:
    def test_case_text_for_embedding(self, sample_cases):
        """Test text concatenation for embedding."""
        case = sample_cases[0]
        text = case.text_for_embedding

        assert case.title in text
        assert case.industry in text
        assert case.problem_description in text
        assert case.actions_taken in text
        assert all(ch in text for ch in case.key_challenges)

    def test_case_structured_features(self, sample_cases):
        """Test structured features extraction."""
        case = sample_cases[0]
        features = case.structured_features

        assert features["industry"] == case.industry
        assert features["company_size"] == case.company_size
        assert features["region"] == case.region
        assert features["strategy_type"] == case.strategy_type
        assert features["num_challenges"] == len(case.key_challenges)
        assert features["num_tags"] == len(case.tags)
        assert features["year"] == case.year
