"""
Integration tests for the full application flow.
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.case_db import load_cases_from_json, embed_cases, build_index, retrieve
from src.predictor import predict
from src.explainer import explain


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test the complete analysis pipeline."""
        # Load cases
        cases = load_cases_from_json()
        assert len(cases) > 0

        # Embed a subset for faster testing
        test_cases = cases[:5]
        embeddings = embed_cases(test_cases)
        assert embeddings.shape[0] == len(test_cases)

        # Build index
        index = build_index(embeddings)
        assert index.ntotal == len(test_cases)

        # Retrieve
        query = "We need to modernize our legacy infrastructure with cloud technology"
        retrieved = retrieve(query, index, test_cases, top_k=3)
        assert len(retrieved) <= 3
        assert all(0 <= score <= 1 for _, score in retrieved)

        # Predict
        recommendation = predict(retrieved)
        assert recommendation.recommended_strategy != "Insufficient data"
        assert recommendation.strategy_confidence > 0
        assert recommendation.estimated_roi_low >= 0
        assert recommendation.avg_time_to_result_months > 0

        # Explain
        query_features = {"industry": "Technology", "company_size": "Medium"}
        explanation = explain(query, query_features, retrieved, recommendation)
        assert len(explanation.case_explanations) == len(retrieved)
        assert len(explanation.strategy_drivers) > 0
        assert len(explanation.confidence_note) > 0

    def test_low_similarity_handling(self):
        """Test system behavior with low similarity matches."""
        cases = load_cases_from_json()[:3]
        embeddings = embed_cases(cases)
        index = build_index(embeddings)

        # Use a very different query
        query = "quantum computing blockchain AI metaverse cryptocurrency NFT"
        retrieved = retrieve(query, index, cases, top_k=3)

        # Even with low similarity, system should still provide recommendations
        recommendation = predict(retrieved)
        assert recommendation.recommended_strategy != "Insufficient data"

        # Explanation should reflect low confidence
        query_features = {}
        explanation = explain(query, query_features, retrieved, recommendation)
        # Low similarity should be noted in confidence assessment
        assert "confidence" in explanation.confidence_note.lower()

    def test_different_query_types(self):
        """Test various query styles."""
        cases = load_cases_from_json()[:5]
        embeddings = embed_cases(cases)
        index = build_index(embeddings)

        queries = [
            "Digital transformation cloud migration",
            "We need to reduce costs and improve efficiency",
            "Expanding into new international markets",
            "Customer retention and satisfaction improvement",
        ]

        for query in queries:
            retrieved = retrieve(query, index, cases, top_k=3)
            recommendation = predict(retrieved)

            # Each query should get a valid recommendation
            assert recommendation.recommended_strategy != "Insufficient data"
            assert recommendation.strategy_confidence > 0
            assert len(recommendation.supporting_rationale) > 0

    def test_edge_case_single_case(self):
        """Test with only one case in database."""
        cases = load_cases_from_json()[:1]
        embeddings = embed_cases(cases)
        index = build_index(embeddings)

        query = "business problem"
        retrieved = retrieve(query, index, cases, top_k=1)

        # Should still work with single case
        assert len(retrieved) == 1

        recommendation = predict(retrieved)
        assert recommendation.recommended_strategy == retrieved[0][0].strategy_type
        assert recommendation.strategy_confidence > 0
