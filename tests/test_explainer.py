"""
Unit tests for the explainability module.
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import Case, OutcomeMetrics
from src.predictor import Recommendation
from src.explainer import explain, _extract_key_phrases, _shared_feature_analysis


@pytest.fixture
def sample_data():
    """Create sample data for explainability tests."""
    cases = [
        (
            Case(
                case_id="TEST-001",
                title="Digital Transformation Initiative",
                industry="Retail",
                company_size="Large",
                region="North America",
                year=2024,
                problem_description="Legacy systems hampering growth",
                key_challenges=["outdated infrastructure", "skill gaps"],
                actions_taken="Cloud migration and training",
                strategy_type="Digital Transformation",
                outcome="Positive",
                outcome_description="Successful transformation",
                outcome_metrics=OutcomeMetrics(
                    revenue_change_pct=15.0,
                    time_to_result_months=12,
                    roi_estimate=2.5
                ),
                tags=["cloud", "digital"]
            ),
            0.85
        ),
    ]

    query = "We need to modernize our legacy infrastructure with cloud technology"
    query_features = {"industry": "Retail", "company_size": "Large"}

    rec = Recommendation(
        recommended_strategy="Digital Transformation",
        strategy_confidence=0.85,
        predicted_outcome="Positive",
        outcome_confidence=0.85,
        estimated_roi_low=2.0,
        estimated_roi_high=3.0,
        avg_time_to_result_months=12.0,
        supporting_rationale="Test rationale"
    )

    return query, query_features, cases, rec


class TestExplainer:
    def test_explain_structure(self, sample_data):
        """Test that explain returns proper structure."""
        query, query_features, cases, rec = sample_data
        expl = explain(query, query_features, cases, rec)

        assert hasattr(expl, 'case_explanations')
        assert hasattr(expl, 'strategy_drivers')
        assert hasattr(expl, 'outcome_factors')
        assert hasattr(expl, 'confidence_note')
        assert hasattr(expl, 'methodology_note')

        assert len(expl.case_explanations) == len(cases)
        assert len(expl.strategy_drivers) > 0
        assert len(expl.outcome_factors) == len(cases)

    def test_case_explanations(self, sample_data):
        """Test per-case explanations."""
        query, query_features, cases, rec = sample_data
        expl = explain(query, query_features, cases, rec)

        ce = expl.case_explanations[0]
        assert ce.case_id == "TEST-001"
        assert ce.title == "Digital Transformation Initiative"
        assert ce.similarity_score == 0.85
        assert isinstance(ce.shared_features, dict)
        assert isinstance(ce.differing_features, dict)
        assert isinstance(ce.key_overlap_phrases, list)

    def test_key_phrase_extraction(self):
        """Test key phrase extraction."""
        text = "We need to modernize our legacy infrastructure with cloud technology"
        phrases = _extract_key_phrases(text, top_n=5)

        assert isinstance(phrases, set)
        assert len(phrases) <= 5
        # Should extract meaningful words, not stopwords
        assert "the" not in phrases
        assert "and" not in phrases
        # Should extract relevant technical terms
        assert any(word in phrases for word in ["modernize", "legacy", "infrastructure", "cloud", "technology"])

    def test_shared_feature_analysis(self, sample_data):
        """Test shared vs differing feature analysis."""
        _, query_features, cases, _ = sample_data
        case = cases[0][0]

        shared, diff = _shared_feature_analysis(query_features, case)

        # Industry and company_size should be shared
        assert "industry" in shared
        assert shared["industry"] == "Retail"
        assert "company_size" in shared
        assert shared["company_size"] == "Large"

    def test_strategy_drivers(self, sample_data):
        """Test strategy driver extraction."""
        query, query_features, cases, rec = sample_data
        expl = explain(query, query_features, cases, rec)

        drivers = expl.strategy_drivers
        assert len(drivers) > 0

        driver = drivers[0]
        assert "factor" in driver
        assert "weight" in driver
        assert "direction" in driver
        assert "case_count" in driver
        assert driver["direction"] in ["supporting", "alternative"]

    def test_outcome_factors(self, sample_data):
        """Test outcome factor extraction."""
        query, query_features, cases, rec = sample_data
        expl = explain(query, query_features, cases, rec)

        factors = expl.outcome_factors
        assert len(factors) == len(cases)

        factor = factors[0]
        assert "case_title" in factor
        assert "outcome" in factor
        assert "roi" in factor
        assert "weight" in factor
        assert "direction" in factor
        assert factor["direction"] in ["positive", "cautionary", "negative"]

    def test_confidence_note_high(self):
        """Test confidence note for high similarity."""
        query = "test query"
        query_features = {}
        cases = [
            (
                Case(
                    case_id="TEST-001",
                    title="Test",
                    industry="Retail",
                    company_size="Large",
                    region="NA",
                    year=2024,
                    problem_description="Test",
                    key_challenges=["test"],
                    actions_taken="Test",
                    strategy_type="Test",
                    outcome="Positive",
                    outcome_description="Test",
                    outcome_metrics=OutcomeMetrics(10.0, 12, 2.0),
                    tags=[]
                ),
                0.75  # High similarity
            )
        ]
        rec = Recommendation("Test", 0.75, "Positive", 0.75, 1.5, 2.5, 12.0, "Test")

        expl = explain(query, query_features, cases, rec)
        assert "High confidence" in expl.confidence_note or "Moderate confidence" in expl.confidence_note

    def test_methodology_note_present(self, sample_data):
        """Test that methodology note is present and descriptive."""
        query, query_features, cases, rec = sample_data
        expl = explain(query, query_features, cases, rec)

        assert len(expl.methodology_note) > 50
        assert "cosine similarity" in expl.methodology_note.lower()
        assert "embedding" in expl.methodology_note.lower()
