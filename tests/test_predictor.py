"""
Unit tests for the prediction module.
"""

import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import Case, OutcomeMetrics
from src.predictor import predict, Recommendation


@pytest.fixture
def sample_retrieved():
    """Create sample retrieved cases with similarity scores."""
    cases = [
        (
            Case(
                case_id="TEST-001",
                title="Digital Transformation A",
                industry="Retail",
                company_size="Large",
                region="North America",
                year=2024,
                problem_description="Legacy systems",
                key_challenges=["infrastructure"],
                actions_taken="Cloud migration",
                strategy_type="Digital Transformation",
                outcome="Positive",
                outcome_description="Success",
                outcome_metrics=OutcomeMetrics(
                    revenue_change_pct=15.0,
                    time_to_result_months=12,
                    roi_estimate=2.5
                ),
                tags=["cloud"]
            ),
            0.85
        ),
        (
            Case(
                case_id="TEST-002",
                title="Digital Transformation B",
                industry="Technology",
                company_size="Medium",
                region="Europe",
                year=2023,
                problem_description="Outdated systems",
                key_challenges=["legacy"],
                actions_taken="Modernization",
                strategy_type="Digital Transformation",
                outcome="Positive",
                outcome_description="Success",
                outcome_metrics=OutcomeMetrics(
                    revenue_change_pct=12.0,
                    time_to_result_months=10,
                    roi_estimate=2.0
                ),
                tags=["modernization"]
            ),
            0.75
        ),
        (
            Case(
                case_id="TEST-003",
                title="Cost Optimization",
                industry="Manufacturing",
                company_size="Large",
                region="Asia",
                year=2022,
                problem_description="High costs",
                key_challenges=["efficiency"],
                actions_taken="Process automation",
                strategy_type="Cost Optimization",
                outcome="Mixed",
                outcome_description="Partial success",
                outcome_metrics=OutcomeMetrics(
                    revenue_change_pct=5.0,
                    time_to_result_months=8,
                    roi_estimate=1.5
                ),
                tags=["automation"]
            ),
            0.60
        ),
    ]
    return cases


class TestPredictor:
    def test_predict_with_results(self, sample_retrieved):
        """Test prediction with valid retrieved cases."""
        rec = predict(sample_retrieved)

        assert isinstance(rec, Recommendation)
        assert rec.recommended_strategy == "Digital Transformation"
        assert 0 <= rec.strategy_confidence <= 1
        assert rec.predicted_outcome in ["Positive", "Mixed", "Negative"]
        assert 0 <= rec.outcome_confidence <= 1
        assert rec.estimated_roi_low >= 0
        assert rec.estimated_roi_high >= rec.estimated_roi_low
        assert rec.avg_time_to_result_months > 0
        assert len(rec.supporting_rationale) > 0

    def test_predict_empty_results(self):
        """Test prediction with no retrieved cases."""
        rec = predict([])

        assert rec.recommended_strategy == "Insufficient data"
        assert rec.strategy_confidence == 0.0
        assert rec.predicted_outcome == "Unknown"
        assert rec.outcome_confidence == 0.0
        assert rec.supporting_rationale == "No similar cases found."

    def test_weighted_vote_strategy(self, sample_retrieved):
        """Test that higher similarity scores have more weight."""
        rec = predict(sample_retrieved)

        # Top 2 cases are "Digital Transformation", bottom is "Cost Optimization"
        # Digital Transformation should win due to higher weights
        assert rec.recommended_strategy == "Digital Transformation"
        assert rec.strategy_confidence > 0.5

    def test_roi_estimation(self, sample_retrieved):
        """Test ROI range estimation."""
        rec = predict(sample_retrieved)

        # ROI estimates are [2.5, 2.0, 1.5]
        # Weighted mean should be closer to 2.5 and 2.0 due to higher scores
        assert rec.estimated_roi_low > 0
        assert rec.estimated_roi_high > rec.estimated_roi_low
        # Check that the range is reasonable given the input
        assert 1.5 <= rec.estimated_roi_high <= 3.0

    def test_time_estimation(self, sample_retrieved):
        """Test timeline estimation."""
        rec = predict(sample_retrieved)

        # Times are [12, 10, 8]
        # Weighted average should favor higher-scored cases
        assert 8 <= rec.avg_time_to_result_months <= 12

    def test_confidence_values(self, sample_retrieved):
        """Test that confidence values are properly normalized."""
        rec = predict(sample_retrieved)

        assert 0 <= rec.strategy_confidence <= 1
        assert 0 <= rec.outcome_confidence <= 1
        # With 3 cases, confidence shouldn't be too extreme
        assert rec.strategy_confidence > 0.3
