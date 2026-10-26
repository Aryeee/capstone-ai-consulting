"""
Tests for the feedback_collector module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from src.feedback_collector import QueryFeedback, store_feedback, get_feedback_stats


class TestQueryFeedback:
    """Tests for QueryFeedback dataclass."""

    def test_valid_feedback_creation(self):
        """Test creating a valid feedback record."""
        feedback = QueryFeedback(
            query_id="Q001",
            recommendation_id="R001",
            user_rating=5,
            helpful=True,
            notes="Great recommendation!"
        )
        assert feedback.query_id == "Q001"
        assert feedback.recommendation_id == "R001"
        assert feedback.user_rating == 5
        assert feedback.helpful is True
        assert feedback.notes == "Great recommendation!"
        assert feedback.timestamp is not None

    def test_feedback_with_empty_notes(self):
        """Test feedback creation with default empty notes."""
        feedback = QueryFeedback(
            query_id="Q001",
            recommendation_id="R001",
            user_rating=3,
            helpful=False
        )
        assert feedback.notes == ""

    def test_invalid_rating_too_high(self):
        """Test that ratings above 5 are rejected."""
        with pytest.raises(ValueError, match="user_rating must be an integer between 1 and 5"):
            QueryFeedback(
                query_id="Q001",
                recommendation_id="R001",
                user_rating=6,
                helpful=True
            )

    def test_invalid_rating_too_low(self):
        """Test that ratings below 1 are rejected."""
        with pytest.raises(ValueError, match="user_rating must be an integer between 1 and 5"):
            QueryFeedback(
                query_id="Q001",
                recommendation_id="R001",
                user_rating=0,
                helpful=True
            )

    def test_invalid_rating_not_integer(self):
        """Test that non-integer ratings are rejected."""
        with pytest.raises(ValueError, match="user_rating must be an integer between 1 and 5"):
            QueryFeedback(
                query_id="Q001",
                recommendation_id="R001",
                user_rating=3.5,
                helpful=True
            )

    def test_invalid_helpful_not_boolean(self):
        """Test that non-boolean helpful values are rejected."""
        with pytest.raises(ValueError, match="helpful must be a boolean"):
            QueryFeedback(
                query_id="Q001",
                recommendation_id="R001",
                user_rating=3,
                helpful="yes"
            )

    def test_invalid_query_id_empty(self):
        """Test that empty query_id is rejected."""
        with pytest.raises(ValueError, match="query_id must be a non-empty string"):
            QueryFeedback(
                query_id="",
                recommendation_id="R001",
                user_rating=3,
                helpful=True
            )

    def test_invalid_recommendation_id_empty(self):
        """Test that empty recommendation_id is rejected."""
        with pytest.raises(ValueError, match="recommendation_id must be a non-empty string"):
            QueryFeedback(
                query_id="Q001",
                recommendation_id="",
                user_rating=3,
                helpful=True
            )

    def test_invalid_notes_not_string(self):
        """Test that non-string notes are rejected."""
        with pytest.raises(ValueError, match="notes must be a string"):
            QueryFeedback(
                query_id="Q001",
                recommendation_id="R001",
                user_rating=3,
                helpful=True,
                notes=123
            )


class TestStoreFeedback:
    """Tests for store_feedback function."""

    def test_store_single_feedback(self):
        """Test storing a single feedback record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            feedback = QueryFeedback(
                query_id="Q001",
                recommendation_id="R001",
                user_rating=5,
                helpful=True,
                notes="Excellent!"
            )

            store_feedback(feedback, str(feedback_file))

            # Verify file was created and contains correct data
            assert feedback_file.exists()
            with open(feedback_file, 'r') as f:
                line = f.readline()
                record = json.loads(line)
                assert record['query_id'] == "Q001"
                assert record['user_rating'] == 5
                assert record['helpful'] is True

    def test_store_multiple_feedbacks(self):
        """Test appending multiple feedback records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            for i in range(3):
                feedback = QueryFeedback(
                    query_id=f"Q{i:03d}",
                    recommendation_id=f"R{i:03d}",
                    user_rating=i + 1,
                    helpful=i % 2 == 0
                )
                store_feedback(feedback, str(feedback_file))

            # Verify all records were stored
            with open(feedback_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 3

    def test_store_feedback_creates_directory(self):
        """Test that store_feedback creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "nested" / "deep" / "feedback.jsonl"

            feedback = QueryFeedback(
                query_id="Q001",
                recommendation_id="R001",
                user_rating=4,
                helpful=True
            )

            store_feedback(feedback, str(feedback_file))
            assert feedback_file.exists()

    def test_store_feedback_invalid_data(self):
        """Test that invalid feedback raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            with pytest.raises(ValueError, match="user_rating must be an integer between 1 and 5"):
                invalid_feedback = QueryFeedback(
                    query_id="Q001",
                    recommendation_id="R001",
                    user_rating=10,  # Invalid
                    helpful=True
                )


class TestGetFeedbackStats:
    """Tests for get_feedback_stats function."""

    def test_stats_on_empty_file(self):
        """Test stats generation on non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "nonexistent.jsonl"

            stats = get_feedback_stats(str(feedback_file))

            assert stats['total_feedback'] == 0
            assert stats['average_rating'] == 0.0
            assert stats['helpful_rate'] == 0.0
            assert stats['helpful_count'] == 0
            assert stats['unhelpful_count'] == 0

    def test_stats_single_record(self):
        """Test stats with a single feedback record."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            feedback = QueryFeedback(
                query_id="Q001",
                recommendation_id="R001",
                user_rating=5,
                helpful=True
            )
            store_feedback(feedback, str(feedback_file))

            stats = get_feedback_stats(str(feedback_file))

            assert stats['total_feedback'] == 1
            assert stats['average_rating'] == 5.0
            assert stats['helpful_count'] == 1
            assert stats['unhelpful_count'] == 0
            assert stats['helpful_rate'] == 100.0
            assert stats['rating_distribution'][5] == 1

    def test_stats_multiple_records(self):
        """Test stats with multiple feedback records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            # Create 5 records with varying ratings
            for i in range(5):
                feedback = QueryFeedback(
                    query_id=f"Q{i:03d}",
                    recommendation_id=f"R{i:03d}",
                    user_rating=i + 1,  # ratings 1-5
                    helpful=i < 3  # first 3 are helpful
                )
                store_feedback(feedback, str(feedback_file))

            stats = get_feedback_stats(str(feedback_file))

            assert stats['total_feedback'] == 5
            assert stats['average_rating'] == 3.0  # (1+2+3+4+5)/5
            assert stats['helpful_count'] == 3
            assert stats['unhelpful_count'] == 2
            assert stats['helpful_rate'] == 60.0
            assert stats['rating_distribution'][1] == 1
            assert stats['rating_distribution'][5] == 1

    def test_stats_unique_queries_and_recommendations(self):
        """Test that stats correctly count unique IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            # Create feedback with repeating query IDs
            for i in range(3):
                feedback = QueryFeedback(
                    query_id="Q001",  # Same query
                    recommendation_id=f"R{i:03d}",  # Different recommendations
                    user_rating=3,
                    helpful=True
                )
                store_feedback(feedback, str(feedback_file))

            stats = get_feedback_stats(str(feedback_file))

            assert stats['total_feedback'] == 3
            assert len(stats['queries_with_feedback']) == 1
            assert len(stats['recommendations_with_feedback']) == 3
            assert stats['queries_with_feedback'][0] == "Q001"

    def test_stats_with_malformed_json(self):
        """Test error handling for malformed JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            # Write malformed JSON
            with open(feedback_file, 'w') as f:
                f.write('invalid json\n')

            with pytest.raises(ValueError, match="Malformed JSON"):
                get_feedback_stats(str(feedback_file))

    def test_stats_with_missing_fields(self):
        """Test error handling for records with missing required fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            # Write valid record followed by incomplete record
            with open(feedback_file, 'w') as f:
                f.write('{"query_id": "Q001", "recommendation_id": "R001", "user_rating": 3, "helpful": true}\n')
                f.write('{"query_id": "Q002"}\n')  # Missing required fields

            with pytest.raises(ValueError, match="Missing required fields"):
                get_feedback_stats(str(feedback_file))

    def test_stats_with_invalid_rating(self):
        """Test error handling for invalid ratings in stored data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            # Write record with invalid rating
            with open(feedback_file, 'w') as f:
                f.write('{"query_id": "Q001", "recommendation_id": "R001", "user_rating": 10, "helpful": true}\n')

            with pytest.raises(ValueError, match="Invalid rating"):
                get_feedback_stats(str(feedback_file))

    def test_stats_skips_empty_lines(self):
        """Test that stats correctly handles empty lines in JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_file = Path(tmpdir) / "feedback.jsonl"

            # Write records with empty lines
            with open(feedback_file, 'w') as f:
                f.write('{"query_id": "Q001", "recommendation_id": "R001", "user_rating": 4, "helpful": true}\n')
                f.write('\n')  # Empty line
                f.write('{"query_id": "Q002", "recommendation_id": "R002", "user_rating": 5, "helpful": true}\n')

            stats = get_feedback_stats(str(feedback_file))

            assert stats['total_feedback'] == 2
            assert stats['average_rating'] == 4.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
