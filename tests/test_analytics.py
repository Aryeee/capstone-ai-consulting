"""
Tests for the analytics module.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
import json

from src import analytics


@pytest.fixture
def temp_logs_dir(monkeypatch):
    """Create a temporary directory for test logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override the logs file path
        logs_file = Path(tmpdir) / "query_logs.jsonl"
        monkeypatch.setattr(analytics, "LOGS_PATH", logs_file)
        monkeypatch.setattr(analytics, "QUERY_LOGS_FILE", logs_file)
        yield Path(tmpdir)


def test_track_query(temp_logs_dir):
    """Test tracking a single query."""
    query_id = analytics.track_query(
        query_text="How can we improve customer retention?",
        industry="Retail",
        retrieved_cases=["CASE-001", "CASE-002", "CASE-003"],
        recommendation="Customer Experience",
        user_rating=4.5,
    )

    assert query_id is not None
    assert analytics.QUERY_LOGS_FILE.exists()

    # Verify the logged data
    df = analytics.load_simplified_query_logs()
    assert len(df) == 1
    assert df.iloc[0]["query_text"] == "How can we improve customer retention?"
    assert df.iloc[0]["industry"] == "Retail"
    assert df.iloc[0]["user_rating"] == 4.5


def test_load_query_logs_empty(temp_logs_dir):
    """Test loading logs when no logs exist."""
    df = analytics.load_simplified_query_logs()
    assert df.empty


def test_query_volume_by_industry(temp_logs_dir):
    """Test computing query volume by industry."""
    # Add test data
    analytics.track_query("Query 1", "Retail", ["CASE-001"], "Strategy-A")
    analytics.track_query("Query 2", "Retail", ["CASE-002"], "Strategy-B")
    analytics.track_query("Query 3", "Healthcare", ["CASE-003"], "Strategy-A")
    analytics.track_query("Query 4", "Technology", ["CASE-004"], "Strategy-C")

    volumes = analytics.query_volume_by_industry()

    assert volumes["Retail"] == 2
    assert volumes["Healthcare"] == 1
    assert volumes["Technology"] == 1


def test_popular_queries(temp_logs_dir):
    """Test finding popular queries."""
    # Add repeated queries
    analytics.track_query("Popular query", "Retail", ["CASE-001"], "Strategy-A")
    analytics.track_query("Popular query", "Retail", ["CASE-002"], "Strategy-A")
    analytics.track_query("Popular query", "Healthcare", ["CASE-003"], "Strategy-B")
    analytics.track_query("Unique query", "Technology", ["CASE-004"], "Strategy-C")

    popular = analytics.popular_queries(top_n=10)

    assert len(popular) > 0
    assert popular[0]["query"] == "Popular query"
    assert popular[0]["frequency"] == 3


def test_recommendation_accuracy_by_strategy(temp_logs_dir):
    """Test computing recommendation accuracy by strategy."""
    # Add queries with ratings
    analytics.track_query("Query 1", "Retail", ["CASE-001"], "Strategy-A", user_rating=5.0)
    analytics.track_query("Query 2", "Retail", ["CASE-002"], "Strategy-A", user_rating=4.0)
    analytics.track_query("Query 3", "Healthcare", ["CASE-003"], "Strategy-B", user_rating=3.0)
    analytics.track_query("Query 4", "Technology", ["CASE-004"], "Strategy-A", user_rating=4.5)

    accuracy = analytics.recommendation_accuracy_by_strategy()

    assert "Strategy-A" in accuracy
    assert accuracy["Strategy-A"]["avg_rating"] == pytest.approx(4.5, abs=0.01)
    assert accuracy["Strategy-A"]["count"] == 3
    assert accuracy["Strategy-B"]["avg_rating"] == 3.0


def test_top_strategies(temp_logs_dir):
    """Test finding top strategies."""
    analytics.track_query("Query 1", "Retail", ["CASE-001"], "Digital Transformation")
    analytics.track_query("Query 2", "Retail", ["CASE-002"], "Digital Transformation")
    analytics.track_query("Query 3", "Healthcare", ["CASE-003"], "Cost Optimization")
    analytics.track_query("Query 4", "Technology", ["CASE-004"], "Digital Transformation")

    top_strats = analytics.top_strategies(top_n=10)

    assert len(top_strats) == 2
    assert top_strats[0]["strategy"] == "Digital Transformation"
    assert top_strats[0]["count"] == 3
    assert top_strats[1]["strategy"] == "Cost Optimization"
    assert top_strats[1]["count"] == 1


def test_export_to_csv(temp_logs_dir):
    """Test exporting to CSV."""
    analytics.track_query("Query 1", "Retail", ["CASE-001"], "Strategy-A")

    csv_path = analytics.export_to_csv()

    assert Path(csv_path).exists()

    # Verify CSV content
    df = __import__('pandas').read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["industry"] == "Retail"


def test_export_to_json(temp_logs_dir):
    """Test exporting to JSON."""
    analytics.track_query("Query 1", "Retail", ["CASE-001"], "Strategy-A")

    json_path = analytics.export_to_json()

    assert Path(json_path).exists()

    # Verify JSON content
    with open(json_path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["industry"] == "Retail"


def test_clear_logs(temp_logs_dir):
    """Test clearing logs."""
    analytics.track_query("Query 1", "Retail", ["CASE-001"], "Strategy-A")
    assert analytics.get_logs_count() == 1

    analytics.clear_logs()
    assert analytics.get_logs_count() == 0


def test_get_logs_count(temp_logs_dir):
    """Test getting logs count."""
    assert analytics.get_logs_count() == 0

    analytics.track_query("Query 1", "Retail", ["CASE-001"], "Strategy-A")
    assert analytics.get_logs_count() == 1

    analytics.track_query("Query 2", "Healthcare", ["CASE-002"], "Strategy-B")
    assert analytics.get_logs_count() == 2
