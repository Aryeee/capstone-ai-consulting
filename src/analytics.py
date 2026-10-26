"""
Analytics tracking: log queries, track metrics, compute statistics.

Stores query logs with feedback for continuous improvement and usage analysis.
Provides functions to:
1. Track query logs (query_id, query_text, industry, retrieved_cases, recommendation, user_rating)
2. Compute metrics (query_volume_by_industry, popular_queries,
   recommendation_accuracy_by_strategy, top_strategies)
3. Export analytics as DataFrame

Query logs are stored in data/query_logs.jsonl for efficient streaming and appending.
"""

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
from collections import Counter

from src.schema import Case
from src.predictor import Recommendation


@dataclass
class QueryLog:
    """Record of a single consultation."""
    query_id: str
    timestamp: str
    query_text: str
    industry: Optional[str]
    company_size: Optional[str]
    retrieved_case_ids: list[str]
    recommended_strategy: str
    strategy_confidence: float
    predicted_outcome: str
    estimated_roi_low: float
    estimated_roi_high: float
    top_k: int


@dataclass
class QueryFeedback:
    """User feedback on a consultation."""
    query_id: str
    timestamp: str
    user_rating: int  # 1-5
    helpful: bool
    notes: Optional[str] = None


LOGS_PATH = Path(__file__).resolve().parent.parent / "data" / "query_logs.jsonl"
FEEDBACK_PATH = Path(__file__).resolve().parent.parent / "data" / "feedback.jsonl"
QUERY_LOGS_FILE = LOGS_PATH  # Alias for consistency


def init_logs():
    """Initialize log files if they don't exist."""
    LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOGS_PATH.touch(exist_ok=True)
    FEEDBACK_PATH.touch(exist_ok=True)


def log_query(
    query_id: str,
    query_text: str,
    retrieved: list[tuple[Case, float]],
    recommendation: Recommendation,
    industry: Optional[str] = None,
    company_size: Optional[str] = None,
    top_k: int = 5,
) -> None:
    """Log a consultation query with its recommendation."""
    init_logs()

    log_entry = QueryLog(
        query_id=query_id,
        timestamp=datetime.utcnow().isoformat(),
        query_text=query_text,
        industry=industry,
        company_size=company_size,
        retrieved_case_ids=[c.case_id for c, _ in retrieved],
        recommended_strategy=recommendation.recommended_strategy,
        strategy_confidence=recommendation.strategy_confidence,
        predicted_outcome=recommendation.predicted_outcome,
        estimated_roi_low=recommendation.estimated_roi_low,
        estimated_roi_high=recommendation.estimated_roi_high,
        top_k=top_k,
    )

    with open(LOGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(log_entry)) + "\n")


def store_feedback(
    query_id: str,
    user_rating: int,
    helpful: bool,
    notes: Optional[str] = None,
) -> None:
    """Store user feedback on a consultation."""
    init_logs()

    if not 1 <= user_rating <= 5:
        raise ValueError("user_rating must be 1-5")

    feedback = QueryFeedback(
        query_id=query_id,
        timestamp=datetime.utcnow().isoformat(),
        user_rating=user_rating,
        helpful=helpful,
        notes=notes,
    )

    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(feedback)) + "\n")


def load_query_logs() -> list[QueryLog]:
    """Load all query logs from disk."""
    init_logs()
    logs = []
    if LOGS_PATH.stat().st_size == 0:
        return logs

    with open(LOGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                logs.append(QueryLog(**json.loads(line)))
    return logs


def load_feedback() -> list[QueryFeedback]:
    """Load all feedback from disk."""
    init_logs()
    feedback = []
    if FEEDBACK_PATH.stat().st_size == 0:
        return feedback

    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                feedback.append(QueryFeedback(**json.loads(line)))
    return feedback


def get_analytics_stats() -> dict:
    """Compute key analytics statistics."""
    logs = load_query_logs()
    feedback = load_feedback()

    if not logs:
        return {
            "total_queries": 0,
            "top_strategies": [],
            "industry_distribution": {},
            "avg_confidence": 0.0,
            "avg_roi_range": (0.0, 0.0),
            "feedback_count": 0,
            "avg_rating": 0.0,
            "helpful_rate": 0.0,
        }

    # Strategy distribution
    strategy_counts = {}
    for log in logs:
        strategy_counts[log.recommended_strategy] = strategy_counts.get(
            log.recommended_strategy, 0
        ) + 1

    # Industry distribution
    industry_counts = {}
    for log in logs:
        if log.industry:
            industry_counts[log.industry] = industry_counts.get(log.industry, 0) + 1

    # Confidence metrics
    avg_confidence = sum(log.strategy_confidence for log in logs) / len(logs)

    # ROI metrics
    roi_lows = [log.estimated_roi_low for log in logs]
    roi_highs = [log.estimated_roi_high for log in logs]
    avg_roi_low = sum(roi_lows) / len(roi_lows)
    avg_roi_high = sum(roi_highs) / len(roi_highs)

    # Feedback metrics
    if feedback:
        avg_rating = sum(f.user_rating for f in feedback) / len(feedback)
        helpful_count = sum(1 for f in feedback if f.helpful)
        helpful_rate = helpful_count / len(feedback)
    else:
        avg_rating = 0.0
        helpful_rate = 0.0

    return {
        "total_queries": len(logs),
        "top_strategies": sorted(
            strategy_counts.items(), key=lambda x: x[1], reverse=True
        )[:5],
        "industry_distribution": dict(
            sorted(industry_counts.items(), key=lambda x: x[1], reverse=True)
        ),
        "avg_confidence": round(avg_confidence, 3),
        "avg_roi_range": (round(avg_roi_low, 2), round(avg_roi_high, 2)),
        "feedback_count": len(feedback),
        "avg_rating": round(avg_rating, 2),
        "helpful_rate": round(helpful_rate, 3),
    }


def get_recommendation_accuracy_by_strategy() -> dict[str, dict]:
    """Compute accuracy metrics per strategy (based on feedback)."""
    logs = load_query_logs()
    feedback = load_feedback()

    # Map query_id → feedback
    feedback_map = {f.query_id: f for f in feedback}

    # Group logs by strategy
    by_strategy = {}
    for log in logs:
        if log.query_id in feedback_map:
            fb = feedback_map[log.query_id]
            if log.recommended_strategy not in by_strategy:
                by_strategy[log.recommended_strategy] = []
            by_strategy[log.recommended_strategy].append(
                {"rating": fb.user_rating, "helpful": fb.helpful}
            )

    # Compute stats per strategy
    stats = {}
    for strategy, ratings in by_strategy.items():
        if ratings:
            avg_rating = sum(r["rating"] for r in ratings) / len(ratings)
            helpful_count = sum(1 for r in ratings if r["helpful"])
            helpful_rate = helpful_count / len(ratings)
            stats[strategy] = {
                "feedback_count": len(ratings),
                "avg_rating": round(avg_rating, 2),
                "helpful_rate": round(helpful_rate, 3),
            }

    return stats


def export_analytics_dataframe() -> pd.DataFrame:
    """Export analytics as a pandas DataFrame for further analysis."""
    logs = load_query_logs()

    if not logs:
        return pd.DataFrame()

    data = []
    for log in logs:
        data.append(
            {
                "query_id": log.query_id,
                "timestamp": log.timestamp,
                "industry": log.industry,
                "company_size": log.company_size,
                "strategy": log.recommended_strategy,
                "confidence": log.strategy_confidence,
                "outcome": log.predicted_outcome,
                "roi_low": log.estimated_roi_low,
                "roi_high": log.estimated_roi_high,
                "num_retrieved": len(log.retrieved_case_ids),
            }
        )

    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────
# New Analytics Functions (Improvements)
# ─────────────────────────────────────────────────────────────


def track_query(
    query_text: str,
    industry: str,
    retrieved_cases: List[str],
    recommendation: str,
    user_rating: Optional[float] = None,
) -> str:
    """
    Track a query and store it in the query logs.

    Args:
        query_text: The user's query text
        industry: Industry context for the query
        retrieved_cases: List of case IDs retrieved by the system
        recommendation: The recommended strategy
        user_rating: Optional user rating (0.0-5.0) for the recommendation

    Returns:
        query_id: Unique identifier for the logged query
    """
    LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    query_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Create a simple log entry (simplified format for this function)
    log_entry = {
        "query_id": query_id,
        "timestamp": timestamp,
        "query_text": query_text,
        "industry": industry,
        "retrieved_cases": retrieved_cases,
        "recommendation": recommendation,
        "user_rating": user_rating,
    }

    with open(LOGS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return query_id


def load_simplified_query_logs() -> pd.DataFrame:
    """
    Load simplified query logs (from track_query function) into a DataFrame.

    Returns:
        DataFrame with columns: query_id, timestamp, query_text, industry,
        retrieved_cases, recommendation, user_rating
    """
    if not LOGS_PATH.exists():
        return pd.DataFrame()

    records = []
    with open(LOGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    # Only include simplified format records
                    if "retrieved_cases" in record:
                        records.append(record)
                except json.JSONDecodeError:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def query_volume_by_industry() -> Dict[str, int]:
    """
    Compute the number of queries per industry.

    Returns:
        Dictionary mapping industry names to query counts
    """
    df = load_simplified_query_logs()
    if df.empty:
        return {}

    return df["industry"].value_counts().to_dict()


def popular_queries(top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Return the most popular/frequently asked queries.

    Args:
        top_n: Number of top queries to return

    Returns:
        List of dicts with query_text and frequency
    """
    df = load_simplified_query_logs()
    if df.empty:
        return []

    query_counts = df["query_text"].value_counts().head(top_n)
    return [
        {"query": query, "frequency": int(count)}
        for query, count in query_counts.items()
    ]


def recommendation_accuracy_by_strategy() -> Dict[str, Dict[str, Any]]:
    """
    Compute accuracy metrics for each recommended strategy.

    Accuracy is defined as the average user rating for each strategy.
    Also includes counts of ratings per strategy.

    Returns:
        Dictionary mapping strategy names to:
        - avg_rating: average user rating (0-5)
        - count: number of ratings
        - median_rating: median user rating
    """
    df = load_simplified_query_logs()
    if df.empty:
        return {}

    # Filter only rows with user ratings
    df_rated = df[df["user_rating"].notna()]

    if df_rated.empty:
        return {}

    results = {}
    for strategy, group in df_rated.groupby("recommendation"):
        ratings = group["user_rating"]
        results[strategy] = {
            "avg_rating": float(ratings.mean()),
            "median_rating": float(ratings.median()),
            "count": int(len(ratings)),
            "min_rating": float(ratings.min()),
            "max_rating": float(ratings.max()),
        }

    return results


def top_strategies(top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Return the most frequently recommended strategies.

    Args:
        top_n: Number of top strategies to return

    Returns:
        List of dicts with strategy name and recommendation count
    """
    df = load_simplified_query_logs()
    if df.empty:
        return []

    strategy_counts = df["recommendation"].value_counts().head(top_n)
    return [
        {"strategy": strategy, "count": int(count)}
        for strategy, count in strategy_counts.items()
    ]


def clear_logs() -> None:
    """Clear all query logs (for testing/reset purposes)."""
    if LOGS_PATH.exists():
        LOGS_PATH.unlink()


def get_logs_count() -> int:
    """Get the total number of logged queries."""
    df = load_simplified_query_logs()
    return len(df)


def export_to_csv(filepath: Optional[Path] = None) -> str:
    """
    Export analytics DataFrame to CSV file.

    Args:
        filepath: Path to save CSV file. If None, saves to data/analytics_export.csv

    Returns:
        Path to the saved CSV file
    """
    if filepath is None:
        filepath = LOGS_PATH.parent / "analytics_export.csv"

    df = load_simplified_query_logs()

    LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    return str(filepath)


def export_to_json(filepath: Optional[Path] = None) -> str:
    """
    Export analytics DataFrame to JSON file.

    Args:
        filepath: Path to save JSON file. If None, saves to data/analytics_export.json

    Returns:
        Path to the saved JSON file
    """
    if filepath is None:
        filepath = LOGS_PATH.parent / "analytics_export.json"

    df = load_simplified_query_logs()

    LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(filepath, orient="records", date_format="iso", indent=2)

    return str(filepath)
