"""
Feedback collection and analysis module for AI Strategy Consulting Assistant.

Manages user feedback on query recommendations, stores feedback to JSONL,
and provides analytics on recommendation accuracy.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional
from datetime import datetime, timezone
import json
import os
from pathlib import Path


@dataclass
class QueryFeedback:
    """Represents user feedback on a query recommendation.

    Attributes:
        query_id: Unique identifier for the query
        recommendation_id: Unique identifier for the recommendation
        user_rating: User's rating of recommendation quality (1-5 scale)
        helpful: Whether the recommendation was helpful
        notes: Optional user comments or additional feedback
        timestamp: When the feedback was recorded (auto-set)
    """
    query_id: str
    recommendation_id: str
    user_rating: int
    helpful: bool
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        """Validate feedback data on initialization."""
        if not isinstance(self.user_rating, int) or not (1 <= self.user_rating <= 5):
            raise ValueError("user_rating must be an integer between 1 and 5")

        if not isinstance(self.helpful, bool):
            raise ValueError("helpful must be a boolean")

        if not isinstance(self.notes, str):
            raise ValueError("notes must be a string")

        if not self.query_id or not isinstance(self.query_id, str):
            raise ValueError("query_id must be a non-empty string")

        if not self.recommendation_id or not isinstance(self.recommendation_id, str):
            raise ValueError("recommendation_id must be a non-empty string")


def store_feedback(feedback: QueryFeedback, feedback_file: str = "data/feedback.jsonl") -> None:
    """Store feedback record to JSONL file.

    Appends a single feedback record to the feedback log. Creates the file
    and parent directory if they don't exist.

    Args:
        feedback: QueryFeedback instance to store
        feedback_file: Path to the feedback JSONL file

    Raises:
        ValueError: If feedback validation fails
        IOError: If file operations fail

    Example:
        >>> feedback = QueryFeedback(
        ...     query_id="Q001",
        ...     recommendation_id="R001",
        ...     user_rating=5,
        ...     helpful=True,
        ...     notes="Excellent recommendation!"
        ... )
        >>> store_feedback(feedback)
    """
    try:
        # Ensure parent directory exists
        feedback_path = Path(feedback_file)
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert feedback to dictionary and serialize to JSONL
        feedback_dict = asdict(feedback)

        # Append to file
        with open(feedback_path, 'a', encoding='utf-8') as f:
            json.dump(feedback_dict, f, ensure_ascii=False)
            f.write('\n')

    except ValueError as e:
        raise ValueError(f"Invalid feedback data: {str(e)}") from e
    except IOError as e:
        raise IOError(f"Failed to write feedback to {feedback_file}: {str(e)}") from e
    except Exception as e:
        raise IOError(f"Unexpected error storing feedback: {str(e)}") from e


def get_feedback_stats(feedback_file: str = "data/feedback.jsonl") -> dict:
    """Generate accuracy and quality statistics from feedback records.

    Analyzes all stored feedback to provide metrics on recommendation quality,
    user satisfaction, and helpfulness rates.

    Args:
        feedback_file: Path to the feedback JSONL file

    Returns:
        Dictionary containing:
            - total_feedback: Total number of feedback records
            - average_rating: Mean user rating (1-5)
            - helpful_count: Number of helpful recommendations
            - unhelpful_count: Number of unhelpful recommendations
            - helpful_rate: Percentage of helpful recommendations (0-100)
            - rating_distribution: Dict with counts per rating (1-5)
            - queries_with_feedback: Unique query IDs with feedback
            - recommendations_with_feedback: Unique recommendation IDs

    Raises:
        IOError: If file cannot be read
        ValueError: If JSONL data is malformed

    Example:
        >>> stats = get_feedback_stats()
        >>> print(f"Average rating: {stats['average_rating']:.2f}")
        >>> print(f"Helpful rate: {stats['helpful_rate']:.1f}%")
    """
    stats = {
        'total_feedback': 0,
        'average_rating': 0.0,
        'helpful_count': 0,
        'unhelpful_count': 0,
        'helpful_rate': 0.0,
        'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        'queries_with_feedback': set(),
        'recommendations_with_feedback': set(),
    }

    try:
        feedback_path = Path(feedback_file)

        # Return empty stats if file doesn't exist
        if not feedback_path.exists():
            stats['queries_with_feedback'] = []
            stats['recommendations_with_feedback'] = []
            return stats

        # Read and process feedback records
        ratings = []

        with open(feedback_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    record = json.loads(line)

                    # Validate required fields
                    if not all(k in record for k in ['query_id', 'recommendation_id', 'user_rating', 'helpful']):
                        raise ValueError(f"Missing required fields in record at line {line_num}")

                    rating = record['user_rating']
                    helpful = record['helpful']

                    # Validate rating range
                    if not isinstance(rating, int) or not (1 <= rating <= 5):
                        raise ValueError(f"Invalid rating {rating} at line {line_num}")

                    ratings.append(rating)
                    stats['rating_distribution'][rating] += 1
                    stats['total_feedback'] += 1

                    if helpful:
                        stats['helpful_count'] += 1
                    else:
                        stats['unhelpful_count'] += 1

                    stats['queries_with_feedback'].add(record['query_id'])
                    stats['recommendations_with_feedback'].add(record['recommendation_id'])

                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON at line {line_num}: {str(e)}") from e
                except (KeyError, TypeError) as e:
                    raise ValueError(f"Invalid record structure at line {line_num}: {str(e)}") from e

        # Calculate aggregate metrics
        if stats['total_feedback'] > 0:
            stats['average_rating'] = sum(ratings) / len(ratings)
            stats['helpful_rate'] = (stats['helpful_count'] / stats['total_feedback']) * 100

        # Convert sets to lists for JSON serialization
        stats['queries_with_feedback'] = list(stats['queries_with_feedback'])
        stats['recommendations_with_feedback'] = list(stats['recommendations_with_feedback'])

        return stats

    except IOError as e:
        raise IOError(f"Failed to read feedback from {feedback_file}: {str(e)}") from e
    except ValueError as e:
        raise ValueError(f"Error processing feedback data: {str(e)}") from e
    except Exception as e:
        raise IOError(f"Unexpected error generating feedback stats: {str(e)}") from e
