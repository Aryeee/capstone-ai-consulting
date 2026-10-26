"""
Query cache: store embeddings and recommendations to avoid recomputation.

Features:
  1. Cache query embeddings and retrieved results (recommendations)
  2. Track hit/miss statistics for analytics
  3. Invalidate cache when new cases are added to the database
  4. Use SQLite for durable persistence across restarts
"""

import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.schema import Case
from src.predictor import Recommendation


# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
CACHE_DB_PATH = _ROOT / "data" / "query_cache.db"

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Thread-safe SQLite connection
# ---------------------------------------------------------------------------
_cache_lock = threading.RLock()
_cache_conn: Optional[sqlite3.Connection] = None


def get_cache_connection() -> sqlite3.Connection:
    """Get or create a thread-local cache database connection."""
    global _cache_conn
    if _cache_conn is None:
        _cache_conn = sqlite3.connect(str(CACHE_DB_PATH), check_same_thread=False)
        _cache_conn.row_factory = sqlite3.Row
        _init_schema()
    return _cache_conn


def _init_schema():
    """Initialize cache tables if they don't exist."""
    conn = get_cache_connection()
    cursor = conn.cursor()

    # Metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Query embeddings cache
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_embeddings (
            query_hash TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # Recommendations cache
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            query_hash TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            top_k INTEGER NOT NULL,
            recommended_strategy TEXT NOT NULL,
            strategy_confidence REAL NOT NULL,
            predicted_outcome TEXT NOT NULL,
            outcome_confidence REAL NOT NULL,
            estimated_roi_low REAL NOT NULL,
            estimated_roi_high REAL NOT NULL,
            avg_time_to_result_months REAL NOT NULL,
            supporting_rationale TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(query_hash) REFERENCES query_embeddings(query_hash)
        )
    """)

    # Retrieved cases for a recommendation
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS retrieved_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_hash TEXT NOT NULL,
            case_id TEXT NOT NULL,
            similarity_score REAL NOT NULL,
            rank INTEGER NOT NULL,
            FOREIGN KEY(query_hash) REFERENCES recommendations(query_hash)
        )
    """)

    # Cache statistics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_stats (
            stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stat_type TEXT NOT NULL,
            count INTEGER NOT NULL,
            recorded_at TEXT NOT NULL
        )
    """)

    # Initialize metadata
    cursor.execute("INSERT OR IGNORE INTO cache_meta (key, value) VALUES (?, ?)",
                   ("schema_version", str(SCHEMA_VERSION)))

    conn.commit()


# ---------------------------------------------------------------------------
# Query hash utilities
# ---------------------------------------------------------------------------
def _hash_query(query_text: str) -> str:
    """Generate a stable hash for a query string."""
    return hashlib.sha256(query_text.encode()).hexdigest()


def _hash_retrieval_params(query_text: str, top_k: int) -> str:
    """Generate a hash that includes query and top_k params."""
    params = f"{query_text}|{top_k}"
    return hashlib.sha256(params.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------
def cache_query_embedding(query_text: str, embedding: np.ndarray) -> None:
    """Store an embedded query in the cache.

    Parameters
    ----------
    query_text : str
        The query string
    embedding : np.ndarray
        The (1, D) embedding array
    """
    with _cache_lock:
        query_hash = _hash_query(query_text)
        embedding_bytes = embedding.astype(np.float32).tobytes()

        conn = get_cache_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO query_embeddings
            (query_hash, query_text, embedding, created_at)
            VALUES (?, ?, ?, ?)
        """, (query_hash, query_text, embedding_bytes, datetime.utcnow().isoformat()))

        conn.commit()


def get_cached_embedding(query_text: str) -> Optional[np.ndarray]:
    """Retrieve a cached query embedding, or None if not found.

    Parameters
    ----------
    query_text : str
        The query string

    Returns
    -------
    Optional[np.ndarray]
        The (1, D) embedding array, or None if not cached
    """
    with _cache_lock:
        query_hash = _hash_query(query_text)
        conn = get_cache_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT embedding FROM query_embeddings WHERE query_hash = ?
        """, (query_hash,))

        row = cursor.fetchone()
        if row is None:
            _record_stat("embedding_miss", 1)
            return None

        embedding_bytes = row[0]
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(1, -1)
        _record_stat("embedding_hit", 1)
        return embedding


# ---------------------------------------------------------------------------
# Recommendation cache
# ---------------------------------------------------------------------------
def cache_recommendation(query_text: str, top_k: int,
                        recommendation: Recommendation,
                        retrieved_cases: list[tuple[Case, float]]) -> None:
    """Store a recommendation and its retrieved cases in the cache.

    Parameters
    ----------
    query_text : str
        The original query
    top_k : int
        Number of top results retrieved
    recommendation : Recommendation
        The recommendation result
    retrieved_cases : list[tuple[Case, float]]
        List of (Case, similarity_score) tuples
    """
    with _cache_lock:
        query_hash = _hash_retrieval_params(query_text, top_k)
        conn = get_cache_connection()
        cursor = conn.cursor()

        # Insert or replace the recommendation
        cursor.execute("""
            INSERT OR REPLACE INTO recommendations
            (query_hash, query_text, top_k,
             recommended_strategy, strategy_confidence,
             predicted_outcome, outcome_confidence,
             estimated_roi_low, estimated_roi_high,
             avg_time_to_result_months, supporting_rationale,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query_hash, query_text, top_k,
            recommendation.recommended_strategy, recommendation.strategy_confidence,
            recommendation.predicted_outcome, recommendation.outcome_confidence,
            recommendation.estimated_roi_low, recommendation.estimated_roi_high,
            recommendation.avg_time_to_result_months,
            recommendation.supporting_rationale,
            datetime.utcnow().isoformat()
        ))

        # Delete old retrieved cases for this query_hash
        cursor.execute("DELETE FROM retrieved_cases WHERE query_hash = ?", (query_hash,))

        # Insert retrieved cases
        for rank, (case, score) in enumerate(retrieved_cases, start=1):
            cursor.execute("""
                INSERT INTO retrieved_cases
                (query_hash, case_id, similarity_score, rank)
                VALUES (?, ?, ?, ?)
            """, (query_hash, case.case_id, float(score), rank))

        conn.commit()


def get_cached_recommendation(query_text: str, top_k: int) -> Optional[dict]:
    """Retrieve a cached recommendation, or None if not found.

    Returns a dict with keys:
      - recommendation: Recommendation object
      - retrieved_cases: list of (case_id, similarity_score, rank) tuples

    Parameters
    ----------
    query_text : str
        The original query
    top_k : int
        Number of top results retrieved

    Returns
    -------
    Optional[dict]
        Cache entry with recommendation and retrieved cases, or None
    """
    with _cache_lock:
        query_hash = _hash_retrieval_params(query_text, top_k)
        conn = get_cache_connection()
        cursor = conn.cursor()

        # Fetch recommendation
        cursor.execute("""
            SELECT
                recommended_strategy, strategy_confidence,
                predicted_outcome, outcome_confidence,
                estimated_roi_low, estimated_roi_high,
                avg_time_to_result_months, supporting_rationale
            FROM recommendations WHERE query_hash = ?
        """, (query_hash,))

        rec_row = cursor.fetchone()
        if rec_row is None:
            _record_stat("recommendation_miss", 1)
            return None

        recommendation = Recommendation(
            recommended_strategy=rec_row[0],
            strategy_confidence=rec_row[1],
            predicted_outcome=rec_row[2],
            outcome_confidence=rec_row[3],
            estimated_roi_low=rec_row[4],
            estimated_roi_high=rec_row[5],
            avg_time_to_result_months=rec_row[6],
            supporting_rationale=rec_row[7],
        )

        # Fetch retrieved cases
        cursor.execute("""
            SELECT case_id, similarity_score, rank
            FROM retrieved_cases
            WHERE query_hash = ?
            ORDER BY rank ASC
        """, (query_hash,))

        retrieved_cases = [(row[0], row[1]) for row in cursor.fetchall()]

        _record_stat("recommendation_hit", 1)
        return {
            "recommendation": recommendation,
            "retrieved_cases": retrieved_cases,
        }


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------
def invalidate_cache() -> None:
    """Clear the entire cache. Call this when the case database is updated.

    Typically invoked after adding new cases to cases.json and rebuilding
    the FAISS index.
    """
    with _cache_lock:
        conn = get_cache_connection()
        cursor = conn.cursor()

        # Clear all cache tables
        cursor.execute("DELETE FROM retrieved_cases")
        cursor.execute("DELETE FROM recommendations")
        cursor.execute("DELETE FROM query_embeddings")

        conn.commit()
        _record_stat("cache_invalidation", 1)


def invalidate_recommendations_only() -> None:
    """Clear only the recommendations and retrieved cases (not embeddings).

    Use this when you've recomputed recommendations but cached embeddings
    are still valid.
    """
    with _cache_lock:
        conn = get_cache_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM retrieved_cases")
        cursor.execute("DELETE FROM recommendations")

        conn.commit()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _record_stat(stat_type: str, count: int = 1) -> None:
    """Internal: record a cache statistic."""
    with _cache_lock:
        conn = get_cache_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO cache_stats (stat_type, count, recorded_at)
            VALUES (?, ?, ?)
        """, (stat_type, count, datetime.utcnow().isoformat()))

        conn.commit()


def get_cache_stats() -> dict:
    """Get aggregated cache statistics.

    Returns a dict with keys:
      - embedding_hits: int
      - embedding_misses: int
      - recommendation_hits: int
      - recommendation_misses: int
      - cache_invalidations: int
      - hit_rate_embedding: float (0-1)
      - hit_rate_recommendation: float (0-1)
      - total_queries_processed: int
    """
    with _cache_lock:
        conn = get_cache_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT stat_type, SUM(count) as total
            FROM cache_stats
            GROUP BY stat_type
        """)

        stats_by_type = {row[0]: row[1] for row in cursor.fetchall()}

        emb_hits = stats_by_type.get("embedding_hit", 0)
        emb_misses = stats_by_type.get("embedding_miss", 0)
        rec_hits = stats_by_type.get("recommendation_hit", 0)
        rec_misses = stats_by_type.get("recommendation_miss", 0)
        invalidations = stats_by_type.get("cache_invalidation", 0)

        emb_total = emb_hits + emb_misses
        rec_total = rec_hits + rec_misses

        return {
            "embedding_hits": emb_hits,
            "embedding_misses": emb_misses,
            "recommendation_hits": rec_hits,
            "recommendation_misses": rec_misses,
            "cache_invalidations": invalidations,
            "hit_rate_embedding": emb_hits / emb_total if emb_total > 0 else 0.0,
            "hit_rate_recommendation": rec_hits / rec_total if rec_total > 0 else 0.0,
            "total_queries_processed": emb_total,
        }


def reset_cache_stats() -> None:
    """Clear all statistics. Useful for benchmarking or testing."""
    with _cache_lock:
        conn = get_cache_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cache_stats")
        conn.commit()


# ---------------------------------------------------------------------------
# Utility: close connection (for testing/cleanup)
# ---------------------------------------------------------------------------
def close_cache():
    """Close the cache database connection. Primarily for testing."""
    global _cache_conn
    with _cache_lock:
        if _cache_conn is not None:
            _cache_conn.close()
            _cache_conn = None


if __name__ == "__main__":
    # Simple demo: initialize schema and print stats
    _init_schema()
    print("Cache database initialized at:", CACHE_DB_PATH)
    print("Current stats:", get_cache_stats())
