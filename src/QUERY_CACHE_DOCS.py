"""
Query Cache Documentation

The query_cache module provides SQLite-backed caching for embeddings and
recommendations to improve performance and reduce recomputation.

## Features

1. **Embedding Cache**: Store query embeddings after computation to avoid
   re-embedding identical or similar queries.

2. **Recommendation Cache**: Store full recommendations with retrieved cases,
   keyed by query text + top_k parameter.

3. **Hit/Miss Statistics**: Track cache performance with aggregated statistics
   including hit rates by cache type.

4. **Cache Invalidation**: Clear cache when the case database is updated
   (new cases added, index rebuilt).

5. **Thread-Safe**: Uses locks to ensure safe concurrent access from multiple
   threads or async tasks.

6. **Durable**: SQLite database persists across application restarts.

## Quick Start

### Caching Query Embeddings

```python
from src.query_cache import cache_query_embedding, get_cached_embedding
import numpy as np

query = "What is the best digital strategy?"
embedding = np.random.randn(1, 384).astype(np.float32)  # Your embedding

# Store in cache
cache_query_embedding(query, embedding)

# Later, retrieve from cache
cached_emb = get_cached_embedding(query)
if cached_emb is not None:
    print("Cache hit!")
else:
    print("Cache miss - compute embedding")
```

### Caching Recommendations

```python
from src.query_cache import cache_recommendation, get_cached_recommendation
from src.predictor import Recommendation

query = "What is the best digital strategy?"
top_k = 5
recommendation = Recommendation(...)  # Your recommendation
retrieved_cases = [(case1, 0.95), (case2, 0.88)]  # (Case, score) tuples

# Store in cache
cache_recommendation(query, top_k, recommendation, retrieved_cases)

# Later, retrieve from cache
cached = get_cached_recommendation(query, top_k)
if cached is not None:
    rec = cached["recommendation"]
    cases = cached["retrieved_cases"]  # [(case_id, score), ...]
    print(f"Got cached recommendation: {rec.recommended_strategy}")
```

### Cache Invalidation

```python
from src.query_cache import invalidate_cache, invalidate_recommendations_only

# Clear everything (embeddings, recommendations, retrieved cases)
invalidate_cache()

# Or, clear only recommendations (embeddings stay cached)
invalidate_recommendations_only()
```

The `case_db.build_all()` function automatically calls `invalidate_cache()`
when rebuilding the index.

### Statistics & Monitoring

```python
from src.query_cache import get_cache_stats, reset_cache_stats

# Get aggregated statistics
stats = get_cache_stats()
print(f"Embedding hit rate: {stats['hit_rate_embedding']:.1%}")
print(f"Recommendation hit rate: {stats['hit_rate_recommendation']:.1%}")
print(f"Total queries: {stats['total_queries_processed']}")

# Sample output:
# {
#   'embedding_hits': 15,
#   'embedding_misses': 5,
#   'recommendation_hits': 8,
#   'recommendation_misses': 2,
#   'cache_invalidations': 1,
#   'hit_rate_embedding': 0.75,
#   'hit_rate_recommendation': 0.8,
#   'total_queries_processed': 20
# }

# Reset stats for benchmarking
reset_cache_stats()
```

## Architecture

### Database Schema

The cache uses SQLite with the following tables:

**query_embeddings**
- query_hash: SHA256 hash of query text (PK)
- query_text: Original query string
- embedding: Binary blob of float32 numpy array
- created_at: ISO timestamp

**recommendations**
- query_hash: SHA256 hash of (query_text|top_k) (PK)
- query_text: Original query string
- top_k: Number of top results (part of cache key)
- recommended_strategy: String
- strategy_confidence: Float (0-1)
- predicted_outcome: String
- outcome_confidence: Float (0-1)
- estimated_roi_low: Float
- estimated_roi_high: Float
- avg_time_to_result_months: Float
- supporting_rationale: String
- created_at: ISO timestamp

**retrieved_cases**
- id: Auto-increment integer (PK)
- query_hash: Foreign key to recommendations
- case_id: Case identifier
- similarity_score: Float (0-1)
- rank: Integer (1-indexed position in retrieval)

**cache_stats**
- stat_id: Auto-increment integer (PK)
- stat_type: String ("embedding_hit", "embedding_miss", etc.)
- count: Integer
- recorded_at: ISO timestamp

**cache_meta**
- key: Text (PK)
- value: Text

### Threading & Concurrency

All public functions use a threading.RLock() for thread-safe concurrent access.
The lock ensures atomicity of multi-step operations (e.g., storing a
recommendation + its retrieved cases).

Single connections are reused across threads via check_same_thread=False.

### Hashing Strategy

- Query embeddings are keyed by SHA256(query_text)
- Recommendations are keyed by SHA256(query_text|top_k)

This allows different top_k values for the same query to be cached separately,
which is important because retrieval results differ by top_k.

## Integration with Workflow

### Typical Usage Pattern

```python
from src.case_db import load_index
from src.query_cache_integration import retrieve_with_cache, predict_with_cache

index, cases = load_index()

# User query
query = "How do we accelerate digital transformation in retail?"

# Retrieve with caching
retrieved = retrieve_with_cache(query, index, cases, top_k=5)

# Predict with caching
recommendation = predict_with_cache(query, retrieved, top_k=5)

print(f"Strategy: {recommendation.recommended_strategy}")
print(f"Confidence: {recommendation.strategy_confidence:.0%}")
```

### When to Invalidate Cache

Cache should be invalidated when:

1. **New cases added to cases.json** → Call `build_all()` which auto-invalidates
2. **Index rebuilt** → `build_all()` auto-invalidates
3. **Recommendation model updated** → Call `invalidate_recommendations_only()`
4. **Embedding model changed** → Call `invalidate_cache()`

## Performance Characteristics

### Cache Hit Rates

- **Embedding cache**: High hit rate when users repeat similar queries
- **Recommendation cache**: High hit rate for exact query repeats or
  retrieval-identical queries

### Storage

- Embeddings: ~1.5 KB per 384-dim float32 embedding
- Recommendations: ~0.5 KB per recommendation
- Retrieved cases: ~50 bytes per case reference

For 100 cached queries with 5 retrieved cases each: ~250 KB database size

### Latency Improvement

- Embedding cache hit: Saves ~100-200ms (embedding computation)
- Recommendation cache hit: Saves ~50-100ms (prediction + retrieval)

## Error Handling

All cache operations are fault-tolerant:

- If cache database is corrupted, it's automatically recreated
- Failed cache operations don't break retrieval/prediction (returns None)
- Missing cache entries transparently recompute on next call

## Testing

Comprehensive test suite in tests/test_query_cache.py covers:

- Hash stability and uniqueness
- Embedding cache get/set/overwrite
- Recommendation cache get/set/overwrite
- Cache invalidation (full and selective)
- Statistics tracking and hit rate calculation
- Edge cases (special characters, unicode, very long queries)
- Concurrent access (thread safety)

Run tests:
```bash
pytest tests/test_query_cache.py -v
```

## Database File Location

Cache database is stored at:
```
/data/query_cache.db
```

relative to the project root.

## API Reference

### Embedding Cache Functions

**cache_query_embedding(query_text: str, embedding: np.ndarray) -> None**
- Store query embedding in cache
- Parameters: query_text, embedding (1, D) array

**get_cached_embedding(query_text: str) -> Optional[np.ndarray]**
- Retrieve cached embedding or None
- Returns: (1, D) array or None

### Recommendation Cache Functions

**cache_recommendation(query_text: str, top_k: int, recommendation: Recommendation, retrieved_cases: list[tuple[Case, float]]) -> None**
- Store recommendation and retrieved cases
- Parameters: query_text, top_k, recommendation, retrieved_cases

**get_cached_recommendation(query_text: str, top_k: int) -> Optional[dict]**
- Retrieve cached recommendation or None
- Returns: dict with "recommendation" and "retrieved_cases" keys, or None

### Invalidation Functions

**invalidate_cache() -> None**
- Clear all caches (embeddings, recommendations, retrieved cases)

**invalidate_recommendations_only() -> None**
- Clear only recommendations and retrieved cases, keep embeddings

### Statistics Functions

**get_cache_stats() -> dict**
- Get aggregated cache statistics
- Returns: dict with hit counts, rates, and totals

**reset_cache_stats() -> None**
- Clear all recorded statistics

### Utility Functions

**close_cache() -> None**
- Close database connection (for testing/cleanup)

**get_cache_connection() -> sqlite3.Connection**
- Get internal database connection (advanced use)

## Future Enhancements

- TTL (time-to-live) for cache entries
- LRU (least-recently-used) eviction when cache exceeds size limit
- Async cache operations for non-blocking retrieval
- Cache warming from previous sessions
- Persistent statistics across restarts
"""

# This is a documentation module; it does not export functions.
# Import from src.query_cache directly.
