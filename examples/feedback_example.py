"""
Example usage of the feedback_collector module.

This demonstrates how to collect user feedback on recommendations
and analyze feedback statistics.
"""

from src.feedback_collector import QueryFeedback, store_feedback, get_feedback_stats


def main():
    """Example: Collect feedback and generate stats."""

    # Example 1: Create and store a single feedback record
    print("=" * 60)
    print("Example 1: Store a single feedback record")
    print("=" * 60)

    feedback = QueryFeedback(
        query_id="Q001",
        recommendation_id="REC-digital-001",
        user_rating=5,
        helpful=True,
        notes="This recommendation was exactly what we needed!"
    )

    store_feedback(feedback)
    print(f"Stored feedback: {feedback}\n")

    # Example 2: Store multiple feedback records
    print("=" * 60)
    print("Example 2: Store multiple feedback records")
    print("=" * 60)

    feedback_list = [
        QueryFeedback(
            query_id="Q002",
            recommendation_id="REC-ai-002",
            user_rating=4,
            helpful=True,
            notes="Good insights, but needed more detail on implementation"
        ),
        QueryFeedback(
            query_id="Q003",
            recommendation_id="REC-cloud-003",
            user_rating=3,
            helpful=False,
            notes="Not applicable to our specific use case"
        ),
        QueryFeedback(
            query_id="Q004",
            recommendation_id="REC-ai-002",
            user_rating=5,
            helpful=True,
            notes="Excellent strategy for our situation"
        ),
    ]

    for fb in feedback_list:
        store_feedback(fb)
        print(f"Stored: Query {fb.query_id} -> Rating {fb.user_rating}/5")

    print()

    # Example 3: Generate feedback statistics
    print("=" * 60)
    print("Example 3: Generate feedback statistics")
    print("=" * 60)

    stats = get_feedback_stats()

    print(f"Total feedback records: {stats['total_feedback']}")
    print(f"Average rating: {stats['average_rating']:.2f}/5.0")
    print(f"Helpful recommendations: {stats['helpful_count']}")
    print(f"Unhelpful recommendations: {stats['unhelpful_count']}")
    print(f"Helpfulness rate: {stats['helpful_rate']:.1f}%")
    print()

    print("Rating distribution:")
    for rating, count in sorted(stats['rating_distribution'].items()):
        print(f"  {rating} stars: {count}")

    print()
    print(f"Unique queries with feedback: {len(stats['queries_with_feedback'])}")
    print(f"Unique recommendations: {len(stats['recommendations_with_feedback'])}")
    print()

    print("Queries with feedback:", stats['queries_with_feedback'])
    print("Recommendations:", stats['recommendations_with_feedback'])


if __name__ == "__main__":
    main()
