"""
Example usage of the confidence_calibration module.

Demonstrates:
1. Training calibration model from feedback data
2. Applying calibration to raw confidence scores
3. Analyzing calibration metrics
"""

import numpy as np
import json
from pathlib import Path

from src.confidence_calibration import (
    train_calibration,
    get_calibrated_confidence,
    get_calibrated_batch,
    compute_calibration_metrics,
    _load_feedback_data,
    _extract_confidence_labels,
)


def generate_sample_feedback(output_file: str = "data/feedback_sample.jsonl", num_records: int = 100):
    """
    Generate sample feedback data for demonstration.

    Creates synthetic feedback with varying confidence and correctness levels.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    records = []

    for i in range(num_records):
        # Generate confidence score (0-1)
        confidence = np.random.uniform(0.0, 1.0)

        # Correctness is correlated with confidence but with some noise
        noise = np.random.normal(0, 0.2)
        correctness_prob = np.clip(confidence + noise, 0, 1)
        is_helpful = np.random.random() < correctness_prob

        # Convert to user_rating (1-5 scale)
        if is_helpful:
            user_rating = np.random.randint(4, 6)  # 4 or 5
        else:
            user_rating = np.random.randint(1, 4)  # 1, 2, or 3

        record = {
            "query_id": f"Q{i:04d}",
            "recommendation_id": f"R{i:04d}",
            "user_rating": user_rating,
            "helpful": is_helpful,
            "confidence": float(confidence),
            "notes": "Sample feedback for calibration training",
        }
        records.append(record)

    # Write to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"Generated {num_records} feedback records in {output_file}")
    return output_file


def example_1_train_calibration():
    """Example 1: Train calibration model from feedback."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Train Calibration Model from Feedback")
    print("="*70)

    feedback_file = "data/feedback_sample.jsonl"

    # Generate sample feedback if it doesn't exist
    if not Path(feedback_file).exists():
        generate_sample_feedback(feedback_file)

    print(f"\nTraining calibration model from {feedback_file}...")

    try:
        iso_reg, metrics = train_calibration(feedback_file)

        print(f"\nCalibration Training Complete!")
        print(f"  Samples used: {metrics.num_samples}")
        print(f"  Expected Calibration Error (ECE): {metrics.expected_calibration_error:.4f}")
        print(f"  Max Calibration Error: {metrics.max_calibration_error:.4f}")
        print(f"  Brier Score: {metrics.brier_score:.4f}")

    except ValueError as e:
        print(f"Error: {e}")


def example_2_apply_single_calibration():
    """Example 2: Apply calibration to a single confidence score."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Apply Calibration to Single Prediction")
    print("="*70)

    # Example raw confidence scores from a recommendation
    raw_confidences = [0.3, 0.5, 0.7, 0.9]

    print(f"\nRaw confidence scores: {raw_confidences}")
    print("\nCalibrated confidences:")

    for raw_conf in raw_confidences:
        calibrated_conf = get_calibrated_confidence(raw_conf)
        print(f"  {raw_conf:.1f} → {calibrated_conf:.4f}")


def example_3_apply_batch_calibration():
    """Example 3: Apply calibration to multiple confidence scores."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Apply Calibration to Batch of Predictions")
    print("="*70)

    # Example: 10 recommendations with varying confidence
    raw_confidences = np.random.uniform(0.2, 0.95, 10)

    print(f"\nApplying calibration to {len(raw_confidences)} predictions...")

    calibrated_confidences = get_calibrated_batch(raw_confidences)

    print(f"\nResults:")
    print(f"{'Raw':<8} {'Calibrated':<12} {'Change':<10}")
    print("-" * 30)

    for raw, cal in zip(raw_confidences, calibrated_confidences):
        change = cal - raw
        direction = "↑" if change > 0.01 else "↓" if change < -0.01 else "→"
        print(f"{raw:.3f}    {cal:.4f}        {direction} {abs(change):+.4f}")


def example_4_analyze_calibration_curve():
    """Example 4: Analyze and visualize calibration curve."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Analyze Calibration Curve")
    print("="*70)

    feedback_file = "data/feedback_sample.jsonl"

    if not Path(feedback_file).exists():
        print(f"Generating sample feedback...")
        generate_sample_feedback(feedback_file)

    # Load and extract data
    feedback_records = _load_feedback_data(feedback_file)
    raw_confidences, correctness_labels = _extract_confidence_labels(feedback_records)

    # Compute calibration curve
    calibration_curve, metrics = compute_calibration_metrics(
        raw_confidences, correctness_labels, num_bins=5
    )

    print(f"\nCalibration Curve Analysis (5 bins):")
    print(f"{'Bin':<6} {'Conf Range':<15} {'Avg Conf':<12} {'Accuracy':<12} {'Samples':<8}")
    print("-" * 55)

    bin_edges = calibration_curve["bin_edges"]
    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        avg_conf = calibration_curve["bin_confidences"][i]
        accuracy = calibration_curve["bin_accuracies"][i]
        count = calibration_curve["bin_counts"][i]

        if avg_conf is not None and accuracy is not None:
            print(f"{i+1:<6} [{lower:.2f}, {upper:.2f}]     {avg_conf:.4f}       "
                  f"{accuracy:.4f}       {count:<8}")
        else:
            print(f"{i+1:<6} [{lower:.2f}, {upper:.2f}]     -              "
                  f"-              {count:<8}")


def example_5_integration_with_recommender():
    """Example 5: Integration pattern for recommender system."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Integration Pattern for Recommender System")
    print("="*70)

    print("\nIntegration workflow:")
    print("""
    1. DURING TRAINING (periodic, e.g., nightly):
       - Collect user feedback from database
       - Call: iso_reg, metrics = train_calibration()
       - Save model for inference

    2. DURING INFERENCE (per-request):
       - Generate recommendation with raw confidence
       - Call: calibrated_conf = get_calibrated_confidence(raw_conf)
       - Return recommendation with calibrated confidence

    3. EXAMPLE CODE:

       from src.predictor import predict
       from src.confidence_calibration import get_calibrated_confidence

       # Generate recommendation
       recommendation = predict(retrieved_cases)

       # Calibrate confidence
       calibrated_strategy_conf = get_calibrated_confidence(
           recommendation.strategy_confidence
       )
       calibrated_outcome_conf = get_calibrated_confidence(
           recommendation.outcome_confidence
       )

       # Return recommendation with calibrated confidences
       recommendation.strategy_confidence = calibrated_strategy_conf
       recommendation.outcome_confidence = calibrated_outcome_conf

       return recommendation
    """)

    print("\nBENEFITS:")
    print("  ✓ User confidence in recommendations matches actual accuracy")
    print("  ✓ Better decision-making with well-calibrated uncertainty")
    print("  ✓ Identifies when model is over/under-confident")
    print("  ✓ Monotonic calibration preserves ranking quality")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONFIDENCE CALIBRATION MODULE - USAGE EXAMPLES")
    print("="*70)

    # Run examples
    example_1_train_calibration()
    example_2_apply_single_calibration()
    example_3_apply_batch_calibration()
    example_4_analyze_calibration_curve()
    example_5_integration_with_recommender()

    print("\n" + "="*70)
    print("Examples Complete!")
    print("="*70)
