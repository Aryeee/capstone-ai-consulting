"""
Unit tests for confidence_calibration module.

Tests calibration curve computation, isotonic regression training,
and confidence score calibration with both real and synthetic feedback.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path

from src.confidence_calibration import (
    _load_feedback_data,
    _extract_confidence_labels,
    compute_calibration_metrics,
    train_isotonic_regressor,
    calibrate_confidence,
    calibrate_batch,
    train_calibration,
    get_calibrated_confidence,
    get_calibrated_batch,
    save_calibration_model,
    load_calibration_model,
    CalibrationMetrics,
)


class TestLoadFeedbackData:
    """Tests for loading feedback from JSONL files."""

    def test_load_empty_file(self):
        """Should return empty list if file does not exist."""
        result = _load_feedback_data("nonexistent_file.jsonl")
        assert result == []

    def test_load_valid_feedback(self, tmp_path):
        """Should load valid feedback records."""
        feedback_file = tmp_path / "feedback.jsonl"
        records = [
            {"query_id": "Q1", "recommendation_id": "R1", "user_rating": 5, "helpful": True},
            {"query_id": "Q2", "recommendation_id": "R2", "user_rating": 3, "helpful": False},
        ]
        with open(feedback_file, "w") as f:
            for record in records:
                json.dump(record, f)
                f.write("\n")

        result = _load_feedback_data(str(feedback_file))
        assert len(result) == 2
        assert result[0]["query_id"] == "Q1"
        assert result[1]["user_rating"] == 3

    def test_load_malformed_json(self, tmp_path):
        """Should raise ValueError for malformed JSON."""
        feedback_file = tmp_path / "feedback.jsonl"
        with open(feedback_file, "w") as f:
            f.write("{invalid json\n")

        with pytest.raises(ValueError, match="Malformed JSON"):
            _load_feedback_data(str(feedback_file))

    def test_load_missing_fields(self, tmp_path):
        """Should raise ValueError for missing required fields."""
        feedback_file = tmp_path / "feedback.jsonl"
        with open(feedback_file, "w") as f:
            json.dump({"query_id": "Q1"}, f)  # Missing other required fields
            f.write("\n")

        with pytest.raises(ValueError, match="Missing required fields"):
            _load_feedback_data(str(feedback_file))

    def test_load_skips_empty_lines(self, tmp_path):
        """Should skip empty lines in JSONL file."""
        feedback_file = tmp_path / "feedback.jsonl"
        with open(feedback_file, "w") as f:
            json.dump(
                {"query_id": "Q1", "recommendation_id": "R1", "user_rating": 5, "helpful": True},
                f,
            )
            f.write("\n\n\n")
            json.dump(
                {"query_id": "Q2", "recommendation_id": "R2", "user_rating": 3, "helpful": False},
                f,
            )
            f.write("\n")

        result = _load_feedback_data(str(feedback_file))
        assert len(result) == 2


class TestExtractConfidenceLabels:
    """Tests for extracting confidence-correctness pairs from feedback."""

    def test_extract_with_explicit_confidence(self):
        """Should use explicit confidence if available."""
        feedback = [
            {
                "query_id": "Q1",
                "recommendation_id": "R1",
                "user_rating": 3,
                "helpful": False,
                "confidence": 0.7,
            },
        ]
        confidences, labels = _extract_confidence_labels(feedback)
        assert len(confidences) == 1
        assert confidences[0] == 0.7
        assert labels[0] == 0

    def test_extract_confidence_from_rating(self):
        """Should derive confidence from user_rating if not explicit."""
        feedback = [
            {"query_id": "Q1", "recommendation_id": "R1", "user_rating": 1, "helpful": False},
            {"query_id": "Q2", "recommendation_id": "R2", "user_rating": 3, "helpful": False},
            {"query_id": "Q3", "recommendation_id": "R3", "user_rating": 5, "helpful": True},
        ]
        confidences, labels = _extract_confidence_labels(feedback)

        # Rating 1 -> (1-1)/4 = 0.0
        assert np.isclose(confidences[0], 0.0)
        # Rating 3 -> (3-1)/4 = 0.5
        assert np.isclose(confidences[1], 0.5)
        # Rating 5 -> (5-1)/4 = 1.0
        assert np.isclose(confidences[2], 1.0)

    def test_extract_correctness_labels(self):
        """Should mark as correct if helpful=True or rating >= 4."""
        feedback = [
            {"query_id": "Q1", "recommendation_id": "R1", "user_rating": 5, "helpful": False},  # Correct (rating >= 4)
            {"query_id": "Q2", "recommendation_id": "R2", "user_rating": 3, "helpful": True},  # Correct (helpful)
            {"query_id": "Q3", "recommendation_id": "R3", "user_rating": 2, "helpful": False},  # Incorrect
            {"query_id": "Q4", "recommendation_id": "R4", "user_rating": 4, "helpful": False},  # Correct (rating >= 4)
        ]
        confidences, labels = _extract_confidence_labels(feedback)

        assert labels[0] == 1
        assert labels[1] == 1
        assert labels[2] == 0
        assert labels[3] == 1

    def test_extract_empty_feedback(self):
        """Should handle empty feedback list."""
        feedback = []
        confidences, labels = _extract_confidence_labels(feedback)
        assert len(confidences) == 0
        assert len(labels) == 0

    def test_extract_malformed_record(self):
        """Should raise ValueError for malformed record."""
        feedback = [{"query_id": "Q1"}]  # Missing user_rating
        with pytest.raises(ValueError):
            _extract_confidence_labels(feedback)


class TestComputeCalibrationMetrics:
    """Tests for computing calibration curves and metrics."""

    def test_perfect_calibration(self):
        """Should have low ECE for well-calibrated predictions."""
        raw_confidences = np.array([0.0, 0.0, 1.0, 1.0])
        correctness_labels = np.array([0, 0, 1, 1])  # Perfect match

        calibration_curve, metrics = compute_calibration_metrics(
            raw_confidences, correctness_labels, num_bins=2
        )

        assert metrics.expected_calibration_error == 0.0
        assert metrics.brier_score == 0.0

    def test_perfectly_miscalibrated(self):
        """Should have high ECE for inversely calibrated predictions."""
        raw_confidences = np.array([0.1, 0.2, 0.8, 0.9])
        correctness_labels = np.array([1, 1, 0, 0])  # Opposite of confidence

        calibration_curve, metrics = compute_calibration_metrics(
            raw_confidences, correctness_labels, num_bins=2
        )

        assert metrics.expected_calibration_error > 0.7
        assert metrics.brier_score > 0.5

    def test_calibration_metrics_structure(self):
        """Should return properly structured calibration metrics."""
        raw_confidences = np.array([0.3, 0.5, 0.7])
        correctness_labels = np.array([0, 1, 1])

        calibration_curve, metrics = compute_calibration_metrics(
            raw_confidences, correctness_labels, num_bins=3
        )

        assert isinstance(metrics, CalibrationMetrics)
        assert 0.0 <= metrics.expected_calibration_error <= 1.0
        assert 0.0 <= metrics.max_calibration_error <= 1.0
        assert 0.0 <= metrics.brier_score <= 1.0
        assert metrics.num_samples == 3

    def test_calibration_curve_structure(self):
        """Should return properly structured calibration curve."""
        raw_confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        correctness_labels = np.array([0, 0, 1, 1, 1])

        calibration_curve, _ = compute_calibration_metrics(
            raw_confidences, correctness_labels, num_bins=5
        )

        assert "bin_edges" in calibration_curve
        assert "bin_confidences" in calibration_curve
        assert "bin_accuracies" in calibration_curve
        assert "bin_counts" in calibration_curve

        assert len(calibration_curve["bin_edges"]) == 6  # num_bins + 1
        assert len(calibration_curve["bin_confidences"]) == 5
        assert len(calibration_curve["bin_accuracies"]) == 5
        assert len(calibration_curve["bin_counts"]) == 5

    def test_empty_input_raises_error(self):
        """Should raise ValueError for empty input."""
        raw_confidences = np.array([])
        correctness_labels = np.array([])

        with pytest.raises(ValueError, match="empty"):
            compute_calibration_metrics(raw_confidences, correctness_labels)

    def test_mismatched_lengths_raises_error(self):
        """Should raise ValueError for mismatched array lengths."""
        raw_confidences = np.array([0.1, 0.2])
        correctness_labels = np.array([0, 1, 1])

        with pytest.raises(ValueError, match="same length"):
            compute_calibration_metrics(raw_confidences, correctness_labels)


class TestIsotonicRegression:
    """Tests for isotonic regression training and application."""

    def test_train_isotonic_regressor(self):
        """Should train isotonic regressor successfully."""
        raw_confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        correctness_labels = np.array([0, 0, 1, 1, 1])

        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)

        assert iso_reg is not None
        assert hasattr(iso_reg, "predict")

    def test_isotonic_preserves_order(self):
        """Should preserve monotonicity: higher input -> higher/equal output."""
        raw_confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        correctness_labels = np.array([0, 0, 1, 1, 1])

        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)
        calibrated = iso_reg.predict(raw_confidences.reshape(-1, 1))

        # Check monotonicity
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1]

    def test_isotonic_bounds_output(self):
        """Should bound output to [0, 1]."""
        raw_confidences = np.array([0.1, 0.5, 0.9])
        correctness_labels = np.array([0, 1, 1])

        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)
        calibrated = iso_reg.predict(raw_confidences.reshape(-1, 1))

        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_isotonic_empty_input_raises_error(self):
        """Should raise ValueError for empty input."""
        raw_confidences = np.array([])
        correctness_labels = np.array([])

        with pytest.raises(ValueError, match="empty"):
            train_isotonic_regressor(raw_confidences, correctness_labels)

    def test_isotonic_mismatched_lengths_raises_error(self):
        """Should raise ValueError for mismatched array lengths."""
        raw_confidences = np.array([0.1, 0.2])
        correctness_labels = np.array([0, 1, 1])

        with pytest.raises(ValueError, match="same length"):
            train_isotonic_regressor(raw_confidences, correctness_labels)


class TestCalibrateConfidence:
    """Tests for applying calibration to single and batch predictions."""

    def test_calibrate_single_confidence(self):
        """Should calibrate a single confidence score."""
        raw_confidences = np.array([0.1, 0.5, 0.9])
        correctness_labels = np.array([0, 1, 1])

        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)
        calibrated = calibrate_confidence(0.5, iso_reg)

        assert 0.0 <= calibrated <= 1.0
        assert isinstance(calibrated, float)

    def test_calibrate_single_invalid_range(self):
        """Should raise ValueError for confidence outside [0, 1]."""
        raw_confidences = np.array([0.1, 0.5, 0.9])
        correctness_labels = np.array([0, 1, 1])
        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            calibrate_confidence(-0.1, iso_reg)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            calibrate_confidence(1.5, iso_reg)

    def test_calibrate_batch(self):
        """Should calibrate multiple confidence scores."""
        raw_confidences = np.array([0.1, 0.5, 0.9])
        correctness_labels = np.array([0, 1, 1])

        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)
        calibrated = calibrate_batch(raw_confidences, iso_reg)

        assert len(calibrated) == len(raw_confidences)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)
        assert calibrated.dtype == np.float32

    def test_calibrate_batch_invalid_range(self):
        """Should raise ValueError for any confidence outside [0, 1]."""
        raw_confidences = np.array([0.1, 0.5, 0.9])
        correctness_labels = np.array([0, 1, 1])
        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)

        invalid_batch = np.array([-0.1, 0.5, 0.9])

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            calibrate_batch(invalid_batch, iso_reg)


class TestCalibrationPersistence:
    """Tests for saving and loading calibration models."""

    def test_save_and_load_model(self, tmp_path):
        """Should save and load calibration model successfully."""
        raw_confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        correctness_labels = np.array([0, 0, 1, 1, 1])

        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)
        model_path = tmp_path / "calibration_model.pkl"

        save_calibration_model(iso_reg, model_path)
        loaded_model = load_calibration_model(model_path)

        assert loaded_model is not None

        # Verify both models produce same output
        original_output = iso_reg.predict(raw_confidences.reshape(-1, 1))
        loaded_output = loaded_model.predict(raw_confidences.reshape(-1, 1))

        assert np.allclose(original_output, loaded_output)

    def test_load_nonexistent_model(self, tmp_path):
        """Should return None if model file does not exist."""
        model_path = tmp_path / "nonexistent_model.pkl"
        loaded_model = load_calibration_model(model_path)
        assert loaded_model is None


class TestEndToEndCalibration:
    """End-to-end tests for calibration training and inference."""

    def test_train_calibration_from_feedback(self, tmp_path):
        """Should train calibration model from feedback data."""
        feedback_file = tmp_path / "feedback.jsonl"

        # Create synthetic feedback
        feedback_records = [
            {"query_id": f"Q{i}", "recommendation_id": f"R{i}", "user_rating": 5, "helpful": True}
            for i in range(20)
        ] + [
            {"query_id": f"Q{i}", "recommendation_id": f"R{i}", "user_rating": 1, "helpful": False}
            for i in range(20, 40)
        ]

        with open(feedback_file, "w") as f:
            for record in feedback_records:
                json.dump(record, f)
                f.write("\n")

        model_path = tmp_path / "calibration_model.pkl"
        iso_reg, metrics = train_calibration(str(feedback_file), model_path)

        assert iso_reg is not None
        assert isinstance(metrics, CalibrationMetrics)
        assert metrics.num_samples == 40

    def test_get_calibrated_confidence_no_model(self):
        """Should return raw confidence if no model exists."""
        raw_conf = 0.7
        calibrated = get_calibrated_confidence(raw_conf)
        assert calibrated == 0.7

    def test_get_calibrated_batch_no_model(self):
        """Should return raw confidences if no model exists."""
        raw_confs = np.array([0.3, 0.5, 0.7])
        calibrated = get_calibrated_batch(raw_confs)
        assert np.allclose(calibrated, raw_confs)

    def test_get_calibrated_confidence_with_model(self, tmp_path):
        """Should load and apply calibration model."""
        # Train and save model
        raw_confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        correctness_labels = np.array([0, 0, 1, 1, 1])

        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)
        model_path = tmp_path / "calibration_model.pkl"
        save_calibration_model(iso_reg, model_path)

        # Apply calibration
        calibrated = get_calibrated_confidence(0.5, model_path=model_path)

        assert isinstance(calibrated, float)
        assert 0.0 <= calibrated <= 1.0

    def test_calibration_improves_over_baseline(self, tmp_path):
        """Should reduce calibration error compared to raw predictions."""
        # Create feedback with systematic miscalibration
        raw_confidences = np.array([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
        # Labels are inverted for low confidence, normal for high confidence
        correctness_labels = np.array([1, 1, 1, 0, 0, 0])

        _, metrics_before = compute_calibration_metrics(raw_confidences, correctness_labels)

        # Train calibration
        iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)
        calibrated = calibrate_batch(raw_confidences, iso_reg)

        # Compute metrics after calibration
        _, metrics_after = compute_calibration_metrics(calibrated, correctness_labels)

        # Calibration should improve ECE
        assert metrics_after.expected_calibration_error <= metrics_before.expected_calibration_error
