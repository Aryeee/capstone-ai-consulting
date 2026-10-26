"""
Confidence calibration module for AI Strategy Consulting Assistant.

Computes calibration curves for confidence vs actual correctness,
applies isotonic regression to map predicted confidence to calibrated probability,
and returns calibrated confidence scores for recommendations.

Uses historical feedback data to improve confidence estimates over time.
"""

from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path
import json
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import pickle


CALIBRATION_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration_model.pkl"


@dataclass
class CalibrationMetrics:
    """Metrics for calibration quality assessment."""
    expected_calibration_error: float  # Average absolute gap between confidence and accuracy
    max_calibration_error: float       # Maximum gap
    brier_score: float                 # Mean squared error of predictions
    num_samples: int                   # Number of samples used


@dataclass
class CalibratedPrediction:
    """A prediction with calibrated confidence."""
    raw_confidence: float              # Original confidence from model
    calibrated_confidence: float       # Calibrated confidence after isotonic regression
    prediction_id: str                 # Unique identifier for tracking


def _load_feedback_data(feedback_file: str = "data/feedback.jsonl") -> list[dict]:
    """
    Load feedback records from JSONL file.

    Parameters
    ----------
    feedback_file : str
        Path to the feedback JSONL file.

    Returns
    -------
    list[dict]
        List of feedback records with keys: query_id, recommendation_id, user_rating,
        helpful, notes, timestamp.

    Raises
    ------
    IOError
        If file cannot be read.
    ValueError
        If JSONL data is malformed.
    """
    feedback_path = Path(feedback_file)

    if not feedback_path.exists():
        return []

    feedback_records = []

    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    record = json.loads(line)

                    # Validate required fields
                    if not all(
                        k in record
                        for k in ["query_id", "recommendation_id", "user_rating", "helpful"]
                    ):
                        raise ValueError(
                            f"Missing required fields in record at line {line_num}"
                        )

                    feedback_records.append(record)

                except json.JSONDecodeError as e:
                    raise ValueError(f"Malformed JSON at line {line_num}: {str(e)}") from e
                except (KeyError, TypeError) as e:
                    raise ValueError(
                        f"Invalid record structure at line {line_num}: {str(e)}"
                    ) from e

        return feedback_records

    except IOError as e:
        raise IOError(f"Failed to read feedback from {feedback_file}: {str(e)}") from e
    except ValueError as e:
        raise ValueError(f"Error processing feedback data: {str(e)}") from e
    except Exception as e:
        raise IOError(f"Unexpected error loading feedback: {str(e)}") from e


def _extract_confidence_labels(
    feedback_records: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract confidence scores and binary correctness labels from feedback.

    A recommendation is considered correct if:
    - user_rating >= 4 (on 1-5 scale), OR
    - helpful == True

    Parameters
    ----------
    feedback_records : list[dict]
        List of feedback records.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (confidence_scores, correctness_labels) where correctness_labels are 0/1.

    Raises
    ------
    ValueError
        If feedback records lack required fields.
    """
    confidences = []
    labels = []

    for record in feedback_records:
        try:
            # Extract confidence from metadata or estimate from rating
            # Priority: explicit confidence > derived from rating
            if "confidence" in record:
                confidence = float(record["confidence"])
            else:
                # Derive confidence from user_rating (1-5 scale)
                # Map to 0-1 range: 1->0.2, 5->1.0
                rating = float(record["user_rating"])
                confidence = min(1.0, max(0.0, (rating - 1) / 4.0))

            # Determine correctness label
            helpful = record.get("helpful", False)
            rating = int(record.get("user_rating", 3))
            is_correct = helpful or rating >= 4

            confidences.append(confidence)
            labels.append(1 if is_correct else 0)

        except (ValueError, KeyError, TypeError) as e:
            raise ValueError(
                f"Error extracting data from feedback record: {str(e)}"
            ) from e

    return np.array(confidences, dtype=np.float32), np.array(labels, dtype=np.int32)


def compute_calibration_metrics(
    raw_confidences: np.ndarray, correctness_labels: np.ndarray, num_bins: int = 10
) -> tuple[dict, CalibrationMetrics]:
    """
    Compute calibration curve and metrics.

    Divides predictions into bins by confidence, compares predicted confidence
    to observed accuracy in each bin.

    Parameters
    ----------
    raw_confidences : np.ndarray
        Raw confidence scores (0-1) from model.
    correctness_labels : np.ndarray
        Binary labels (0/1) indicating actual correctness.
    num_bins : int, optional
        Number of bins for calibration curve (default 10).

    Returns
    -------
    tuple[dict, CalibrationMetrics]
        (calibration_curve_dict, metrics)

        calibration_curve_dict has keys:
        - bin_edges: Confidence bin boundaries
        - bin_confidences: Mean predicted confidence in each bin
        - bin_accuracies: Observed accuracy in each bin
        - bin_counts: Number of samples in each bin

    Raises
    ------
    ValueError
        If input arrays are empty or mismatched.
    """
    if len(raw_confidences) == 0:
        raise ValueError("raw_confidences array is empty")

    if len(raw_confidences) != len(correctness_labels):
        raise ValueError("raw_confidences and correctness_labels must have same length")

    # Compute overall metrics
    expected_calibration_error = float(
        np.mean(np.abs(raw_confidences - correctness_labels))
    )
    brier_score = float(np.mean((raw_confidences - correctness_labels) ** 2))

    # Bin-based calibration curve
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Include upper edge only for last bin
        mask = (raw_confidences >= lower) & (
            raw_confidences < upper if i < num_bins - 1 else raw_confidences <= upper
        )

        if np.sum(mask) > 0:
            bin_confidences.append(float(np.mean(raw_confidences[mask])))
            bin_accuracies.append(float(np.mean(correctness_labels[mask])))
            bin_counts.append(int(np.sum(mask)))
        else:
            bin_confidences.append(None)
            bin_accuracies.append(None)
            bin_counts.append(0)

    calibration_curve = {
        "bin_edges": bin_edges.tolist(),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }

    max_calibration_error = float(
        max(
            [
                abs(c - a)
                for c, a in zip(bin_confidences, bin_accuracies)
                if c is not None and a is not None
            ],
            default=0.0,
        )
    )

    metrics = CalibrationMetrics(
        expected_calibration_error=expected_calibration_error,
        max_calibration_error=max_calibration_error,
        brier_score=brier_score,
        num_samples=len(raw_confidences),
    )

    return calibration_curve, metrics


def train_isotonic_regressor(
    raw_confidences: np.ndarray, correctness_labels: np.ndarray
) -> IsotonicRegression:
    """
    Train isotonic regression model to map raw confidence to calibrated probability.

    Isotonic regression is non-parametric and enforces monotonicity:
    higher raw confidence → higher calibrated confidence (or equal).

    Parameters
    ----------
    raw_confidences : np.ndarray
        Raw confidence scores (0-1) from model.
    correctness_labels : np.ndarray
        Binary labels (0/1) indicating actual correctness.

    Returns
    -------
    IsotonicRegression
        Fitted isotonic regression model.

    Raises
    ------
    ValueError
        If input arrays are empty or mismatched.
    """
    if len(raw_confidences) == 0:
        raise ValueError("raw_confidences array is empty")

    if len(raw_confidences) != len(correctness_labels):
        raise ValueError("raw_confidences and correctness_labels must have same length")

    # Train isotonic regression with out_of_bounds handling
    iso_reg = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_reg.fit(raw_confidences, correctness_labels)

    return iso_reg


def calibrate_confidence(
    raw_confidence: float, iso_reg: IsotonicRegression
) -> float:
    """
    Apply isotonic regression to calibrate a single confidence score.

    Parameters
    ----------
    raw_confidence : float
        Raw confidence score from model (0-1).
    iso_reg : IsotonicRegression
        Fitted isotonic regression model.

    Returns
    -------
    float
        Calibrated confidence score (0-1).

    Raises
    ------
    ValueError
        If raw_confidence is outside [0, 1].
    """
    if not 0.0 <= raw_confidence <= 1.0:
        raise ValueError(f"raw_confidence must be in [0, 1], got {raw_confidence}")

    # IsotonicRegression expects 2D input for predict
    calibrated = float(iso_reg.predict([[raw_confidence]])[0])
    return np.clip(calibrated, 0.0, 1.0)


def calibrate_batch(
    raw_confidences: np.ndarray, iso_reg: IsotonicRegression
) -> np.ndarray:
    """
    Apply isotonic regression to calibrate multiple confidence scores.

    Parameters
    ----------
    raw_confidences : np.ndarray
        Raw confidence scores from model (0-1).
    iso_reg : IsotonicRegression
        Fitted isotonic regression model.

    Returns
    -------
    np.ndarray
        Calibrated confidence scores (0-1).

    Raises
    ------
    ValueError
        If any raw confidence is outside [0, 1].
    """
    if np.any((raw_confidences < 0.0) | (raw_confidences > 1.0)):
        raise ValueError("All raw_confidences must be in [0, 1]")

    # Reshape for isotonic regression prediction
    raw_confidences_reshaped = raw_confidences.reshape(-1, 1)
    calibrated = iso_reg.predict(raw_confidences_reshaped)
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def save_calibration_model(iso_reg: IsotonicRegression, path: Path = CALIBRATION_MODEL_PATH) -> None:
    """
    Save isotonic regression model to disk.

    Parameters
    ----------
    iso_reg : IsotonicRegression
        Fitted isotonic regression model.
    path : Path, optional
        Path to save model (default: data/calibration_model.pkl).

    Raises
    ------
    IOError
        If file cannot be written.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(iso_reg, f)
    except IOError as e:
        raise IOError(f"Failed to save calibration model to {path}: {str(e)}") from e
    except Exception as e:
        raise IOError(f"Unexpected error saving calibration model: {str(e)}") from e


def load_calibration_model(path: Path = CALIBRATION_MODEL_PATH) -> Optional[IsotonicRegression]:
    """
    Load isotonic regression model from disk.

    Parameters
    ----------
    path : Path, optional
        Path to load model from (default: data/calibration_model.pkl).

    Returns
    -------
    IsotonicRegression or None
        Fitted isotonic regression model, or None if not found.

    Raises
    ------
    IOError
        If file cannot be read.
    """
    try:
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except IOError as e:
        raise IOError(f"Failed to load calibration model from {path}: {str(e)}") from e
    except Exception as e:
        raise IOError(f"Unexpected error loading calibration model: {str(e)}") from e


def train_calibration(
    feedback_file: str = "data/feedback.jsonl",
    model_path: Path = CALIBRATION_MODEL_PATH,
) -> tuple[IsotonicRegression, CalibrationMetrics]:
    """
    End-to-end calibration training from feedback data.

    Loads feedback, extracts confidence-correctness pairs, trains isotonic
    regression, and saves the model.

    Parameters
    ----------
    feedback_file : str, optional
        Path to feedback JSONL file (default: data/feedback.jsonl).
    model_path : Path, optional
        Path to save calibration model (default: data/calibration_model.pkl).

    Returns
    -------
    tuple[IsotonicRegression, CalibrationMetrics]
        (trained_model, metrics)

    Raises
    ------
    ValueError
        If feedback is insufficient or malformed.
    IOError
        If file operations fail.
    """
    # Load feedback
    feedback_records = _load_feedback_data(feedback_file)

    if len(feedback_records) == 0:
        raise ValueError("No feedback records found. Cannot train calibration model.")

    # Extract confidence and labels
    raw_confidences, correctness_labels = _extract_confidence_labels(feedback_records)

    # Compute calibration metrics and curve
    calibration_curve, metrics = compute_calibration_metrics(
        raw_confidences, correctness_labels
    )

    # Train isotonic regression
    iso_reg = train_isotonic_regressor(raw_confidences, correctness_labels)

    # Save model
    save_calibration_model(iso_reg, model_path)

    return iso_reg, metrics


def get_calibrated_confidence(
    raw_confidence: float,
    iso_reg: Optional[IsotonicRegression] = None,
    model_path: Path = CALIBRATION_MODEL_PATH,
) -> float:
    """
    Get calibrated confidence for a single prediction.

    If no trained model is provided, loads from disk. If no model exists,
    returns raw confidence (pass-through).

    Parameters
    ----------
    raw_confidence : float
        Raw confidence score from model (0-1).
    iso_reg : IsotonicRegression, optional
        Fitted isotonic regression model. If None, loads from disk.
    model_path : Path, optional
        Path to calibration model (default: data/calibration_model.pkl).

    Returns
    -------
    float
        Calibrated confidence score (0-1).

    Raises
    ------
    ValueError
        If raw_confidence is outside [0, 1].
    """
    if not 0.0 <= raw_confidence <= 1.0:
        raise ValueError(f"raw_confidence must be in [0, 1], got {raw_confidence}")

    # Load model if not provided
    if iso_reg is None:
        iso_reg = load_calibration_model(model_path)

    # If no model exists, return raw confidence
    if iso_reg is None:
        return raw_confidence

    return calibrate_confidence(raw_confidence, iso_reg)


def get_calibrated_batch(
    raw_confidences: np.ndarray,
    iso_reg: Optional[IsotonicRegression] = None,
    model_path: Path = CALIBRATION_MODEL_PATH,
) -> np.ndarray:
    """
    Get calibrated confidences for multiple predictions.

    If no trained model is provided, loads from disk. If no model exists,
    returns raw confidences (pass-through).

    Parameters
    ----------
    raw_confidences : np.ndarray
        Raw confidence scores from model (0-1).
    iso_reg : IsotonicRegression, optional
        Fitted isotonic regression model. If None, loads from disk.
    model_path : Path, optional
        Path to calibration model (default: data/calibration_model.pkl).

    Returns
    -------
    np.ndarray
        Calibrated confidence scores (0-1).

    Raises
    ------
    ValueError
        If any raw confidence is outside [0, 1].
    """
    if np.any((raw_confidences < 0.0) | (raw_confidences > 1.0)):
        raise ValueError("All raw_confidences must be in [0, 1]")

    # Load model if not provided
    if iso_reg is None:
        iso_reg = load_calibration_model(model_path)

    # If no model exists, return raw confidences
    if iso_reg is None:
        return raw_confidences

    return calibrate_batch(raw_confidences, iso_reg)
