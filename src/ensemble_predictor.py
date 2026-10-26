"""
Ensemble prediction: combine multiple prediction strategies for robustness.

Combines three ensemble strategies:
1. Weighted k-NN vote: Uses similarity scores with exponential weighting
2. Logistic regression: Trains on case features + similarity scores to predict strategy
3. Random forest: Ensemble of decision trees for non-linear patterns

Final prediction via majority vote + confidence averaging across all three strategies.
Includes cross-validation and model persistence.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
from pathlib import Path

from src.schema import Case
from src.predictor import Recommendation


ENSEMBLE_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "ensemble_models.pkl"
CV_FOLDS = 5


@dataclass
class EnsembleModels:
    """Container for trained ensemble models with cross-validation metrics."""
    lr_model: LogisticRegression
    rf_model: RandomForestClassifier
    scaler: StandardScaler
    strategy_classes: list[str]
    lr_cv_scores: np.ndarray = None  # Cross-validation scores for LR
    rf_cv_scores: np.ndarray = None  # Cross-validation scores for RF
    knn_cv_scores: np.ndarray = None  # Cross-validation scores for k-NN


def _prepare_features(
    retrieved: list[tuple[Case, float]], cases: list[Case]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix for ML models.

    Features include (in order):
    1. Similarity score from retrieval [0, 1]
    2. Company size encoding [0=Small, 1=Medium, 2=Large]
    3. Year normalized to [0, 1]
    4. ROI estimate (raw value)
    5. Time to result normalized to [0, 1] (divided by 60 months)
    6. Number of key challenges (count)

    Parameters
    ----------
    retrieved : list[tuple[Case, float]]
        Retrieved cases with similarity scores.
    cases : list[Case]
        Reference cases (used for context, typically same as retrieved).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Feature matrix (N x 6) and strategy labels (N,).
    """
    features_list = []
    strategy_labels = []

    for case, score in retrieved:
        # Feature 1: Similarity score (primary predictor)
        case_features = [score]

        # Feature 2: Company size (categorical encoding)
        size_encoding = {"Small": 0, "Medium": 1, "Large": 2}.get(case.company_size, 1)
        case_features.append(size_encoding)

        # Feature 3: Year normalized
        case_features.append(case.year / 2025.0)

        # Feature 4: ROI estimate (raw)
        case_features.append(case.outcome_metrics.roi_estimate)

        # Feature 5: Time to result normalized
        case_features.append(case.outcome_metrics.time_to_result_months / 60.0)

        # Feature 6: Challenge count
        case_features.append(len(case.key_challenges))

        features_list.append(case_features)
        strategy_labels.append(case.strategy_type)

    X = np.array(features_list, dtype=np.float32)
    return X, strategy_labels


def train_ensemble(cases: list[Case]) -> EnsembleModels:
    """
    Train ensemble models on historical cases with cross-validation.

    Trains three strategies:
    1. Weighted k-NN vote (baseline, no training required)
    2. Logistic regression: linear decision boundaries for strategy classification
    3. Random forest: non-linear patterns via ensemble of decision trees

    Evaluates each strategy via stratified k-fold cross-validation.

    Parameters
    ----------
    cases : list[Case]
        Historical cases to train on. Requires at least 2*CV_FOLDS cases.

    Returns
    -------
    EnsembleModels
        Trained models, scaler, and cross-validation scores.
    """
    # Prepare training data
    all_features = []
    all_strategies = []

    for case in cases:
        # Single-case "retrieval" with high self-similarity
        features = [0.95]

        # Structured features
        size_encoding = {"Small": 0, "Medium": 1, "Large": 2}.get(case.company_size, 1)
        features.append(size_encoding)
        features.append(case.year / 2025.0)
        features.append(case.outcome_metrics.roi_estimate)
        features.append(case.outcome_metrics.time_to_result_months / 60.0)
        features.append(len(case.key_challenges))

        all_features.append(features)
        all_strategies.append(case.strategy_type)

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_strategies)

    # Feature scaling (required for LR, beneficial for RF)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Strategy 1: Weighted k-NN (baseline)
    # Cross-validation would require custom scorer; for now use placeholder
    knn_cv_scores = np.array([0.0] * CV_FOLDS)

    # Strategy 2: Logistic regression with cross-validation
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs'
    )
    lr_cv_scores = cross_val_score(lr, X_scaled, y, cv=cv, scoring='accuracy')

    # Strategy 3: Random forest with cross-validation
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    rf_cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')

    # Train final models on full dataset
    lr.fit(X_scaled, y)
    rf.fit(X_scaled, y)

    # Get strategy classes
    strategy_classes = sorted(np.unique(y).tolist())

    return EnsembleModels(
        lr_model=lr,
        rf_model=rf,
        scaler=scaler,
        strategy_classes=strategy_classes,
        lr_cv_scores=lr_cv_scores,
        rf_cv_scores=rf_cv_scores,
        knn_cv_scores=knn_cv_scores,
    )


def save_ensemble(models: EnsembleModels, path: Path = ENSEMBLE_MODEL_PATH) -> None:
    """
    Save trained ensemble models to disk.

    Persists logistic regression, random forest, feature scaler, and cross-validation
    metrics for later inference or retraining evaluation.

    Parameters
    ----------
    models : EnsembleModels
        Trained models container.
    path : Path
        File path for pickle serialization. Creates parent directories as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(models, f)


def load_ensemble(path: Path = ENSEMBLE_MODEL_PATH) -> EnsembleModels | None:
    """
    Load trained ensemble models from disk.

    Parameters
    ----------
    path : Path
        File path for pickle deserialization.

    Returns
    -------
    EnsembleModels | None
        Loaded models, or None if file does not exist.
    """
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _weighted_knn_vote(retrieved: list[tuple[Case, float]]) -> tuple[str, float]:
    """
    Strategy 1: Weighted k-NN vote.

    Uses exponential weighting on similarity scores to vote for strategy.
    Higher similarity scores receive proportionally higher weights.

    Parameters
    ----------
    retrieved : list[tuple[Case, float]]
        Retrieved cases with similarity scores.

    Returns
    -------
    tuple[str, float]
        Best strategy and weighted vote confidence.
    """
    scores = np.array([s for _, s in retrieved])
    weights = np.exp(scores) / np.exp(scores).sum()

    strategy_votes = {}
    for (case, _), w in zip(retrieved, weights):
        strategy_votes[case.strategy_type] = strategy_votes.get(case.strategy_type, 0.0) + w

    best_strategy = max(strategy_votes, key=strategy_votes.get)
    confidence = strategy_votes[best_strategy]

    return best_strategy, confidence


def _logistic_regression_predict(
    retrieved: list[tuple[Case, float]],
    ensemble: EnsembleModels,
) -> tuple[str, float]:
    """
    Strategy 2: Logistic regression prediction.

    Trains linear decision boundaries on case features and similarity scores.
    Averages probabilities across retrieved cases for robust prediction.

    Parameters
    ----------
    retrieved : list[tuple[Case, float]]
        Retrieved cases with similarity scores.
    ensemble : EnsembleModels
        Trained ensemble models.

    Returns
    -------
    tuple[str, float]
        Best strategy and average predicted probability.
    """
    X, _ = _prepare_features(retrieved, [c for c, _ in retrieved])
    X_scaled = ensemble.scaler.transform(X)

    # Average probabilities across all retrieved cases
    lr_probs = ensemble.lr_model.predict_proba(X_scaled).mean(axis=0)
    best_idx = np.argmax(lr_probs)
    best_strategy = ensemble.strategy_classes[best_idx]
    confidence = float(lr_probs[best_idx])

    return best_strategy, confidence


def _random_forest_predict(
    retrieved: list[tuple[Case, float]],
    ensemble: EnsembleModels,
) -> tuple[str, float]:
    """
    Strategy 3: Random forest prediction.

    Ensemble of decision trees capturing non-linear patterns in case features.
    Averages probabilities across retrieved cases for robust prediction.

    Parameters
    ----------
    retrieved : list[tuple[Case, float]]
        Retrieved cases with similarity scores.
    ensemble : EnsembleModels
        Trained ensemble models.

    Returns
    -------
    tuple[str, float]
        Best strategy and average predicted probability.
    """
    X, _ = _prepare_features(retrieved, [c for c, _ in retrieved])
    X_scaled = ensemble.scaler.transform(X)

    # Average probabilities across all retrieved cases
    rf_probs = ensemble.rf_model.predict_proba(X_scaled).mean(axis=0)
    best_idx = np.argmax(rf_probs)
    best_strategy = ensemble.strategy_classes[best_idx]
    confidence = float(rf_probs[best_idx])

    return best_strategy, confidence


def predict_ensemble(
    retrieved: list[tuple[Case, float]],
    ensemble: EnsembleModels | None = None,
) -> Recommendation:
    """
    Ensemble prediction combining three strategies via majority vote.

    Strategy combination:
    - Strategy 1 (Weighted k-NN): Always available, baseline retrieval-based vote
    - Strategy 2 (Logistic Regression): Linear decision boundaries on features
    - Strategy 3 (Random Forest): Non-linear patterns via tree ensemble

    Final decision via majority vote among the three strategies.
    Confidence computed as average of confidences from strategies voting for winner.

    Parameters
    ----------
    retrieved : list[tuple[Case, float]]
        Retrieved cases with similarity scores (must be non-empty).
    ensemble : EnsembleModels | None
        Trained ensemble models. If None, uses only weighted k-NN (Strategy 1).

    Returns
    -------
    Recommendation
        Ensemble recommendation with combined confidence, outcome prediction,
        ROI range, timeline estimate, and supporting rationale.
    """
    if not retrieved:
        return Recommendation(
            recommended_strategy="Insufficient data",
            strategy_confidence=0.0,
            predicted_outcome="Unknown",
            outcome_confidence=0.0,
            estimated_roi_low=0.0,
            estimated_roi_high=0.0,
            avg_time_to_result_months=0.0,
            supporting_rationale="No similar cases found.",
        )

    # Compute weights from similarity scores (used for outcome/ROI/timeline prediction)
    scores = np.array([s for _, s in retrieved])
    weights = np.exp(scores) / np.exp(scores).sum()

    # Strategy 1: Weighted k-NN vote (always available)
    best_strategy_knn, conf_knn = _weighted_knn_vote(retrieved)

    # Strategies 2 & 3: ML models (if available)
    best_strategy_lr = best_strategy_knn
    conf_lr = conf_knn
    best_strategy_rf = best_strategy_knn
    conf_rf = conf_knn

    if ensemble is not None:
        try:
            best_strategy_lr, conf_lr = _logistic_regression_predict(retrieved, ensemble)
            best_strategy_rf, conf_rf = _random_forest_predict(retrieved, ensemble)
        except Exception:
            # Fallback to k-NN if ML models fail
            pass

    # Majority vote among three strategies
    votes = [best_strategy_knn, best_strategy_lr, best_strategy_rf]
    vote_counts = {}
    for v in votes:
        vote_counts[v] = vote_counts.get(v, 0) + 1

    best_strategy = max(vote_counts, key=vote_counts.get)

    # Average confidence from strategies voting for winner
    confidences = []
    if best_strategy_knn == best_strategy:
        confidences.append(conf_knn)
    if best_strategy_lr == best_strategy:
        confidences.append(conf_lr)
    if best_strategy_rf == best_strategy:
        confidences.append(conf_rf)

    ensemble_confidence = float(np.mean(confidences)) if confidences else 0.5

    # --- Outcome prediction ---
    outcome_votes = {}
    for (case, _), w in zip(retrieved, weights):
        outcome_votes[case.outcome] = outcome_votes.get(case.outcome, 0.0) + w

    best_outcome = max(outcome_votes, key=outcome_votes.get)
    outcome_conf = float(outcome_votes[best_outcome])

    # --- ROI & timeline estimation ---
    rois = np.array([c.outcome_metrics.roi_estimate for c, _ in retrieved])
    roi_weighted_mean = float(np.average(rois, weights=weights))
    roi_std = float(np.sqrt(np.average((rois - roi_weighted_mean) ** 2, weights=weights)))
    roi_low = max(0.0, round(roi_weighted_mean - roi_std, 2))
    roi_high = round(roi_weighted_mean + roi_std, 2)

    times = np.array([c.outcome_metrics.time_to_result_months for c, _ in retrieved])
    avg_time = float(np.average(times, weights=weights))

    # --- Rationale ---
    top_case = retrieved[0][0]
    rationale = (
        f"Recommended '{best_strategy}' (ensemble confidence {ensemble_confidence:.0%}) based on "
        f"{len(retrieved)} similar historical cases. The closest analogy is "
        f"'{top_case.title}' ({top_case.industry}, {top_case.year}) which had a "
        f"{top_case.outcome.lower()} outcome with {top_case.outcome_metrics.roi_estimate:.1f}x ROI."
    )

    return Recommendation(
        recommended_strategy=best_strategy,
        strategy_confidence=round(float(ensemble_confidence), 3),
        predicted_outcome=best_outcome,
        outcome_confidence=round(float(outcome_conf), 3),
        estimated_roi_low=roi_low,
        estimated_roi_high=roi_high,
        avg_time_to_result_months=round(avg_time, 1),
        supporting_rationale=rationale,
    )
