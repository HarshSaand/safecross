"""Safety-risk and evaluation metrics for time-aligned road interactions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import MultiHypothesisPrediction


@dataclass(frozen=True)
class ApproachResult:
    """Closest separation between two aligned trajectories."""

    distance_m: float
    time_s: float
    index: int


def closest_approach(
    times: np.ndarray, ego_positions: np.ndarray, actor_positions: np.ndarray
) -> ApproachResult:
    """Return minimum Euclidean separation and its timestamp."""

    t, ego, actor = _validate_aligned(times, ego_positions, actor_positions)
    distances = np.linalg.norm(ego - actor, axis=1)
    index = int(np.argmin(distances))
    return ApproachResult(float(distances[index]), float(t[index]), index)


def time_to_collision(
    times: np.ndarray,
    ego_positions: np.ndarray,
    actor_positions: np.ndarray,
    *,
    collision_radius_m: float = 1.5,
) -> float:
    """Return time of first safety-envelope overlap, or infinity if none."""

    if collision_radius_m <= 0:
        raise ValueError("collision_radius_m must be positive")
    t, ego, actor = _validate_aligned(times, ego_positions, actor_positions)
    colliding = np.flatnonzero(np.linalg.norm(ego - actor, axis=1) <= collision_radius_m)
    return float(t[colliding[0]]) if len(colliding) else float("inf")


def conflict_probability(
    ego_positions: np.ndarray,
    prediction: MultiHypothesisPrediction,
    *,
    safety_radius_m: float = 2.0,
) -> float:
    """Probability mass of hypotheses entering the ego safety envelope."""

    ego = np.asarray(ego_positions, dtype=float)
    if ego.shape != (len(prediction.times), 2):
        raise ValueError("ego_positions must align with prediction times")
    if safety_radius_m <= 0:
        raise ValueError("safety_radius_m must be positive")
    min_distances = np.linalg.norm(prediction.positions - ego[None, :, :], axis=2).min(axis=1)
    return float(prediction.probabilities[min_distances <= safety_radius_m].sum())


def early_warning_lead_time(
    times: np.ndarray,
    risk_scores: np.ndarray,
    *,
    event_time_s: float,
    threshold: float = 0.5,
) -> float:
    """Seconds between the first threshold crossing and a conflict event.

    Returns NaN if no warning occurs before the event.  Warnings after the
    event are intentionally not rewarded.
    """

    t = np.asarray(times, dtype=float)
    scores = np.asarray(risk_scores, dtype=float)
    if t.ndim != 1 or scores.shape != t.shape or np.any(np.diff(t) <= 0):
        raise ValueError("times and risk_scores must be aligned 1D arrays")
    if not 0 <= threshold <= 1 or np.any((scores < 0) | (scores > 1)):
        raise ValueError("risk scores and threshold must be in [0, 1]")
    warning = np.flatnonzero((scores >= threshold) & (t <= event_time_s))
    return float(event_time_s - t[warning[0]]) if len(warning) else float("nan")


def brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error of binary event probabilities (lower is better)."""

    probs, labels = _validate_probabilities(probabilities, outcomes)
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error(
    probabilities: np.ndarray, outcomes: np.ndarray, *, bins: int = 10
) -> float:
    """Binary expected calibration error using equal-width confidence bins."""

    if bins <= 0:
        raise ValueError("bins must be positive")
    probs, labels = _validate_probabilities(probabilities, outcomes)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probs >= lower) & (probs < upper if index < bins - 1 else probs <= upper)
        if np.any(mask):
            error += float(mask.mean()) * abs(float(probs[mask].mean() - labels[mask].mean()))
    return error


def warning_classification(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Precision, recall, F1 and false-warning rate at a fixed threshold."""

    probs, labels = _validate_probabilities(probabilities, outcomes)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    predicted = probs >= threshold
    positive = labels.astype(bool)
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    tn = int(np.sum(~predicted & ~positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    false_warning_rate = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_warning_rate": false_warning_rate,
    }


def _validate_aligned(
    times: np.ndarray, ego_positions: np.ndarray, actor_positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    ego = np.asarray(ego_positions, dtype=float)
    actor = np.asarray(actor_positions, dtype=float)
    if t.ndim != 1 or len(t) == 0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must be a non-empty increasing 1D array")
    if ego.shape != (len(t), 2) or actor.shape != (len(t), 2):
        raise ValueError("position arrays must have shape (T, 2)")
    return t, ego, actor


def _validate_probabilities(
    probabilities: np.ndarray, outcomes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=float)
    labels = np.asarray(outcomes, dtype=float)
    if probs.ndim != 1 or labels.shape != probs.shape or len(probs) == 0:
        raise ValueError("probabilities and outcomes must be aligned non-empty 1D arrays")
    if np.any((probs < 0) | (probs > 1)) or np.any(~np.isin(labels, [0, 1])):
        raise ValueError("probabilities must be in [0,1] and outcomes must be binary")
    return probs, labels
