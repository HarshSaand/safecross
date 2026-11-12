"""Inspectable forecasting baselines and a multi-hypothesis result schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MultiHypothesisPrediction:
    """Future positions and probabilities for K mutually exclusive modes."""

    times: np.ndarray
    positions: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        positions = np.asarray(self.positions, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        if times.ndim != 1 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing")
        if positions.ndim != 3 or positions.shape[1:] != (len(times), 2):
            raise ValueError("positions must have shape (K, T, 2)")
        if probabilities.shape != (positions.shape[0],):
            raise ValueError("probabilities must have shape (K,)")
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("probabilities must be non-negative and sum to one")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "probabilities", probabilities)


def constant_velocity(
    observed_times: np.ndarray,
    observed_positions: np.ndarray,
    future_times: np.ndarray,
) -> np.ndarray:
    """Extrapolate the final observed velocity into the future.

    A least-squares velocity over up to the final five observations is used to
    reduce sensitivity to one noisy finite difference.
    """

    times = np.asarray(observed_times, dtype=float)
    positions = np.asarray(observed_positions, dtype=float)
    future = np.asarray(future_times, dtype=float)
    if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("observed_times must contain at least two increasing values")
    if positions.shape != (len(times), 2):
        raise ValueError("observed_positions must have shape (T, 2)")
    if future.ndim != 1 or np.any(future <= times[-1]) or np.any(np.diff(future) <= 0):
        raise ValueError("future_times must be increasing and after the observations")

    window = min(5, len(times))
    local_t = times[-window:] - times[-1]
    design = np.column_stack((local_t, np.ones(window)))
    velocity = np.linalg.lstsq(design, positions[-window:], rcond=None)[0][0]
    return positions[-1] + (future - times[-1])[:, None] * velocity


def constant_velocity_hypotheses(
    observed_times: np.ndarray,
    observed_positions: np.ndarray,
    future_times: np.ndarray,
    *,
    speed_scales: tuple[float, ...] = (0.65, 1.0, 1.35),
    probabilities: tuple[float, ...] = (0.2, 0.6, 0.2),
) -> MultiHypothesisPrediction:
    """Create speed-varied futures around a constant-velocity baseline."""

    if len(speed_scales) != len(probabilities) or not speed_scales:
        raise ValueError("speed_scales and probabilities must be non-empty and equal length")
    base = constant_velocity(observed_times, observed_positions, future_times)
    origin = np.asarray(observed_positions, dtype=float)[-1]
    displacement = base - origin
    positions = np.stack([origin + scale * displacement for scale in speed_scales])
    probs = np.asarray(probabilities, dtype=float)
    probs = probs / probs.sum()
    return MultiHypothesisPrediction(np.asarray(future_times), positions, probs)
