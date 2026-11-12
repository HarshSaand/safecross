import math
import unittest

import numpy as np

from safecross.baselines import MultiHypothesisPrediction
from safecross.risk import (
    brier_score,
    closest_approach,
    conflict_probability,
    early_warning_lead_time,
    expected_calibration_error,
    time_to_collision,
    warning_classification,
)


class RiskTests(unittest.TestCase):
    def test_approach_and_collision_time(self):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        ego = np.column_stack((times, np.zeros_like(times)))
        actor = np.array([[2.0, 3.0], [2.0, 2.0], [2.0, 0.5], [2.0, -2.0]])
        approach = closest_approach(times, ego, actor)
        self.assertAlmostEqual(approach.distance_m, 0.5)
        self.assertEqual(approach.time_s, 2.0)
        self.assertEqual(time_to_collision(times, ego, actor, collision_radius_m=0.75), 2.0)
        self.assertTrue(math.isinf(time_to_collision(times, ego, actor, collision_radius_m=0.1)))

    def test_conflict_probability_sums_only_unsafe_modes(self):
        times = np.array([1.0, 2.0])
        prediction = MultiHypothesisPrediction(
            times,
            np.array([
                [[1.0, 0.2], [2.0, 0.2]],
                [[1.0, 5.0], [2.0, 5.0]],
            ]),
            np.array([0.7, 0.3]),
        )
        ego = np.array([[1.0, 0.0], [2.0, 0.0]])
        self.assertAlmostEqual(conflict_probability(ego, prediction, safety_radius_m=1.0), 0.7)

    def test_warning_lead_time_uses_first_pre_event_warning(self):
        lead = early_warning_lead_time(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.1, 0.6, 0.8, 0.9]),
            event_time_s=3.0,
            threshold=0.5,
        )
        self.assertEqual(lead, 2.0)

    def test_calibration_metrics_and_classification(self):
        probabilities = np.array([0.1, 0.2, 0.8, 0.9])
        outcomes = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(brier_score(probabilities, outcomes), 0.025)
        self.assertAlmostEqual(expected_calibration_error(probabilities, outcomes, bins=2), 0.15)
        metrics = warning_classification(probabilities, outcomes, threshold=0.5)
        self.assertEqual(metrics, {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "false_warning_rate": 0.0,
        })


if __name__ == "__main__":
    unittest.main()
