import unittest

import numpy as np

from safecross.baselines import constant_velocity, constant_velocity_hypotheses
from safecross.scenarios import generate_interaction, generate_suite


class ScenarioTests(unittest.TestCase):
    def test_interaction_generation_is_deterministic(self):
        first = generate_interaction("occluded_crossing", seed=7)
        second = generate_interaction("occluded_crossing", seed=7)
        np.testing.assert_allclose(first.ego_positions, second.ego_positions)
        np.testing.assert_allclose(first.actor_positions, second.actor_positions)
        self.assertEqual(first.scenario_id, second.scenario_id)

    def test_occluded_actor_becomes_visible(self):
        scenario = generate_interaction("occluded_crossing", seed=2)
        self.assertEqual(scenario.visibility[0], 0)
        self.assertEqual(scenario.visibility[-1], 1)
        self.assertTrue(np.all(np.diff(scenario.visibility) >= 0))

    def test_suite_is_balanced(self):
        suite = generate_suite(seeds=3)
        self.assertEqual(len(suite), 12)
        self.assertEqual(
            {item.kind for item in suite},
            {"crossing", "occluded_crossing", "cyclist_merge", "non_crossing"},
        )

    def test_constant_velocity_reconstructs_linear_motion(self):
        times = np.array([0.0, 1.0, 2.0])
        positions = np.column_stack((2 * times, -times))
        future = np.array([3.0, 4.0])
        prediction = constant_velocity(times, positions, future)
        np.testing.assert_allclose(prediction, [[6.0, -3.0], [8.0, -4.0]])

    def test_multi_hypothesis_probabilities_are_normalized(self):
        result = constant_velocity_hypotheses(
            np.array([0.0, 1.0]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([2.0, 3.0]),
            probabilities=(1.0, 2.0, 1.0),
        )
        self.assertEqual(result.positions.shape, (3, 2, 2))
        self.assertAlmostEqual(result.probabilities.sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
