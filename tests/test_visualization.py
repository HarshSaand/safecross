from __future__ import annotations

import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safecross.visualization import (
    ScenarioView,
    create_comparison_animation,
    generate_reference_scenario,
    scenario_from_core,
    simulate_policy,
)


def test_reference_scenario_is_deterministic_and_valid() -> None:
    first = generate_reference_scenario(duration=3.0, dt=0.1)
    second = generate_reference_scenario(duration=3.0, dt=0.1)
    np.testing.assert_allclose(first.actor_positions, second.actor_positions)
    assert first.times.shape == (31,)


def test_core_adapter_accepts_mapping() -> None:
    value = {
        "times": [0.0, 0.5, 1.0],
        "ego_positions": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        "actor_positions": [[2.0, 2.0], [2.0, 1.0], [2.0, 0.0]],
        "actor_kind": "cyclist",
        "occluders": [],
    }
    scenario = scenario_from_core(value)
    assert scenario.actor_kind == "cyclist"
    assert scenario.ego_positions.shape == (3, 2)
    assert scenario.actor_positions[-1, 0] == 1.5


def test_safecross_acts_earlier_than_baseline() -> None:
    scenario = generate_reference_scenario()
    baseline = simulate_policy(scenario, "baseline")
    safe = simulate_policy(scenario, "safecross")
    baseline_action = next(i for i, decision in enumerate(baseline.decisions) if decision != "CONTINUE")
    safe_action = next(i for i, decision in enumerate(safe.decisions) if decision != "CONTINUE")
    assert safe_action < baseline_action


def test_animation_can_render_first_frame() -> None:
    import matplotlib.pyplot as plt

    scenario = generate_reference_scenario(duration=0.4, dt=0.1)
    figure, animation = create_comparison_animation(scenario, fps=5)
    animation._func(0)
    assert len(figure.axes) == 2
    animation._draw_was_started = True
    plt.close(figure)


def test_invalid_scenario_shape_is_rejected() -> None:
    try:
        ScenarioView(
            times=np.array([0.0, 1.0, 2.0]),
            ego_positions=np.zeros((2, 2)),
            actor_positions=np.zeros((3, 2)),
        )
    except ValueError as error:
        assert "shape" in str(error)
    else:
        raise AssertionError("invalid shape should raise ValueError")
