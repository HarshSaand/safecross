"""Deterministic synthetic interactions for SafeCross development and testing.

The generator deliberately stays small and inspectable.  It is not intended to
replace real motion datasets or a closed-loop simulator; it supplies physically
plausible fixtures for validating forecasting and safety-metric code before
large datasets are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ActorKind = Literal["pedestrian", "cyclist"]
ScenarioKind = Literal[
    "crossing",
    "occluded_crossing",
    "cyclist_merge",
    "non_crossing",
]


@dataclass(frozen=True)
class InteractionScenario:
    """A time-aligned ego/actor interaction in a planar road frame.

    Coordinates are metres.  The ego vehicle travels predominantly along +x;
    +y is to its left.  ``visibility`` is the observable fraction of the actor
    at each timestep and ranges from 0 (fully occluded) to 1 (fully visible).
    """

    scenario_id: str
    kind: ScenarioKind
    actor_kind: ActorKind
    times: np.ndarray
    ego_positions: np.ndarray
    actor_positions: np.ndarray
    visibility: np.ndarray
    metadata: dict[str, float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        ego = np.asarray(self.ego_positions, dtype=float)
        actor = np.asarray(self.actor_positions, dtype=float)
        visibility = np.asarray(self.visibility, dtype=float)
        n = len(times)
        if times.ndim != 1 or n < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("times must be a strictly increasing 1D array")
        if ego.shape != (n, 2) or actor.shape != (n, 2):
            raise ValueError("ego_positions and actor_positions must have shape (T, 2)")
        if visibility.shape != (n,) or np.any((visibility < 0) | (visibility > 1)):
            raise ValueError("visibility must have shape (T,) and values in [0, 1]")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "ego_positions", ego)
        object.__setattr__(self, "actor_positions", actor)
        object.__setattr__(self, "visibility", visibility)


def _smooth_start(times: np.ndarray, start: float, duration: float) -> np.ndarray:
    """Return a smooth 0..1 motion fraction beginning at ``start``."""

    u = np.clip((times - start) / duration, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def generate_interaction(
    kind: ScenarioKind = "occluded_crossing",
    *,
    seed: int = 0,
    horizon_s: float = 8.0,
    dt: float = 0.1,
) -> InteractionScenario:
    """Generate one deterministic interaction with mild seeded variation.

    Parameters
    ----------
    kind:
        Interaction geometry to produce.
    seed:
        Controls speeds and start times.  Equal arguments always produce equal
        arrays, supporting reproducible tests and rendered examples.
    horizon_s, dt:
        Duration and temporal resolution in seconds.
    """

    if horizon_s <= 1.0 or dt <= 0 or dt > horizon_s:
        raise ValueError("require horizon_s > 1 and 0 < dt <= horizon_s")
    if kind not in {"crossing", "occluded_crossing", "cyclist_merge", "non_crossing"}:
        raise ValueError(f"unsupported scenario kind: {kind}")

    rng = np.random.default_rng(seed)
    times = np.arange(0.0, horizon_s + dt / 2.0, dt)
    ego_speed = float(rng.uniform(5.5, 7.5))
    ego = np.column_stack((ego_speed * times, np.zeros_like(times)))
    conflict_x = float(rng.uniform(27.0, 33.0))

    if kind in {"crossing", "occluded_crossing", "non_crossing"}:
        actor_kind: ActorKind = "pedestrian"
        crossing_start = float(rng.uniform(1.8, 2.5))
        crossing_duration = float(rng.uniform(3.0, 4.0))
        fraction = _smooth_start(times, crossing_start, crossing_duration)
        start_y = -5.5
        if kind == "non_crossing":
            # Walk parallel to the road edge and never enter the ego lane.
            actor = np.column_stack(
                (conflict_x - 4.0 + 1.1 * times, np.full_like(times, start_y))
            )
        else:
            actor = np.column_stack(
                (np.full_like(times, conflict_x), start_y + 11.0 * fraction)
            )
        visibility = np.ones_like(times)
        if kind == "occluded_crossing":
            reveal_start = crossing_start + crossing_duration * 0.38
            visibility = np.clip((times - reveal_start) / 0.8, 0.0, 1.0)
    else:
        actor_kind = "cyclist"
        merge_start = float(rng.uniform(1.3, 2.0))
        merge_fraction = _smooth_start(times, merge_start, 2.4)
        cyclist_speed = float(rng.uniform(4.0, 5.5))
        actor = np.column_stack(
            (12.0 + cyclist_speed * times, -3.5 + 3.5 * merge_fraction)
        )
        visibility = np.ones_like(times)

    return InteractionScenario(
        scenario_id=f"synthetic-{kind}-seed-{seed}",
        kind=kind,
        actor_kind=actor_kind,
        times=times,
        ego_positions=ego,
        actor_positions=actor,
        visibility=visibility,
        metadata={
            "ego_speed_mps": ego_speed,
            "conflict_x_m": conflict_x,
            "seed": float(seed),
        },
    )


def generate_suite(*, seeds: int = 10) -> list[InteractionScenario]:
    """Generate a balanced, deterministic suite across all scenario kinds."""

    if seeds <= 0:
        raise ValueError("seeds must be positive")
    kinds: tuple[ScenarioKind, ...] = (
        "crossing",
        "occluded_crossing",
        "cyclist_merge",
        "non_crossing",
    )
    return [generate_interaction(kind, seed=seed) for seed in range(seeds) for kind in kinds]
