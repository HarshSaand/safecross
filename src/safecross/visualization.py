"""Visual storytelling tools for the SafeCross closed-loop demo.

The renderer deliberately keeps the simulation lightweight. It is a visual
benchmark harness, not a vehicle dynamics claim. The same scenario is shown
twice: a reactive constant-velocity baseline and an uncertainty-aware policy
that reasons about a vulnerable road user's possible future paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import shutil

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle


@dataclass(frozen=True)
class ScenarioView:
    """Minimal time-aligned scene representation consumed by the renderer."""

    times: np.ndarray
    ego_positions: np.ndarray
    actor_positions: np.ndarray
    actor_kind: str = "pedestrian"
    title: str = "Occluded pedestrian crossing"
    occluders: tuple[tuple[float, float, float, float], ...] = ((-1.6, 1.2, 3.2, 2.4),)

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        ego = np.asarray(self.ego_positions, dtype=float)
        actor = np.asarray(self.actor_positions, dtype=float)
        if times.ndim != 1 or times.size < 3:
            raise ValueError("times must be a one-dimensional array with at least 3 samples")
        if ego.shape != (times.size, 2) or actor.shape != (times.size, 2):
            raise ValueError("ego_positions and actor_positions must have shape (T, 2)")
        if not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "ego_positions", ego)
        object.__setattr__(self, "actor_positions", actor)


@dataclass(frozen=True)
class PolicyTrace:
    """Closed-loop trajectory and decision telemetry for one policy."""

    positions: np.ndarray
    speeds: np.ndarray
    risk: np.ndarray
    time_to_conflict: np.ndarray
    decisions: tuple[str, ...]
    minimum_distance: float


def _read_field(value: Any, names: Iterable[str]) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    raise AttributeError(f"none of the fields {tuple(names)} were found")


def scenario_from_core(scenario: Any) -> ScenarioView:
    """Adapt a core scenario object or mapping to :class:`ScenarioView`.

    The adapter accepts the field names used by SafeCross's core generator as
    well as plain dictionaries, making the visual layer easy to reuse in tests
    and notebooks.
    """

    times = _read_field(scenario, ("times", "timestamps", "t"))
    ego = _read_field(scenario, ("ego_positions", "ego_trajectory", "ego_xy"))
    actor = _read_field(
        scenario,
        ("actor_positions", "vru_positions", "agent_positions", "actor_trajectory"),
    )
    kind = getattr(scenario, "actor_kind", None)
    if kind is None and isinstance(scenario, dict):
        kind = scenario.get("actor_kind", "pedestrian")
    title = getattr(scenario, "title", None)
    if title is None and isinstance(scenario, dict):
        title = scenario.get("title", "Vulnerable road-user interaction")
    if title is None and hasattr(scenario, "kind"):
        title = str(getattr(scenario, "kind")).replace("_", " ").title()
    occluders = getattr(scenario, "occluders", None)
    if occluders is None and isinstance(scenario, dict):
        occluders = scenario.get("occluders")
    ego_array = np.asarray(ego, dtype=float).copy()
    actor_array = np.asarray(actor, dtype=float).copy()
    # Core fixtures use absolute road coordinates while the comparison canvas
    # is centred on the interaction. Re-centre both trajectories on the
    # actor's closest approach to the ego lane without changing their geometry.
    conflict_index = int(np.argmin(np.abs(actor_array[:, 1])))
    x_shift = actor_array[conflict_index, 0] - 1.5
    ego_array[:, 0] -= x_shift
    actor_array[:, 0] -= x_shift
    return ScenarioView(
        times=np.asarray(times),
        ego_positions=ego_array,
        actor_positions=actor_array,
        actor_kind=str(kind or "pedestrian"),
        title=str(title or "Vulnerable road-user interaction"),
        occluders=tuple(occluders or ((-1.6, 1.2, 3.2, 2.4),)),
    )


def generate_reference_scenario(duration: float = 6.0, dt: float = 0.08) -> ScenarioView:
    """Create the deterministic occluded-crossing scene used in public demos."""

    times = np.arange(0.0, duration + 1e-9, dt)
    ego = np.column_stack((-23.0 + 8.0 * times, np.zeros_like(times)))
    # The pedestrian reaches the lane at almost the same time as the
    # constant-speed vehicle, creating a genuine near-conflict rather than a
    # decorative crossing animation.
    actor_y = np.where(times < 0.9, 5.0, 5.0 - 2.2 * (times - 0.9))
    actor = np.column_stack((np.full_like(times, 1.5), actor_y))
    return ScenarioView(
        times=times,
        ego_positions=ego,
        actor_positions=actor,
        title="Pedestrian emerging from behind a parked vehicle",
        occluders=((-0.9, 1.15, 4.1, 2.2),),
    )


def _actor_velocity(scenario: ScenarioView, index: int) -> np.ndarray:
    previous = max(0, index - 1)
    following = min(index + 1, scenario.times.size - 1)
    elapsed = scenario.times[following] - scenario.times[previous]
    return (scenario.actor_positions[following] - scenario.actor_positions[previous]) / elapsed


def _conflict_estimate(
    ego_position: np.ndarray,
    ego_speed: float,
    actor_position: np.ndarray,
    actor_velocity: np.ndarray,
) -> tuple[float, float]:
    if ego_speed <= 0.05 or abs(actor_velocity[1]) <= 0.05:
        return np.inf, 0.0
    ego_time = (actor_position[0] - ego_position[0]) / ego_speed
    actor_time = (0.0 - actor_position[1]) / actor_velocity[1]
    if ego_time <= 0 or actor_time <= 0:
        return np.inf, 0.0
    timing_gap = abs(ego_time - actor_time)
    risk = float(np.exp(-0.5 * (timing_gap / 0.75) ** 2) * np.exp(-0.06 * max(actor_position[1], 0.0)))
    return max(ego_time, actor_time), float(np.clip(risk, 0.0, 1.0))


def simulate_policy(scenario: ScenarioView, mode: str) -> PolicyTrace:
    """Run a transparent closed-loop baseline or SafeCross-style controller."""

    if mode not in {"baseline", "safecross"}:
        raise ValueError("mode must be 'baseline' or 'safecross'")
    count = scenario.times.size
    positions = np.zeros((count, 2), dtype=float)
    speeds = np.zeros(count, dtype=float)
    risks = np.zeros(count, dtype=float)
    ttc = np.full(count, np.inf, dtype=float)
    decisions: list[str] = []
    positions[0] = scenario.ego_positions[0]
    speeds[0] = 8.0
    target_speed = 8.0
    for index in range(count):
        actor_position = scenario.actor_positions[index]
        velocity = _actor_velocity(scenario, index)
        current_ttc, current_risk = _conflict_estimate(
            positions[index], speeds[index], actor_position, velocity
        )
        # A reactive baseline only treats the crossing as actionable when the
        # pedestrian has almost cleared the occluding vehicle.
        visible = actor_position[1] <= 2.15
        if mode == "baseline":
            effective_risk = current_risk if visible else 0.0
            if effective_risk >= 0.50:
                acceleration, decision = -6.0, "EMERGENCY BRAKE"
            elif effective_risk >= 0.24:
                acceleration, decision = -2.6, "BRAKE"
            else:
                acceleration, decision = 1.2 if speeds[index] < target_speed else 0.0, "CONTINUE"
        else:
            # SafeCross can act on the partially observed track and calibrated
            # uncertainty before the pedestrian is fully visible.
            uncertainty_boost = 0.20 if 2.15 < actor_position[1] < 3.75 else 0.0
            effective_risk = float(np.clip(current_risk + uncertainty_boost, 0.0, 1.0))
            if effective_risk >= 0.58:
                acceleration, decision = -3.9, "YIELD"
            elif effective_risk >= 0.28:
                acceleration, decision = -2.2, "SLOW"
            else:
                acceleration, decision = 1.0 if speeds[index] < target_speed else 0.0, "CONTINUE"
        risks[index] = effective_risk
        ttc[index] = current_ttc
        decisions.append(decision)
        if index + 1 < count:
            dt = scenario.times[index + 1] - scenario.times[index]
            next_speed = float(np.clip(speeds[index] + acceleration * dt, 0.0, target_speed))
            positions[index + 1] = positions[index] + np.array([(speeds[index] + next_speed) * 0.5 * dt, 0.0])
            speeds[index + 1] = next_speed
    distances = np.linalg.norm(positions - scenario.actor_positions, axis=1)
    return PolicyTrace(
        positions=positions,
        speeds=speeds,
        risk=risks,
        time_to_conflict=ttc,
        decisions=tuple(decisions),
        minimum_distance=float(np.min(distances)),
    )


def _style_axis(axis: plt.Axes, title: str) -> None:
    axis.set_xlim(-32, 24)
    axis.set_ylim(-7.5, 7.5)
    axis.set_aspect("equal")
    axis.set_facecolor("#10151c")
    axis.set_title(title, color="white", fontsize=14, fontweight="bold", pad=12)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.add_patch(Rectangle((-32, -3.0), 56, 6.0, color="#2b333d", zorder=0))
    axis.plot([-32, 24], [0, 0], linestyle=(0, (6, 8)), color="#89939e", linewidth=1.0, alpha=0.6)
    axis.add_patch(Rectangle((-2.6, -6.4), 8.1, 12.8, color="#d8d8d4", alpha=0.11, zorder=0))
    for x in np.linspace(-2.2, 4.8, 7):
        axis.add_patch(Rectangle((x, -3.0), 0.55, 6.0, color="#e8e8e2", alpha=0.22, zorder=0))


def _draw_occluders(axis: plt.Axes, scenario: ScenarioView) -> None:
    for x, y, width, height in scenario.occluders:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.08,rounding_size=0.35",
                facecolor="#59616b",
                edgecolor="#9ba3aa",
                linewidth=1.2,
                zorder=4,
            )
        )
        axis.text(x + width / 2, y + height / 2, "PARKED", ha="center", va="center", color="#c8cdd2", fontsize=7, zorder=5)


def _draw_vehicle(axis: plt.Axes, position: np.ndarray, color: str) -> None:
    x, y = position
    body = FancyBboxPatch(
        (x - 1.65, y - 0.85), 3.3, 1.7,
        boxstyle="round,pad=0.05,rounding_size=0.38",
        facecolor=color, edgecolor="#f4f6f8", linewidth=1.3, zorder=7,
    )
    axis.add_patch(body)
    axis.add_patch(Rectangle((x + 0.35, y - 0.62), 0.72, 1.24, facecolor="#162435", alpha=0.82, zorder=8))


def _predicted_paths(scenario: ScenarioView, index: int) -> list[np.ndarray]:
    position = scenario.actor_positions[index]
    velocity = _actor_velocity(scenario, index)
    horizon = np.linspace(0.0, 2.5, 24)
    straight = position + horizon[:, None] * velocity
    hesitate = position + horizon[:, None] * velocity * 0.38
    continue_sidewalk = position + np.column_stack((0.8 * horizon, np.zeros_like(horizon)))
    return [straight, hesitate, continue_sidewalk]


def create_comparison_animation(
    scenario: ScenarioView | Any,
    *,
    fps: int = 15,
    frame_stride: int = 1,
) -> tuple[plt.Figure, FuncAnimation]:
    """Create a baseline-versus-SafeCross animation and return figure/animation."""

    if not isinstance(scenario, ScenarioView):
        scenario = scenario_from_core(scenario)
    baseline = simulate_policy(scenario, "baseline")
    safe = simulate_policy(scenario, "safecross")
    indices = np.arange(0, scenario.times.size, max(1, frame_stride))
    figure, axes = plt.subplots(1, 2, figsize=(14.4, 6.8), facecolor="#090d12")
    figure.suptitle(
        "SafeCross | Early risk prediction for vulnerable road users",
        color="white", fontsize=18, fontweight="bold", y=0.98,
    )
    figure.text(
        0.5, 0.025,
        "Closed-loop simulation prototype | identical seeded scenario | not validated on a real vehicle",
        ha="center", color="#8996a3", fontsize=9,
    )

    def update(frame_number: int) -> list[Any]:
        index = int(indices[frame_number])
        for axis, title, trace, accent in (
            (axes[0], "Reactive baseline", baseline, "#ff6b6b"),
            (axes[1], "SafeCross", safe, "#4dd4ac"),
        ):
            axis.clear()
            _style_axis(axis, title)
            _draw_occluders(axis, scenario)
            _draw_vehicle(axis, trace.positions[index], accent)
            actor = scenario.actor_positions[index]
            axis.add_patch(Circle(actor, 0.42, facecolor="#ffd166", edgecolor="white", linewidth=1.0, zorder=9))
            axis.text(actor[0], actor[1] + 0.85, scenario.actor_kind.upper(), ha="center", color="#ffe7a5", fontsize=7, zorder=10)
            axis.plot(
                trace.positions[: index + 1, 0], trace.positions[: index + 1, 1],
                color=accent, linewidth=2.0, alpha=0.58, zorder=3,
            )
            paths = _predicted_paths(scenario, index)
            probabilities = (0.62, 0.23, 0.15)
            for path, probability, color in zip(paths, probabilities, ("#ff8c70", "#ffd166", "#6fb7ff")):
                axis.plot(path[:, 0], path[:, 1], color=color, linewidth=2.2, alpha=0.82, zorder=6)
                axis.text(path[-1, 0], path[-1, 1], f"{probability:.0%}", color=color, fontsize=7, zorder=9)
            distance = float(np.linalg.norm(trace.positions[index] - actor))
            finite_ttc = trace.time_to_conflict[index]
            ttc_text = f"{finite_ttc:.1f} s" if np.isfinite(finite_ttc) else "--"
            telemetry = (
                f"DECISION  {trace.decisions[index]}\n"
                f"RISK      {trace.risk[index]:.0%}\n"
                f"TTC       {ttc_text}\n"
                f"SPEED     {trace.speeds[index]:.1f} m/s\n"
                f"DISTANCE  {distance:.1f} m"
            )
            axis.text(
                0.025, 0.965, telemetry,
                transform=axis.transAxes, ha="left", va="top",
                color="white", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.55", facecolor="#070a0e", edgecolor=accent, alpha=0.91),
                zorder=20,
            )
            axis.text(
                0.975, 0.035,
                f"t = {scenario.times[index]:.2f} s\nmin separation {trace.minimum_distance:.2f} m",
                transform=axis.transAxes, ha="right", va="bottom",
                color="#c5ccd3", fontsize=8,
            )
        return []

    animation = FuncAnimation(
        figure,
        update,
        frames=len(indices),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    return figure, animation


def render_comparison_video(
    scenario: ScenarioView | Any,
    output: str | Path,
    *,
    fps: int = 15,
    dpi: int = 120,
    frame_stride: int = 1,
) -> Path:
    """Render an MP4 when FFmpeg exists, otherwise fall back to GIF."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, animation = create_comparison_animation(scenario, fps=fps, frame_stride=frame_stride)
    try:
        if output_path.suffix.lower() == ".mp4" and shutil.which("ffmpeg"):
            writer = FFMpegWriter(fps=fps, codec="h264", bitrate=2400, extra_args=["-pix_fmt", "yuv420p"])
            animation.save(output_path, writer=writer, dpi=dpi)
        else:
            if output_path.suffix.lower() != ".gif":
                output_path = output_path.with_suffix(".gif")
            animation.save(output_path, writer=PillowWriter(fps=fps), dpi=min(dpi, 100))
    finally:
        plt.close(figure)
    return output_path

