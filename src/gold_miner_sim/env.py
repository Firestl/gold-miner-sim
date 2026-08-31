"""Minimal, deterministic Gold Miner Gymnasium environment (Milestone 1).

One ``step()`` advances the simulation by exactly one physics tick (1/60 s).
No wall-clock time, no reward shaping. The map layout depends on
``map_mode``: ``"fixed"`` (default) is fully deterministic, while
``"random"`` draws three spawn points without replacement from
RANDOM_SPAWN_POINTS (or from a caller-supplied ``spawn_points`` pool)
through the Gymnasium-seeded ``np_random`` RNG, so a given seed always
reproduces the same maps.
"""

from __future__ import annotations

import enum
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

if TYPE_CHECKING:
    from gold_miner_sim.renderer import HumanRenderer

# ---------------------------------------------------------------------------
# Canvas / timing
# ---------------------------------------------------------------------------
WIDTH = 900
HEIGHT = 600
ANCHOR = (450.0, 70.0)  # Hook anchor point in px (x, y).
SIM_FPS = 60
DT = 1.0 / SIM_FPS
EPISODE_TIME = 60.0  # seconds per episode

# ---------------------------------------------------------------------------
# Hook parameters (angles in degrees, lengths in px, speeds px/s)
# ---------------------------------------------------------------------------
INITIAL_ANGLE = -70.0
MIN_ANGLE = -70.0
MAX_ANGLE = 70.0
SWING_ANGULAR_SPEED = 60.0  # deg/s
MIN_ROPE_LENGTH = 50.0
MAX_ROPE_LENGTH = 460.0
HOOK_RADIUS = 6.0
EXTENSION_SPEED = 320.0
EMPTY_RETRACT_SPEED = 360.0

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
WAIT = 0
FIRE = 1


class HookState(enum.Enum):
    """Phases of the hook. Only SWINGING accepts the FIRE action."""

    SWINGING = "swinging"
    EXTENDING = "extending"
    RETRACT_EMPTY = "retract_empty"
    RETRACT_LOADED = "retract_loaded"


class ObjectType(enum.Enum):
    GOLD = "gold"
    DIAMOND = "diamond"
    ROCK = "rock"


@dataclass
class GameObject:
    """A collectible circular object on the map."""

    type: ObjectType
    x: float
    y: float
    radius: float
    value: float
    retract_speed: float  # Loaded retract speed (px/s).
    active: bool = True


# ---------------------------------------------------------------------------
# Map layout (shared by both map modes)
# ---------------------------------------------------------------------------
# Object specs in fixed slot order: (type, radius, value, retract_speed).
# Only the center position varies between map modes.
_OBJECT_SPECS: tuple[tuple[ObjectType, float, float, float], ...] = (
    (ObjectType.GOLD, 30.0, 250.0, 140.0),
    (ObjectType.DIAMOND, 18.0, 500.0, 280.0),
    (ObjectType.ROCK, 34.0, 50.0, 90.0),
)

# Fixed-map centers (V0 contract) in the same slot order; do not change.
_FIXED_POSITIONS: tuple[tuple[float, float], ...] = (
    (315.0, 300.0),
    (465.0, 450.0),
    (610.0, 340.0),
)

# Candidate centers (x, y) in px for map_mode="random": each reset draws
# three points without replacement and assigns them, in slot order, to
# GOLD / DIAMOND / ROCK.
RANDOM_SPAWN_POINTS = (
    (173.0, 230.0),
    (159.0, 314.0),
    (283.0, 269.0),
    (250.0, 416.0),
    (347.0, 352.0),
    (387.0, 425.0),
    (450.0, 490.0),
    (499.0, 346.0),
    (583.0, 436.0),
    (615.0, 356.0),
    (714.0, 384.0),
    (672.0, 256.0),
)


def _freeze_spawn_pool(
    spawn_points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Validate a caller-supplied random-mode spawn pool and freeze it.

    Returns a fresh ``tuple`` of ``(x, y)`` float pairs so later mutation of
    the caller's sequence cannot affect the environment. Raises
    ``ValueError`` unless ``spawn_points`` is an iterable of at least three
    numeric length-2 ``(x, y)`` elements. Coordinates are converted with
    ``float()`` (value-exact, never rounded).
    """
    try:
        points = list(spawn_points)
    except TypeError:
        raise ValueError(
            f"spawn_points must be an iterable of (x, y) pairs, got {spawn_points!r}"
        ) from None
    if len(points) < 3:
        raise ValueError(
            f"spawn_points must contain at least 3 points, got {len(points)}"
        )
    frozen: list[tuple[float, float]] = []
    for point in points:
        try:
            x, y = point
        except (TypeError, ValueError):
            raise ValueError(
                f"each spawn point must be an (x, y) pair, got {point!r}"
            ) from None
        if not isinstance(x, numbers.Real) or not isinstance(y, numbers.Real):
            # ValueError is the documented issue #21 contract for pool
            # validation (same as map_mode/render_mode), not a TypeError.
            raise ValueError(  # noqa: TRY004
                f"spawn point coordinates must be numeric, got {point!r}"
            )
        frozen.append((float(x), float(y)))
    return tuple(frozen)


def sweep_circle_hit(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    cx: float,
    cy: float,
    r: float,
) -> float | None:
    """Return the first-hit parameter ``t`` in [0, 1] of the segment
    ``(x0, y0) -> (x1, y1)`` against the circle ``(cx, cy, r)``,
    or ``None`` when the segment does not intersect the circle.

    Solves ``|P0 + t*d - C|^2 = r^2`` for the smallest root in [0, 1].
    Returns 0.0 when P0 already lies inside the circle.
    """
    dx = x1 - x0
    dy = y1 - y0
    fx = x0 - cx
    fy = y0 - cy

    # Start point already inside the circle.
    c = fx * fx + fy * fy - r * r
    if c <= 0.0:
        return 0.0

    a = dx * dx + dy * dy
    if a <= 1e-12:
        # Degenerate segment: reduce to a point-in-circle test.
        return 0.0 if c <= 0.0 else None

    b = 2.0 * (fx * dx + fy * dy)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    if 0.0 <= t1 <= 1.0:
        return t1
    t2 = (-b + sqrt_disc) / (2.0 * a)
    if 0.0 <= t2 <= 1.0:
        return t2
    return None


class GoldMinerEnv(gymnasium.Env[NDArray[np.float32], int]):
    """Single-agent Gold Miner environment (V0) with a fixed or random map."""

    metadata = {"render_modes": ["human"], "render_fps": 60}  # noqa: RUF012

    def __init__(
        self,
        render_mode: str | None = None,
        map_mode: str = "fixed",
        spawn_points: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError(
                f"render_mode must be None or 'human', got {render_mode!r}"
            )
        if map_mode not in ("fixed", "random"):
            raise ValueError(f"map_mode must be 'fixed' or 'random', got {map_mode!r}")
        if map_mode == "fixed" and spawn_points is not None:
            raise ValueError(
                "spawn_points is only supported with map_mode='random', got "
                f"map_mode='fixed' with spawn_points={spawn_points!r}"
            )
        self.render_mode = render_mode
        self.map_mode = map_mode
        # Spawn pool for map_mode="random": either the historical
        # RANDOM_SPAWN_POINTS constant (spawn_points=None, bitwise-identical
        # legacy behavior) or a caller-supplied pool frozen at construction
        # time so later caller mutation cannot affect the environment.
        if spawn_points is None:
            self._spawn_pool: tuple[tuple[float, float], ...] = RANDOM_SPAWN_POINTS
        else:
            self._spawn_pool = _freeze_spawn_pool(spawn_points)
        self.action_space = spaces.Discrete(2)
        # 8 hook/global values + 3 objects * 6 values, all normalized to [-1, 1].
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(26,), dtype=np.float32
        )

        # Public game state; renderers and tests read these directly.
        self.score: float = 0.0
        self.remaining_time: float = EPISODE_TIME
        self.hook_state: HookState = HookState.SWINGING
        self.angle: float = INITIAL_ANGLE  # degrees; 0 = straight down
        self.swing_direction: int = 1
        self.rope_length: float = MIN_ROPE_LENGTH
        self.attached_object: GameObject | None = None
        self.objects: list[GameObject] = []

        self._ticks: int = 0  # Integer tick counter; avoids float time drift.
        self._renderer: HumanRenderer | None = None

    @property
    def hook_tip(self) -> tuple[float, float]:
        """Current hook tip position (px)."""
        rad = math.radians(self.angle)
        x = ANCHOR[0] + self.rope_length * math.sin(rad)
        y = ANCHOR[1] + self.rope_length * math.cos(rad)
        return (x, y)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.score = 0.0
        self.remaining_time = EPISODE_TIME
        self.hook_state = HookState.SWINGING
        self.angle = INITIAL_ANGLE
        self.swing_direction = 1
        self.rope_length = MIN_ROPE_LENGTH
        self.attached_object = None
        self._ticks = 0
        # Fresh dataclass instances on every reset, fixed slot order
        # (GOLD / DIAMOND / ROCK); map_mode only picks the centers.
        positions: tuple[tuple[float, float], ...]
        if self.map_mode == "fixed":
            positions = _FIXED_POSITIONS
        else:
            # Three distinct spawn points without replacement, drawn from the
            # Gymnasium-seeded RNG so a given seed always yields the same
            # maps. Coordinates are used verbatim (exact tuple constants).
            # self._spawn_pool is RANDOM_SPAWN_POINTS unless a custom pool was
            # injected, so the legacy RNG call is unchanged in the default
            # path (same choice() arguments, same single RNG draw).
            indices = self.np_random.choice(
                len(self._spawn_pool), size=3, replace=False
            )
            positions = tuple(self._spawn_pool[int(i)] for i in indices)
        self.objects = [
            GameObject(
                obj_type,
                x=x,
                y=y,
                radius=radius,
                value=value,
                retract_speed=speed,
            )
            for (obj_type, radius, value, speed), (x, y) in zip(
                _OBJECT_SPECS, positions
            )
        ]
        # First frame for human mode (a no-op when render_mode is None).
        self.render()
        return self._observation(), self._info()

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # Validate against action_space so both Python ints and NumPy
        # integers (Discrete.sample() returns np.int64) are accepted, and
        # out-of-range actions fail loudly instead of acting as WAIT.
        if not self.action_space.contains(action):
            raise ValueError(
                f"invalid action {action!r}, expected 0 (WAIT) or 1 (FIRE)"
            )

        old_score = self.score

        # 1) Action handling. Only SWINGING accepts FIRE; entering EXTENDING
        #    freezes the angle immediately (no swing update on this tick).
        if self.hook_state is HookState.SWINGING and action == FIRE:
            self.hook_state = HookState.EXTENDING

        # 2) Advance exactly one physics tick.
        if self.hook_state is HookState.SWINGING:
            self._advance_swinging()
        elif self.hook_state is HookState.EXTENDING:
            self._advance_extending()
        elif self.hook_state is HookState.RETRACT_EMPTY:
            self._advance_retract_empty()
        else:  # HookState.RETRACT_LOADED
            self._advance_retract_loaded()

        # 3) Time via tick counter so remaining_time is exactly 0.0 at
        #    tick 3600 (ticks/60 instead of accumulating -dt).
        self._ticks += 1
        self.remaining_time = max(0.0, EPISODE_TIME - self._ticks / SIM_FPS)

        # 4) Reward, episode-end flags, observation, info.
        reward = self.score - old_score
        truncated = self.remaining_time <= 0.0
        # Auto-render for human mode (no-op when render_mode is None), so
        # callers no longer need to invoke render() after every step.
        self.render()
        return self._observation(), reward, False, truncated, self._info()

    def render(self) -> None:
        if self.render_mode != "human":
            return
        if self._renderer is None:
            # Imported lazily to avoid a circular import with the renderer.
            from gold_miner_sim.renderer import HumanRenderer

            self._renderer = HumanRenderer(self)
        self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()

    @property
    def window_closed(self) -> bool:
        """True after the human-render window has been closed by the user."""
        return self._renderer is not None and self._renderer.closed

    # ------------------------------------------------------------------
    # Per-state physics (one tick each)
    # ------------------------------------------------------------------
    def _advance_swinging(self) -> None:
        self.angle += self.swing_direction * SWING_ANGULAR_SPEED * DT
        if self.angle >= MAX_ANGLE:
            self.angle = MAX_ANGLE
            self.swing_direction = -1
        elif self.angle <= MIN_ANGLE:
            self.angle = MIN_ANGLE
            self.swing_direction = 1

    def _advance_extending(self) -> None:
        prev_rope = self.rope_length
        prev_tip = self.hook_tip
        # Clamp this tick's candidate rope length first: the collision sweep
        # must only cover the legal path (up to MAX_ROPE_LENGTH), never the
        # overshoot a raw ``prev + speed * dt`` step would produce.
        new_rope = min(prev_rope + EXTENSION_SPEED * DT, MAX_ROPE_LENGTH)
        rad = math.radians(self.angle)
        new_tip_x = ANCHOR[0] + new_rope * math.sin(rad)
        new_tip_y = ANCHOR[1] + new_rope * math.cos(rad)

        # Swept collision over this tick's tip path; take the earliest hit.
        best_t: float = math.inf
        hit_object: GameObject | None = None
        for obj in self.objects:
            if not obj.active:
                continue
            t = sweep_circle_hit(
                prev_tip[0],
                prev_tip[1],
                new_tip_x,
                new_tip_y,
                obj.x,
                obj.y,
                obj.radius + HOOK_RADIUS,
            )
            if t is not None and t < best_t:
                best_t = t
                hit_object = obj

        if hit_object is not None:
            # Stop the rope at the first-contact position, then immediately
            # snap the object onto the (new) hook tip: its center must follow
            # the tip from the hit tick on, so the observation / render of
            # this step is already consistent. Order matters: hook_tip is
            # read only after rope_length has been updated.
            self.rope_length = prev_rope + best_t * (new_rope - prev_rope)
            self.attached_object = hit_object
            hit_object.x, hit_object.y = self.hook_tip
            self.hook_state = HookState.RETRACT_LOADED
        elif new_rope >= MAX_ROPE_LENGTH:
            self.rope_length = MAX_ROPE_LENGTH
            self.hook_state = HookState.RETRACT_EMPTY
        else:
            self.rope_length = new_rope

    def _advance_retract_empty(self) -> None:
        self.rope_length -= EMPTY_RETRACT_SPEED * DT
        if self.rope_length <= MIN_ROPE_LENGTH:
            self.rope_length = MIN_ROPE_LENGTH
            # Swing direction is kept from before the launch (no reset).
            self.hook_state = HookState.SWINGING

    def _advance_retract_loaded(self) -> None:
        obj = self.attached_object
        assert obj is not None  # RETRACT_LOADED always carries an object.
        self.rope_length -= obj.retract_speed * DT
        reached_top = self.rope_length <= MIN_ROPE_LENGTH
        self.rope_length = max(self.rope_length, MIN_ROPE_LENGTH)
        # The attached object's center always follows the hook tip.
        obj.x, obj.y = self.hook_tip
        if reached_top:
            # Score only when the object is fully retracted to the top.
            self.score += obj.value
            obj.active = False
            self.attached_object = None
            self.hook_state = HookState.SWINGING

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------
    def _observation(self) -> NDArray[np.float32]:
        # Global / hook: 8 values.
        obs: list[float] = [
            self.angle / MAX_ANGLE,  # normalized hook angle
            float(self.swing_direction),  # -1 or +1
            1.0 if self.hook_state is HookState.SWINGING else 0.0,
            1.0 if self.hook_state is HookState.EXTENDING else 0.0,
            1.0 if self.hook_state is HookState.RETRACT_EMPTY else 0.0,
            1.0 if self.hook_state is HookState.RETRACT_LOADED else 0.0,
            self.rope_length / MAX_ROPE_LENGTH,  # normalized rope length
            self.remaining_time / EPISODE_TIME,  # normalized remaining time
        ]
        # Objects: 3 fixed slots (GOLD / DIAMOND / ROCK) * 6 values.
        for obj in self.objects:
            if obj.active:
                obs.extend(
                    [
                        obj.x / WIDTH,
                        obj.y / HEIGHT,
                        obj.radius / 100.0,
                        obj.value / 500.0,
                        obj.retract_speed / EMPTY_RETRACT_SPEED,
                        1.0,  # active flag
                    ]
                )
            else:
                obs.extend([0.0] * 6)
        return np.asarray(obs, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "remaining_time": self.remaining_time,
            "hook_state": self.hook_state.name,
        }
