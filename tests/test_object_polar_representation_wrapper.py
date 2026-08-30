"""行为测试 for :class:`ObjectPolarRepresentationWrapper` (Issue #17).

覆盖：shape/dtype、角度约定（atan2(dx, dy)，0° 正下 / 左负 / 右正）、
3-4-5 距离、active object 转换、inactive block 保持全 0、其余 21 维
不变、reward/flags/info 透传、不原地修改、observation_space.contains、
Cartesian ↔ Polar round-trip、Full/Polar lockstep 与 factory 集成。
"""

from __future__ import annotations

import math
from typing import Any, cast

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces
from numpy.typing import NDArray

from gold_miner_sim.benchmark import OBSERVATION_MODES, make_benchmark_env
from gold_miner_sim.env import (
    ANCHOR,
    FIRE,
    HEIGHT,
    MAX_ANGLE,
    MAX_ROPE_LENGTH,
    RANDOM_SPAWN_POINTS,
    WAIT,
    WIDTH,
    GoldMinerEnv,
)
from gold_miner_sim.wrappers import (
    OBJECT_POLAR_BLOCKS,
    OBJECT_POSITION_INDICES,
    FireBudgetWrapper,
    ObjectPolarRepresentationWrapper,
    ObjectPositionMaskWrapper,
    SwingAdvanceDecisionWrapper,
)

# The 21 observation slots that must survive the polar transform untouched.
OTHER_INDICES = [i for i in range(27) if i not in OBJECT_POSITION_INDICES]

GOLD_BLOCK = OBJECT_POLAR_BLOCKS[0]
DIAMOND_BLOCK = OBJECT_POLAR_BLOCKS[1]
ROCK_BLOCK = OBJECT_POLAR_BLOCKS[2]


def cartesian_to_polar(x_px: float, y_px: float) -> tuple[float, float]:
    """Forward reference transform: normalized (angle, distance) from px."""
    dx = x_px - ANCHOR[0]
    dy = y_px - ANCHOR[1]
    angle_deg = math.degrees(math.atan2(dx, dy))
    distance_px = math.hypot(dx, dy)
    return angle_deg / MAX_ANGLE, distance_px / MAX_ROPE_LENGTH


def polar_to_cartesian(
    angle_feature: float, distance_feature: float
) -> tuple[float, float]:
    """Inverse transform from the Issue #17 definition (px x, y)."""
    angle_deg = angle_feature * MAX_ANGLE
    distance = distance_feature * MAX_ROPE_LENGTH
    rad = math.radians(angle_deg)
    x_px = ANCHOR[0] + distance * math.sin(rad)
    y_px = ANCHOR[1] + distance * math.cos(rad)
    return x_px, y_px


def build_observation(
    positions: tuple[tuple[float, float] | None, ...],
) -> NDArray[np.float32]:
    """Build a 27-dim inner observation from per-block center positions.

    ``positions`` holds one entry per object block (GOLD / DIAMOND / ROCK);
    ``None`` marks the block inactive and zeroes it exactly like
    ``GoldMinerEnv``. Active blocks store normalized px x/y plus an active
    flag of 1.0; every non-position slot gets a non-zero 0.25 filler so
    passthrough regressions are visible.
    """
    observation = np.full(27, 0.25, dtype=np.float32)
    for (x_index, y_index, active_index), position in zip(
        OBJECT_POLAR_BLOCKS, positions
    ):
        if position is None:
            observation[x_index] = 0.0
            observation[y_index] = 0.0
            observation[active_index] = 0.0
        else:
            observation[x_index] = position[0] / WIDTH
            observation[y_index] = position[1] / HEIGHT
            observation[active_index] = 1.0
    return observation


class ScriptedPolarEnv(gymnasium.Env[NDArray[np.float32], int]):
    """Deterministic 27-dim inner env with a fixed observation.

    Returns the same observation on every reset/step and yields scripted
    (reward, terminated, truncated, info) transitions in step order
    (defaults 0.0 / False / False / {}). Passes reset seed/options through
    for passthrough assertions.
    """

    def __init__(
        self,
        observation: NDArray[np.float32],
        transitions: list[tuple[float, bool, bool, dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(27,), dtype=np.float32
        )
        self._observation = observation
        self.transitions = transitions if transitions is not None else []
        self.actions: list[int] = []
        self.reset_seed: int | None = None
        self.reset_options: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.reset_seed = seed
        self.reset_options = options
        self.actions = []
        return self._observation, {"inner_reset": True}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        self.actions.append(int(action))
        index = len(self.actions) - 1
        if index < len(self.transitions):
            reward, terminated, truncated, info = self.transitions[index]
        else:
            reward, terminated, truncated, info = 0.0, False, False, {}
        return (
            self._observation,
            reward,
            terminated,
            truncated,
            dict(info),
        )


class PersistentObsEnv(gymnasium.Env[NDArray[np.float32], int]):
    """Inner env returning the same persistent observation buffer object."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(27,), dtype=np.float32
        )
        self.buffer: NDArray[np.float32] = build_observation(
            ((315.0, 300.0), None, None)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        return self.buffer, {}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        return self.buffer, 0.0, False, False, {}


class DiscreteObsEnv(gymnasium.Env[NDArray[np.float32], int]):
    """Inner env with a non-Box observation space (rejection test)."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)


def wrapper_chain(
    env: gymnasium.Env[Any, Any],
) -> list[gymnasium.Wrapper[Any, Any, Any, Any]]:
    """Return every wrapper of the chain from outermost to the base env."""
    chain: list[gymnasium.Wrapper[Any, Any, Any, Any]] = []
    current: gymnasium.Env[Any, Any] = env
    while isinstance(current, gymnasium.Wrapper):
        chain.append(current)
        current = current.env
    return chain


def assert_lockstep(
    full_observation: NDArray[np.float32],
    polar_observation: NDArray[np.float32],
) -> None:
    """Assert non-position slots match and polar slots invert to full x/y."""
    assert np.array_equal(
        full_observation[OTHER_INDICES], polar_observation[OTHER_INDICES]
    )
    for x_index, y_index, active_index in OBJECT_POLAR_BLOCKS:
        if full_observation[active_index] <= 0.5:
            assert polar_observation[x_index] == 0.0
            assert polar_observation[y_index] == 0.0
            continue
        x_px, y_px = polar_to_cartesian(
            float(polar_observation[x_index]),
            float(polar_observation[y_index]),
        )
        assert x_px == pytest.approx(float(full_observation[x_index]) * WIDTH, abs=1e-2)
        assert y_px == pytest.approx(
            float(full_observation[y_index]) * HEIGHT, abs=1e-2
        )


# ---------------------------------------------------------------------------
# 20.1 shape / dtype
# ---------------------------------------------------------------------------


def test_reset_shape_and_dtype() -> None:
    wrapped = make_benchmark_env("polar")

    observation, _info = wrapped.reset(seed=42)

    assert observation.shape == (27,)
    assert observation.dtype == np.float32


def test_step_shape_and_dtype() -> None:
    wrapped = make_benchmark_env("polar")
    wrapped.reset(seed=42)

    observation, _reward, _terminated, _truncated, _info = wrapped.step(WAIT)

    assert observation.shape == (27,)
    assert observation.dtype == np.float32


# ---------------------------------------------------------------------------
# 20.2 angle convention (0 = straight down, negative left, positive right)
# ---------------------------------------------------------------------------


def test_angle_zero_directly_below_anchor() -> None:
    # (450, 170) is exactly 100 px below ANCHOR (450, 70): angle must be 0,
    # not the 90 deg that the wrong atan2(dy, dx) convention would produce.
    inner = ScriptedPolarEnv(build_observation(((450.0, 170.0), None, None)))
    wrapped = ObjectPolarRepresentationWrapper(inner)

    observation, _info = wrapped.reset()

    assert observation[8] == pytest.approx(0.0)
    assert observation[9] == pytest.approx(100.0 / MAX_ROPE_LENGTH)


def test_angle_negative_left_of_anchor() -> None:
    # (350, 170): dx=-100, dy=100 -> -45 deg. The wrong atan2(dy, dx)
    # convention would yield +135 deg, i.e. a value above 1.0.
    inner = ScriptedPolarEnv(build_observation(((350.0, 170.0), None, None)))
    wrapped = ObjectPolarRepresentationWrapper(inner)

    observation, _info = wrapped.reset()

    assert observation[8] == pytest.approx(-45.0 / MAX_ANGLE)
    assert observation[8] < 0.0


def test_angle_positive_right_of_anchor() -> None:
    # (550, 170): dx=100, dy=100 -> +45 deg.
    inner = ScriptedPolarEnv(build_observation(((550.0, 170.0), None, None)))
    wrapped = ObjectPolarRepresentationWrapper(inner)

    observation, _info = wrapped.reset()

    assert observation[8] == pytest.approx(45.0 / MAX_ANGLE)


# ---------------------------------------------------------------------------
# 20.3 distance (3-4-5 triangle)
# ---------------------------------------------------------------------------


def test_distance_345_triangle() -> None:
    # (510, 150): dx=60, dy=80 -> 3-4-5 triangle scaled by 20 => 100 px.
    inner = ScriptedPolarEnv(build_observation(((510.0, 150.0), None, None)))
    wrapped = ObjectPolarRepresentationWrapper(inner)

    observation, _info = wrapped.reset()

    assert observation[9] == pytest.approx(100.0 / MAX_ROPE_LENGTH)
    # atan2(3, 4) = 36.8699 deg in the environment's angle convention.
    assert observation[8] == pytest.approx(36.8699 / MAX_ANGLE)


# ---------------------------------------------------------------------------
# 20.4 active objects are transformed
# ---------------------------------------------------------------------------


def test_all_active_objects_transformed() -> None:
    positions = ((315.0, 300.0), (465.0, 450.0), (610.0, 340.0))
    inner = ScriptedPolarEnv(build_observation(positions))
    wrapped = ObjectPolarRepresentationWrapper(inner)

    observation, _info = wrapped.reset()

    for (x_index, y_index, _active_index), (x_px, y_px) in zip(
        OBJECT_POLAR_BLOCKS, positions
    ):
        # Cartesian values must actually be replaced.
        assert observation[x_index] != pytest.approx(x_px / WIDTH)
        assert observation[y_index] != pytest.approx(y_px / HEIGHT)
        # And the written polar values must round-trip to the same center.
        angle_feature = float(observation[x_index])
        distance_feature = float(observation[y_index])
        assert -1.0 <= angle_feature <= 1.0
        assert 0.0 <= distance_feature <= 1.0
        got_x, got_y = polar_to_cartesian(angle_feature, distance_feature)
        assert got_x == pytest.approx(x_px, abs=1e-2)
        assert got_y == pytest.approx(y_px, abs=1e-2)


def test_transform_applies_on_every_step() -> None:
    inner = ScriptedPolarEnv(build_observation(((450.0, 170.0), None, None)))
    wrapped = ObjectPolarRepresentationWrapper(inner)
    wrapped.reset()

    observation, _reward, _terminated, _truncated, _info = wrapped.step(WAIT)

    assert observation[8] == pytest.approx(0.0)
    assert observation[9] == pytest.approx(100.0 / MAX_ROPE_LENGTH)


# ---------------------------------------------------------------------------
# 20.5 inactive blocks stay all zero / untouched
# ---------------------------------------------------------------------------


def test_inactive_blocks_remain_all_zero() -> None:
    # Realistic inner contract: inactive blocks are entirely zero while
    # GOLD is active; the zero blocks must never gain polar features.
    inner_observation = build_observation(((315.0, 300.0), None, None))
    inner = ScriptedPolarEnv(inner_observation)
    wrapped = ObjectPolarRepresentationWrapper(inner)

    observation, _info = wrapped.reset()

    for x_index, y_index, active_index in (DIAMOND_BLOCK, ROCK_BLOCK):
        assert observation[active_index] == 0.0
        assert observation[x_index] == 0.0
        assert observation[y_index] == 0.0


def test_inactive_block_is_never_polarized() -> None:
    # Even if an inactive block carried non-zero x/y (a contract violation
    # the wrapper must not amplify), the wrapper must leave the block
    # untouched instead of interpreting (x, y) as a target.
    inner_observation = build_observation((None, None, None))
    for x_index, y_index, active_index in OBJECT_POLAR_BLOCKS:
        inner_observation[x_index] = 0.7
        inner_observation[y_index] = 0.6
        inner_observation[active_index] = 0.0
    inner = ScriptedPolarEnv(inner_observation)
    wrapped = ObjectPolarRepresentationWrapper(inner)

    observation, _info = wrapped.reset()

    for x_index, y_index, active_index in OBJECT_POLAR_BLOCKS:
        assert observation[active_index] == 0.0
        assert observation[x_index] == 0.7
        assert observation[y_index] == 0.6


# ---------------------------------------------------------------------------
# 20.6 the other 21 dimensions are unchanged
# ---------------------------------------------------------------------------


def test_other_dimensions_unchanged() -> None:
    inner_observation = build_observation(((315.0, 300.0), (465.0, 450.0), None))
    inner = ScriptedPolarEnv(inner_observation)
    wrapped = ObjectPolarRepresentationWrapper(inner)

    observation, _info = wrapped.reset()

    assert np.array_equal(observation[OTHER_INDICES], inner_observation[OTHER_INDICES])
    # Includes the radius/value/speed slots of active blocks (10-12, 16-18).
    for block_index in (10, 11, 12, 16, 17, 18):
        assert observation[block_index] == inner_observation[block_index]


# ---------------------------------------------------------------------------
# 20.7 reward / flags / info passthrough
# ---------------------------------------------------------------------------


def test_reward_passthrough() -> None:
    inner = ScriptedPolarEnv(
        build_observation(((315.0, 300.0), None, None)),
        transitions=[(1.5, False, False, {})],
    )
    wrapped = ObjectPolarRepresentationWrapper(inner)
    wrapped.reset()

    _observation, reward, _terminated, _truncated, _info = wrapped.step(WAIT)

    assert reward is inner.transitions[0][0]  # same float object, no copy
    assert reward == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [(True, False), (False, True)],
)
def test_terminated_truncated_passthrough(terminated: bool, truncated: bool) -> None:
    inner = ScriptedPolarEnv(
        build_observation(((315.0, 300.0), None, None)),
        transitions=[(2.0, terminated, truncated, {})],
    )
    wrapped = ObjectPolarRepresentationWrapper(inner)
    wrapped.reset()

    _observation, _reward, out_terminated, out_truncated, _info = wrapped.step(WAIT)

    assert out_terminated is terminated
    assert out_truncated is truncated


def test_info_passthrough() -> None:
    inner = ScriptedPolarEnv(
        build_observation(((315.0, 300.0), None, None)),
        transitions=[(0.0, False, False, {"marker": "x"})],
    )
    wrapped = ObjectPolarRepresentationWrapper(inner)
    wrapped.reset()

    _observation, _reward, _terminated, _truncated, info = wrapped.step(WAIT)

    assert info == {"marker": "x"}


def test_reset_passes_seed_options_and_info_through() -> None:
    inner = ScriptedPolarEnv(build_observation((None, None, None)))
    wrapped = ObjectPolarRepresentationWrapper(inner)

    _observation, info = wrapped.reset(seed=17, options={"episode": 2})

    assert inner.reset_seed == 17
    assert inner.reset_options == {"episode": 2}
    assert info == {"inner_reset": True}


def test_action_space_unchanged() -> None:
    inner = ScriptedPolarEnv(build_observation((None, None, None)))
    wrapped = ObjectPolarRepresentationWrapper(inner)

    assert wrapped.action_space == inner.action_space
    assert wrapped.action_space == spaces.Discrete(2)


# ---------------------------------------------------------------------------
# 20.8 / 20.9 spaces and buffers
# ---------------------------------------------------------------------------


def test_observation_space_contract() -> None:
    inner = ScriptedPolarEnv(build_observation((None, None, None)))
    wrapped = ObjectPolarRepresentationWrapper(inner)

    space = wrapped.observation_space
    assert isinstance(space, spaces.Box)
    assert space.shape == (27,)
    assert space.dtype == np.float32
    observation, _info = wrapped.reset()
    assert space.contains(observation)
    inner_space = inner.observation_space
    assert isinstance(inner_space, spaces.Box)
    assert np.array_equal(space.low, inner_space.low)
    assert np.array_equal(space.high, inner_space.high)


def test_no_inplace_modification_of_inner_observation() -> None:
    inner = PersistentObsEnv()
    wrapped = ObjectPolarRepresentationWrapper(inner)

    reset_observation, _info = wrapped.reset(seed=1)
    assert reset_observation is not inner.buffer
    # Inner buffer keeps Cartesian data; returned obs holds polar data.
    assert inner.buffer[8] == pytest.approx(315.0 / WIDTH)
    assert inner.buffer[9] == pytest.approx(300.0 / HEIGHT)
    assert reset_observation[8] != pytest.approx(315.0 / WIDTH)
    reset_observation[8] = 99.0  # mutating the returned obs must not leak

    step_observation, _reward, _terminated, _truncated, _info = wrapped.step(WAIT)
    assert step_observation is not inner.buffer
    assert step_observation is not reset_observation
    assert inner.buffer[8] == pytest.approx(315.0 / WIDTH)
    assert inner.buffer[9] == pytest.approx(300.0 / HEIGHT)
    assert step_observation[8] != pytest.approx(315.0 / WIDTH)


def test_outputs_contained_in_observation_space() -> None:
    """Real-chain reset and step outputs (incl. FIREs) stay in the space."""
    env = make_benchmark_env("polar")
    observation, _info = env.reset(seed=42)
    assert env.observation_space.contains(observation)
    for action in (WAIT, WAIT, FIRE, WAIT, FIRE, FIRE, WAIT):
        observation, _reward, terminated, truncated, _info = env.step(action)
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# 20.10 Cartesian <-> Polar round trip
# ---------------------------------------------------------------------------


def test_round_trip_over_all_random_spawn_points() -> None:
    """Wrapper output inverts back to every benchmark spawn center."""
    for spawn_x, spawn_y in RANDOM_SPAWN_POINTS:
        inner = ScriptedPolarEnv(build_observation(((spawn_x, spawn_y), None, None)))
        wrapped = ObjectPolarRepresentationWrapper(inner)

        observation, _info = wrapped.reset()

        x_px, y_px = polar_to_cartesian(float(observation[8]), float(observation[9]))
        assert x_px == pytest.approx(spawn_x, abs=1e-2)
        assert y_px == pytest.approx(spawn_y, abs=1e-2)


def test_round_trip_matches_reference_forward_transform() -> None:
    position = (583.0, 436.0)
    expected_angle, expected_distance = cartesian_to_polar(*position)
    inner = ScriptedPolarEnv(build_observation((position, None, None)))
    wrapped = ObjectPolarRepresentationWrapper(inner)

    observation, _info = wrapped.reset()

    assert observation[8] == pytest.approx(expected_angle)
    assert observation[9] == pytest.approx(expected_distance)


# ---------------------------------------------------------------------------
# 20.11 Full / Polar lockstep
# ---------------------------------------------------------------------------


def test_full_polar_lockstep_dynamics_unchanged() -> None:
    """Same seed + same actions: identical rewards, flags, info and slots."""
    full_env = make_benchmark_env("full")
    polar_env = make_benchmark_env("polar")
    full_obs, full_info = full_env.reset(seed=42)
    polar_obs, polar_info = polar_env.reset(seed=42)
    assert polar_info == full_info
    assert_lockstep(full_obs, polar_obs)
    # Sanity: the full chain's position slots are real Cartesian values, so
    # the lockstep comparison is actually meaningful.
    assert np.any(full_obs[list(OBJECT_POSITION_INDICES)] != 0.0)

    for action in (WAIT, WAIT, FIRE, WAIT, FIRE, FIRE, WAIT):
        full_obs, full_reward, full_done, full_trunc, full_info = full_env.step(action)
        polar_obs, polar_reward, polar_done, polar_trunc, polar_info = polar_env.step(
            action
        )
        assert full_reward == polar_reward
        assert full_done == polar_done
        assert full_trunc == polar_trunc
        assert polar_info == full_info
        assert_lockstep(full_obs, polar_obs)


# ---------------------------------------------------------------------------
# 20.12 factory regression
# ---------------------------------------------------------------------------


def test_factory_observation_modes_tuple() -> None:
    assert OBSERVATION_MODES == ("full", "blind", "polar")


def test_factory_polar_chain_wraps_representation_outside_budget() -> None:
    env = make_benchmark_env("polar")

    assert isinstance(env, ObjectPolarRepresentationWrapper)
    assert isinstance(env.env, FireBudgetWrapper)
    assert isinstance(env.env.env, SwingAdvanceDecisionWrapper)


def test_factory_full_chain_has_no_representation_wrapper() -> None:
    env = make_benchmark_env("full")

    chain = wrapper_chain(env)
    assert isinstance(env, FireBudgetWrapper)
    assert isinstance(env.env, SwingAdvanceDecisionWrapper)
    assert not any(
        isinstance(
            wrapper, (ObjectPositionMaskWrapper, ObjectPolarRepresentationWrapper)
        )
        for wrapper in chain
    )


def test_factory_blind_chain_wraps_mask_outside_budget() -> None:
    env = make_benchmark_env("blind")

    assert isinstance(env, ObjectPositionMaskWrapper)
    assert isinstance(env.env, FireBudgetWrapper)
    assert isinstance(env.env.env, SwingAdvanceDecisionWrapper)


def test_factory_observation_space_shape() -> None:
    for mode in OBSERVATION_MODES:
        env = make_benchmark_env(mode)
        assert env.observation_space.shape == (27,)
        assert cast(gymnasium.spaces.Space[Any], env.observation_space).dtype == (
            np.float32
        )


def test_factory_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        make_benchmark_env("banana")


# ---------------------------------------------------------------------------
# constructor contract
# ---------------------------------------------------------------------------


def test_rejects_non_box_observation_space() -> None:
    with pytest.raises(TypeError):
        ObjectPolarRepresentationWrapper(DiscreteObsEnv())


def test_rejects_wrong_observation_shape() -> None:
    inner_observation = np.zeros(26, dtype=np.float32)
    inner = ScriptedPolarEnv(inner_observation)
    inner.observation_space = spaces.Box(
        low=-1.0, high=1.0, shape=(26,), dtype=np.float32
    )

    with pytest.raises(ValueError):
        ObjectPolarRepresentationWrapper(inner)


def test_real_env_inactive_final_observation_stays_zero() -> None:
    """Collect all three fixed-map objects, then check the final observation."""
    inner_env = GoldMinerEnv(map_mode="fixed")
    env = ObjectPolarRepresentationWrapper(
        FireBudgetWrapper(SwingAdvanceDecisionWrapper(inner_env), max_fires=3)
    )
    observation, _info = env.reset()
    target_angles = (-30.0, 2.0, 31.0)
    target_index = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        angle = float(observation[0]) * MAX_ANGLE
        action = WAIT
        if (
            target_index < len(target_angles)
            and abs(angle - target_angles[target_index]) <= 3.0
        ):
            action = FIRE
            target_index += 1
        observation, _reward, terminated, truncated, _info = env.step(action)

    hook_env = cast(GoldMinerEnv, env.unwrapped)
    assert all(obj.active is False for obj in hook_env.objects)
    assert observation.shape == (27,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    for x_index, y_index, active_index in OBJECT_POLAR_BLOCKS:
        assert observation[active_index] == 0.0
        assert observation[x_index] == 0.0
        assert observation[y_index] == 0.0
