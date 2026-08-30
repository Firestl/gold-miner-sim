"""行为测试 for :class:`ObjectPositionMaskWrapper` (Issue #13)."""

from __future__ import annotations

from typing import Any, cast

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces
from numpy.typing import NDArray

from gold_miner_sim.benchmark import OBSERVATION_MODES, make_benchmark_env
from gold_miner_sim.env import FIRE, MAX_ANGLE, WAIT, GoldMinerEnv
from gold_miner_sim.wrappers import (
    OBJECT_POSITION_INDICES,
    FireBudgetWrapper,
    ObjectPositionMaskWrapper,
    SwingAdvanceDecisionWrapper,
)

# Fixed launch targets for the fixed map (GOLD / DIAMOND / ROCK); with the
# 10-tick WAIT batching of SwingAdvanceDecisionWrapper the decision angles
# are -70, -60, ... so the tolerance must exceed half the 10 deg step to
# always land a decision near each target (fires at -30, 0 and +30 deg).
TARGET_ANGLES: tuple[float, ...] = (-30.0, 2.0, 31.0)
FIRE_TOLERANCE_DEG = 3.0

# The 21 observation slots that must survive masking untouched.
OTHER_INDICES = [i for i in range(27) if i not in OBJECT_POSITION_INDICES]


class ScriptedMaskEnv(gymnasium.Env[NDArray[np.float32], int]):
    """Small deterministic 27-dim inner env for mask and passthrough tests."""

    def __init__(
        self,
        transitions: list[tuple[float, bool, bool, dict[str, Any]]],
        observation_size: int = 27,
    ) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(observation_size,), dtype=np.float32
        )
        self._observation_shape: tuple[int, ...] = (observation_size,)
        self.transitions = transitions
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
        return np.full(self._observation_shape, 0.5, dtype=np.float32), {
            "inner_reset": True
        }

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        self.actions.append(int(action))
        index = len(self.actions) - 1
        if index < len(self.transitions):
            reward, terminated, truncated, info = self.transitions[index]
        else:
            reward, terminated, truncated, info = 0.0, False, False, {}
        # Distinct observations per step; non-zero everywhere including the
        # masked position slots, within the [-1, 1] space bounds.
        observation = np.full(
            self._observation_shape,
            0.25 + 0.05 * (index % 12),
            dtype=np.float32,
        )
        return observation, reward, terminated, truncated, dict(info)


class PersistentObsEnv(gymnasium.Env[NDArray[np.float32], int]):
    """Inner env returning the same persistent observation buffer object."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(27,), dtype=np.float32
        )
        self.buffer: NDArray[np.float32] = np.full(27, 0.75, dtype=np.float32)

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
    blind_observation: NDArray[np.float32],
) -> None:
    """Assert masked slots are 0 and all other slots match the full chain."""
    assert np.all(blind_observation[list(OBJECT_POSITION_INDICES)] == 0.0)
    assert np.array_equal(
        full_observation[OTHER_INDICES], blind_observation[OTHER_INDICES]
    )


def test_reset_shape_and_dtype() -> None:
    wrapped = make_benchmark_env("blind")

    observation, _info = wrapped.reset(seed=42)

    assert observation.shape == (27,)
    assert observation.dtype == np.float32


def test_reset_masks_position_indices() -> None:
    wrapped = make_benchmark_env("blind")

    observation, _info = wrapped.reset(seed=42)

    assert np.all(observation[list(OBJECT_POSITION_INDICES)] == 0.0)


def test_reset_preserves_other_dimensions() -> None:
    """Lockstep full/blind chains: reset and steps only differ in the mask.

    Stepping the blind chain with identical actions also proves the mask
    keeps positions zeroed on every step (merged step-masking check).
    """
    full_env = make_benchmark_env("full")
    blind_env = make_benchmark_env("blind")
    full_obs, _full_info = full_env.reset(seed=42)
    blind_obs, _blind_info = blind_env.reset(seed=42)
    assert_lockstep(full_obs, blind_obs)
    # Sanity: the full chain's position slots are not all zero, so the
    # lockstep comparison is actually meaningful.
    assert np.any(full_obs[list(OBJECT_POSITION_INDICES)] != 0.0)

    for action in (WAIT, WAIT, FIRE, WAIT, FIRE, FIRE):
        full_obs, _reward, full_done, _trunc, _full_info = full_env.step(action)
        blind_obs, _reward_b, blind_done, _trunc_b, _blind_info = blind_env.step(action)
        assert full_done == blind_done
        assert_lockstep(full_obs, blind_obs)


def test_reward_passthrough() -> None:
    inner = ScriptedMaskEnv([(1.5, False, False, {})])
    wrapped = ObjectPositionMaskWrapper(inner)
    wrapped.reset()

    _observation, reward, _terminated, _truncated, _info = wrapped.step(WAIT)

    assert reward is inner.transitions[0][0]  # same float object, no copy
    assert reward == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("terminated", "truncated"),
    [(True, False), (False, True)],
)
def test_terminated_truncated_passthrough(terminated: bool, truncated: bool) -> None:
    inner = ScriptedMaskEnv([(2.0, terminated, truncated, {})])
    wrapped = ObjectPositionMaskWrapper(inner)
    wrapped.reset()

    _observation, _reward, out_terminated, out_truncated, _info = wrapped.step(WAIT)

    assert out_terminated is terminated
    assert out_truncated is truncated


def test_info_passthrough() -> None:
    inner = ScriptedMaskEnv([(0.0, False, False, {"marker": "x"})])
    wrapped = ObjectPositionMaskWrapper(inner)
    wrapped.reset()

    _observation, _reward, _terminated, _truncated, info = wrapped.step(WAIT)

    assert info == {"marker": "x"}


def test_reset_passes_seed_options_and_info_through() -> None:
    inner = ScriptedMaskEnv([])
    wrapped = ObjectPositionMaskWrapper(inner)

    observation, info = wrapped.reset(seed=17, options={"episode": 2})

    assert inner.reset_seed == 17
    assert inner.reset_options == {"episode": 2}
    assert info == {"inner_reset": True}
    assert np.all(observation[list(OBJECT_POSITION_INDICES)] == 0.0)


def test_action_space_unchanged() -> None:
    inner = ScriptedMaskEnv([])
    wrapped = ObjectPositionMaskWrapper(inner)

    assert wrapped.action_space == inner.action_space
    assert wrapped.action_space == spaces.Discrete(2)


def test_observation_space_contract() -> None:
    inner = ScriptedMaskEnv([])
    wrapped = ObjectPositionMaskWrapper(inner)

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
    wrapped = ObjectPositionMaskWrapper(inner)

    reset_observation, _info = wrapped.reset(seed=1)
    assert reset_observation is not inner.buffer
    assert np.all(inner.buffer[list(OBJECT_POSITION_INDICES)] == 0.75)
    assert np.all(reset_observation[list(OBJECT_POSITION_INDICES)] == 0.0)
    reset_observation[8] = 99.0  # mutating the returned obs must not leak

    step_observation, _reward, _terminated, _truncated, _info = wrapped.step(WAIT)
    assert step_observation is not inner.buffer
    assert step_observation is not reset_observation
    assert np.all(inner.buffer[list(OBJECT_POSITION_INDICES)] == 0.75)
    assert np.all(step_observation[list(OBJECT_POSITION_INDICES)] == 0.0)


def test_inactive_objects_still_valid_observation() -> None:
    """Collect all three fixed-map objects, then check the final observation."""
    inner_env = GoldMinerEnv(map_mode="fixed")
    env = ObjectPositionMaskWrapper(
        FireBudgetWrapper(SwingAdvanceDecisionWrapper(inner_env), max_fires=3)
    )
    observation, _info = env.reset()
    target_index = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        angle = float(observation[0]) * MAX_ANGLE
        action = WAIT
        if (
            target_index < len(TARGET_ANGLES)
            and abs(angle - TARGET_ANGLES[target_index]) <= FIRE_TOLERANCE_DEG
        ):
            action = FIRE
            # One launch per target; SwingAdvanceDecisionWrapper returns
            # only while the hook is SWINGING, so the next target is
            # considered at the next decision point.
            target_index += 1
        observation, _reward, terminated, truncated, _info = env.step(action)

    hook_env = cast(GoldMinerEnv, env.unwrapped)
    assert all(obj.active is False for obj in hook_env.objects)
    assert observation.shape == (27,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert np.all(observation[list(OBJECT_POSITION_INDICES)] == 0.0)


def test_rejects_non_box_observation_space() -> None:
    with pytest.raises(TypeError):
        ObjectPositionMaskWrapper(DiscreteObsEnv())


def test_rejects_wrong_observation_shape() -> None:
    inner = ScriptedMaskEnv([], observation_size=26)

    with pytest.raises(ValueError):
        ObjectPositionMaskWrapper(inner)


def test_factory_full_chain_has_no_mask_wrapper() -> None:
    env = make_benchmark_env("full")

    chain = wrapper_chain(env)
    assert isinstance(env, FireBudgetWrapper)
    assert isinstance(env.env, SwingAdvanceDecisionWrapper)
    assert not any(isinstance(wrapper, ObjectPositionMaskWrapper) for wrapper in chain)


def test_factory_blind_chain_wraps_mask_outside_budget() -> None:
    env = make_benchmark_env("blind")

    assert isinstance(env, ObjectPositionMaskWrapper)
    assert isinstance(env.env, FireBudgetWrapper)
    assert isinstance(env.env.env, SwingAdvanceDecisionWrapper)


def test_factory_observation_space_shape() -> None:
    for mode in OBSERVATION_MODES:
        env = make_benchmark_env(mode)
        assert env.observation_space.shape == (27,)


def test_factory_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        make_benchmark_env("banana")
