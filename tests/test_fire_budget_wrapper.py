"""行为测试 for :class:`FireBudgetWrapper` (Issue #11)."""

from __future__ import annotations

from typing import Any, cast

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces
from numpy.typing import NDArray

from gold_miner_sim.env import FIRE, WAIT, GoldMinerEnv
from gold_miner_sim.wrappers import (
    FireBudgetWrapper,
    SwingAdvanceDecisionWrapper,
)


class ScriptedBudgetEnv(gymnasium.Env[NDArray[np.float32], int]):
    """Small deterministic inner env for budget and propagation tests."""

    def __init__(
        self,
        transitions: list[tuple[float, bool, bool, dict[str, Any]]],
    ) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(26,), dtype=np.float32
        )
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
        return np.zeros(26, dtype=np.float32), {"inner_reset": True}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        self.actions.append(int(action))
        index = len(self.actions) - 1
        if index < len(self.transitions):
            reward, terminated, truncated, info = self.transitions[index]
        else:
            reward, terminated, truncated, info = 0.0, False, False, {}
        observation = np.full(26, index, dtype=np.float32)
        return observation, reward, terminated, truncated, dict(info)


@pytest.mark.parametrize("wrapper_cls", [SwingAdvanceDecisionWrapper])
def test_swing_fire_supports_nested_time_limit(wrapper_cls: Any) -> None:
    inner = GoldMinerEnv()
    limited = gymnasium.wrappers.TimeLimit(inner, max_episode_steps=1_000)
    wrapped = wrapper_cls(limited)
    wrapped.reset(seed=0)
    for _ in range(4):
        wrapped.step(WAIT)

    _observation, reward, terminated, truncated, _info = wrapped.step(FIRE)

    assert reward == pytest.approx(250.0)
    assert inner.score == pytest.approx(250.0)
    assert terminated is False and truncated is False


def test_reset_initializes_budget_and_observation() -> None:
    wrapped = FireBudgetWrapper(ScriptedBudgetEnv([]))

    observation, info = wrapped.reset()

    assert wrapped.max_fires == 3
    assert wrapped.fires_used == 0
    assert wrapped.fires_remaining == 3
    assert observation.shape == (27,)
    assert observation.dtype == np.float32
    assert observation[-1] == pytest.approx(1.0)
    assert wrapped.observation_space.shape == (27,)
    assert wrapped.observation_space.contains(observation)
    assert info["inner_reset"] is True
    assert info["fires_used"] == 0
    assert info["fires_remaining"] == 3


def test_reset_passes_seed_options_and_preserves_inner_info() -> None:
    inner = ScriptedBudgetEnv([])
    wrapped = FireBudgetWrapper(inner)

    _observation, info = wrapped.reset(seed=17, options={"episode": 2})

    assert inner.reset_seed == 17
    assert inner.reset_options == {"episode": 2}
    assert info["inner_reset"] is True
    assert info["fires_used"] == 0
    assert info["fires_remaining"] == 3


def test_wait_is_forwarded_without_consuming_budget() -> None:
    inner = ScriptedBudgetEnv([(1.25, False, False, {"marker": "wait"})])
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()

    observation, reward, terminated, truncated, info = wrapped.step(WAIT)

    assert inner.actions == [WAIT]
    assert wrapped.fires_used == 0
    assert wrapped.fires_remaining == 3
    assert observation[-1] == pytest.approx(1.0)
    assert reward == pytest.approx(1.25)
    assert terminated is False and truncated is False
    assert info["marker"] == "wait"
    assert info["fires_used"] == 0
    assert info["fires_remaining"] == 3


def test_first_fire_decrements_budget_before_return() -> None:
    inner = ScriptedBudgetEnv([(2.5, False, False, {})])
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()

    observation, reward, terminated, truncated, info = wrapped.step(FIRE)

    assert inner.actions == [FIRE]
    assert wrapped.fires_used == 1
    assert wrapped.fires_remaining == 2
    assert observation[-1] == pytest.approx(2.0 / 3.0)
    assert reward == pytest.approx(2.5)
    assert terminated is False and truncated is False
    assert info["fires_used"] == 1
    assert info["fires_remaining"] == 2


def test_second_fire_has_one_budget_unit_left() -> None:
    inner = ScriptedBudgetEnv([(0.0, False, False, {}), (0.0, False, False, {})])
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()

    wrapped.step(FIRE)
    observation, _reward, terminated, truncated, info = wrapped.step(FIRE)

    assert inner.actions == [FIRE, FIRE]
    assert wrapped.fires_used == 2
    assert wrapped.fires_remaining == 1
    assert observation[-1] == pytest.approx(1.0 / 3.0)
    assert terminated is False and truncated is False
    assert info["fires_used"] == 2
    assert info["fires_remaining"] == 1


def test_empty_hook_fire_consumes_budget() -> None:
    inner = GoldMinerEnv()
    wrapped = FireBudgetWrapper(SwingAdvanceDecisionWrapper(inner))
    wrapped.reset(seed=0)

    observation, reward, terminated, truncated, info = wrapped.step(FIRE)

    assert reward == pytest.approx(0.0)
    assert inner.score == pytest.approx(0.0)
    assert inner._ticks == 146 + 10
    assert wrapped.fires_used == 1
    assert wrapped.fires_remaining == 2
    assert observation[-1] == pytest.approx(2.0 / 3.0)
    assert terminated is False and truncated is False
    assert info["fires_used"] == 1
    assert info["fires_remaining"] == 2


def test_successful_fire_consumes_budget_and_keeps_reward() -> None:
    inner = GoldMinerEnv()
    wrapped = FireBudgetWrapper(SwingAdvanceDecisionWrapper(inner))
    wrapped.reset(seed=0)
    for _ in range(4):
        wrapped.step(WAIT)

    observation, reward, terminated, truncated, info = wrapped.step(FIRE)

    assert reward == pytest.approx(250.0)
    assert inner.score == pytest.approx(250.0)
    assert inner.objects[0].active is False
    assert wrapped.fires_used == 1
    assert wrapped.fires_remaining == 2
    assert observation[-1] == pytest.approx(2.0 / 3.0)
    assert terminated is False and truncated is False
    assert info["fires_used"] == 1
    assert info["fires_remaining"] == 2


def test_third_fire_runs_inner_transition_before_terminating() -> None:
    inner = ScriptedBudgetEnv(
        [
            (0.0, False, False, {}),
            (0.0, False, False, {}),
            (500.0, False, False, {"inner_complete": True}),
        ]
    )
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()
    wrapped.step(FIRE)
    wrapped.step(FIRE)

    observation, reward, terminated, truncated, info = wrapped.step(FIRE)

    assert inner.actions == [FIRE, FIRE, FIRE]
    assert reward == pytest.approx(500.0)
    assert observation[-1] == pytest.approx(0.0)
    assert terminated is True
    assert truncated is False
    assert info["inner_complete"] is True
    assert info["fires_used"] == 3
    assert info["fires_remaining"] == 0


def test_third_fire_final_reward_is_not_dropped() -> None:
    inner = ScriptedBudgetEnv(
        [(0.0, False, False, {}), (0.0, False, False, {}), (7.75, False, False, {})]
    )
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()

    total_reward = 0.0
    for _ in range(3):
        _observation, reward, _terminated, _truncated, _info = wrapped.step(FIRE)
        total_reward += reward

    assert total_reward == pytest.approx(7.75)


def test_third_fire_inner_timeout_has_priority_over_budget_termination() -> None:
    inner = ScriptedBudgetEnv(
        [
            (0.0, False, False, {}),
            (0.0, False, False, {}),
            (4.0, False, True, {"timed_out": True}),
        ]
    )
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()
    wrapped.step(FIRE)
    wrapped.step(FIRE)

    _observation, reward, terminated, truncated, info = wrapped.step(FIRE)

    assert reward == pytest.approx(4.0)
    assert terminated is False
    assert truncated is True
    assert info["timed_out"] is True
    assert info["fires_used"] == 3
    assert info["fires_remaining"] == 0


def test_inner_terminated_result_is_preserved() -> None:
    inner = ScriptedBudgetEnv([(3.0, True, False, {"inner_end": True})])
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()

    _observation, reward, terminated, truncated, info = wrapped.step(FIRE)

    assert reward == pytest.approx(3.0)
    assert terminated is True
    assert truncated is False
    assert info["inner_end"] is True
    assert wrapped.fires_used == 1
    assert wrapped.fires_remaining == 2


def test_info_fields_are_merged_with_inner_info() -> None:
    inner = ScriptedBudgetEnv([(0.0, False, False, {"score": 12, "custom": "x"})])
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()

    _observation, _reward, _terminated, _truncated, info = wrapped.step(WAIT)

    assert info["score"] == 12
    assert info["custom"] == "x"
    assert info["fires_used"] == 0
    assert info["fires_remaining"] == 3


@pytest.mark.parametrize("bad_action", [2, -1, 3, np.int64(7), np.int32(-2), 0.0, 1.0])
def test_invalid_actions_are_rejected_without_consuming_budget(
    bad_action: object,
) -> None:
    inner = ScriptedBudgetEnv([])
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()

    with pytest.raises(ValueError):
        wrapped.step(cast("int | np.integer[Any]", bad_action))

    assert inner.actions == []
    assert wrapped.fires_used == 0
    assert wrapped.fires_remaining == 3


def test_budget_resets_for_a_new_episode() -> None:
    inner = ScriptedBudgetEnv([(0.0, False, False, {})])
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()
    wrapped.step(FIRE)
    wrapped.step(FIRE)

    observation, info = wrapped.reset(seed=9)

    assert wrapped.fires_used == 0
    assert wrapped.fires_remaining == 3
    assert observation[-1] == pytest.approx(1.0)
    assert info["fires_used"] == 0
    assert info["fires_remaining"] == 3


def test_custom_max_fires_is_normalized_and_terminates_on_last_fire() -> None:
    inner = ScriptedBudgetEnv([(0.0, False, False, {}), (2.0, False, False, {})])
    wrapped = FireBudgetWrapper(inner, max_fires=2)
    observation, _info = wrapped.reset()
    assert observation[-1] == pytest.approx(1.0)

    first_observation, _reward, terminated, truncated, _info = wrapped.step(FIRE)
    assert first_observation[-1] == pytest.approx(0.5)
    assert terminated is False and truncated is False

    last_observation, reward, terminated, truncated, info = wrapped.step(FIRE)
    assert last_observation[-1] == pytest.approx(0.0)
    assert reward == pytest.approx(2.0)
    assert terminated is True and truncated is False
    assert info["fires_remaining"] == 0


@pytest.mark.parametrize("bad_max_fires", [0, -1])
def test_max_fires_must_be_positive(bad_max_fires: int) -> None:
    with pytest.raises(ValueError):
        FireBudgetWrapper(ScriptedBudgetEnv([]), max_fires=bad_max_fires)


def test_fourth_fire_is_rejected_after_budget_exhaustion() -> None:
    inner = ScriptedBudgetEnv(
        [(0.0, False, False, {}), (0.0, False, False, {}), (0.0, False, False, {})]
    )
    wrapped = FireBudgetWrapper(inner)
    wrapped.reset()
    for _ in range(3):
        wrapped.step(FIRE)

    with pytest.raises(ValueError, match="budget exhausted"):
        wrapped.step(FIRE)

    assert inner.actions == [FIRE, FIRE, FIRE]


def test_episode_reward_sum_matches_final_score_with_three_fires() -> None:
    inner = GoldMinerEnv()
    wrapped = FireBudgetWrapper(SwingAdvanceDecisionWrapper(inner))
    wrapped.reset(seed=0)
    total_reward = 0.0

    def step(action: int) -> tuple[bool, bool]:
        nonlocal total_reward
        _observation, reward, terminated, truncated, _info = wrapped.step(action)
        total_reward += reward
        return terminated, truncated

    for _ in range(4):
        step(WAIT)
    terminated, truncated = step(FIRE)
    assert terminated is False and truncated is False

    for _ in range(2):
        step(WAIT)
    terminated, truncated = step(FIRE)
    assert terminated is False and truncated is False

    for _ in range(2):
        step(WAIT)
    terminated, truncated = step(FIRE)

    assert terminated is True and truncated is False
    assert inner.score == pytest.approx(800.0)
    assert total_reward == pytest.approx(inner.score)
    assert total_reward == pytest.approx(800.0)
