"""``DecisionIntervalWrapper`` 行为测试。

wrapper 每个决策推进恰好 DECISION_INTERVAL = 10 个底层 physics tick：
第 1 个 tick 使用 Agent 动作，其余 tick 固定 WAIT（0）；reward 累加这
10 个 tick；任一底层 tick 结束（terminated/truncated）即提前返回，
不再推进剩余 tick。

除真实 ``GoldMinerEnv`` 外，另用一个脚本化假环境（ScriptedStubEnv）
按预设序列返回 reward/terminated/truncated 并记录执行的每个动作，
以精确验证 reward 累加与提前结束语义。
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces
from numpy.typing import NDArray

from gold_miner_sim.env import (
    EPISODE_TIME,
    FIRE,
    SIM_FPS,
    WAIT,
    GoldMinerEnv,
    HookState,
)
from gold_miner_sim.wrappers import DECISION_INTERVAL, DecisionIntervalWrapper


class ScriptedStubEnv(gymnasium.Env):
    """脚本化假环境：按预设序列返回结果并记录每次执行的动作。

    第 i 个 step 记录 action，并返回 rewards[i] / terminateds[i] /
    truncateds[i]（列表耗尽时分别取 0.0 / False / False）。observation
    恒为全 0 向量，action/observation space 与真实环境相同。
    """

    def __init__(
        self,
        rewards: list[float],
        terminateds: list[bool],
        truncateds: list[bool],
    ) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(26,), dtype=np.float32
        )
        self._rewards = rewards
        self._terminateds = terminateds
        self._truncateds = truncateds
        self.executed_actions: list[int | np.integer[Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        self.executed_actions = []
        return np.zeros((26,), dtype=np.float32), {}

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        i = len(self.executed_actions)
        self.executed_actions.append(action)
        reward = self._rewards[i] if i < len(self._rewards) else 0.0
        terminated = self._terminateds[i] if i < len(self._terminateds) else False
        truncated = self._truncateds[i] if i < len(self._truncateds) else False
        return np.zeros((26,), dtype=np.float32), reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# A. WAIT：一次决策推进恰好 10 个底层 tick
# ---------------------------------------------------------------------------
def test_wait_advances_exactly_ten_underlying_ticks() -> None:
    """目的：一次 WAIT 决策应推进恰好 10 个底层 physics tick。

    输入：真实环境包上 wrapper，reset 后执行一次 step(WAIT)。
    输出：底层内部 tick 计数为 10，remaining_time 精确减少 10/60 s；
    WAIT 每 tick 摆动 1°，故角度恰为 -60°；reward=0、未结束；返回的
    info 反映最后一次底层 tick；action/observation space 与底层一致。
    """
    env = GoldMinerEnv()
    wrapped = DecisionIntervalWrapper(env)
    wrapped.reset()
    _, reward, terminated, truncated, info = wrapped.step(WAIT)

    assert DECISION_INTERVAL == 10
    assert env._ticks == 10  # 白盒：底层 tick 计数
    assert env.remaining_time == pytest.approx(EPISODE_TIME - 10.0 / SIM_FPS)
    assert env.angle == pytest.approx(-70.0 + 10.0)
    assert reward == 0.0
    assert terminated is False and truncated is False
    assert info["remaining_time"] == pytest.approx(EPISODE_TIME - 10.0 / SIM_FPS)
    assert wrapped.action_space == env.action_space
    assert wrapped.observation_space == env.observation_space


def test_reset_passes_through_seed_and_options() -> None:
    """目的：reset 透传 seed/options 并直接返回底层结果。

    输入：两个相同 seed 的 wrapper 环境，options 各传一个标记值。
    输出：两次 reset 的 observation 逐位相等；options 原样进入底层
    env.unwrapped 的 options 参数（假环境记录透传值）。
    """
    wrapped_a = DecisionIntervalWrapper(GoldMinerEnv())
    obs_a, info_a = wrapped_a.reset(seed=7)
    wrapped_b = DecisionIntervalWrapper(GoldMinerEnv())
    obs_b, info_b = wrapped_b.reset(seed=7)

    assert np.array_equal(obs_a, obs_b)
    assert info_a == info_b

    class OptionsRecordingStub(ScriptedStubEnv):
        """记录 reset 透传 seed/options 的假环境。"""

        def __init__(self) -> None:
            super().__init__(rewards=[], terminateds=[], truncateds=[])
            self.reset_seed: int | None = None
            self.reset_options: dict[str, Any] | None = None

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[NDArray[np.float32], dict[str, Any]]:
            self.reset_seed = seed
            self.reset_options = options
            return super().reset(seed=seed, options=options)

    stub = OptionsRecordingStub()
    DecisionIntervalWrapper(stub).reset(seed=42, options={"k": 1})
    assert stub.reset_seed == 42
    assert stub.reset_options == {"k": 1}


# ---------------------------------------------------------------------------
# B. FIRE 等价性：wrapper 一步 == 手工 1×FIRE + 9×WAIT
# ---------------------------------------------------------------------------
def test_fire_wrapper_step_matches_manual_tick_sequence() -> None:
    """目的：wrapper 一次 FIRE 与裸环境手工执行 FIRE + 9×WAIT 完全等价。

    输入：两个相同 seed 的真实环境；一个经 wrapper 执行 step(FIRE)，
    另一个裸环境依次 step(FIRE, WAIT × 9)。
    输出：observation 逐元素相等、reward 相等、结束标志相等，且
    hook_state 一致、angle / rope_length / remaining_time（approx）相等。
    """
    inner = GoldMinerEnv()
    wrapped = DecisionIntervalWrapper(inner)
    wrapped.reset(seed=3)
    obs_w, reward_w, terminated_w, truncated_w, _ = wrapped.step(FIRE)

    env_b = GoldMinerEnv()
    env_b.reset(seed=3)
    obs_b, reward_b, terminated_b, truncated_b, _ = env_b.step(FIRE)
    for _ in range(DECISION_INTERVAL - 1):
        obs_b, reward_b, terminated_b, truncated_b, _ = env_b.step(WAIT)

    assert obs_w == pytest.approx(obs_b)  # 逐元素
    assert reward_w == pytest.approx(reward_b)
    assert terminated_w == terminated_b
    assert truncated_w == truncated_b

    assert inner.hook_state is env_b.hook_state
    assert inner.angle == pytest.approx(env_b.angle)
    assert inner.rope_length == pytest.approx(env_b.rope_length)
    assert inner.remaining_time == pytest.approx(env_b.remaining_time)


# ---------------------------------------------------------------------------
# C. Reward 累加：10 个底层 tick 的 reward 求和返回
# ---------------------------------------------------------------------------
def test_reward_accumulated_across_ten_ticks() -> None:
    """目的：一次 wrapper step 返回 10 个底层 tick 的 reward 之和。

    输入：脚本化假环境按 tick 依次返回 reward
    [0.5, -0.25, 1.0, 0, 2.5, 0, 0.75, 0, 0.25, 1.0]（全程不结束）；
    wrapper 执行 step(FIRE)。
    输出：wrapper reward 等于 10 个 reward 之和（5.75）；假环境恰好执行
    10 次，动作序列为 FIRE + 9×WAIT；terminated/truncated 均为 False。
    """
    rewards = [0.5, -0.25, 1.0, 0.0, 2.5, 0.0, 0.75, 0.0, 0.25, 1.0]
    stub = ScriptedStubEnv(rewards=rewards, terminateds=[], truncateds=[])
    wrapped = DecisionIntervalWrapper(stub)
    wrapped.reset()
    _, reward, terminated, truncated, _ = wrapped.step(FIRE)

    assert reward == pytest.approx(sum(rewards))
    assert reward == pytest.approx(5.75)
    assert terminated is False and truncated is False
    assert len(stub.executed_actions) == 10
    assert stub.executed_actions == [FIRE] + [WAIT] * (DECISION_INTERVAL - 1)


# ---------------------------------------------------------------------------
# D. Episode 提前结束
# ---------------------------------------------------------------------------
def test_stub_stops_early_on_truncated_tick() -> None:
    """目的：任一底层 tick truncated 时 wrapper 立即返回并停止推进。

    输入：脚本化假环境令第 3 个底层 tick 返回 truncated=True；wrapper
    执行一次 step(FIRE)。
    输出：假环境恰好执行 3 次（不再推进剩余 7 个 tick）；wrapper 返回
    truncated=True、terminated=False，reward 只累加前 3 个 tick 的
    reward（1 + 2 + 3 = 6）。
    """
    stub = ScriptedStubEnv(
        rewards=[1.0, 2.0, 3.0, 4.0, 5.0],
        terminateds=[False, False, False],
        truncateds=[False, False, True],
    )
    wrapped = DecisionIntervalWrapper(stub)
    wrapped.reset()
    _, reward, terminated, truncated, _ = wrapped.step(FIRE)

    assert len(stub.executed_actions) == 3
    assert stub.executed_actions == [FIRE, WAIT, WAIT]
    assert truncated is True
    assert terminated is False
    assert reward == pytest.approx(6.0)


def test_stub_stops_early_on_terminated_tick() -> None:
    """目的：底层 tick terminated 时同样立即返回并停止推进。

    输入：脚本化假环境令第 2 个底层 tick 返回 terminated=True；wrapper
    执行一次 step(WAIT)。
    输出：假环境恰好执行 2 次；wrapper 返回 terminated=True、
    truncated=False，reward 只累加前 2 个 tick（0.5 + 1.5 = 2.0）。
    """
    stub = ScriptedStubEnv(
        rewards=[0.5, 1.5, 2.5],
        terminateds=[False, True, False],
        truncateds=[False, False, False],
    )
    wrapped = DecisionIntervalWrapper(stub)
    wrapped.reset()
    _, reward, terminated, truncated, _ = wrapped.step(WAIT)

    assert len(stub.executed_actions) == 2
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(2.0)


def test_stub_stops_immediately_when_first_tick_truncated() -> None:
    """目的：第 1 个底层 tick（agent action tick）即 truncated 时，不得再
    执行任何后续 WAIT tick。

    输入：脚本化假环境令第 1 个底层 tick（index 0）返回 truncated=True；
    wrapper 执行一次 step(FIRE)。
    输出：假环境恰好执行 1 次（动作即 agent 的 FIRE）；wrapper 返回
    truncated=True、terminated=False，reward 只含第 1 个 tick（1.5）。
    """
    stub = ScriptedStubEnv(rewards=[1.5], terminateds=[], truncateds=[True])
    wrapped = DecisionIntervalWrapper(stub)
    wrapped.reset()
    _, reward, terminated, truncated, _ = wrapped.step(FIRE)

    assert stub.executed_actions == [FIRE]
    assert truncated is True
    assert terminated is False
    assert reward == pytest.approx(1.5)


def test_stub_stops_immediately_when_first_tick_terminated() -> None:
    """目的：第 1 个底层 tick（agent action tick）即 terminated 时，同样
    立即停止，不再推进剩余 tick。

    输入：脚本化假环境令第 1 个底层 tick（index 0）返回 terminated=True；
    wrapper 执行一次 step(WAIT)。
    输出：假环境恰好执行 1 次；wrapper 返回 terminated=True、
    truncated=False，reward 只含第 1 个 tick（0.25）。
    """
    stub = ScriptedStubEnv(rewards=[0.25], terminateds=[True], truncateds=[])
    wrapped = DecisionIntervalWrapper(stub)
    wrapped.reset()
    _, reward, terminated, truncated, _ = wrapped.step(WAIT)

    assert stub.executed_actions == [WAIT]
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(0.25)


def test_real_env_full_episode_truncates_with_consistent_reward() -> None:
    """目的：真实环境经 wrapper 整局步进到 truncated，reward 累计与最终
    score 一致（reward 累加一致性）。

    输入：wrapper 环境；先 4 次 step(WAIT)（40 tick，摆到 -30°），再
    step(FIRE) 抓取 GOLD，此后一直 step(WAIT) 直到 truncated。
    输出：结束时 remaining_time == 0.0（精确）、truncated=True、
    terminated=False；全部分步 reward 之和等于最终 score（250.0，恰好
    收取 GOLD 一次），GOLD 已失效。
    """
    inner = GoldMinerEnv()
    wrapped = DecisionIntervalWrapper(inner)
    wrapped.reset(seed=0)

    for _ in range(4):
        wrapped.step(WAIT)
    assert inner.angle == pytest.approx(-30.0)  # 40 tick × 1°/tick
    wrapped.step(FIRE)  # 第 1 个底层 tick 在 -30° 发射，命中 GOLD
    assert inner.hook_state is not HookState.SWINGING  # 发射已生效

    total_reward = 0.0
    truncated = False
    steps = 0
    while not truncated:
        _, reward, terminated, truncated, _ = wrapped.step(WAIT)
        assert terminated is False
        total_reward += reward
        steps += 1
        assert steps < 400, "episode never truncated"

    assert truncated is True
    assert inner.remaining_time == 0.0  # 精确：60 - 3600/60
    assert inner.score == pytest.approx(250.0)
    assert total_reward == pytest.approx(250.0)
    assert inner.objects[0].active is False
