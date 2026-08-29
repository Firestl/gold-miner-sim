"""``SwingDecisionWrapper`` 行为测试。

wrapper 的核心不变量：除 episode 结束的那次返回外，每次 ``step()``
返回时底层 hook 必须处于 ``HookState.SWINGING``。WAIT 决策推进至多
``SWING_WAIT_INTERVAL = 10`` 个底层 tick（与 ``DecisionIntervalWrapper``
的 WAIT 等价）；FIRE 决策先执行 1 个底层 FIRE tick（SWINGING→EXTENDING），
再自动推进 WAIT tick，直到 hook 恰好回到 SWINGING（不多跑一个 tick）
或任一底层 tick 结束 episode。

全部使用 fixed map（完全确定）：初始角 -70°，SWINGING 每 tick 恰好
+1°，摆动一个来回恰好 280 tick，故每次 WAIT 决策推进 10°，决策点角度
为 -70, -60, ..., +70 循环。固定物体：GOLD(315,300)/DIAMOND(465,450)/
ROCK(610,340)。发射时的关键几何（已实测）：
- -30° 射线距 GOLD 中心 1.9px（< r36）必命中：伸 34 tick + 载收回
  78 tick = 112 tick；
- 0° 垂直射线距 DIAMOND 中心 15px（< r24）必命中：59 + 67 = 126 tick；
- +30° 射线距 ROCK 中心约 3.6px（< r40）必命中：42 + 150 = 192 tick；
- -70° 射线不命中任何物体：空钩伸 77 tick + 空收 69 tick = 146 tick。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from gold_miner_sim.env import (
    EPISODE_TIME,
    FIRE,
    MIN_ROPE_LENGTH,
    SIM_FPS,
    WAIT,
    GoldMinerEnv,
    HookState,
)
from gold_miner_sim.wrappers import (
    SWING_WAIT_INTERVAL,
    SwingDecisionWrapper,
)


# ---------------------------------------------------------------------------
# A. WAIT 等价性：一次决策 == 裸环境连跑 10 个 WAIT tick
# ---------------------------------------------------------------------------
def test_wait_matches_ten_bare_wait_ticks() -> None:
    """目的：wrapper 的 WAIT 决策应与裸环境连跑 10 次 step(WAIT) 完全等价。

    输入：两个相同 seed 的 fixed map 环境，一个包 wrapper 执行
    step(WAIT)，另一个裸环境依次执行 10 次 step(WAIT)。
    输出：observation 逐位相等；reward（均 0）、hook_state（均
    SWINGING）、angle（-60°）、rope_length、remaining_time 全部一致
    （remaining_time 精确相等：底层按 tick 计数计算）；结束标志均为
    False；SWING_WAIT_INTERVAL == 10。
    """
    assert SWING_WAIT_INTERVAL == 10
    wrapped = SwingDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    obs_w, reward_w, terminated_w, truncated_w, info_w = wrapped.step(WAIT)

    env_b = GoldMinerEnv()
    env_b.reset(seed=0)
    obs_b: NDArray[np.float32] = np.zeros(26, dtype=np.float32)
    reward_b = 0.0
    terminated_b = truncated_b = False
    for _ in range(10):
        obs_b, reward_b, terminated_b, truncated_b, info_b = env_b.step(WAIT)

    assert np.array_equal(obs_w, obs_b)  # 逐位相等
    assert reward_w == reward_b == 0.0
    assert terminated_w is terminated_b is False
    assert truncated_w is truncated_b is False

    inner = wrapped.unwrapped
    assert inner.hook_state is HookState.SWINGING is env_b.hook_state
    assert inner.angle == pytest.approx(env_b.angle)  # -60°
    assert inner.rope_length == env_b.rope_length
    assert inner.remaining_time == env_b.remaining_time
    assert info_w == info_b


# ---------------------------------------------------------------------------
# B. FIRE 空钩：一次决策内部完成 EXTENDING→RETRACT_EMPTY→SWINGING
# ---------------------------------------------------------------------------
def test_fire_empty_hook_full_cycle_at_minus_seventy() -> None:
    """目的：-70°（不命中任何固定物体）FIRE 应在一次决策内完成空钩往返。

    输入：reset 后（决策点 -70°）直接 step(FIRE)。
    输出：返回时 hook_state 为 SWINGING、reward==0、rope 回到
    MIN_ROPE_LENGTH；底层恰好消耗 146 个 tick（伸 77 含 FIRE tick +
    空收 69），故 remaining_time 精确等于 EPISODE_TIME - 146/SIM_FPS
    （回到 SWINGING 后不多跑一个 tick）；角度保持 -70°、摆动方向保持。
    """
    wrapped = SwingDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    assert inner.angle == pytest.approx(-70.0)

    obs, reward, terminated, truncated, info = wrapped.step(FIRE)

    assert inner.hook_state is HookState.SWINGING
    assert reward == 0.0
    assert terminated is False and truncated is False
    assert inner.rope_length == MIN_ROPE_LENGTH
    assert inner._ticks == 146  # 白盒：77 伸出（含 FIRE tick）+ 69 空收
    assert inner.remaining_time == EPISODE_TIME - 146 / SIM_FPS  # 精确
    assert inner.angle == pytest.approx(-70.0)  # 飞行期间角度冻结
    assert inner.swing_direction == 1  # 摆动方向保持
    assert info["hook_state"] == "SWINGING"  # info 存 HookState.name
    assert np.isclose(float(obs[7]), inner.remaining_time / EPISODE_TIME)


# ---------------------------------------------------------------------------
# C. FIRE 命中并回收：内部经过 RETRACT_LOADED，回收完毕才返回
# ---------------------------------------------------------------------------
def test_fire_catches_gold_and_returns_on_swinging_tick() -> None:
    """目的：-30° FIRE 命中 GOLD 后应载物收回、计分，并恰在回到 SWINGING
    的 tick 返回（无多余 tick）。

    输入：reset 后 4 次 step(WAIT)（-70°→-30°），再 step(FIRE)；另用裸
    环境手工执行同样的 tick 序列（40 个 WAIT tick + FIRE + WAIT 直到
    SWINGING）作对照。
    输出：wrapper 一次 FIRE 返回 reward==250、score==250、GOLD inactive、
    attached_object 为 None、hook_state 为 SWINGING；底层 tick 数与手工
    序列完全一致（40 + 34 伸 + 78 载收 = 152），remaining_time 精确相等，
    且最终 observation 逐位相等。
    """
    wrapped = SwingDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    for _ in range(4):
        wrapped.step(WAIT)
    assert inner.angle == pytest.approx(-30.0)  # 4 × 10 tick × 1°/tick

    obs_w, reward_w, terminated_w, truncated_w, _ = wrapped.step(FIRE)

    assert inner.hook_state is HookState.SWINGING
    assert reward_w == pytest.approx(250.0)
    assert inner.score == pytest.approx(250.0)
    assert inner.objects[0].active is False  # GOLD 已回收
    assert inner.attached_object is None
    assert terminated_w is False and truncated_w is False

    # 对照：裸环境手工 tick 序列（同样的发射时机与早停条件）。
    env_b = GoldMinerEnv()
    env_b.reset(seed=0)
    for _ in range(40):
        obs_b, reward_b, _, _, _ = env_b.step(WAIT)
    obs_b, reward_b, terminated_b, truncated_b, _ = env_b.step(FIRE)
    while env_b.hook_state is not HookState.SWINGING:
        obs_b, reward_b, terminated_b, truncated_b, _ = env_b.step(WAIT)

    assert inner._ticks == env_b._ticks == 152  # 40 + 34 + 78，无多余 tick
    assert inner.remaining_time == env_b.remaining_time  # 精确
    assert np.array_equal(obs_w, obs_b)  # 逐位相等
    assert reward_w == pytest.approx(reward_b)
    assert terminated_w is terminated_b and truncated_w is truncated_b


# ---------------------------------------------------------------------------
# D. 自动推进中 episode 超时：立即在 truncated 的底层 tick 返回
# ---------------------------------------------------------------------------
def test_fire_truncated_mid_retract_scores_nothing() -> None:
    """目的：剩余时间不足以完成 FIRE 时，wrapper 应在 truncated 的底层
    tick 立即返回，未回收完成的物体不计分，且允许返回非 SWINGING。

    输入：reset 后连续 346 次 step(WAIT)（恰好 3460 tick，摆回 +30°、
    剩余 140 tick），再 step(FIRE)（+30° 射线命中 ROCK：伸 42 tick +
    载收需 150 tick > 剩余时间）。
    输出：FIRE 决策返回 truncated==True、terminated==False、
    remaining_time==0.0（精确）、hook_state 为 RETRACT_LOADED（episode
    已结束，允许非 SWINGING）、reward==0、score==0、ROCK 仍 attached 且
    三个物体均 active；底层 tick 恰为 3600。
    """
    wrapped = SwingDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    for _ in range(346):  # 3460 tick：280 tick/来回 + 100 → +30°，方向 +1
        _, reward, terminated, truncated, _ = wrapped.step(WAIT)
        assert terminated is False and truncated is False
        assert inner.hook_state is HookState.SWINGING  # WAIT 不变量
    assert inner.angle == pytest.approx(30.0)
    assert inner._ticks == 3460
    assert inner.remaining_time == EPISODE_TIME - 3460 / SIM_FPS

    _, reward, terminated, truncated, _ = wrapped.step(FIRE)

    assert truncated is True
    assert terminated is False
    assert inner.remaining_time == 0.0  # 精确：60 - 3600/60
    assert inner._ticks == 3600
    assert inner.hook_state is HookState.RETRACT_LOADED  # 超时于载物收回途中
    assert inner.attached_object is inner.objects[2]  # ROCK 仍挂在钩上
    assert reward == 0.0  # 未收回到顶，不加分
    assert inner.score == 0.0
    assert all(obj.active for obj in inner.objects)  # 三个物体均未回收


# ---------------------------------------------------------------------------
# E. 整局 reward 一致性：三分全收，sum(reward) == final score == 800
# ---------------------------------------------------------------------------
def test_full_episode_reward_consistency_three_objects() -> None:
    """目的：脚本化策略经 wrapper 跑完整局，reward 累计与最终 score 一致，
    且每个非 episode 结束的返回点 hook 均为 SWINGING（核心不变量）。

    输入：fixed map；-30° FIRE 抓 GOLD（250）→ 回到 SWINGING（方向保持）
    后 WAIT×3 到 0° FIRE 抓 DIAMOND（500）→ WAIT×3 到 +30° FIRE 抓 ROCK
    （50）→ 之后持续 WAIT 直到 truncated。
    输出：三次 FIRE 的分步 reward 分别为 250/500/50；每次 wrapper.step
    返回（episode 结束前）底层 hook_state 均为 SWINGING；sum(所有分步
    reward) == final score == 800；三个物体均 inactive；最终
    truncated==True、terminated==False、remaining_time==0.0。
    """
    wrapped = SwingDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped

    total_reward = 0.0

    def check_step(
        expected_reward: float,
    ) -> tuple[bool, bool]:
        """执行一次 WAIT 决策，校验不变量并累计 reward。"""
        nonlocal total_reward
        _, reward, terminated, truncated, _ = wrapped.step(WAIT)
        total_reward += reward
        if not (terminated or truncated):
            assert inner.hook_state is HookState.SWINGING  # 核心不变量
        assert reward == pytest.approx(expected_reward)
        return terminated, truncated

    for _ in range(4):
        check_step(0.0)
    assert inner.angle == pytest.approx(-30.0)

    _, reward, terminated, truncated, _ = wrapped.step(FIRE)
    total_reward += reward
    assert inner.hook_state is HookState.SWINGING  # 非 episode 结束
    assert reward == pytest.approx(250.0)
    assert inner.angle == pytest.approx(-30.0)  # 方向保持，继续向 0° 摆
    assert terminated is False and truncated is False

    for _ in range(3):
        check_step(0.0)
    assert inner.angle == pytest.approx(0.0)

    _, reward, terminated, truncated, _ = wrapped.step(FIRE)
    total_reward += reward
    assert inner.hook_state is HookState.SWINGING
    assert reward == pytest.approx(500.0)
    assert inner.angle == pytest.approx(0.0)
    assert terminated is False and truncated is False

    for _ in range(3):
        check_step(0.0)
    assert inner.angle == pytest.approx(30.0)

    _, reward, terminated, truncated, _ = wrapped.step(FIRE)
    total_reward += reward
    assert inner.hook_state is HookState.SWINGING
    assert reward == pytest.approx(50.0)
    assert terminated is False and truncated is False
    assert all(not obj.active for obj in inner.objects)  # 三分全收

    steps = 0
    while not truncated:
        terminated, truncated = check_step(0.0)
        steps += 1
        assert steps < 400, "episode never truncated"

    assert terminated is False and truncated is True
    assert inner.remaining_time == 0.0
    assert total_reward == pytest.approx(inner.score)  # reward 累计一致性
    assert total_reward == pytest.approx(800.0)
    assert inner.score == pytest.approx(800.0)


# ---------------------------------------------------------------------------
# F. 非法 action：fail loudly，不得静默当作 WAIT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_action", [2, -1, 3, np.int64(7), np.int32(-2), 0.0, 1.0]
)
def test_invalid_action_raises_value_error(bad_action: object) -> None:
    """目的：action_space 之外的 action 应抛 ValueError 而非当作 WAIT。

    输入：reset 后的 wrapper，分别以若干非法动作调用 step：越界整数
    （含 np.integer 类型）以及 0.0/1.0 —— 后者与 WAIT/FIRE 数值相等，
    但不属于 Discrete(2)，底层 env 会拒绝，wrapper 不得放宽契约。
    输出：每次调用均抛 ValueError，且底层未消耗任何 tick。
    """
    wrapped = SwingDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    with pytest.raises(ValueError):
        wrapped.step(bad_action)  # type: ignore[arg-type]
    assert inner._ticks == 0  # 底层未执行任何 tick


# ---------------------------------------------------------------------------
# G. 决策状态不变量：数千随机决策后 hook 仍只在 SWINGING 时交还决策权
# ---------------------------------------------------------------------------
def test_decision_invariant_under_random_actions() -> None:
    """目的：连续随机（seed 固定）的 WAIT/FIRE 决策序列驱动数千 step，
    证明 wrapper 结构上不允许在非 SWINGING 状态获得决策机会。

    输入：fixed map，np.random.default_rng(123) 生成 np.int64 动作序列
    （仅 WAIT/FIRE），共 3000 次 wrapper.step；episode 结束则以相同
    seed reset 后继续。
    输出：每个返回点若 episode 未结束则底层 hook_state 必为 SWINGING；
    若 episode 结束则 terminated/truncated 恰有一个为 True。
    """
    wrapped = SwingDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=123)
    inner = wrapped.unwrapped
    rng = np.random.default_rng(123)

    for _ in range(3000):
        action = np.int64(rng.integers(0, 2))  # Discrete.sample() 同型
        _, _, terminated, truncated, _ = wrapped.step(action)
        if terminated or truncated:
            assert terminated != truncated  # 恰有一个为 True
            wrapped.reset(seed=123)
        else:
            assert inner.hook_state is HookState.SWINGING  # 核心不变量
