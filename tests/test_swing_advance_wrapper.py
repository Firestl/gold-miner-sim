"""``SwingAdvanceDecisionWrapper`` 行为测试（Milestone 5，issue #9）。

在 ``SwingDecisionWrapper`` 语义之上增加一条规则：FIRE 的完整出钩/
回收周期正常结束并回到 SWINGING 后，再自动推进
``ADVANCE_INTERVAL = 10`` 个 WAIT physics ticks 才返回。核心不变量：
（1）除 episode 结束的那次返回外，每次 ``step()`` 返回时底层 hook 仍
必须处于 ``HookState.SWINGING``；（2）FIRE 正常返回时角度已离开原
发射角（angle pinning 结构性消除，约 ±10°，非边界场景）。

全部使用 fixed map（完全确定）：初始角 -70°，SWINGING 每 tick 恰好
+1°，摆动一个来回恰好 280 tick，WAIT 决策每次推进 10°，决策点角度为
-70, -60, ..., +70。固定物体：GOLD(315,300)/DIAMOND(465,450)/
ROCK(610,340)。发射时的关键几何（已实测，均含 FIRE tick 本身）：
- -70° 射线不命中任何物体：空钩伸 77 tick + 空收 69 tick = 146 tick；
- -30° 射线距 GOLD 中心 1.9px（< r36）必命中：伸 34 + 载收 78 = 112；
- 0° 垂直射线距 DIAMOND 中心 15px（< r24）必命中：59 + 67 = 126；
- +20° 射线不命中任何物体（三个物体都不在该射线 40px 内）：146 tick；
- +30° 射线距 ROCK 中心约 3.6px（< r40）必命中：伸 42 + 载收 150 = 192；
- +60° 射线不命中任何物体：146 tick；advance 期间 61..70°，恰好最后
  一个 tick 到达 +70° 并按底层物理反向（swing_direction 翻转为 -1）。
底层边界语义：摆动 tick 中 ``angle >= 70`` 即夹紧到 70 并当场翻转
方向（到达 ±70° 的那一个 tick 方向已翻转）。
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
    ADVANCE_INTERVAL,
    SWING_WAIT_INTERVAL,
    SwingAdvanceDecisionWrapper,
    SwingDecisionWrapper,
)


# ---------------------------------------------------------------------------
# 14.1 WAIT 等价性：一次决策 == 裸环境连跑 10 个 WAIT tick == SDW.step(WAIT)
# ---------------------------------------------------------------------------
def test_wait_matches_ten_bare_wait_ticks() -> None:
    """目的：WAIT 语义与 SwingDecisionWrapper 完全一致（推进 10 tick）。

    输入：三个相同 seed 的 fixed map 环境：新 wrapper 执行 step(WAIT)、
    SwingDecisionWrapper 执行 step(WAIT)、裸环境依次执行 10 次
    step(WAIT)。
    输出：三者 observation 逐位相等；reward（均 0）、terminated/
    truncated（均 False）、info 相等；angle（-60°）、hook_state（均
    SWINGING）、rope_length、remaining_time 全部一致；
    ADVANCE_INTERVAL == SWING_WAIT_INTERVAL == 10。
    """
    assert ADVANCE_INTERVAL == 10
    assert SWING_WAIT_INTERVAL == 10
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    obs_w, reward_w, terminated_w, truncated_w, info_w = wrapped.step(WAIT)

    sdw = SwingDecisionWrapper(GoldMinerEnv())
    sdw.reset(seed=0)
    obs_s, reward_s, terminated_s, truncated_s, info_s = sdw.step(WAIT)

    env_b = GoldMinerEnv()
    env_b.reset(seed=0)
    obs_b: NDArray[np.float32] = np.zeros(26, dtype=np.float32)
    reward_b = 0.0
    terminated_b = truncated_b = False
    for _ in range(10):
        obs_b, reward_b, terminated_b, truncated_b, info_b = env_b.step(WAIT)

    assert np.array_equal(obs_w, obs_b)  # 与裸环境逐位相等
    assert np.array_equal(obs_w, obs_s)  # 与 Milestone 4 wrapper 逐位相等
    assert reward_w == reward_b == reward_s == 0.0
    assert terminated_w is terminated_b is terminated_s is False
    assert truncated_w is truncated_b is truncated_s is False
    assert info_w == info_b == info_s

    inner = wrapped.unwrapped
    assert inner.hook_state is HookState.SWINGING is env_b.hook_state
    assert inner.angle == pytest.approx(env_b.angle)  # -60°
    assert inner.rope_length == env_b.rope_length
    assert inner.remaining_time == env_b.remaining_time


# ---------------------------------------------------------------------------
# 14.2 空钩 FIRE + advance：一次决策内 EXTENDING→RETRACT_EMPTY→SWINGING→WAIT×10
# ---------------------------------------------------------------------------
def test_fire_empty_hook_full_cycle_plus_advance() -> None:
    """目的：-70°（不命中任何固定物体）FIRE 应在一次决策内完成空钩往返，
    再推进 10 个 SWINGING WAIT tick 才返回。

    输入：reset 后（决策点 -70°）直接 step(FIRE)。
    输出：返回时 hook_state 为 SWINGING、reward==0、rope 回到
    MIN_ROPE_LENGTH；底层恰好消耗 146（FIRE cycle）+ 10（advance）
    = 156 个 tick，remaining_time 精确等于 EPISODE_TIME - 156/SIM_FPS；
    角度已离开发射角：-70° -> -60°（方向 +1 保持，远离边界）。
    """
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    assert inner.angle == pytest.approx(-70.0)

    obs, reward, terminated, truncated, info = wrapped.step(FIRE)

    assert inner.hook_state is HookState.SWINGING
    assert reward == 0.0
    assert terminated is False and truncated is False
    assert inner.rope_length == MIN_ROPE_LENGTH
    assert inner._ticks == 146 + ADVANCE_INTERVAL  # 白盒：cycle + advance
    assert inner.remaining_time == EPISODE_TIME - 156 / SIM_FPS  # 精确
    assert inner.angle == pytest.approx(-60.0)  # 已离开发射角 -70°
    assert inner.swing_direction == 1  # 摆动方向保持
    assert info["hook_state"] == "SWINGING"  # info 存 HookState.name
    assert np.isclose(float(obs[7]), inner.remaining_time / EPISODE_TIME)


# ---------------------------------------------------------------------------
# 14.3 载钩 FIRE + advance：回收计分后再推进 10 tick
# ---------------------------------------------------------------------------
def test_fire_catches_gold_then_advances_ten_ticks() -> None:
    """目的：-30° FIRE 命中 GOLD 后应载物收回、计分，回到 SWINGING 后
    再推进恰好 10 个 WAIT tick 才返回。

    输入：reset 后 4 次 step(WAIT)（-70°→-30°，tick 40），再 step(FIRE)。
    输出：一次 FIRE 返回 reward==250、score==250、GOLD inactive、
    attached_object 为 None、hook_state 为 SWINGING、rope 最短；
    底层 tick 数 40 + 112（FIRE cycle）+ 10（advance）= 162；
    返回角度 -20°，不再是发射角 -30°。
    """
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    for _ in range(4):
        wrapped.step(WAIT)
    assert inner.angle == pytest.approx(-30.0)  # 4 × 10 tick × 1°/tick

    _obs, reward, terminated, truncated, _ = wrapped.step(FIRE)

    assert inner.hook_state is HookState.SWINGING
    assert reward == pytest.approx(250.0)
    assert inner.score == pytest.approx(250.0)
    assert inner.objects[0].active is False  # GOLD 已回收
    assert inner.attached_object is None
    assert inner.rope_length == MIN_ROPE_LENGTH
    assert terminated is False and truncated is False
    assert inner._ticks == 152 + ADVANCE_INTERVAL  # 40 + 112 + 10
    assert inner.angle == pytest.approx(-20.0)  # 回收完成后又摆过 10°


# ---------------------------------------------------------------------------
# 14.4 Angle pinning 回归：返回角 != 发射角，且与裸环境重放完全一致
# ---------------------------------------------------------------------------
def test_fire_return_angle_differs_from_fire_angle() -> None:
    """目的：非边界发射角 FIRE 正常返回后，决策角必须离开原发射角
    （消除 M4 的同角循环结构），且推进完全由底层物理驱动。

    输入：wrapper 在 -30°（非边界）FIRE；裸环境以相同 seed 手工执行
    完整 FIRE cycle（FIRE + WAIT 直到 SWINGING）再 WAIT x 10 作对照。
    输出：returned_angle（-20°）!= fire_angle（-30°）；与裸环境对照的
    最终 observation 逐位相等，angle / rope / remaining_time / tick 数
    精确相等（wrapper 没有自行计算任何角度）。
    """
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    for _ in range(4):
        wrapped.step(WAIT)
    fire_angle = inner.angle
    assert fire_angle == pytest.approx(-30.0)  # 非边界发射角

    obs_w, _reward, _terminated, _truncated, _ = wrapped.step(FIRE)
    returned_angle = inner.angle

    assert returned_angle != fire_angle  # 核心：不再停留在发射角
    assert returned_angle == pytest.approx(-20.0)  # 摆动方向 +1，前进 10°

    # 对照：裸环境完整 FIRE cycle + WAIT x 10，逐位一致。
    env_b = GoldMinerEnv()
    env_b.reset(seed=0)
    for _ in range(40):
        obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(WAIT)
    obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(FIRE)
    while env_b.hook_state is not HookState.SWINGING:
        obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(WAIT)
    for _ in range(ADVANCE_INTERVAL):
        obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(WAIT)

    assert inner._ticks == env_b._ticks == 162
    assert inner.angle == env_b.angle  # 精确相等，非近似
    assert inner.rope_length == env_b.rope_length
    assert inner.remaining_time == env_b.remaining_time
    assert inner.swing_direction == env_b.swing_direction
    assert np.array_equal(obs_w, obs_b)  # 逐位相等


# ---------------------------------------------------------------------------
# 14.5 边界反射：advance 期间到达 +70° 由底层物理当场反向
# ---------------------------------------------------------------------------
def test_post_fire_advance_reflects_at_max_angle() -> None:
    """目的：发射结束点接近 +70° 且方向向右时，post-FIRE advance 必须按
    底层 swing physics 走到 +70° 当场反向，wrapper 不得手工改角度。

    输入：reset 后 13 次 step(WAIT)（tick 130，决策点 +60°，方向 +1），
    step(FIRE)：+60° 射线为空钩（146 tick，回到 +60° 方向冻结 +1），
    advance 期间角度 61..70°，最后一个 tick 恰好到达 +70° 反向；随后
    再 step(WAIT) 继续 10 tick 应向左摆回 +60°。另以裸环境手工执行
    相同 tick 序列作对照。
    输出：FIRE 返回时 angle==+70°、swing_direction==-1（底层反向）、
    tick==276+10；随后一次 WAIT 后 angle==+60°、方向仍 -1（向左继续）；
    两个返回点的 obs 与裸环境对照逐位相等。
    """
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    for _ in range(13):
        wrapped.step(WAIT)
    assert inner.angle == pytest.approx(60.0)
    assert inner.swing_direction == 1  # 向右，即将奔向 +70° 边界

    obs_w1, reward, terminated, truncated, _ = wrapped.step(FIRE)
    assert reward == 0.0  # 空钩
    assert terminated is False and truncated is False
    assert inner._ticks == 276 + ADVANCE_INTERVAL  # 130 + 146 + 10
    assert inner.angle == pytest.approx(70.0)  # advance 末 tick 到达边界
    assert inner.swing_direction == -1  # 底层物理已当场反向

    obs_w2, _, _, _, _ = wrapped.step(WAIT)
    assert inner.angle == pytest.approx(60.0)  # 反向后向左继续摆动
    assert inner.swing_direction == -1

    # 对照：裸环境相同 tick 序列（130 WAIT + FIRE cycle + WAIT x 20）。
    env_b = GoldMinerEnv()
    env_b.reset(seed=0)
    for _ in range(130):
        obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(WAIT)
    obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(FIRE)
    while env_b.hook_state is not HookState.SWINGING:
        obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(WAIT)
    for _ in range(ADVANCE_INTERVAL):  # post-FIRE advance
        obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(WAIT)
    assert np.array_equal(obs_w1, obs_b)
    obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(WAIT)
    for _ in range(9):  # 补齐第二个 WAIT 决策的 10 tick
        obs_b, _reward_b, _terminated_b, _truncated_b, _ = env_b.step(WAIT)
    assert np.array_equal(obs_w2, obs_b)
    assert env_b.angle == pytest.approx(60.0)
    assert env_b.swing_direction == -1


# ---------------------------------------------------------------------------
# 14.6 FIRE cycle 内超时：立即返回，不执行 post-FIRE advance
# ---------------------------------------------------------------------------
def test_fire_timeout_during_cycle_skips_advance() -> None:
    """目的：剩余时间不足以完成 FIRE 时，应在 truncated 的底层 tick
    立即返回，未回收完成的物体不计分，且绝不执行 post-FIRE advance。

    输入：reset 后连续 346 次 step(WAIT)（tick 3460，摆到 +30°、剩 140
    tick），再 step(FIRE)：+30° 射线命中 ROCK（伸 42 + 载收 150 = 192
    tick > 剩余 140）。
    输出：FIRE 决策返回 truncated==True、terminated==False、
    remaining_time==0.0（精确）、底层 tick 恰为 3600；hook_state 为
    RETRACT_LOADED（若错误地执行了 advance，此处不可能为非 SWINGING）；
    reward==0、score==0、ROCK 仍 attached、三个物体均 active。
    """
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    for _ in range(346):  # 3460 tick：280 tick/来回 + 100 -> +30°
        _, reward, terminated, truncated, _ = wrapped.step(WAIT)
        assert terminated is False and truncated is False
        assert inner.hook_state is HookState.SWINGING  # WAIT 不变量
    assert inner.angle == pytest.approx(30.0)
    assert inner._ticks == 3460

    _, reward, terminated, truncated, _ = wrapped.step(FIRE)

    assert truncated is True
    assert terminated is False
    assert inner.remaining_time == 0.0  # 精确：60 - 3600/60
    assert inner._ticks == 3600  # episode 上限，无多余 tick
    assert inner.hook_state is HookState.RETRACT_LOADED  # 超时于载物收回途中
    assert inner.attached_object is inner.objects[2]  # ROCK 仍挂在钩上
    assert reward == 0.0  # 未收回到顶，不加分
    assert inner.score == 0.0
    assert all(obj.active for obj in inner.objects)  # 三个物体均未回收


# ---------------------------------------------------------------------------
# 14.7 post-FIRE advance 内超时：在 timeout tick 立即返回
# ---------------------------------------------------------------------------
def test_fire_timeout_during_advance_returns_immediately() -> None:
    """目的：FIRE cycle 已完整结束并回到 SWINGING、但剩余时间不足 10
    tick 时，advance 应在第一个 truncated tick 立即停止。

    输入：reset 后连续 345 次 step(WAIT)（tick 3450，决策点 +20°，剩
    150 tick），再 step(FIRE)：+20° 射线为空钩（146 tick，tick 3596 回
    到 SWINGING，仅剩 4 tick < ADVANCE_INTERVAL）。
    输出：FIRE 决策返回 truncated==True、terminated==False、
    reward==0、remaining_time==0.0；底层 tick 最终恰为 episode 上限
    3600（146 + 4，未执行剩余 WAIT）；hook_state 仍为 SWINGING（超时
    发生在 advance 的摆动中）；角度从发射角 +20° 推进到 +24°（恰好 4
    个 tick 的摆动量，证明执行了部分 advance 而非 0 个或 10 个）。
    """
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    for _ in range(345):  # 3450 tick：280 x 12 + 90 -> +20°，剩 150 tick
        _, _, terminated, truncated, _ = wrapped.step(WAIT)
        assert terminated is False and truncated is False
    assert inner.angle == pytest.approx(20.0)
    assert inner.remaining_time == pytest.approx(2.5)  # 150 tick

    _, reward, terminated, truncated, _ = wrapped.step(FIRE)

    assert truncated is True
    assert terminated is False
    assert reward == 0.0  # 空钩，本就无分
    assert inner.remaining_time == 0.0
    assert inner._ticks == 3600  # 3450 + 146 + 4 == episode 上限
    assert inner.hook_state is HookState.SWINGING  # 超时于 advance 摆动中
    assert inner.angle == pytest.approx(24.0)  # +20° 摆过 4 tick
    assert inner.swing_direction == 1


# ---------------------------------------------------------------------------
# 14.8 整局 reward 一致性：三分全收，sum(reward) == final score == 800
# ---------------------------------------------------------------------------
def test_full_episode_reward_consistency_three_objects() -> None:
    """目的：脚本化策略经 wrapper 跑完整局，reward 累计与最终 score
    一致，且每个非 episode 结束的返回点 hook 均为 SWINGING。

    输入：fixed map；-30° FIRE 抓 GOLD（250）→ advance 落在 -20° →
    WAIT×2 到 0° FIRE 抓 DIAMOND（500）→ advance 落在 +10° → WAIT×2
    到 +30° FIRE 抓 ROCK（50）→ advance 落在 +40° → 之后持续 WAIT
    直到 truncated。
    输出：三次 FIRE 的分步 reward 分别为 250/500/50；每次 wrapper.step
    返回（episode 结束前）底层 hook_state 均为 SWINGING；每次 FIRE 后
    的决策角都离开发射角；sum(所有分步 reward) == final score == 800；
    三个物体均 inactive；最终 truncated==True、remaining_time==0.0。
    """
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped

    total_reward = 0.0

    def fire_and_check(expected_reward: float) -> None:
        """执行一次 FIRE 决策，校验不变量、推进角与分步 reward。"""
        nonlocal total_reward
        fire_angle = inner.angle
        _, reward, terminated, truncated, _ = wrapped.step(FIRE)
        total_reward += reward
        assert terminated is False and truncated is False
        assert inner.hook_state is HookState.SWINGING  # 核心不变量
        assert inner.angle != fire_angle  # 不再停留在发射角
        assert reward == pytest.approx(expected_reward)

    for _ in range(4):
        _, reward, terminated, truncated, _ = wrapped.step(WAIT)
        total_reward += reward
        assert reward == 0.0 and not (terminated or truncated)
    assert inner.angle == pytest.approx(-30.0)

    fire_and_check(250.0)  # GOLD
    assert inner.angle == pytest.approx(-20.0)  # advance 落点

    for _ in range(2):
        _, reward, terminated, truncated, _ = wrapped.step(WAIT)
        total_reward += reward
        assert reward == 0.0 and not (terminated or truncated)
    assert inner.angle == pytest.approx(0.0)

    fire_and_check(500.0)  # DIAMOND
    assert inner.angle == pytest.approx(10.0)

    for _ in range(2):
        _, reward, terminated, truncated, _ = wrapped.step(WAIT)
        total_reward += reward
        assert reward == 0.0 and not (terminated or truncated)
    assert inner.angle == pytest.approx(30.0)

    fire_and_check(50.0)  # ROCK
    assert inner.angle == pytest.approx(40.0)
    assert all(not obj.active for obj in inner.objects)  # 三分全收

    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = wrapped.step(WAIT)
        total_reward += reward
        assert reward == 0.0
        if not (terminated or truncated):
            assert inner.hook_state is HookState.SWINGING
        steps += 1
        assert steps < 400, "episode never truncated"

    assert terminated is False and truncated is True
    assert inner.remaining_time == 0.0
    assert total_reward == pytest.approx(inner.score)  # reward 累计一致性
    assert total_reward == pytest.approx(800.0)
    assert inner.score == pytest.approx(800.0)


# ---------------------------------------------------------------------------
# 非法 action：fail loudly，不得静默当作 WAIT（issue §4）
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
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
    wrapped.reset(seed=0)
    inner = wrapped.unwrapped
    with pytest.raises(ValueError):
        wrapped.step(bad_action)  # type: ignore[arg-type]
    assert inner._ticks == 0  # 底层未执行任何 tick


# ---------------------------------------------------------------------------
# 决策状态不变量：数千随机决策后 hook 仍只在 SWINGING 时交还决策权
# ---------------------------------------------------------------------------
def test_decision_invariant_under_random_actions() -> None:
    """目的：连续随机（seed 固定）的 WAIT/FIRE 决策序列驱动数千 step，
    证明 wrapper 结构上不允许在非 SWINGING 状态获得决策机会，且
    np.int64 动作（Discrete.sample() 的返回类型）始终被接受。

    输入：fixed map，np.random.default_rng(123) 生成 np.int64 动作序列
    （仅 WAIT/FIRE），共 3000 次 wrapper.step；episode 结束则以相同
    seed reset 后继续。
    输出：每个返回点若 episode 未结束则底层 hook_state 必为 SWINGING；
    若 episode 结束则 terminated/truncated 恰有一个为 True。
    """
    wrapped = SwingAdvanceDecisionWrapper(GoldMinerEnv())
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
