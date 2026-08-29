"""``GoldMinerEnv``（Milestone 1，issue #1 第 15 节）行为测试。

环境完全确定，因此多数断言精确到 tick 算术：一个 tick 为 DT = 1/60 s，
WAIT 每 tick 恰好摆动 1 度，并在 ±70 度处钳制。

仅测试 ``gold_miner_sim.env`` 的公开行为，另含一个白盒集成测试：
把物体移到另一物体的发射射线上，验证"最近优先"碰撞选择。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from gold_miner_sim.env import (
    ANCHOR,
    DT,
    EMPTY_RETRACT_SPEED,
    EPISODE_TIME,
    EXTENSION_SPEED,
    FIRE,
    MAX_ANGLE,
    MAX_ROPE_LENGTH,
    MIN_ANGLE,
    MIN_ROPE_LENGTH,
    SIM_FPS,
    SWING_ANGULAR_SPEED,
    WAIT,
    GameObject,
    GoldMinerEnv,
    HookState,
    ObjectType,
    sweep_circle_hit,
)

INITIAL_ANGLE = -70.0
# 每 tick 恰好摆动 1 度，故 0.6 度容差可无歧义地选中摆动跨越目标角的那一 tick。
ANGLE_TOL = 0.6

# 固定地图（issue #1 第 4 节）：槽位顺序 GOLD/DIAMOND/ROCK。
MAP = (
    (ObjectType.GOLD, (315.0, 300.0), 30.0, 250.0, 140.0),
    (ObjectType.DIAMOND, (465.0, 450.0), 18.0, 500.0, 280.0),
    (ObjectType.ROCK, (610.0, 340.0), 34.0, 50.0, 90.0),
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _wait_until_swinging(env: GoldMinerEnv, max_steps: int = 400) -> None:
    """目的：不断 step(WAIT) 直到钩子回到 SWINGING。

    输入：环境 env，可选步数上限 max_steps（默认 400）。
    输出：无返回值；结束时 hook_state 为 SWINGING，超限则断言失败。
    """
    steps = 0
    while env.hook_state is not HookState.SWINGING:
        env.step(WAIT)
        steps += 1
        assert steps < max_steps, "hook never returned to SWINGING"


def _fire_at_angle(env: GoldMinerEnv, target_angle: float, tol: float = ANGLE_TOL) -> None:
    """目的：WAIT 等待摆动到目标角附近后发射。

    输入：环境 env、目标角度 target_angle、容差 tol（默认 0.6°）。
    输出：无返回值；FIRE 后 hook_state 为 EXTENDING。
    """
    _wait_until_swinging(env)
    steps = 0
    while abs(env.angle - target_angle) > tol:
        env.step(WAIT)
        steps += 1
        assert steps < 200, f"swing never reached {target_angle} degrees"
    env.step(FIRE)
    assert env.hook_state is HookState.EXTENDING


def _run_loaded_retract(
    env: GoldMinerEnv,
    slot: int,
    expected_type: ObjectType,
    expected_value: float,
    expected_speed: float,
) -> None:
    """目的：从发射后开始，验证抓取、带载收回、计分的完整周期。

    输入：环境 env、目标物体槽位 slot，以及期望的类型/分值/收回速度。
    输出：无返回值；逐 tick 断言命中后物体附着且中心锁定钩尖、绳长按
    expected_speed 匀速递减、收回期间 reward=0；到达顶部恰好一次性得
    expected_value 分，物体失效，回到 SWINGING 且绳长恢复最小值。
    """
    # 伸出阶段，直到命中。
    steps = 0
    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)
        steps += 1
        assert steps < 200, "hook never hit the expected object"
    assert env.hook_state is HookState.RETRACT_LOADED
    attached = env.attached_object
    assert attached is not None and attached is env.objects[slot]
    assert attached.type is expected_type

    # 带载收回：不计分、绳长每 tick 匀速缩短、物体中心每 tick 锁定钩尖。
    score_at_catch = env.score
    scoring_rewards: list[float] = []
    prev_rope = env.rope_length
    steps = 0
    while env.hook_state is HookState.RETRACT_LOADED:
        _, reward, terminated, truncated, _ = env.step(WAIT)
        assert not terminated and not truncated
        if env.hook_state is HookState.RETRACT_LOADED:
            assert reward == 0.0
            assert env.score == pytest.approx(score_at_catch)
            assert (attached.x, attached.y) == env.hook_tip  # 精确相等：本 tick 刚赋值
            assert env.rope_length == pytest.approx(prev_rope - expected_speed * DT)
        else:
            scoring_rewards.append(reward)
        prev_rope = env.rope_length
        steps += 1
        assert steps < 600, "loaded retract never finished"

    # 到达顶部时恰好计分一次，分值等于物体价值。
    assert len(scoring_rewards) == 1
    assert scoring_rewards[0] == pytest.approx(expected_value)
    assert env.score == pytest.approx(score_at_catch + expected_value)
    assert attached.active is False
    assert env.attached_object is None
    assert env.hook_state is HookState.SWINGING
    assert env.rope_length == pytest.approx(MIN_ROPE_LENGTH)


def _place_on_ray(obj: GameObject, angle_deg: float, rope: float) -> None:
    """目的（白盒）：把物体中心放到 angle_deg 方向的发射射线上，
    距锚点径向距离 rope。

    输入：物体对象 obj、角度 angle_deg、径向距离 rope。
    输出：无返回值；obj.x/obj.y 被改写为射线上的对应坐标。
    """
    rad = math.radians(angle_deg)
    obj.x = ANCHOR[0] + rope * math.sin(rad)
    obj.y = ANCHOR[1] + rope * math.cos(rad)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
def test_reset_initial_state() -> None:
    """目的：校验 reset 后的完整初始状态。

    输入：默认环境，reset(seed=42)。
    输出：分数 0、剩余时间=EPISODE_TIME、钩子 SWINGING、角度 -70°、
    方向 +1、绳长=MIN_ROPE_LENGTH、无附着物体；3 个物体按固定地图
    GOLD/DIAMOND/ROCK 生成且全部 active；obs 为 float32、shape=(26,)、
    落在观察空间内；info 含 score、remaining_time、hook_state="SWINGING"。
    """
    env = GoldMinerEnv()
    obs, info = env.reset(seed=42)

    assert env.score == 0.0
    assert env.remaining_time == pytest.approx(EPISODE_TIME)
    assert env.hook_state is HookState.SWINGING
    assert env.angle == pytest.approx(INITIAL_ANGLE)
    assert env.swing_direction == 1
    assert env.rope_length == pytest.approx(MIN_ROPE_LENGTH)
    assert env.attached_object is None

    assert len(env.objects) == 3
    for obj, (obj_type, (x, y), radius, value, speed) in zip(env.objects, MAP):
        assert isinstance(obj, GameObject)
        assert obj.type is obj_type
        assert (obj.x, obj.y) == (x, y)
        assert obj.radius == radius
        assert obj.value == value
        assert obj.retract_speed == speed
        assert obj.active is True

    assert obs.shape == (26,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["score"] == 0.0
    assert info["remaining_time"] == pytest.approx(EPISODE_TIME)
    assert info["hook_state"] == "SWINGING"


def test_reset_clears_state_from_previous_episode() -> None:
    """目的：reset 应彻底清除上一局的残留状态。

    输入：先部分游玩——抓取 GOLD 使绳长/角度/附着/物体状态变化；
    再以不同 seed 调用 reset。
    输出：分数/时间/角度/方向/绳长/钩子状态全部回到初始值，所有物体恢复
    active 且 GOLD 位置不变；reset 后的 obs 与全新环境的 obs 逐位相等。
    """
    env = GoldMinerEnv()
    obs_fresh, _ = env.reset(seed=1)

    # 先部分游玩：抓取 GOLD，使绳长/角度/附着/物体状态发生变化。
    _fire_at_angle(env, -30.0)
    steps = 0
    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)
        steps += 1
    assert env.hook_state is HookState.RETRACT_LOADED
    assert env.attached_object is not None

    obs, _ = env.reset(seed=7)  # 不同 seed：固定地图不应改变
    assert env.score == 0.0
    assert env.remaining_time == pytest.approx(EPISODE_TIME)
    assert env.hook_state is HookState.SWINGING
    assert env.angle == pytest.approx(INITIAL_ANGLE)
    assert env.swing_direction == 1
    assert env.rope_length == pytest.approx(MIN_ROPE_LENGTH)
    assert env.attached_object is None
    assert all(obj.active for obj in env.objects)
    gold = env.objects[0]
    assert (gold.x, gold.y) == (315.0, 300.0)
    assert env.observation_space.contains(obs)
    # 无残留：reset 后的观测与全新环境逐位相等。
    other = GoldMinerEnv()
    obs_other, _ = other.reset(seed=99)
    assert np.array_equal(obs, obs_other)
    assert np.array_equal(obs_fresh, obs_other)


# ---------------------------------------------------------------------------
# 摆动
# ---------------------------------------------------------------------------
def test_swing_updates_angle_one_degree_per_wait() -> None:
    """目的：每个 WAIT tick 钩子恰好转动 1 度。

    输入：默认环境，reset 后执行一次 step(WAIT)。
    输出：角度精确等于 -70° + SWING_ANGULAR_SPEED * DT（即 -69°），方向仍为 +1。
    """
    env = GoldMinerEnv()
    env.reset()
    env.step(WAIT)
    assert env.angle == pytest.approx(INITIAL_ANGLE + SWING_ANGULAR_SPEED * DT)
    assert env.angle == pytest.approx(-69.0)
    assert env.swing_direction == 1


def test_swing_reverses_at_boundaries() -> None:
    """目的：摆动到达 ±70° 边界时精确钳制并反向。

    输入：默认环境，连续 WAIT 先摆到 +70°，再摆回 -70°。
    输出：+70° 处方向变为 -1；-70° 处方向变回 +1；下一 tick 角度为 -69°，无漂移。
    """
    env = GoldMinerEnv()
    env.reset()

    # 向右摆到 +70° 钳制（无漂移：恰好 140 tick）。
    for _ in range(200):
        if env.swing_direction == -1:
            break
        env.step(WAIT)
    assert env.angle == pytest.approx(MAX_ANGLE)
    assert env.swing_direction == -1

    # 向左摆到 -70° 钳制，然后再次向右。
    for _ in range(200):
        if env.swing_direction == 1:
            break
        env.step(WAIT)
    assert env.angle == pytest.approx(MIN_ANGLE)
    assert env.swing_direction == 1
    env.step(WAIT)
    assert env.angle == pytest.approx(MIN_ANGLE + 1.0)


def test_swing_never_exceeds_bounds() -> None:
    """目的：随机动作序列下角度始终不越界。

    输入：默认环境，以固定种子随机生成 1000 次 WAIT/FIRE。
    输出：每步均满足 MIN_ANGLE ≤ angle ≤ MAX_ANGLE，方向始终为 ±1。
    """
    env = GoldMinerEnv()
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(1000):
        env.step(int(rng.integers(0, 2)))  # 随机 WAIT/FIRE 动作序列
        assert MIN_ANGLE <= env.angle <= MAX_ANGLE
        assert env.swing_direction in (-1, 1)


# ---------------------------------------------------------------------------
# 发射
# ---------------------------------------------------------------------------
def test_fire_enters_extending_and_freezes_angle() -> None:
    """目的：FIRE 进入 EXTENDING 并冻结角度。

    输入：默认环境，step(FIRE) 后连续 20 次 WAIT（远未触及 460 px 绳长上限）。
    输出：钩子保持 EXTENDING，角度不变，绳长每 tick 精确增加
    EXTENSION_SPEED * DT。
    """
    env = GoldMinerEnv()
    env.reset()
    env.step(FIRE)
    assert env.hook_state is HookState.EXTENDING
    frozen_angle = env.angle
    rope = env.rope_length
    for _ in range(20):  # 连同 FIRE tick 共 21 个伸出 tick：远离 460 px 钳制
        _, _, _, truncated, _ = env.step(WAIT)
        assert not truncated
        assert env.angle == frozen_angle
        assert env.rope_length == pytest.approx(rope + EXTENSION_SPEED * DT)
        rope = env.rope_length


def test_fire_has_no_effect_outside_swinging() -> None:
    """目的：非 SWINGING 状态下 FIRE 应被忽略。

    输入：-70° 空射后，分别在 EXTENDING 阶段反复 FIRE、在 RETRACT_EMPTY
    阶段 FIRE，最后回到 SWINGING 再 FIRE。
    输出：EXTENDING/RETRACT_EMPTY 中的 FIRE 不改变状态且 reward=0，绳长按
    原速度变化；回到 SWINGING 后 FIRE 正常触发 EXTENDING。
    """
    env = GoldMinerEnv()
    env.reset()
    env.step(FIRE)  # -70° 空射：不会命中任何物体
    for _ in range(5):
        env.step(WAIT)

    # EXTENDING 中的 FIRE 不得重启或改变本次发射。
    _, reward, _, truncated, _ = env.step(FIRE)
    assert not truncated
    assert reward == 0.0
    assert env.hook_state is HookState.EXTENDING
    for _ in range(3):
        rope = env.rope_length
        env.step(FIRE)  # 重复 FIRE 持续被忽略
        assert env.hook_state is HookState.EXTENDING
        assert env.rope_length == pytest.approx(rope + EXTENSION_SPEED * DT)

    # RETRACT_EMPTY 中的 FIRE 同样不得重新发射。
    _wait_until_state(env, HookState.RETRACT_EMPTY)
    rope = env.rope_length
    env.step(FIRE)
    assert env.hook_state is HookState.RETRACT_EMPTY
    assert env.rope_length == pytest.approx(rope - EMPTY_RETRACT_SPEED * DT)

    # 回到 SWINGING 后 FIRE 正常生效。
    _wait_until_swinging(env)
    env.step(FIRE)
    assert env.hook_state is HookState.EXTENDING


def _wait_until_state(env: GoldMinerEnv, state: HookState, max_steps: int = 400) -> None:
    """目的：不断 step(WAIT) 直到 hook_state 达到指定状态。

    输入：环境 env、目标状态 state，可选步数上限 max_steps（默认 400）。
    输出：无返回值；达到目标状态返回，超限则断言失败。
    """
    steps = 0
    while env.hook_state is not state:
        env.step(WAIT)
        steps += 1
        assert steps < max_steps, f"hook never entered {state}"


# ---------------------------------------------------------------------------
# 空钩收回
# ---------------------------------------------------------------------------
def test_empty_retract_full_cycle() -> None:
    """目的：空钩发射的伸出-收回完整周期精确到 tick。

    输入：默认环境，reset 后在 -70° 直接 FIRE（射线不经过任何物体）。
    输出：77 个伸出 tick 后绳长钳制在 MAX_ROPE_LENGTH 并转入
    RETRACT_EMPTY，角度冻结为 -70°、分数 0；再经 69 个收回 tick 回到
    SWINGING、绳长=MIN_ROPE_LENGTH，全程 reward=0、分数不变。
    """
    env = GoldMinerEnv()
    env.reset()
    env.step(FIRE)  # -70° 射线不经过任何物体

    # 伸出：50 + 77 * (320/60) 越过 460 -> 钳制并切换状态。
    steps = 0
    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)
        steps += 1
    assert steps == 76  # 加上 FIRE tick 本身共 77 个伸出 tick
    assert env.hook_state is HookState.RETRACT_EMPTY
    assert env.rope_length == pytest.approx(MAX_ROPE_LENGTH)
    assert env.angle == pytest.approx(INITIAL_ANGLE)  # 已冻结
    assert env.score == 0.0

    # 空钩收回速度 360 px/s：69 tick 回到最小绳长。
    steps = 0
    while env.hook_state is HookState.RETRACT_EMPTY:
        rope = env.rope_length
        _, reward, _, truncated, _ = env.step(WAIT)
        assert reward == 0.0 and not truncated
        expected = max(rope - EMPTY_RETRACT_SPEED * DT, MIN_ROPE_LENGTH)
        assert env.rope_length == pytest.approx(expected)
        steps += 1
    assert steps == 69
    assert env.hook_state is HookState.SWINGING
    assert env.rope_length == pytest.approx(MIN_ROPE_LENGTH)
    assert env.angle == pytest.approx(INITIAL_ANGLE)
    assert env.score == 0.0


def test_empty_retract_preserves_swing_direction() -> None:
    """目的：空钩收回后保持发射前的摆动方向。

    输入：先摆到 +70°（方向变为 -1），再摆到 +40° 时 FIRE，空钩收回至 SWINGING。
    输出：方向仍为 -1、角度仍为 40°，下一 tick 变为 39°（继续向左）。
    """
    env = GoldMinerEnv()
    env.reset()
    for _ in range(140):  # 向右摆到 +70° 钳制
        env.step(WAIT)
    assert env.swing_direction == -1
    for _ in range(30):  # 此时 +40° 向左摆动；射线不命中任何物体
        env.step(WAIT)
    assert env.angle == pytest.approx(40.0)
    env.step(FIRE)
    _wait_until_swinging(env)
    # 保持发射前的方向（仍向左）。
    assert env.swing_direction == -1
    assert env.angle == pytest.approx(40.0)
    env.step(WAIT)
    assert env.angle == pytest.approx(39.0)


# ---------------------------------------------------------------------------
# 带载收回（每种物体各一次完整抓取）
# ---------------------------------------------------------------------------
def test_loaded_retract_gold() -> None:
    """目的：完整验证 GOLD 的抓取-带载收回-计分周期。

    输入：-30° 射线发射（距 GOLD 中心 1.9 px）。
    输出：命中后按 140 px/s 带载收回，到达顶部一次性得 +250 分并回到
    SWINGING；后续 tick 不重复计分。
    """
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, -30.0)  # -30° 射线距 GOLD 中心 1.9 px
    _run_loaded_retract(env, slot=0, expected_type=ObjectType.GOLD,
                        expected_value=250.0, expected_speed=140.0)
    # +250 奖励不得触发第二次。
    _, reward, _, truncated, _ = env.step(WAIT)
    assert reward == 0.0 and not truncated
    assert env.score == pytest.approx(250.0)


def test_loaded_retract_diamond() -> None:
    """目的：完整验证 DIAMOND 的抓取-带载收回-计分周期。

    输入：2° 射线发射。
    输出：命中 DIAMOND，按 280 px/s 带载收回，到达顶部一次性得 +500 分。
    """
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, 2.0)
    _run_loaded_retract(env, slot=1, expected_type=ObjectType.DIAMOND,
                        expected_value=500.0, expected_speed=280.0)


def test_loaded_retract_rock() -> None:
    """目的：完整验证 ROCK 的抓取-带载收回-计分周期。

    输入：31° 射线发射。
    输出：命中 ROCK，按 90 px/s 带载收回，到达顶部一次性得 +50 分。
    """
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, 31.0)
    _run_loaded_retract(env, slot=2, expected_type=ObjectType.ROCK,
                        expected_value=50.0, expected_speed=90.0)


def test_hit_step_syncs_attached_object_to_hook_tip() -> None:
    """目的：命中当帧物体坐标应立即同步到钩尖（issue #1 第 5.4 节），
    而不是停留在地图原位置。

    输入：-30° 射线正常抓取 GOLD，step 直到退出 EXTENDING。
    输出：转入 RETRACT_LOADED 的同一 tick，附着物体坐标等于钩尖坐标。
    """
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, -30.0)  # 正常抓取 GOLD

    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)

    assert env.hook_state is HookState.RETRACT_LOADED
    attached = env.attached_object
    assert attached is not None and attached is env.objects[0]
    tip_x, tip_y = env.hook_tip
    assert attached.x == pytest.approx(tip_x)
    assert attached.y == pytest.approx(tip_y)


# ---------------------------------------------------------------------------
# 碰撞
# ---------------------------------------------------------------------------
def test_sweep_circle_hit_through_center() -> None:
    """目的：射线穿过圆心时扫掠碰撞返回正确的命中参数。

    输入：线段 (0,0)→(10,0)，圆心 (5,0)、半径 2。
    输出：返回 t≈0.3，且 0 < t < 1。
    """
    t = sweep_circle_hit(0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 2.0)
    assert t is not None
    assert 0.0 < t < 1.0
    assert t == pytest.approx(0.3)


def test_sweep_circle_hit_near_miss() -> None:
    """目的：擦边、段末端之外、起点后方的圆均不应命中。

    输入：线段 (0,0)→(10,0)、半径 2 的圆，分别置于垂直距离 2.5（> 半径 2）、
    段末端外 (15,0)、起点后方 (-5,0)。
    输出：三种情况均返回 None。
    """
    # 垂直距离 2.5 > r = 2：擦边不算命中。
    assert sweep_circle_hit(0.0, 0.0, 10.0, 0.0, 5.0, 2.5, 2.0) is None
    # 圆在段末端之外 / 起点后方。
    assert sweep_circle_hit(0.0, 0.0, 10.0, 0.0, 15.0, 0.0, 2.0) is None
    assert sweep_circle_hit(0.0, 0.0, 10.0, 0.0, -5.0, 0.0, 2.0) is None


def test_sweep_circle_hit_edge_cases() -> None:
    """目的：起点在圆内与零长度线段的退化情形。

    输入：起点 (0,0) 已在圆内（半径 2）；零长度线段分别位于圆内 (5,0)
    与圆外 (5,3)。
    输出：起点在圆内返回 t=0.0；零长段在圆内返回 t=0.0（退化为点在圆内
    判定），在圆外返回 None。
    """
    # 起点已在圆内。
    assert sweep_circle_hit(0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 2.0) == 0.0
    # 零长度退化线段：退化为点在圆内判定。
    assert sweep_circle_hit(5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 2.0) == 0.0
    assert sweep_circle_hit(5.0, 3.0, 5.0, 3.0, 5.0, 0.0, 2.0) is None


def test_collision_picks_nearest_object_on_same_ray() -> None:
    """目的：同一射线上命中最近的物体（白盒）。

    输入：把 DIAMOND 移到 -30° 射线上 GOLD 后方 400 px 处，按该角度连续两次发射。
    输出：第一发只抓到 GOLD，DIAMOND 保持 active；第二发（角度冻结不变）
    抓到 DIAMOND，ROCK 仍 active，总分 750。
    """
    env = GoldMinerEnv()
    env.reset()
    # 白盒：把 DIAMOND 移到 -30° 射线上、GOLD 后方远处。
    rad = math.radians(-30.0)
    far_rope = 400.0
    diamond = env.objects[1]
    diamond.x = ANCHOR[0] + far_rope * math.sin(rad)
    diamond.y = ANCHOR[1] + far_rope * math.cos(rad)

    _fire_at_angle(env, -30.0)
    _run_loaded_retract(env, slot=0, expected_type=ObjectType.GOLD,
                        expected_value=250.0, expected_speed=140.0)
    assert diamond.active is True  # 远处物体未被抓取

    # 角度冻结不变，下一次发射抓到（现在是最近的）DIAMOND。
    env.step(FIRE)
    assert env.hook_state is HookState.EXTENDING
    _run_loaded_retract(env, slot=1, expected_type=ObjectType.DIAMOND,
                        expected_value=500.0, expected_speed=280.0)
    assert env.objects[2].active is True
    assert env.score == pytest.approx(750.0)


def test_max_rope_boundary_collision_ignores_overshoot() -> None:
    """目的：仅与最后一 tick 超出绳长上限的越界路径相切的物体不算命中
    （扫掠按钳制到 MAX_ROPE_LENGTH 的路径计算，防止越界误抓）。

    输入：GOLD 半径改为 24.4，中心放在 -70° 射线上、超出绳长上限的
    overshoot 段处（有效接触半径 30.4 px 的近边缘位于越界段内、但严格
    大于 MAX_ROPE_LENGTH），从初始 -70° 直接发射。
    输出：先做几何自检——合法路径不命中、未钳制的越界路径才命中；
    实际发射后结果为 RETRACT_EMPTY，绳长=MAX_ROPE_LENGTH，无附着物体，
    GOLD 仍 active。
    """
    env = GoldMinerEnv()
    env.reset()

    # 复现伸出阶段的 tick 算术，找出最后一个伸出 tick：
    # prev_rope -> min(prev_rope + EXTENSION_SPEED * DT, MAX_ROPE_LENGTH)。
    prev_rope = MIN_ROPE_LENGTH
    while prev_rope + EXTENSION_SPEED * DT < MAX_ROPE_LENGTH:
        prev_rope += EXTENSION_SPEED * DT
    overshoot_end = prev_rope + EXTENSION_SPEED * DT
    assert MAX_ROPE_LENGTH < overshoot_end  # 未钳制终点越过上限

    # 接触圆 = radius + HOOK_RADIUS = 30.4 px。其圆心在未钳制终点之外
    # 30 px，近边缘（overshoot_end - 0.4）位于越界段内、但严格大于
    # MAX_ROPE_LENGTH：合法扫掠路径够不到它，未钳制的（有 bug 的）路径
    # 才会命中。
    gold = env.objects[0]
    gold.radius = 24.4
    _place_on_ray(gold, INITIAL_ANGLE, overshoot_end + 30.0)

    # 几何自检：使用与环境相同的扫掠原语。
    rad = math.radians(INITIAL_ANGLE)
    sin_a, cos_a = math.sin(rad), math.cos(rad)
    tip_prev = (ANCHOR[0] + prev_rope * sin_a, ANCHOR[1] + prev_rope * cos_a)
    tip_legal = (
        ANCHOR[0] + MAX_ROPE_LENGTH * sin_a,
        ANCHOR[1] + MAX_ROPE_LENGTH * cos_a,
    )
    tip_buggy = (ANCHOR[0] + overshoot_end * sin_a, ANCHOR[1] + overshoot_end * cos_a)
    eff_radius = gold.radius + 6.0  # HOOK_RADIUS
    assert sweep_circle_hit(*tip_prev, *tip_legal, gold.x, gold.y, eff_radius) is None
    assert sweep_circle_hit(*tip_legal, *tip_buggy, gold.x, gold.y, eff_radius) is not None

    env.step(FIRE)  # 从初始 -70° 摆动直接发射
    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)

    assert env.hook_state is HookState.RETRACT_EMPTY  # 未命中，而非抓取
    assert env.rope_length == pytest.approx(MAX_ROPE_LENGTH)
    assert env.attached_object is None
    assert gold.active is True


def test_collision_on_legal_path_near_max_rope_hits() -> None:
    """目的：overshoot 测试的对照组——合法路径上的物体应被正常抓取。

    输入：GOLD 半径改为 24.4，中心放在 -70° 射线上距锚点
    MAX_ROPE_LENGTH - 20 px 处（在合法路径内）。
    输出：命中并转入 RETRACT_LOADED，attached_object 即该 GOLD。
    """
    env = GoldMinerEnv()
    env.reset()
    gold = env.objects[0]
    gold.radius = 24.4
    _place_on_ray(gold, INITIAL_ANGLE, MAX_ROPE_LENGTH - 20.0)

    env.step(FIRE)
    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)

    assert env.hook_state is HookState.RETRACT_LOADED
    assert env.attached_object is gold


# ---------------------------------------------------------------------------
# 观测契约（issue #1 第 8 节：固定 26 槽位布局）
# ---------------------------------------------------------------------------
def test_observation_contract_reset_exact_26_values() -> None:
    """目的：reset 后 26 维观测完全符合固定布局契约。

    输入：默认环境，reset 后取 obs。
    输出：obs 逐位等于期望向量——归一化角度 -1、方向 1、one-hot SWINGING、
    归一化绳长 50/460、剩余时间 1，以及 GOLD/DIAMOND/ROCK 三个槽位的
    (x/900, y/600, r/100, value/500, speed/360, 1)。
    """
    env = GoldMinerEnv()
    obs, _ = env.reset()
    expected = np.array(
        [
            INITIAL_ANGLE / MAX_ANGLE,  # 0: 归一化角度 = -1
            1.0,  # 1: 摆动方向
            1.0, 0.0, 0.0, 0.0,  # 2-5: one-hot SWINGING
            MIN_ROPE_LENGTH / MAX_ROPE_LENGTH,  # 6: 归一化绳长 50/460
            EPISODE_TIME / EPISODE_TIME,  # 7: 剩余时间 60/60 = 1
            # GOLD 槽位 (8-13)
            315.0 / 900.0, 300.0 / 600.0, 30.0 / 100.0,
            250.0 / 500.0, 140.0 / 360.0, 1.0,
            # DIAMOND 槽位 (14-19)
            465.0 / 900.0, 450.0 / 600.0, 18.0 / 100.0,
            500.0 / 500.0, 280.0 / 360.0, 1.0,
            # ROCK 槽位 (20-25)
            610.0 / 900.0, 340.0 / 600.0, 34.0 / 100.0,
            50.0 / 500.0, 90.0 / 360.0, 1.0,
        ],
        dtype=np.float32,
    )
    assert obs.shape == (26,)
    assert obs == pytest.approx(expected)


def test_observation_contract_tracks_state_changes() -> None:
    """目的：观测随物理状态逐 tick 变化。

    输入：默认环境，依次 step(FIRE)、step(WAIT)。
    输出：FIRE tick 进入 EXTENDING 且已完成一步物理——one-hot 变为
    EXTENDING、绳长增加一步伸长、时间消耗 1/3600；再 WAIT 后绳长与时间
    各再推进一步，角度保持冻结。
    """
    env = GoldMinerEnv()
    env.reset()

    # FIRE tick 进入 EXTENDING 且已推进一步物理：one-hot 变化、绳长增加
    # 一步伸长、消耗 1/3600 的归一化时间预算。
    obs, _, _, truncated, _ = env.step(FIRE)
    assert not truncated
    assert obs[0] == pytest.approx(INITIAL_ANGLE / MAX_ANGLE)  # 角度冻结
    assert obs[1] == 1.0
    assert list(obs[2:6]) == [0.0, 1.0, 0.0, 0.0]  # one-hot EXTENDING
    assert obs[6] == pytest.approx(
        (MIN_ROPE_LENGTH + EXTENSION_SPEED * DT) / MAX_ROPE_LENGTH
    )
    assert obs[7] == pytest.approx((EPISODE_TIME - DT) / EPISODE_TIME)

    obs, _, _, truncated, _ = env.step(WAIT)
    assert not truncated
    assert list(obs[2:6]) == [0.0, 1.0, 0.0, 0.0]
    assert obs[6] == pytest.approx(
        (MIN_ROPE_LENGTH + 2.0 * EXTENSION_SPEED * DT) / MAX_ROPE_LENGTH
    )
    assert obs[7] == pytest.approx((EPISODE_TIME - 2.0 * DT) / EPISODE_TIME)


def test_observation_contract_collected_object_slots_zeroed() -> None:
    """目的：被抓取物体的观测槽位整体清零。

    输入：-30° 抓取 GOLD，step 到计分并回到 SWINGING 的那一 tick 取 obs。
    输出：分数 250；槽位 8-13 恰好全 0，DIAMOND/ROCK 槽位保持契约值不变。
    """
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, -30.0)
    obs: np.ndarray | None = None
    while True:
        obs, _, _, truncated, _ = env.step(WAIT)
        assert not truncated
        if env.hook_state is HookState.SWINGING:
            break  # 计分 GOLD 并回到 SWINGING 的那一 tick
    assert obs is not None
    assert env.score == pytest.approx(250.0)
    # 被收取物体的槽位恰好全 0。
    assert list(obs[8:14]) == [0.0] * 6
    # 其余槽位保持契约值不变。
    assert obs[14:20] == pytest.approx(
        np.array([465.0 / 900.0, 450.0 / 600.0, 18.0 / 100.0,
                  500.0 / 500.0, 280.0 / 360.0, 1.0], dtype=np.float32)
    )
    assert obs[20:26] == pytest.approx(
        np.array([610.0 / 900.0, 340.0 / 600.0, 34.0 / 100.0,
                  50.0 / 500.0, 90.0 / 360.0, 1.0], dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# 超时
# ---------------------------------------------------------------------------
def test_full_episode_runs_exactly_3600_ticks_then_truncates() -> None:
    """目的：整局恰好在第 3600 tick 超时截断。

    输入：默认环境，连续 step(WAIT) 3600 次。
    输出：前 3599 步 truncated=False，第 3600 步 truncated=True；
    terminated 恒为 False、reward 恒为 0、info["score"] 恒为 0；
    结束后 remaining_time=0.0，所有物体仍 active。
    """
    env = GoldMinerEnv()
    env.reset()
    for tick in range(1, SIM_FPS * int(EPISODE_TIME) + 1):  # 3600 ticks
        _, reward, terminated, truncated, info = env.step(WAIT)
        assert terminated is False
        assert reward == 0.0
        assert info["score"] == 0.0
        if tick < 3600:
            assert truncated is False
        else:
            assert truncated is True
    assert env.remaining_time == 0.0  # 精确：60 - 3600/60
    assert env.score == 0.0
    assert all(obj.active for obj in env.objects)


def test_timeout_discards_unfinished_loaded_retract() -> None:
    """目的：超时丢弃未完成的带载收回，不计分。

    输入：先空射一次平移摆动相位，等到 episode 尾段（剩余时间 0.65~1.5 s）
    再次出现 -30° 方向时发射抓取 GOLD。
    输出：episode 在 RETRACT_LOADED 中途 truncated；remaining_time=0、
    score=0，GOLD 仍 active 且坐标锁定在钩尖。
    """
    env = GoldMinerEnv()
    env.reset()
    gold = env.objects[0]

    # 从 -70° 空射一次使摆动相位平移 146 tick，让 -30° 方向在 episode
    # 尾段（剩余不足 1 s）再次出现。
    env.step(FIRE)
    _wait_until_swinging(env)

    fired = False
    truncated = False
    info: dict = {}
    for _ in range(4000):
        ready = (
            not fired
            and env.hook_state is HookState.SWINGING
            and abs(env.angle - (-30.0)) <= ANGLE_TOL
            and 0.65 <= env.remaining_time <= 1.5
        )
        if ready:
            # GOLD 约需 34 tick 命中 + 78 tick 收回（共 1.87 s）；剩余
            # ≤ 1.5 s 时 episode 必在收回途中结束。
            _, _, _, truncated, info = env.step(FIRE)
            fired = True
        else:
            _, _, _, truncated, info = env.step(WAIT)
        if truncated:
            break

    assert fired, "never reached a -30 deg swing with little time left"
    assert truncated is True
    assert env.remaining_time == 0.0
    # episode 在搬运 GOLD 途中结束：不计分，物体完好。
    assert env.hook_state is HookState.RETRACT_LOADED
    assert env.attached_object is gold
    assert gold.active is True
    assert env.score == 0.0
    assert info["score"] == 0.0
    assert (gold.x, gold.y) == env.hook_tip


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------
def _record(env: GoldMinerEnv, traj: list[tuple]) -> None:
    """目的：记录一步后的关键状态，用于逐位对比两次轨迹。

    输入：环境 env、轨迹列表 traj。
    输出：无返回值；向 traj 追加一条元组（角度、绳长、钩子状态、分数、
    各物体存活情况）。
    """
    traj.append(
        (
            env.angle,
            env.rope_length,
            env.hook_state,
            env.score,
            tuple(obj.active for obj in env.objects),
        )
    )


def _run_fixed_policy(
    env: GoldMinerEnv,
    actions: list[int],
    traj: list[tuple],
) -> None:
    """目的：驱动固定演示策略并记录每个动作及每步后的轨迹
    （依次抓取 GOLD、DIAMOND、ROCK，然后 WAIT 到超时）。

    输入：环境 env、动作列表 actions（就地追加）、轨迹列表 traj（就地追加）。
    输出：无返回值；actions/traj 被填满，直到 truncated 结束。
    """
    for target_angle in (-30.0, 2.0, 31.0):
        while (
            env.hook_state is HookState.SWINGING
            and abs(env.angle - target_angle) > ANGLE_TOL
        ):
            actions.append(WAIT)
            env.step(WAIT)
            _record(env, traj)
        actions.append(FIRE)
        env.step(FIRE)
        _record(env, traj)
        while env.hook_state is not HookState.SWINGING:
            actions.append(WAIT)
            env.step(WAIT)
            _record(env, traj)
    while True:
        actions.append(WAIT)
        _, _, _, truncated, _ = env.step(WAIT)
        _record(env, traj)
        if truncated:
            break


def test_determinism_replaying_recorded_action_list() -> None:
    """目的：重放完全相同的动作序列必须复现完整轨迹（issue #1 第 14 节）。

    输入：固定策略脚本记录的每 tick 动作列表（依次抓 GOLD、DIAMOND、ROCK，
    再 WAIT 到超时）；全新 reset 后逐条重放（seed 不同，但环境无随机性）。
    输出：两次轨迹记录逐位相等；整局恰为 3600 个动作，总分 800
    （250 + 500 + 50），剩余时间 0，三个物体均被收取。
    """
    env = GoldMinerEnv()
    env.reset(seed=1)
    actions: list[int] = []
    traj_a: list[tuple] = []
    _run_fixed_policy(env, actions, traj_a)

    # 整个 60 s episode 每 tick 一个动作，三个物体全部收取。
    assert len(actions) == SIM_FPS * int(EPISODE_TIME)
    assert env.score == pytest.approx(800.0)  # 250 + 500 + 50

    # 严格重放：全新 reset 后逐条执行记录的动作列表。
    env.reset(seed=2)  # seed 无关紧要：环境没有随机性
    traj_b: list[tuple] = []
    for action in actions:
        env.step(action)
        _record(env, traj_b)

    assert len(traj_a) == len(traj_b)
    assert traj_a == traj_b  # 浮点项逐位相等
    assert env.score == pytest.approx(800.0)
    assert env.remaining_time == 0.0
    assert tuple(obj.active for obj in env.objects) == (False, False, False)


# ---------------------------------------------------------------------------
# Gymnasium API 合规
# ---------------------------------------------------------------------------
def test_gymnasium_env_checker() -> None:
    """目的：通过 Gymnasium 官方 env_checker 合规检查。

    输入：默认环境（跳过渲染检查）。
    输出：check_env 不抛异常，close 正常返回。
    """
    env = GoldMinerEnv()
    check_env(env, skip_render_check=True)
    env.close()
