"""``GoldMinerEnv`` map_mode="random"（Milestone 3，issue #5）行为测试。

固定地图契约回归 + 随机地图语义：随机布局只改变三个物体的 center，
物体属性、观测契约、物理确定性均不变；同 seed 重放必须逐位一致。
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from gold_miner_sim.env import (
    ANCHOR,
    FIRE,
    HEIGHT,
    MAX_ROPE_LENGTH,
    RANDOM_SPAWN_POINTS,
    WAIT,
    WIDTH,
    GoldMinerEnv,
    ObjectType,
)

# 固定地图契约（issue #1）：槽位顺序 GOLD/DIAMOND/ROCK。
FIXED_POSITIONS = ((315.0, 300.0), (465.0, 450.0), (610.0, 340.0))
FIXED_SPECS = (
    (ObjectType.GOLD, 30.0, 250.0, 140.0),
    (ObjectType.DIAMOND, 18.0, 500.0, 280.0),
    (ObjectType.ROCK, 34.0, 50.0, 90.0),
)

SPAWN_SET = set(RANDOM_SPAWN_POINTS)
# 两个最大半径（ROCK r=34）同时占据时圆面也不得重叠。
MIN_POINT_DISTANCE = 2.0 * 34.0


# ---------------------------------------------------------------------------
# 默认行为 / 模式校验
# ---------------------------------------------------------------------------
def test_default_env_is_fixed_map() -> None:
    """目的：默认构造即 fixed 模式，物体与 V0 契约逐位一致。

    输入：默认环境 GoldMinerEnv()，reset(seed=0)。
    输出：map_mode == "fixed"；三个物体类型顺序 GOLD/DIAMOND/ROCK，
    center 精确等于 (315,300)/(465,450)/(610,340)，
    radius/value/retract_speed 为 30/250/140、18/500/280、34/50/90，
    且全部 active。
    """
    env = GoldMinerEnv()
    assert env.map_mode == "fixed"
    env.reset(seed=0)

    assert len(env.objects) == 3
    for obj, (obj_type, radius, value, speed), (x, y) in zip(
        env.objects, FIXED_SPECS, FIXED_POSITIONS
    ):
        assert obj.type is obj_type
        assert (obj.x, obj.y) == (x, y)
        assert obj.radius == radius
        assert obj.value == value
        assert obj.retract_speed == speed
        assert obj.active is True


def test_explicit_fixed_mode_matches_default() -> None:
    """目的：显式 map_mode="fixed" 与默认构造完全一致。

    输入：默认环境与 GoldMinerEnv(map_mode="fixed")，均 reset(seed=7)。
    输出：两者 obs 逐位相等；三个物体 center 与全部属性逐一相等。
    """
    default = GoldMinerEnv()
    explicit = GoldMinerEnv(map_mode="fixed")
    obs_default, _ = default.reset(seed=7)
    obs_explicit, _ = explicit.reset(seed=7)

    assert explicit.map_mode == "fixed"
    assert np.array_equal(obs_default, obs_explicit)
    for obj_d, obj_e in zip(default.objects, explicit.objects):
        assert (obj_d.x, obj_d.y) == (obj_e.x, obj_e.y)
        assert obj_d.type is obj_e.type
        assert obj_d.radius == obj_e.radius
        assert obj_d.value == obj_e.value
        assert obj_d.retract_speed == obj_e.retract_speed


def test_invalid_map_mode_raises() -> None:
    """目的：非法 map_mode 应立刻抛 ValueError，不得静默回退。

    输入：map_mode 取 "banana"、"Random"（大小写敏感）、""。
    输出：三种取值均抛 ValueError。
    """
    for bad in ("banana", "Random", ""):
        with pytest.raises(ValueError):
            GoldMinerEnv(map_mode=bad)


# ---------------------------------------------------------------------------
# 随机布局语义
# ---------------------------------------------------------------------------
def test_random_positions_come_only_from_spawn_points() -> None:
    """目的：random 模式每次 reset 的物体 center 只能取自常量表。

    输入：GoldMinerEnv(map_mode="random") 连续 reset 30 次（seed=0..29）。
    输出：每次三个 center 精确属于 RANDOM_SPAWN_POINTS 且互不相同，
    类型顺序仍为 GOLD/DIAMOND/ROCK；30 次累计覆盖至少 6 个不同
    spawn point（防止退化成固定 3 点）。
    """
    env = GoldMinerEnv(map_mode="random")
    covered: set[tuple[float, float]] = set()
    for seed in range(30):
        env.reset(seed=seed)
        centers = [(obj.x, obj.y) for obj in env.objects]
        assert len(set(centers)) == 3  # 三个 center 互不相同
        for center in centers:
            assert center in SPAWN_SET
        covered.update(centers)
        assert [obj.type for obj in env.objects] == [
            ObjectType.GOLD,
            ObjectType.DIAMOND,
            ObjectType.ROCK,
        ]
    assert len(covered) >= 6


def test_random_mode_preserves_object_properties() -> None:
    """目的：random 模式只随机 center，物体属性保持契约不变。

    输入：GoldMinerEnv(map_mode="random") 以多个 seed reset。
    输出：每次三物体 type/radius/value/retract_speed 与固定契约一致
    （分值 250/500/50），且全部 active。
    """
    env = GoldMinerEnv(map_mode="random")
    for seed in (0, 1, 2, 3, 42, 123):
        env.reset(seed=seed)
        for obj, (obj_type, radius, value, speed) in zip(
            env.objects, FIXED_SPECS
        ):
            assert obj.type is obj_type
            assert obj.radius == radius
            assert obj.value == value
            assert obj.retract_speed == speed
            assert obj.active is True


def test_random_mode_observation_contract_unchanged() -> None:
    """目的：random 布局下观测契约不变（26 维 float32、固定槽位）。

    输入：GoldMinerEnv(map_mode="random") reset(seed=5) 后的 obs 与物体。
    输出：obs.shape==(26,)、dtype==float32、落在 observation_space 内；
    槽位顺序 GOLD/DIAMOND/ROCK，各槽 6 维依次为
    x/WIDTH、y/HEIGHT、radius/100、value/500、speed/360、active=1.0，
    且与物体属性精确一致（float32 舍入后相等）。
    """
    env = GoldMinerEnv(map_mode="random")
    obs, _ = env.reset(seed=5)

    assert obs.shape == (26,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    for slot, obj in enumerate(env.objects):
        base = 8 + 6 * slot
        assert obs[base] == np.float32(obj.x / WIDTH)
        assert obs[base + 1] == np.float32(obj.y / HEIGHT)
        assert obs[base + 2] == np.float32(obj.radius / 100.0)
        assert obs[base + 3] == np.float32(obj.value / 500.0)
        assert obs[base + 4] == np.float32(obj.retract_speed / 360.0)
        assert obs[base + 5] == 1.0  # active flag
    assert [obj.type for obj in env.objects] == [
        ObjectType.GOLD,
        ObjectType.DIAMOND,
        ObjectType.ROCK,
    ]


# ---------------------------------------------------------------------------
# 种子可复现性
# ---------------------------------------------------------------------------
def test_same_seed_same_map() -> None:
    """目的：同 seed 的两个独立 random 环境应得到同一张地图。

    输入：两个 GoldMinerEnv(map_mode="random") 各 reset(seed=123)。
    输出：三个物体 center 逐一相等，obs 逐位相等。
    """
    env_a = GoldMinerEnv(map_mode="random")
    env_b = GoldMinerEnv(map_mode="random")
    obs_a, _ = env_a.reset(seed=123)
    obs_b, _ = env_b.reset(seed=123)

    for obj_a, obj_b in zip(env_a.objects, env_b.objects):
        assert (obj_a.x, obj_a.y) == (obj_b.x, obj_b.y)
    assert np.array_equal(obs_a, obs_b)


def test_same_seed_same_map_sequence() -> None:
    """目的：同初始 seed 的两局 reset 序列逐局一致，且确实逐局变化。

    输入：两个 GoldMinerEnv(map_mode="random") 各执行
    reset(seed=123)、reset()、reset()、reset() 共 4 局。
    输出：两环境的 4 局布局序列逐局相等；且序列中至少出现 2 种不同
    layout（证明 np_random 状态跨局推进，而非每局重新固定）。
    """
    env_a = GoldMinerEnv(map_mode="random")
    env_b = GoldMinerEnv(map_mode="random")
    layouts_a: list[tuple[tuple[float, float], ...]] = []
    layouts_b: list[tuple[tuple[float, float], ...]] = []
    for episode in range(4):
        if episode == 0:
            env_a.reset(seed=123)
            env_b.reset(seed=123)
        else:
            env_a.reset()
            env_b.reset()
        layouts_a.append(tuple((o.x, o.y) for o in env_a.objects))
        layouts_b.append(tuple((o.x, o.y) for o in env_b.objects))

    assert layouts_a == layouts_b
    assert len(set(layouts_a)) >= 2


# ---------------------------------------------------------------------------
# 常量几何约束 / 物理确定性
# ---------------------------------------------------------------------------
def test_spawn_points_geometry() -> None:
    """目的：静态校验 RANDOM_SPAWN_POINTS 本身的几何可行性。

    输入：全部 12 个 spawn point。
    输出：每点都在画布内（0<=x<=WIDTH，0<=y<=HEIGHT）；每点到 ANCHOR
    的欧氏距离 <= MAX_ROPE_LENGTH（绳可及）；任意两点间距 >
    2*34=68 px（两个最大半径 ROCK 同时占据也不圆重叠）。
    """
    for x, y in RANDOM_SPAWN_POINTS:
        assert 0.0 <= x <= WIDTH
        assert 0.0 <= y <= HEIGHT
        assert math.dist((x, y), ANCHOR) <= MAX_ROPE_LENGTH
    for p1, p2 in itertools.combinations(RANDOM_SPAWN_POINTS, 2):
        assert math.dist(p1, p2) > MIN_POINT_DISTANCE


def test_random_mode_replay_deterministic() -> None:
    """目的：同 seed 同动作脚本下 random 模式逐 tick 确定性重放。

    输入：两个 GoldMinerEnv(map_mode="random") 各 reset(seed=7)（该
    布局把 GOLD 放在 30° 发射射线上），执行同一动作脚本：前 100 tick
    WAIT、第 101 tick FIRE、其后 WAIT，共 600 tick。
    输出：每 tick 两环境 angle/rope_length/hook_state/score 精确相等，
    obs 逐位相等；两环境均抓住 GOLD 并各得 250 分（命中/带载收回/
    计分路径也被逐 tick 覆盖），证明 np_random 只影响初始布局，
    不影响物理确定性。
    """
    env_a = GoldMinerEnv(map_mode="random")
    env_b = GoldMinerEnv(map_mode="random")
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    assert np.array_equal(obs_a, obs_b)

    for tick in range(600):
        action = FIRE if tick == 100 else WAIT
        obs_a, _, _, trunc_a, _ = env_a.step(action)
        obs_b, _, _, trunc_b, _ = env_b.step(action)
        assert env_a.angle == env_b.angle
        assert env_a.rope_length == env_b.rope_length
        assert env_a.hook_state is env_b.hook_state
        assert env_a.score == env_b.score
        assert np.array_equal(obs_a, obs_b)
        assert trunc_a == trunc_b

    # 该 seed 下钩子必然抓住 GOLD：确定性重放同时覆盖命中与计分路径。
    assert env_a.score == 250.0 and env_b.score == 250.0
    assert env_a.attached_object is None and env_b.attached_object is None
