"""Geometry evaluation（Milestone 11，issue #21）核心工程测试。

覆盖：GoldMinerEnv 自定义 spawn pool 契约与历史行为逐位回归、
geometry_eval 的 ID/rot3/rot3_scale105 pool 变换与角度约定、
几何合法性、paired seed 的 point-index 对应关系、以及
make_geometry_eval_env factory 回归。
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

from gold_miner_sim.benchmark import make_benchmark_env, make_geometry_eval_env
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
from gold_miner_sim.geometry_eval import (
    GEOMETRY_MODES,
    geometry_spawn_pool,
    transform_point,
)
from gold_miner_sim.wrappers import FireBudgetWrapper, SwingAdvanceDecisionWrapper

# 物体契约（槽位顺序 GOLD/DIAMOND/ROCK）：(type, radius, value, retract_speed)。
OBJECT_SPECS = (
    (ObjectType.GOLD, 30.0, 250.0, 140.0),
    (ObjectType.DIAMOND, 18.0, 500.0, 280.0),
    (ObjectType.ROCK, 34.0, 50.0, 90.0),
)
MAX_OBJECT_RADIUS = 34.0  # ROCK 的半径，几何合法性用最大圆检验。
MIN_PAIRWISE_DISTANCE = 2.0 * MAX_OBJECT_RADIUS
# 逆变换映射回源点的容差（issue §15.8：容差 1e-6）。
INVERSE_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def _center_angle_deg(x: float, y: float) -> float:
    """项目角度约定下的 center 角（度）：atan2(dx, dy)，0° = 正下方。"""
    return math.degrees(math.atan2(x - ANCHOR[0], y - ANCHOR[1]))


def _wrap180(angle_deg: float) -> float:
    """把角度 wrap 到 [-180, 180)。"""
    return (angle_deg + 180.0) % 360.0 - 180.0


def _inverse_transform_point(
    x: float, y: float, rotation_deg: float, radial_scale: float
) -> tuple[float, float]:
    """transform_point 的逆变换：转回原角度、除回 radial scale。"""
    dx = x - ANCHOR[0]
    dy = y - ANCHOR[1]
    a2 = math.atan2(dx, dy)
    r2 = math.hypot(dx, dy)
    a = a2 - math.radians(rotation_deg)
    r = r2 / radial_scale
    return (ANCHOR[0] + r * math.sin(a), ANCHOR[1] + r * math.cos(a))


def _match_source_index(point: tuple[float, float], *, exact: bool) -> int:
    """把一个 center 匹配回 RANDOM_SPAWN_POINTS 的源 index。

    exact=True 时要求与源点精确相等（id pool 逐位复用常量）；
    否则要求最近距离 < INVERSE_TOLERANCE（逆变换的浮点误差内）。
    """
    distances = [math.dist(point, source) for source in RANDOM_SPAWN_POINTS]
    best = min(range(len(distances)), key=lambda i: distances[i])
    if exact:
        assert distances[best] == 0.0
    else:
        assert distances[best] < INVERSE_TOLERANCE
    return best


# ---------------------------------------------------------------------------
# 15.1 历史回归：显式传入默认 pool 与不传逐位一致
# ---------------------------------------------------------------------------
def test_explicit_default_pool_bitwise_matches_legacy_random() -> None:
    """目的：random 模式显式传 spawn_points=RANDOM_SPAWN_POINTS 与历史
    不传行为逐位一致（两者的 RNG 消耗必须完全相同）。

    输入：两个 GoldMinerEnv(map_mode="random")，一个不传 spawn_points、
    一个显式传 RANDOM_SPAWN_POINTS；seeds 0/1/7/42/123/3007 各 reset
    一局，随后各再连续 3 次 reset()（无 seed，验证 RNG 状态推进一致）。
    输出：每一局两环境 obs 逐位相等，三个物体 center 精确相等。
    """
    legacy = GoldMinerEnv(map_mode="random")
    explicit = GoldMinerEnv(map_mode="random", spawn_points=RANDOM_SPAWN_POINTS)

    def assert_same_layout() -> None:
        for obj_legacy, obj_explicit in zip(legacy.objects, explicit.objects):
            assert (obj_legacy.x, obj_legacy.y) == (obj_explicit.x, obj_explicit.y)

    for seed in (0, 1, 7, 42, 123, 3007):
        obs_legacy, _ = legacy.reset(seed=seed)
        obs_explicit, _ = explicit.reset(seed=seed)
        assert np.array_equal(obs_legacy, obs_explicit)
        assert_same_layout()
    for _ in range(3):  # 无 seed reset：RNG 状态推进路径也必须一致。
        obs_legacy, _ = legacy.reset()
        obs_explicit, _ = explicit.reset()
        assert np.array_equal(obs_legacy, obs_explicit)
        assert_same_layout()


# ---------------------------------------------------------------------------
# 15.2 自定义 spawn pool 契约
# ---------------------------------------------------------------------------
CUSTOM_POOL = (
    (100.0, 200.0),
    (300.0, 250.0),
    (500.0, 300.0),
    (650.0, 380.0),
    (760.0, 210.0),
)


def test_custom_pool_works_with_distinct_points() -> None:
    """目的：合法自定义 pool 正常工作，一局使用 3 个不同 index 且物体属性不变。

    输入：GoldMinerEnv(map_mode="random", spawn_points=CUSTOM_POOL)
    （5 点 pool），连续 reset 多个 seed。
    输出：每局三个 center 均属于 CUSTOM_POOL 且互不相同（即 3 个不同
    index）；type/radius/value/retract_speed 槽位契约与历史完全一致，
    且全部 active。
    """
    env = GoldMinerEnv(map_mode="random", spawn_points=CUSTOM_POOL)
    pool_set = set(CUSTOM_POOL)
    covered: set[tuple[float, float]] = set()
    for seed in range(20):
        env.reset(seed=seed)
        centers = [(obj.x, obj.y) for obj in env.objects]
        assert len(set(centers)) == 3
        for center in centers:
            assert center in pool_set
        covered.update(centers)
        for obj, (obj_type, radius, value, speed) in zip(env.objects, OBJECT_SPECS):
            assert obj.type is obj_type
            assert obj.radius == radius
            assert obj.value == value
            assert obj.retract_speed == speed
            assert obj.active is True
    assert len(covered) >= 4  # 多个 seed 确实抽取了不同组合。


def test_custom_pool_same_seed_reproduces_indices() -> None:
    """目的：同 seed 下自定义 pool 的选中 index 完全可复现（含物理确定性）。

    输入：两个相同自定义 pool 的 GoldMinerEnv(map_mode="random")，各
    reset(seed=9) 后执行同一动作脚本（前 50 tick WAIT、第 51 tick FIRE、
    其后 WAIT，共 200 tick）。
    输出：reset 后三 center 精确相等；每 tick obs 逐位相等。
    """
    env_a = GoldMinerEnv(map_mode="random", spawn_points=CUSTOM_POOL)
    env_b = GoldMinerEnv(map_mode="random", spawn_points=CUSTOM_POOL)
    obs_a, _ = env_a.reset(seed=9)
    obs_b, _ = env_b.reset(seed=9)
    for obj_a, obj_b in zip(env_a.objects, env_b.objects):
        assert (obj_a.x, obj_a.y) == (obj_b.x, obj_b.y)

    for tick in range(200):
        action = FIRE if tick == 50 else WAIT
        obs_a, _, _, _, _ = env_a.step(action)
        obs_b, _, _, _, _ = env_b.step(action)
        assert np.array_equal(obs_a, obs_b)


def test_custom_pool_frozen_against_caller_mutation() -> None:
    """目的：构造后 caller 修改传入的 list 不得影响 env 内部 pool。

    输入：以 list of lists（可变结构）构造 random 环境，构造后原地修改
    元素并向 list 追加新点，再 reset(seed=3)。
    输出：物体 center 仍精确来自构造时的原始三点，不含被篡改后的坐标。
    """
    # 可变 list of lists 也是合法的可迭代 pool（运行时按元素解包校验）；
    # cast 仅为静态检查，重点验证构造后 caller 的原地修改不影响 env。
    pool_list: list[list[float]] = [[173.0, 230.0], [159.0, 314.0], [283.0, 269.0]]
    env = GoldMinerEnv(
        map_mode="random",
        spawn_points=cast(Sequence[tuple[float, float]], pool_list),
    )
    pool_list[0][0] = 999.0  # 篡改第一个点。
    pool_list.append([1.0, 1.0])  # 追加新点。

    env.reset(seed=3)
    centers = {(obj.x, obj.y) for obj in env.objects}
    assert centers == {(173.0, 230.0), (159.0, 314.0), (283.0, 269.0)}


def test_fixed_mode_with_spawn_points_raises() -> None:
    """目的：map_mode="fixed" 且 spawn_points 非 None 必须抛 ValueError。

    输入：GoldMinerEnv(map_mode="fixed", spawn_points=RANDOM_SPAWN_POINTS)。
    输出：构造时抛 ValueError（避免语义歧义）。
    """
    with pytest.raises(ValueError):
        GoldMinerEnv(map_mode="fixed", spawn_points=RANDOM_SPAWN_POINTS)


@pytest.mark.parametrize(
    "bad_pool",
    [
        pytest.param((), id="empty"),
        pytest.param(((1.0, 2.0), (3.0, 4.0)), id="two-points"),
        pytest.param(42, id="not-iterable"),
        pytest.param(
            ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)), id="wrong-length"
        ),
        pytest.param(((1.0, "x"), (2.0, "y"), (3.0, "z")), id="non-numeric"),
        pytest.param(("ab", "cd", "ef"), id="string-elements"),
    ],
)
def test_invalid_custom_pools_raise(bad_pool: object) -> None:
    """目的：非法 pool（不可迭代 / 少于 3 点 / 元素非长度 2 数值对）必须
    构造时抛 ValueError。

    输入：六种非法 spawn_points（空、2 点、非可迭代、元素长度 3、坐标
    非数值、元素为字符串）。
    输出：全部抛 ValueError。
    """
    with pytest.raises(ValueError):
        GoldMinerEnv(map_mode="random", spawn_points=cast(Any, bad_pool))


# ---------------------------------------------------------------------------
# 15.3 ID pool identity
# ---------------------------------------------------------------------------
def test_id_pool_is_identity() -> None:
    """目的："id" 模式必须直接返回 RANDOM_SPAWN_POINTS 本身。

    输入：geometry_spawn_pool("id")。
    输出：返回对象 is RANDOM_SPAWN_POINTS（无复制、无重排）；
    GEOMETRY_MODES 常量为 ("id", "rot3", "rot3_scale105")；
    未知 mode 抛 ValueError。
    """
    assert GEOMETRY_MODES == ("id", "rot3", "rot3_scale105")
    assert geometry_spawn_pool("id") is RANDOM_SPAWN_POINTS
    with pytest.raises(ValueError):
        geometry_spawn_pool("banana")


# ---------------------------------------------------------------------------
# 15.4 角度约定（atan2(dx, dy)，0° = 正下方）
# ---------------------------------------------------------------------------
def test_transform_angle_convention() -> None:
    """目的：验证 transform_point 使用项目角度约定，防止误写 atan2(dy, dx)。

    输入：合成点——正下方点 (450, 470)（dx=0, dy=400，角 0°）与左下点
    (350, 170)（dx=-100, dy=100，角 -45°），各自 +3° 旋转、radial 不变。
    输出：正下方点变换后 center 角 = +3° 且 x2 > 450（向右偏，排除数学
    约定的向左偏）；左下点变换后 center 角 = -42°；两点到 anchor 距离
    近似不变。
    """
    straight_down = (450.0, 470.0)  # dx=0, dy=400 -> 0°
    lower_left = (350.0, 170.0)  # dx=-100, dy=100 -> -45°
    assert _center_angle_deg(*straight_down) == pytest.approx(0.0)
    assert _center_angle_deg(*lower_left) == pytest.approx(-45.0)

    down_rotated = transform_point(straight_down[0], straight_down[1], 3.0, 1.0)
    left_rotated = transform_point(lower_left[0], lower_left[1], 3.0, 1.0)

    assert _center_angle_deg(*down_rotated) == pytest.approx(3.0)
    assert down_rotated[0] > ANCHOR[0]  # +3° 应向右偏（排除 atan2(dy, dx)）。
    assert _center_angle_deg(*left_rotated) == pytest.approx(-42.0)

    assert math.dist(down_rotated, ANCHOR) == pytest.approx(
        math.dist(straight_down, ANCHOR)
    )
    assert math.dist(left_rotated, ANCHOR) == pytest.approx(
        math.dist(lower_left, ANCHOR)
    )


# ---------------------------------------------------------------------------
# 15.5 / 15.6 rot3 与 rot3_scale105 的不变量
# ---------------------------------------------------------------------------
def _assert_pool_order_preserved(pool: tuple[tuple[float, float], ...]) -> None:
    assert len(pool) == len(RANDOM_SPAWN_POINTS)
    for transformed, original in zip(pool, RANDOM_SPAWN_POINTS):
        # 点序不变：同下标一一对应（距离远小于任意异下标间距）。
        assert math.dist(transformed, original) < MIN_PAIRWISE_DISTANCE


def test_rot3_invariants() -> None:
    """目的：rot3 pool 全部 12 点满足刚性旋转不变量。

    输入：geometry_spawn_pool("rot3") 与 RANDOM_SPAWN_POINTS。
    输出：每点到 anchor 距离与 ID 相等（approx）；center 角 = ID 角 +3°；
    任意两点 pairwise 距离与 ID 相等；点序按 index 一一对应；且每个点
    坐标确实发生了变化。
    """
    pool = geometry_spawn_pool("rot3")
    assert len(pool) == 12
    _assert_pool_order_preserved(pool)

    for i, (transformed, original) in enumerate(zip(pool, RANDOM_SPAWN_POINTS)):
        assert math.dist(transformed, original) > 0.0  # 每个点都被移动。
        assert math.dist(transformed, ANCHOR) == pytest.approx(
            math.dist(original, ANCHOR)
        )
        assert _center_angle_deg(*transformed) == pytest.approx(
            _wrap180(_center_angle_deg(*original) + 3.0)
        )

    for i, j in itertools.combinations(range(12), 2):
        assert math.dist(pool[i], pool[j]) == pytest.approx(
            math.dist(RANDOM_SPAWN_POINTS[i], RANDOM_SPAWN_POINTS[j])
        )


def test_rot3_scale105_invariants() -> None:
    """目的：rot3_scale105 pool 全部 12 点满足旋转 + 放缩不变量。

    输入：geometry_spawn_pool("rot3_scale105") 与 RANDOM_SPAWN_POINTS。
    输出：center 角 = ID 角 +3°；到 anchor 距离 = 1.05 × ID 距离；任意
    两点 pairwise 距离 = 1.05 × ID pairwise 距离；点序按 index 一一对应。
    """
    pool = geometry_spawn_pool("rot3_scale105")
    assert len(pool) == 12
    _assert_pool_order_preserved(pool)

    for transformed, original in zip(pool, RANDOM_SPAWN_POINTS):
        assert _center_angle_deg(*transformed) == pytest.approx(
            _wrap180(_center_angle_deg(*original) + 3.0)
        )
        assert math.dist(transformed, ANCHOR) == pytest.approx(
            1.05 * math.dist(original, ANCHOR)
        )

    for i, j in itertools.combinations(range(12), 2):
        assert math.dist(pool[i], pool[j]) == pytest.approx(
            1.05 * math.dist(RANDOM_SPAWN_POINTS[i], RANDOM_SPAWN_POINTS[j])
        )


# ---------------------------------------------------------------------------
# 15.7 几何合法性（三个 pool）
# ---------------------------------------------------------------------------
def test_geometry_legality_for_all_pools() -> None:
    """目的：三个 geometry pool 均满足画布 / 绳长 / 无重叠 / 唯一性约束。

    输入：GEOMETRY_MODES 中每个 mode 的 pool（各 12 点）。
    输出：center ± 最大半径 34 落在画布 [0,900]×[0,600] 内；每点到
    ANCHOR 距离 < MAX_ROPE_LENGTH(460)；任意两点间距 > 68（两个最大圆
    不重叠）；12 个点唯一。
    """
    for mode in GEOMETRY_MODES:
        pool = geometry_spawn_pool(mode)
        assert len(pool) == 12
        assert len(set(pool)) == 12  # 无重复点。
        for x, y in pool:
            assert MAX_OBJECT_RADIUS <= x <= WIDTH - MAX_OBJECT_RADIUS
            assert MAX_OBJECT_RADIUS <= y <= HEIGHT - MAX_OBJECT_RADIUS
            assert math.dist((x, y), ANCHOR) < MAX_ROPE_LENGTH
        for p1, p2 in itertools.combinations(pool, 2):
            assert math.dist(p1, p2) > MIN_PAIRWISE_DISTANCE


# ---------------------------------------------------------------------------
# 15.8 Paired seed correspondence
# ---------------------------------------------------------------------------
def _selected_source_indices(mode: str, seed: int) -> list[int]:
    """对给定 mode/seed 构建 factory 环境并返回三个槽位的源 point index。"""
    env = make_geometry_eval_env(mode)
    env.reset(seed=seed)
    hook = cast(GoldMinerEnv, env.unwrapped)
    centers = [(obj.x, obj.y) for obj in hook.objects]
    if mode == "id":
        return [_match_source_index(center, exact=True) for center in centers]
    rotation_deg, radial_scale = (3.0, 1.0) if mode == "rot3" else (3.0, 1.05)
    sources = [
        _inverse_transform_point(x, y, rotation_deg, radial_scale) for x, y in centers
    ]
    return [_match_source_index(source, exact=False) for source in sources]


@pytest.mark.parametrize("seed", [3000, 3007, 3042, 3055])
def test_paired_seed_correspondence(seed: int) -> None:
    """目的：同一 map seed 下三个 geometry mode 必须选中相同源 point index。

    输入：seeds 3000/3007/3042/3055，每个 seed 对 id/rot3/rot3_scale105
    分别构建 make_geometry_eval_env 并 reset；rot3/rot3_scale105 的
    center 用逆变换（转回原角度、除回 scale，容差 1e-6）映射回
    RANDOM_SPAWN_POINTS 的源 index。
    输出：三 mode 的 GOLD/DIAMOND/ROCK 源 index 序列完全一致，且槽位
    类型顺序均为 GOLD/DIAMOND/ROCK。
    """
    indices_by_mode = {}
    for mode in GEOMETRY_MODES:
        indices_by_mode[mode] = _selected_source_indices(mode, seed)
    assert (
        indices_by_mode["id"]
        == indices_by_mode["rot3"]
        == indices_by_mode["rot3_scale105"]
    )
    assert len(set(indices_by_mode["id"])) == 3  # 一局内 3 个不同 index。


# ---------------------------------------------------------------------------
# 15.9 Factory 回归
# ---------------------------------------------------------------------------
def test_factory_id_matches_benchmark_env_bitwise() -> None:
    """目的：make_geometry_eval_env("id") 与历史 make_benchmark_env("full")
    在同 seed 下逐位一致，且 wrapper 链结构相同。

    输入：两个 factory 各以 seeds 0/7/42/123 reset；并检查 wrapper 链
    FireBudgetWrapper(SwingAdvanceDecisionWrapper(GoldMinerEnv))。
    输出：每局 27 维 obs 逐位相等；链上每层类型与 map_mode 符合历史契约。
    """
    env_id = make_geometry_eval_env("id")
    env_bench = make_benchmark_env("full")
    assert isinstance(env_id, FireBudgetWrapper)
    assert isinstance(env_id.env, SwingAdvanceDecisionWrapper)
    inner = env_id.env.env  # SwingAdvanceDecisionWrapper.env 就是 GoldMinerEnv。
    assert isinstance(inner, GoldMinerEnv)
    assert inner.map_mode == "random"

    for seed in (0, 7, 42, 123):
        obs_id, _ = env_id.reset(seed=seed)
        obs_bench, _ = env_bench.reset(seed=seed)
        assert obs_id.shape == (27,)
        assert np.array_equal(obs_id, obs_bench)


@pytest.mark.parametrize("mode", ["rot3", "rot3_scale105"])
def test_factory_transformed_modes_place_expected_centers(mode: str) -> None:
    """目的：factory 在 rot3/rot3_scale105 下确实放置了对应变换后的点。

    输入：make_geometry_eval_env(mode) reset(seed=3007)，从
    env.unwrapped.objects 读三个 center；先用逆变换映射回源 index，再与
    geometry_spawn_pool(mode) 中同 index 的变换点比较。
    输出：每个 center 与 pool 中对应下标的变换点一致（approx）；逆变换
    回源 index 与 id 模式同 seed 的选中 index 完全一致。
    """
    env = make_geometry_eval_env(mode)
    env.reset(seed=3007)
    hook = cast(GoldMinerEnv, env.unwrapped)
    pool = geometry_spawn_pool(mode)
    expected_indices = _selected_source_indices("id", 3007)

    for obj, index in zip(hook.objects, expected_indices):
        center = (obj.x, obj.y)
        assert center == pytest.approx(pool[index])
        rotation_deg, radial_scale = (3.0, 1.0) if mode == "rot3" else (3.0, 1.05)
        source = _inverse_transform_point(*center, rotation_deg, radial_scale)
        assert _match_source_index(source, exact=False) == index


def test_factory_unknown_mode_raises() -> None:
    """目的：未知 geometry_mode 必须抛 ValueError。

    输入：make_geometry_eval_env("banana")。
    输出：构造时抛 ValueError。
    """
    with pytest.raises(ValueError):
        make_geometry_eval_env("banana")
