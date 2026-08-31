"""M11 geometry evaluator 测试（issue #21 §15.10–§15.12）。

覆盖：frozen-policy episode runner 的 deterministic 推理契约、
classify_fire 三分类（含 timeout 不算 miss）、observation-only FIRE 诊断
的正确性与只读性、oracle 驱动的小规模集成 sanity、以及 §12/§13 的
paired delta / retention 聚合 helper。不训练任何模型，也不把
100-map gate 放进 pytest。
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from gold_miner_sim.benchmark import make_geometry_eval_env
from gold_miner_sim.env import FIRE, MAX_ANGLE, WAIT, GoldMinerEnv
from gold_miner_sim.geometry_eval import (
    FULL_SCORE,
    GEOMETRY_MODES,
    across_seed_paired_analysis,
    classify_fire,
    fire_observation_diagnostics,
    paired_analysis,
    run_geometry_episode,
    summarize_geometry_results,
)
from gold_miner_sim.oracle import oracle_action, predicted_first_hit_slot

# oracle 在三个 geometry mode 上均可满收集的固定 map seed（issue §15.12
# 允许 pytest 只覆盖 1–3 张固定地图）。
SANITY_MAP_SEEDS = (3000, 3007)
# reset 后首决策角度（INITIAL_ANGLE=-70°）下 predicted_first_hit_slot 为
# None 的 seed：首步 FIRE 必然 miss（id pool, seed 3000 实测验证）。
MISS_MAP_SEED = 3000


class _StrictDeterministicModel:
    """带 SB3 predict 签名的 stub：记录每次 deterministic 值，收到 False 即 fail。

    策略本身是 M8 oracle_action（确定性、observation-only）。
    """

    def __init__(self) -> None:
        self.deterministic_flags: list[bool] = []
        self.actions: list[int] = []

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = False
    ) -> tuple[int, Any]:
        self.deterministic_flags.append(deterministic)
        if not deterministic:
            raise AssertionError(
                "evaluator 必须使用 model.predict(obs, deterministic=True)"
            )
        action = oracle_action(obs)
        self.actions.append(int(action))
        return action, None


class _ScriptedModel:
    """按脚本返回动作的 stub；脚本耗尽后回退 WAIT。记录收到的 obs 副本。"""

    def __init__(self, actions: list[int]) -> None:
        self.actions_script = list(actions)
        self.actions: list[int] = []
        self.observations: list[NDArray[np.float32]] = []
        self.deterministic_flags: list[bool] = []

    def predict(
        self, obs: NDArray[np.float32], deterministic: bool = False
    ) -> tuple[int, Any]:
        self.deterministic_flags.append(deterministic)
        if not deterministic:
            raise AssertionError(
                "evaluator 必须使用 model.predict(obs, deterministic=True)"
            )
        # 深拷贝快照：若 runner 原地修改 obs，这里留存的副本可与 fresh
        # replay 的 obs 对比检出差异。
        self.observations.append(np.array(obs, copy=True))
        index = len(self.actions)
        action = (
            self.actions_script[index] if index < len(self.actions_script) else WAIT
        )
        self.actions.append(int(action))
        return action, None


def _replay_reward_sum(geometry_mode: str, map_seed: int, actions: list[int]) -> float:
    """用动作脚本在全新 env 上手工 replay 并累加 reward（runner 的对照）。"""
    env = make_geometry_eval_env(geometry_mode)
    try:
        _obs, _info = env.reset(seed=map_seed)
        total_reward = 0.0
        terminated = truncated = False
        index = 0
        while not (terminated or truncated):
            action = actions[index] if index < len(actions) else WAIT
            _obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            index += 1
        return total_reward
    finally:
        env.close()


def _replay_observations(
    geometry_mode: str, map_seed: int, actions: list[int]
) -> list[NDArray[np.float32]]:
    """用动作脚本在全新 env 上 replay，返回每个决策点收到的 obs。"""
    env = make_geometry_eval_env(geometry_mode)
    try:
        obs, _info = env.reset(seed=map_seed)
        collected: list[NDArray[np.float32]] = []
        terminated = truncated = False
        index = 0
        while not (terminated or truncated):
            action = actions[index] if index < len(actions) else WAIT
            collected.append(np.array(obs, copy=True))
            obs, _reward, terminated, truncated, _info = env.step(action)
            index += 1
        return collected
    finally:
        env.close()


def _env_public_snapshot(env: Any) -> tuple[Any, ...]:
    """对 env 公开属性（含 wrapper 预算字段）做值快照，用于只读性断言。"""
    hook = cast(GoldMinerEnv, env.unwrapped)
    return (
        hook.score,
        hook.remaining_time,
        hook.hook_state,
        hook.angle,
        hook.swing_direction,
        hook.rope_length,
        hook.attached_object,
        tuple(
            (
                obj.type,
                obj.x,
                obj.y,
                obj.radius,
                obj.value,
                obj.retract_speed,
                obj.active,
            )
            for obj in hook.objects
        ),
        env.fires_used,
        env.fires_remaining,
    )


# ---------------------------------------------------------------------------
# §15.10 deterministic 推理契约
# ---------------------------------------------------------------------------
def test_runner_always_calls_predict_with_deterministic_true() -> None:
    """目的：§15.10 —— runner 对策略的每次调用都必须传 deterministic=True，
    禁止 epsilon / 随机推理。

    输入：make_geometry_eval_env("id") + _StrictDeterministicModel（签名
    默认 deterministic=False，记录每次实际收到的值，False 即 fail），
    map seed 3000 跑一局完整 episode。
    输出：episode 正常结束且至少决策一次；stub 收到的 deterministic 全为
    True；动作序列全部合法（WAIT/FIRE）。
    """
    env = make_geometry_eval_env("id")
    model = _StrictDeterministicModel()
    try:
        result = run_geometry_episode(env, model, map_seed=3000)
    finally:
        env.close()

    assert result.decisions >= 1
    assert len(model.deterministic_flags) == result.decisions
    assert all(model.deterministic_flags)
    assert set(model.actions) <= {WAIT, FIRE}


# ---------------------------------------------------------------------------
# §15.11 classify_fire 三分类
# ---------------------------------------------------------------------------
def test_classify_fire_three_mutually_exclusive_classes() -> None:
    """目的：§15.11 —— classify_fire 的三类互斥语义；timeout 必须单列，
    不得归为普通 miss。

    输入：active flag 1→0（productive，含 truncated=True 变体）、无变化
    且未 truncated（miss）、无变化且 truncated=True（timeout_fire）。
    输出：依次返回 "productive" / "miss" / "timeout_fire"；收集发生时
    即使同 transition 内 timeout 也优先计为 productive。
    """
    assert classify_fire((True, True, True), (False, True, True), False) == (
        "productive"
    )
    assert classify_fire((True, True, True), (False, False, True), True) == (
        "productive"
    )
    assert classify_fire((True, False, True), (True, False, True), False) == "miss"
    assert classify_fire((True, True, True), (True, True, True), True) == (
        "timeout_fire"
    )
    assert classify_fire((False, False, False), (False, False, False), True) == (
        "timeout_fire"
    )


# ---------------------------------------------------------------------------
# §15.11 oracle 策略完整 episode：计数 / 采集率 / score==reward 累加
# ---------------------------------------------------------------------------
def test_oracle_policy_episode_metrics_match_manual_replay() -> None:
    """目的：§15.11(i) —— oracle_action 驱动的最小完整 episode：productive
    计数、per-object collection count/rate、score==reward 累加、
    collected_object_type tally 全部正确。

    输入：make_geometry_eval_env("id")，map seeds 3000/3007，oracle stub；
    seed 3000 另在全新 env 上以相同动作脚本手工 replay 累加 reward。
    输出：两局均 3 次 FIRE 全部 productive、三物体全部收集、
    per-episode tally 各类型恰 1 次；summarize 的 gold/diamond/rock
    collection count==2、rate==1.0、collected_object_type_per_fire 同为
    2；seed 3000 的 score == 手工 reward 和 == 800。
    """
    env = make_geometry_eval_env("id")
    model = _StrictDeterministicModel()
    results = []
    try:
        for map_seed in SANITY_MAP_SEEDS:
            results.append(run_geometry_episode(env, model, map_seed=map_seed))
    finally:
        env.close()

    for result in results:
        assert result.fires_used == 3
        assert result.productive_fire_count == 3
        assert result.miss_fire_count == 0
        assert result.timeout_fire_count == 0
        assert result.collected == (True, True, True)
        assert result.collected_object_type_tally == {
            "gold": 1,
            "diamond": 1,
            "rock": 1,
        }

    summary = summarize_geometry_results(results)
    assert summary["episodes"] == 2
    assert summary["gold_collection_count"] == 2
    assert summary["diamond_collection_count"] == 2
    assert summary["rock_collection_count"] == 2
    assert summary["gold_collection_rate"] == pytest.approx(1.0)
    assert summary["diamond_collection_rate"] == pytest.approx(1.0)
    assert summary["rock_collection_rate"] == pytest.approx(1.0)
    assert summary["collected_object_type_per_fire"] == {
        "gold": 2,
        "diamond": 2,
        "rock": 2,
    }
    assert summary["miss_fire_rate"] == pytest.approx(0.0)
    assert summary["episode_scores"] == [result.score for result in results]

    # score 用 reward 累加：与手工 replay 的 reward 总和逐分一致。
    replay_total = _replay_reward_sum("id", 3000, _replayed_actions("id", 3000))
    assert replay_total == pytest.approx(results[0].score)
    assert results[0].score == pytest.approx(FULL_SCORE)


def _replayed_actions(geometry_mode: str, map_seed: int) -> list[int]:
    """跑一局 oracle episode 并返回动作脚本（供手工 replay 对照）。"""
    env = make_geometry_eval_env(geometry_mode)
    model = _StrictDeterministicModel()
    try:
        run_geometry_episode(env, model, map_seed=map_seed)
    finally:
        env.close()
    return model.actions


# ---------------------------------------------------------------------------
# §15.11 构造必然 miss 的 FIRE + 诊断只读性
# ---------------------------------------------------------------------------
def test_scripted_first_fire_miss_is_counted_once_without_score_change() -> None:
    """目的：§15.11(ii) —— 构造必然 miss 的 FIRE：选 map seed 使 reset 后
    首决策角度下 predicted_first_hit_slot(...) is None，首步即 FIRE、其后
    一直 WAIT；并验证 runner 传给策略的 obs 未被改动。

    输入：make_geometry_eval_env("id")、seed 3000（reset 角度 -70° 下无
    任何命中）、动作脚本 [FIRE]（之后全 WAIT）。
    输出：miss_fire_count==1、productive/timeout_fire==0、fires_used==1、
    score 保持 0；该 FIRE 的 collected_slot 与 predicted_first_hit_slot
    均为 None；stub 收到的 obs 序列与全新 env replay 的 obs 逐位相等。
    """
    env = make_geometry_eval_env("id")
    model = _ScriptedModel([FIRE])
    try:
        reset_obs, _info = env.reset(seed=MISS_MAP_SEED)
        assert (
            predicted_first_hit_slot(reset_obs, float(reset_obs[0]) * MAX_ANGLE) is None
        )
        result = run_geometry_episode(env, model, map_seed=MISS_MAP_SEED)
    finally:
        env.close()

    assert result.fires_used == 1
    assert result.miss_fire_count == 1
    assert result.productive_fire_count == 0
    assert result.timeout_fire_count == 0
    assert result.score == 0.0
    assert len(result.fire_diagnostics) == 1
    fire = result.fire_diagnostics[0]
    assert fire.collected_slot_after_step is None
    assert fire.collected_object_type is None
    assert fire.classification == "miss"
    assert fire.predicted_first_hit_slot is None
    assert fire.score_after_fire == 0.0

    # obs 逐位 pass-through：runner 未修改传入策略的任何 obs。
    replayed = _replay_observations("id", MISS_MAP_SEED, model.actions_script)
    assert len(replayed) == len(model.observations)
    for recorded, expected in zip(model.observations, replayed):
        assert np.array_equal(recorded, expected)


def test_fire_diagnostics_do_not_mutate_obs_or_env_state() -> None:
    """目的：§15.11 —— FIRE 诊断是纯 observation-only：计算前后 obs 逐位
    不变，env 公开属性（score / hook 状态 / objects / FIRE 预算）完全不变。

    输入：make_geometry_eval_env("rot3") reset(seed=3000) 后的 obs、env
    公开属性值快照；对同一 obs 调用 fire_observation_diagnostics。
    输出：obs bitwise 相等；快照逐项相等（objects 的 type/坐标/半径/
    value/速度/active 全部不变）。
    """
    env = make_geometry_eval_env("rot3")
    try:
        obs, _info = env.reset(seed=3000)
        snapshot = _env_public_snapshot(env)
        original = np.array(obs, copy=True)

        fire_observation_diagnostics(obs)

        assert np.array_equal(obs, original)
        assert _env_public_snapshot(env) == snapshot
    finally:
        env.close()


# ---------------------------------------------------------------------------
# §15.12 小规模 oracle 集成 sanity（不放 100-map gate）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("geometry_mode", GEOMETRY_MODES)
@pytest.mark.parametrize("map_seed", SANITY_MAP_SEEDS)
def test_oracle_sanity_full_collection_on_each_geometry_mode(
    geometry_mode: str, map_seed: int
) -> None:
    """目的：§15.12 —— 每个 geometry mode 各取固定 map seed，用 M8 oracle
    跑完整 episode 的小规模集成 sanity。

    输入：make_geometry_eval_env(geometry_mode)、oracle stub（同一
    deterministic 契约），map seeds 3000/3007。
    输出：episode 正常结束（terminated=True、truncated=False）、reward
    总和（手工 replay）== 最终 score == 800、三物体全部收集、3 次 FIRE
    全部 productive 且均命中 predicted_first_hit_slot。
    """
    env = make_geometry_eval_env(geometry_mode)
    model = _StrictDeterministicModel()
    try:
        result = run_geometry_episode(env, model, map_seed=map_seed)
    finally:
        env.close()

    assert result.terminated is True
    assert result.truncated is False
    assert result.score == pytest.approx(FULL_SCORE)
    assert result.collected == (True, True, True)
    assert result.fires_used == 3
    assert result.productive_fire_count == 3

    replay_total = _replay_reward_sum(geometry_mode, map_seed, model.actions)
    assert replay_total == pytest.approx(result.score)
    for fire in result.fire_diagnostics:
        assert fire.classification == "productive"
        assert fire.collected_slot_after_step is not None
        assert fire.predicted_first_hit_slot is not None


# ---------------------------------------------------------------------------
# §13 / §12 paired 分析 helper
# ---------------------------------------------------------------------------
def test_paired_analysis_delta_and_counts() -> None:
    """目的：§13 —— paired_analysis 的小数组行为：逐元素 delta、mean/
    median、improved(>0)/unchanged(==0)/degraded(<0) 计数，以及非法输入。

    输入：id=[800, 400, 250, 0]、ood=[800, 250, 250, 250]（同序配对）；
    另验长度不等与空列表两种非法输入。
    输出：deltas=[0, -150, 0, 250]、mean==25、median==0、improved==1、
    unchanged==2、degraded==1；长度不等与空输入均抛 ValueError。
    """
    analysis = paired_analysis([800.0, 400.0, 250.0, 0.0], [800.0, 250.0, 250.0, 250.0])

    assert analysis["paired_count"] == 4
    assert analysis["mean_paired_delta"] == pytest.approx(25.0)
    assert analysis["median_paired_delta"] == pytest.approx(0.0)
    assert analysis["improved_count"] == 1
    assert analysis["unchanged_count"] == 2
    assert analysis["degraded_count"] == 1

    with pytest.raises(ValueError):
        paired_analysis([800.0], [800.0, 0.0])
    with pytest.raises(ValueError):
        paired_analysis([], [])


def test_across_seed_paired_analysis_drop_retention_and_zero_guard() -> None:
    """目的：§12 —— 跨 training seed 聚合：drop=ood−id、retention=ood/id、
    across-seed mean/std、retention>=0.90 与 <0.60 计数；id_mean==0 时
    retention 定义为 None 且不进入 mean/std 与两个计数。

    输入：id_means=[800, 400, 0]、ood_means=[720, 440, 300]；另验 id 全
    为 0 的极端情形与空/长度不等输入。
    输出：drops=[-80, 40, 300]、retentions=[0.9, 1.1, None]、mean_drop≈
    86.667、std_drop≈158.606、mean_retention==1.0、std_retention≈0.1、
    seeds_retention_ge_090==2、seeds_retention_lt_060==0；id 全 0 时
    mean/std_retention 为 None 且两个计数为 0；非法输入抛 ValueError。
    """
    analysis = across_seed_paired_analysis([800.0, 400.0, 0.0], [720.0, 440.0, 300.0])

    assert analysis["seed_count"] == 3
    assert analysis["per_seed_drop"] == pytest.approx([-80.0, 40.0, 300.0])
    assert analysis["per_seed_retention"][0] == pytest.approx(0.9)
    assert analysis["per_seed_retention"][1] == pytest.approx(1.1)
    assert analysis["per_seed_retention"][2] is None
    assert analysis["mean_drop"] == pytest.approx(260.0 / 3.0)
    assert analysis["std_drop"] == pytest.approx(158.6056, rel=1e-4)
    assert analysis["mean_retention"] == pytest.approx(1.0)
    assert analysis["std_retention"] == pytest.approx(0.1)
    assert analysis["seeds_retention_ge_090"] == 2
    assert analysis["seeds_retention_lt_060"] == 0

    all_zero = across_seed_paired_analysis([0.0, 0.0], [100.0, 200.0])
    assert all_zero["per_seed_retention"] == [None, None]
    assert all_zero["mean_retention"] is None
    assert all_zero["std_retention"] is None
    assert all_zero["seeds_retention_ge_090"] == 0
    assert all_zero["seeds_retention_lt_060"] == 0
    assert all_zero["per_seed_drop"] == pytest.approx([100.0, 200.0])

    with pytest.raises(ValueError):
        across_seed_paired_analysis([800.0], [800.0, 0.0])
    with pytest.raises(ValueError):
        across_seed_paired_analysis([], [])


def test_summarize_geometry_results_rejects_empty_results() -> None:
    """目的：汇总入口对空结果必须显式报错而非返回空洞指标。

    输入：summarize_geometry_results([])。
    输出：抛 ValueError。
    """
    with pytest.raises(ValueError):
        summarize_geometry_results([])
