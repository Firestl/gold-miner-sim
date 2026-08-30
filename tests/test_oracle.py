"""Tests for the observation-only Geometry Oracle."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from gold_miner_sim.env import (
    ANCHOR,
    FIRE,
    HEIGHT,
    HOOK_RADIUS,
    MAX_ANGLE,
    MAX_ROPE_LENGTH,
    MIN_ROPE_LENGTH,
    WAIT,
    WIDTH,
    GoldMinerEnv,
)
from gold_miner_sim.oracle import (
    OBJECT_SLOTS,
    first_active_slot,
    object_geometry_from_observation,
    oracle_action,
    predicted_first_hit_slot,
    ray_circle_entry_distance,
    target_angle_deg,
)
from gold_miner_sim.wrappers import FireBudgetWrapper, SwingAdvanceDecisionWrapper


def _observation(
    *,
    angle: float = 0.0,
    active: tuple[bool, bool, bool] = (False, False, False),
    positions: tuple[tuple[float, float, float], ...] | None = None,
) -> NDArray[np.float32]:
    """Build a valid-shaped full benchmark observation for pure helper tests."""
    obs = np.zeros(27, dtype=np.float32)
    obs[0] = angle / MAX_ANGLE
    if positions is None:
        positions = (
            (ANCHOR[0], ANCHOR[1] + 200.0, 30.0),
            (ANCHOR[0], ANCHOR[1] + 300.0, 18.0),
            (ANCHOR[0], ANCHOR[1] + 400.0, 34.0),
        )
    for is_active, slot, (x_px, y_px, radius_px) in zip(
        active, OBJECT_SLOTS, positions
    ):
        _name, x_index, y_index, radius_index, active_index = slot
        obs[x_index] = x_px / WIDTH
        obs[y_index] = y_px / HEIGHT
        obs[radius_index] = radius_px / 100.0
        obs[active_index] = 1.0 if is_active else 0.0
    return obs


def test_target_angle_uses_environment_coordinate_convention() -> None:
    assert target_angle_deg(ANCHOR[0], ANCHOR[1] + 100.0) == pytest.approx(0.0)
    assert target_angle_deg(ANCHOR[0] - 100.0, ANCHOR[1] + 100.0) == pytest.approx(
        -45.0
    )
    assert target_angle_deg(ANCHOR[0] + 100.0, ANCHOR[1] + 100.0) == pytest.approx(45.0)


def test_object_geometry_denormalizes_observation() -> None:
    obs = _observation(
        active=(True, False, False),
        positions=((123.0, 234.0, 37.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )

    geometry = object_geometry_from_observation(obs, OBJECT_SLOTS[0])

    assert geometry.name == "gold"
    assert geometry.x_px == pytest.approx(123.0)
    assert geometry.y_px == pytest.approx(234.0)
    assert geometry.radius_px == pytest.approx(37.0)
    assert geometry.active is True


def test_inactive_object_is_ignored_even_at_anchor() -> None:
    obs = _observation(
        angle=0.0,
        active=(False, False, False),
        positions=((ANCHOR[0], ANCHOR[1], 100.0),) * 3,
    )

    assert predicted_first_hit_slot(obs, 0.0) is None
    assert oracle_action(obs) == WAIT


def test_ray_through_center_hits() -> None:
    distance = ray_circle_entry_distance(ANCHOR[0], ANCHOR[1] + 200.0, 20.0, 0.0)

    assert distance == pytest.approx(200.0 - 20.0 - HOOK_RADIUS)


def test_ray_tangent_to_expanded_circle_hits() -> None:
    expanded_radius = 20.0 + HOOK_RADIUS
    distance = ray_circle_entry_distance(
        ANCHOR[0] + expanded_radius,
        ANCHOR[1] + 200.0,
        20.0,
        0.0,
    )

    assert distance == pytest.approx(200.0)


def test_ray_outside_expanded_circle_misses() -> None:
    expanded_radius = 20.0 + HOOK_RADIUS
    assert (
        ray_circle_entry_distance(
            ANCHOR[0] + expanded_radius + 0.01,
            ANCHOR[1] + 200.0,
            20.0,
            0.0,
        )
        is None
    )


def test_intersection_before_minimum_rope_length_misses() -> None:
    assert (
        ray_circle_entry_distance(
            ANCHOR[0], ANCHOR[1] + MIN_ROPE_LENGTH - 10.0, 1.0, 0.0
        )
        is None
    )


def test_intersection_after_maximum_rope_length_misses() -> None:
    assert (
        ray_circle_entry_distance(
            ANCHOR[0], ANCHOR[1] + MAX_ROPE_LENGTH + 10.0, 1.0, 0.0
        )
        is None
    )


def test_first_hit_returns_near_object() -> None:
    obs = _observation(
        angle=0.0,
        active=(True, True, False),
        positions=(
            (ANCHOR[0], ANCHOR[1] + 150.0, 10.0),
            (ANCHOR[0], ANCHOR[1] + 300.0, 10.0),
            (0.0, 0.0, 0.0),
        ),
    )

    assert predicted_first_hit_slot(obs, 0.0) == 0


@pytest.mark.parametrize(
    ("active", "expected"),
    [
        ((True, True, True), 0),
        ((False, True, True), 1),
        ((False, False, True), 2),
        ((False, False, False), None),
    ],
)
def test_fixed_target_priority(
    active: tuple[bool, bool, bool], expected: int | None
) -> None:
    obs = _observation(active=active)

    assert first_active_slot(obs) == expected


def test_oracle_action_fires_when_first_hit_is_target() -> None:
    obs = _observation(
        angle=0.0,
        active=(True, False, False),
        positions=(
            (ANCHOR[0], ANCHOR[1] + 200.0, 20.0),
            (0, 0, 0),
            (0, 0, 0),
        ),
    )

    assert oracle_action(obs) == FIRE


def test_oracle_action_waits_when_current_ray_misses_target() -> None:
    obs = _observation(
        angle=0.0,
        active=(True, False, False),
        positions=(
            (ANCHOR[0] + 100.0, ANCHOR[1] + 200.0, 1.0),
            (0, 0, 0),
            (0, 0, 0),
        ),
    )

    assert oracle_action(obs) == WAIT


def test_oracle_action_always_returns_discrete_action() -> None:
    active_cases = (
        (True, True, True),
        (False, True, True),
        (False, False, False),
    )
    for active in active_cases:
        obs = _observation(active=active)
        assert oracle_action(obs) in (WAIT, FIRE)


def test_oracle_helpers_do_not_mutate_observation() -> None:
    obs = _observation(
        angle=0.0,
        active=(True, True, True),
        positions=(
            (ANCHOR[0], ANCHOR[1] + 200.0, 20.0),
            (ANCHOR[0] + 50.0, ANCHOR[1] + 300.0, 18.0),
            (ANCHOR[0] - 50.0, ANCHOR[1] + 400.0, 34.0),
        ),
    )
    original = obs.copy()

    first_active_slot(obs)
    predicted_first_hit_slot(obs, 0.0)
    oracle_action(obs)

    assert np.array_equal(obs, original)


def test_fixed_map_oracle_completes_three_fire_episode() -> None:
    inner = GoldMinerEnv(map_mode="fixed")
    env = FireBudgetWrapper(SwingAdvanceDecisionWrapper(inner), max_fires=3)
    obs, _info = env.reset(seed=0)
    total_reward = 0.0
    fires = 0
    decision_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = oracle_action(obs)
        if action == FIRE:
            fires += 1
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
        decision_count += 1
        assert fires <= 3
        assert env.action_space.contains(action)
        assert decision_count < 100

    assert terminated is True
    assert truncated is False
    assert fires == 3
    assert inner.score == pytest.approx(800.0)
    assert total_reward == pytest.approx(inner.score)
    final_geometries = (
        object_geometry_from_observation(obs, slot) for slot in OBJECT_SLOTS
    )
    assert all(not geometry.active for geometry in final_geometries)
