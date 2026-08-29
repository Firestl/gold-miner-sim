"""Behavior tests for the Milestone 1 ``GoldMinerEnv`` (issue #1, section 15).

The environment is fully deterministic, so most tests assert exact tick
arithmetic (one tick is ``DT = 1/60`` s, the hook swings exactly 1 degree per
``WAIT`` tick and clamps at +/-70 degrees).

Only public behavior of ``gold_miner_sim.env`` is exercised, plus one
white-box integration test that moves an object onto another object's fire
ray to verify nearest-first collision selection.
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
    MAX_ANGLE,
    MAX_ROPE_LENGTH,
    MIN_ANGLE,
    MIN_ROPE_LENGTH,
    SIM_FPS,
    SWING_ANGULAR_SPEED,
    FIRE,
    WAIT,
    GameObject,
    GoldMinerEnv,
    HookState,
    ObjectType,
    sweep_circle_hit,
)

INITIAL_ANGLE = -70.0
ANGLE_TOL = 0.6  # The hook swings exactly 1 degree per tick, so 0.6 deg
# unambiguously selects the tick where the swing crosses the target angle.

# Fixed map from the spec (issue #1 section 4): slot order GOLD/DIAMOND/ROCK.
MAP = (
    (ObjectType.GOLD, (315.0, 300.0), 30.0, 250.0, 140.0),
    (ObjectType.DIAMOND, (465.0, 450.0), 18.0, 500.0, 280.0),
    (ObjectType.ROCK, (610.0, 340.0), 34.0, 50.0, 90.0),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wait_until_swinging(env: GoldMinerEnv, max_steps: int = 400) -> None:
    """Step WAIT until the hook is back in SWINGING."""
    steps = 0
    while env.hook_state is not HookState.SWINGING:
        env.step(WAIT)
        steps += 1
        assert steps < max_steps, "hook never returned to SWINGING"


def _fire_at_angle(env: GoldMinerEnv, target_angle: float, tol: float = ANGLE_TOL) -> None:
    """WAIT until SWINGING near ``target_angle``, then FIRE."""
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
    """From a fired hook, check the full catch + loaded-retract + scoring cycle."""
    # Extension phase until the catch.
    steps = 0
    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)
        steps += 1
        assert steps < 200, "hook never hit the expected object"
    assert env.hook_state is HookState.RETRACT_LOADED
    attached = env.attached_object
    assert attached is not None and attached is env.objects[slot]
    assert attached.type is expected_type

    # Loaded retract: no score, constant per-tick rope shortening, and the
    # object center locked to the hook tip on every tick.
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
            assert (attached.x, attached.y) == env.hook_tip  # exact: assigned this tick
            assert env.rope_length == pytest.approx(prev_rope - expected_speed * DT)
        else:
            scoring_rewards.append(reward)
        prev_rope = env.rope_length
        steps += 1
        assert steps < 600, "loaded retract never finished"

    # Scored exactly once, with the object's value, when reaching the top.
    assert len(scoring_rewards) == 1
    assert scoring_rewards[0] == pytest.approx(expected_value)
    assert env.score == pytest.approx(score_at_catch + expected_value)
    assert attached.active is False
    assert env.attached_object is None
    assert env.hook_state is HookState.SWINGING
    assert env.rope_length == pytest.approx(MIN_ROPE_LENGTH)


def _place_on_ray(obj: GameObject, angle_deg: float, rope: float) -> None:
    """White-box helper: put ``obj``'s center on the fire ray of
    ``angle_deg`` at radial distance ``rope`` from the anchor."""
    rad = math.radians(angle_deg)
    obj.x = ANCHOR[0] + rope * math.sin(rad)
    obj.y = ANCHOR[1] + rope * math.cos(rad)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
def test_reset_initial_state() -> None:
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
    env = GoldMinerEnv()
    obs_fresh, _ = env.reset(seed=1)

    # Partially play: catch GOLD so rope/angle/attached/object state move.
    _fire_at_angle(env, -30.0)
    steps = 0
    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)
        steps += 1
    assert env.hook_state is HookState.RETRACT_LOADED
    assert env.attached_object is not None

    obs, _ = env.reset(seed=7)  # different seed: fixed map must not change
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
    # No residue: the reset observation matches a brand-new env bit for bit.
    other = GoldMinerEnv()
    obs_other, _ = other.reset(seed=99)
    assert np.array_equal(obs, obs_other)
    assert np.array_equal(obs_fresh, obs_other)


# ---------------------------------------------------------------------------
# Swing
# ---------------------------------------------------------------------------
def test_swing_updates_angle_one_degree_per_wait() -> None:
    env = GoldMinerEnv()
    env.reset()
    env.step(WAIT)
    assert env.angle == pytest.approx(INITIAL_ANGLE + SWING_ANGULAR_SPEED * DT)
    assert env.angle == pytest.approx(-69.0)
    assert env.swing_direction == 1


def test_swing_reverses_at_boundaries() -> None:
    env = GoldMinerEnv()
    env.reset()

    # Swing right until the +70 clamp (drift-free: exactly 140 ticks).
    for _ in range(200):
        if env.swing_direction == -1:
            break
        env.step(WAIT)
    assert env.angle == pytest.approx(MAX_ANGLE)
    assert env.swing_direction == -1

    # Swing left until the -70 clamp, then head right again.
    for _ in range(200):
        if env.swing_direction == 1:
            break
        env.step(WAIT)
    assert env.angle == pytest.approx(MIN_ANGLE)
    assert env.swing_direction == 1
    env.step(WAIT)
    assert env.angle == pytest.approx(MIN_ANGLE + 1.0)


def test_swing_never_exceeds_bounds() -> None:
    env = GoldMinerEnv()
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(1000):
        env.step(int(rng.integers(0, 2)))  # random WAIT/FIRE script
        assert MIN_ANGLE <= env.angle <= MAX_ANGLE
        assert env.swing_direction in (-1, 1)


# ---------------------------------------------------------------------------
# Fire
# ---------------------------------------------------------------------------
def test_fire_enters_extending_and_freezes_angle() -> None:
    env = GoldMinerEnv()
    env.reset()
    env.step(FIRE)
    assert env.hook_state is HookState.EXTENDING
    frozen_angle = env.angle
    rope = env.rope_length
    for _ in range(20):  # 21 extending ticks total: far from the 460 px clamp
        _, _, _, truncated, _ = env.step(WAIT)
        assert not truncated
        assert env.angle == frozen_angle
        assert env.rope_length == pytest.approx(rope + EXTENSION_SPEED * DT)
        rope = env.rope_length


def test_fire_has_no_effect_outside_swinging() -> None:
    env = GoldMinerEnv()
    env.reset()
    env.step(FIRE)  # empty shot from -70 deg: hits nothing
    for _ in range(5):
        env.step(WAIT)

    # FIRE during EXTENDING must not restart or alter the launch.
    _, reward, _, truncated, _ = env.step(FIRE)
    assert not truncated
    assert reward == 0.0
    assert env.hook_state is HookState.EXTENDING
    for _ in range(3):
        rope = env.rope_length
        env.step(FIRE)  # repeated FIREs keep being ignored
        assert env.hook_state is HookState.EXTENDING
        assert env.rope_length == pytest.approx(rope + EXTENSION_SPEED * DT)

    # FIRE during RETRACT_EMPTY must not relaunch either.
    _wait_until_state(env, HookState.RETRACT_EMPTY)
    rope = env.rope_length
    env.step(FIRE)
    assert env.hook_state is HookState.RETRACT_EMPTY
    assert env.rope_length == pytest.approx(rope - EMPTY_RETRACT_SPEED * DT)

    # Once SWINGING again, FIRE works normally.
    _wait_until_swinging(env)
    env.step(FIRE)
    assert env.hook_state is HookState.EXTENDING


def _wait_until_state(env: GoldMinerEnv, state: HookState, max_steps: int = 400) -> None:
    """Step WAIT until ``env.hook_state`` is ``state``."""
    steps = 0
    while env.hook_state is not state:
        env.step(WAIT)
        steps += 1
        assert steps < max_steps, f"hook never entered {state}"


# ---------------------------------------------------------------------------
# Empty retract
# ---------------------------------------------------------------------------
def test_empty_retract_full_cycle() -> None:
    env = GoldMinerEnv()
    env.reset()
    env.step(FIRE)  # -70 deg ray passes near no object

    # Extension: 50 + 77 * (320/60) crosses 460 -> clamp and switch state.
    steps = 0
    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)
        steps += 1
    assert steps == 76  # plus the FIRE tick itself = 77 extending ticks
    assert env.hook_state is HookState.RETRACT_EMPTY
    assert env.rope_length == pytest.approx(MAX_ROPE_LENGTH)
    assert env.angle == pytest.approx(INITIAL_ANGLE)  # frozen
    assert env.score == 0.0

    # Empty retract at 360 px/s: 69 ticks back to the minimum length.
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
    env = GoldMinerEnv()
    env.reset()
    for _ in range(140):  # swing right up to the +70 clamp
        env.step(WAIT)
    assert env.swing_direction == -1
    for _ in range(30):  # now at +40 deg moving left; ray misses all objects
        env.step(WAIT)
    assert env.angle == pytest.approx(40.0)
    env.step(FIRE)
    _wait_until_swinging(env)
    # Direction is kept from before the launch (still moving left).
    assert env.swing_direction == -1
    assert env.angle == pytest.approx(40.0)
    env.step(WAIT)
    assert env.angle == pytest.approx(39.0)


# ---------------------------------------------------------------------------
# Loaded retract (one full capture per object type)
# ---------------------------------------------------------------------------
def test_loaded_retract_gold() -> None:
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, -30.0)  # -30 deg ray passes 1.9 px from GOLD center
    _run_loaded_retract(env, slot=0, expected_type=ObjectType.GOLD,
                        expected_value=250.0, expected_speed=140.0)
    # The +250 reward must not fire a second time.
    _, reward, _, truncated, _ = env.step(WAIT)
    assert reward == 0.0 and not truncated
    assert env.score == pytest.approx(250.0)


def test_loaded_retract_diamond() -> None:
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, 2.0)
    _run_loaded_retract(env, slot=1, expected_type=ObjectType.DIAMOND,
                        expected_value=500.0, expected_speed=280.0)


def test_loaded_retract_rock() -> None:
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, 31.0)
    _run_loaded_retract(env, slot=2, expected_type=ObjectType.ROCK,
                        expected_value=50.0, expected_speed=90.0)


def test_hit_step_syncs_attached_object_to_hook_tip() -> None:
    """The step that detects the hit must already report the object at the
    hook tip (issue #1 section 5.4), not at its original map position."""
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, -30.0)  # normal GOLD catch

    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)

    assert env.hook_state is HookState.RETRACT_LOADED
    attached = env.attached_object
    assert attached is not None and attached is env.objects[0]
    tip_x, tip_y = env.hook_tip
    assert attached.x == pytest.approx(tip_x)
    assert attached.y == pytest.approx(tip_y)


# ---------------------------------------------------------------------------
# Collision
# ---------------------------------------------------------------------------
def test_sweep_circle_hit_through_center() -> None:
    t = sweep_circle_hit(0.0, 0.0, 10.0, 0.0, 5.0, 0.0, 2.0)
    assert t is not None
    assert 0.0 < t < 1.0
    assert t == pytest.approx(0.3)


def test_sweep_circle_hit_near_miss() -> None:
    # Perpendicular distance 2.5 > r = 2: a graze that must not connect.
    assert sweep_circle_hit(0.0, 0.0, 10.0, 0.0, 5.0, 2.5, 2.0) is None
    # Circle beyond the segment end / behind the start.
    assert sweep_circle_hit(0.0, 0.0, 10.0, 0.0, 15.0, 0.0, 2.0) is None
    assert sweep_circle_hit(0.0, 0.0, 10.0, 0.0, -5.0, 0.0, 2.0) is None


def test_sweep_circle_hit_edge_cases() -> None:
    # Start point already inside the circle.
    assert sweep_circle_hit(0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 2.0) == 0.0
    # Degenerate zero-length segment: reduces to a point-in-circle test.
    assert sweep_circle_hit(5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 2.0) == 0.0
    assert sweep_circle_hit(5.0, 3.0, 5.0, 3.0, 5.0, 0.0, 2.0) is None


def test_collision_picks_nearest_object_on_same_ray() -> None:
    env = GoldMinerEnv()
    env.reset()
    # White-box: move the DIAMOND onto the -30 deg fire ray, well behind GOLD.
    rad = math.radians(-30.0)
    far_rope = 400.0
    diamond = env.objects[1]
    diamond.x = ANCHOR[0] + far_rope * math.sin(rad)
    diamond.y = ANCHOR[1] + far_rope * math.cos(rad)

    _fire_at_angle(env, -30.0)
    _run_loaded_retract(env, slot=0, expected_type=ObjectType.GOLD,
                        expected_value=250.0, expected_speed=140.0)
    assert diamond.active is True  # the far object was not grabbed

    # Same frozen angle, next launch grabs the (now nearest) DIAMOND.
    env.step(FIRE)
    assert env.hook_state is HookState.EXTENDING
    _run_loaded_retract(env, slot=1, expected_type=ObjectType.DIAMOND,
                        expected_value=500.0, expected_speed=280.0)
    assert env.objects[2].active is True
    assert env.score == pytest.approx(750.0)


def test_max_rope_boundary_collision_ignores_overshoot() -> None:
    """An object only touching the final tick's overshoot path (rope in
    (MAX_ROPE_LENGTH, unclamped end]) must be missed: the sweep is done on
    the path clamped to MAX_ROPE_LENGTH, so the shot ends RETRACT_EMPTY."""
    env = GoldMinerEnv()
    env.reset()

    # Replicate the extension tick arithmetic to find the last extending
    # tick: prev_rope -> min(prev_rope + EXTENSION_SPEED * DT, MAX_ROPE_LENGTH).
    prev_rope = MIN_ROPE_LENGTH
    while prev_rope + EXTENSION_SPEED * DT < MAX_ROPE_LENGTH:
        prev_rope += EXTENSION_SPEED * DT
    overshoot_end = prev_rope + EXTENSION_SPEED * DT
    assert MAX_ROPE_LENGTH < overshoot_end  # unclamped end crosses the limit

    # Contact circle = radius + HOOK_RADIUS = 30.4 px. Its center sits 30 px
    # beyond the unclamped end, so its near edge (overshoot_end - 0.4) lies
    # inside the overshoot band but strictly beyond MAX_ROPE_LENGTH: the
    # legal sweep path cannot reach it, the unclamped (buggy) path would.
    gold = env.objects[0]
    gold.radius = 24.4
    _place_on_ray(gold, INITIAL_ANGLE, overshoot_end + 30.0)

    # Geometry sanity, using the same sweep primitive as the environment.
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

    env.step(FIRE)  # launch straight from the initial -70 deg swing
    while env.hook_state is HookState.EXTENDING:
        env.step(WAIT)

    assert env.hook_state is HookState.RETRACT_EMPTY  # miss, not a catch
    assert env.rope_length == pytest.approx(MAX_ROPE_LENGTH)
    assert env.attached_object is None
    assert gold.active is True


def test_collision_on_legal_path_near_max_rope_hits() -> None:
    """Control for the overshoot test: the same object on the legal path
    (radial distance <= MAX_ROPE_LENGTH) is caught normally."""
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
# Observation contract (issue #1 section 8: fixed 26-slot layout)
# ---------------------------------------------------------------------------
def test_observation_contract_reset_exact_26_values() -> None:
    env = GoldMinerEnv()
    obs, _ = env.reset()
    expected = np.array(
        [
            INITIAL_ANGLE / MAX_ANGLE,  # 0: normalized angle = -1
            1.0,  # 1: swing direction
            1.0, 0.0, 0.0, 0.0,  # 2-5: one-hot SWINGING
            MIN_ROPE_LENGTH / MAX_ROPE_LENGTH,  # 6: normalized rope 50/460
            EPISODE_TIME / EPISODE_TIME,  # 7: remaining time 60/60 = 1
            # GOLD slot (8-13)
            315.0 / 900.0, 300.0 / 600.0, 30.0 / 100.0,
            250.0 / 500.0, 140.0 / 360.0, 1.0,
            # DIAMOND slot (14-19)
            465.0 / 900.0, 450.0 / 600.0, 18.0 / 100.0,
            500.0 / 500.0, 280.0 / 360.0, 1.0,
            # ROCK slot (20-25)
            610.0 / 900.0, 340.0 / 600.0, 34.0 / 100.0,
            50.0 / 500.0, 90.0 / 360.0, 1.0,
        ],
        dtype=np.float32,
    )
    assert obs.shape == (26,)
    assert obs == pytest.approx(expected)


def test_observation_contract_tracks_state_changes() -> None:
    env = GoldMinerEnv()
    env.reset()

    # The FIRE tick enters EXTENDING and already advances one physics tick:
    # the one-hot moves, rope grows by one extension step, one tick of time
    # (1/3600 of the normalized budget) is consumed.
    obs, _, _, truncated, _ = env.step(FIRE)
    assert not truncated
    assert obs[0] == pytest.approx(INITIAL_ANGLE / MAX_ANGLE)  # angle frozen
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
    env = GoldMinerEnv()
    env.reset()
    _fire_at_angle(env, -30.0)
    obs: np.ndarray | None = None
    while True:
        obs, _, _, truncated, _ = env.step(WAIT)
        assert not truncated
        if env.hook_state is HookState.SWINGING:
            break  # the tick that scored GOLD and returned to SWINGING
    assert obs is not None
    assert env.score == pytest.approx(250.0)
    # The whole collected slot is exactly zero.
    assert list(obs[8:14]) == [0.0] * 6
    # The other slots keep their exact contract values.
    assert obs[14:20] == pytest.approx(
        np.array([465.0 / 900.0, 450.0 / 600.0, 18.0 / 100.0,
                  500.0 / 500.0, 280.0 / 360.0, 1.0], dtype=np.float32)
    )
    assert obs[20:26] == pytest.approx(
        np.array([610.0 / 900.0, 340.0 / 600.0, 34.0 / 100.0,
                  50.0 / 500.0, 90.0 / 360.0, 1.0], dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------
def test_full_episode_runs_exactly_3600_ticks_then_truncates() -> None:
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
    assert env.remaining_time == 0.0  # exact: 60 - 3600/60
    assert env.score == 0.0
    assert all(obj.active for obj in env.objects)


def test_timeout_discards_unfinished_loaded_retract() -> None:
    env = GoldMinerEnv()
    env.reset()
    gold = env.objects[0]

    # One empty shot from -70 deg shifts the swing phase by 146 ticks so that
    # the -30 deg heading recurs late in the episode with < 1 s left.
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
            # GOLD needs ~34 ticks to hook + ~78 ticks to retract (1.87 s in
            # total); with <= 1.5 s left the episode must end mid-retract.
            _, _, _, truncated, info = env.step(FIRE)
            fired = True
        else:
            _, _, _, truncated, info = env.step(WAIT)
        if truncated:
            break

    assert fired, "never reached a -30 deg swing with little time left"
    assert truncated is True
    assert env.remaining_time == 0.0
    # The episode ended while still hauling GOLD: no score, object intact.
    assert env.hook_state is HookState.RETRACT_LOADED
    assert env.attached_object is gold
    assert gold.active is True
    assert env.score == 0.0
    assert info["score"] == 0.0
    assert (gold.x, gold.y) == env.hook_tip


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def _record(env: GoldMinerEnv, traj: list[tuple]) -> None:
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
    """Drive the fixed demo policy while recording every action and the
    post-step trajectory entry (capture GOLD, DIAMOND, ROCK, then WAIT to
    timeout)."""
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
    """Issue #1 section 14: replaying the exact same action sequence must
    reproduce the full trajectory (issue review: record first, replay the
    same list second)."""
    env = GoldMinerEnv()
    env.reset(seed=1)
    actions: list[int] = []
    traj_a: list[tuple] = []
    _run_fixed_policy(env, actions, traj_a)

    # One action per tick for the whole 60 s episode, all three objects in.
    assert len(actions) == SIM_FPS * int(EPISODE_TIME)
    assert env.score == pytest.approx(800.0)  # 250 + 500 + 50

    # Strict replay: fresh reset, then follow the recorded list verbatim.
    env.reset(seed=2)  # seeds cannot matter: there is no randomness
    traj_b: list[tuple] = []
    for action in actions:
        env.step(action)
        _record(env, traj_b)

    assert len(traj_a) == len(traj_b)
    assert traj_a == traj_b  # float entries are bit-identical
    assert env.score == pytest.approx(800.0)
    assert env.remaining_time == 0.0
    assert tuple(obj.active for obj in env.objects) == (False, False, False)


# ---------------------------------------------------------------------------
# Gymnasium API compliance
# ---------------------------------------------------------------------------
def test_gymnasium_env_checker() -> None:
    env = GoldMinerEnv()
    check_env(env, skip_render_check=True)
    env.close()
