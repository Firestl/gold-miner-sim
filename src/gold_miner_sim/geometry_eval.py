"""Geometry-evaluation spawn pools for the Milestone 11 stress test.

Pure transform / mode helpers (no wrappers, no reward or observation
changes): each geometry mode maps the official ``RANDOM_SPAWN_POINTS``
pool onto a 12-point evaluation pool around the hook anchor ``ANCHOR``,
preserving point order -- index ``i`` of every pool corresponds to index
``i`` of ``RANDOM_SPAWN_POINTS``.

* ``"id"`` -- the identity control; returns ``RANDOM_SPAWN_POINTS`` itself.
* ``"rot3"`` -- rotates every point by +3 degrees around ``ANCHOR``,
  radial distances unchanged.
* ``"rot3_scale105"`` -- +3 degrees rotation and every radial distance
  scaled by 1.05.

Angles use the project's hook convention (0 deg = straight down,
positive to the right), i.e. ``atan2(dx, dy)`` with ``dx, dy`` measured
from ``ANCHOR`` -- NOT the mathematical ``atan2(dy, dx)``. Transformed
coordinates are exact floats and are never rounded.

The second half of this module (Issue #21 §10-§13) is the frozen-policy
evaluation layer shared by ``scripts/eval_geometry_generalization.py``:
a deterministic episode runner with per-FIRE observation-only diagnostics,
the §10 metric aggregation, and the §12/§13 paired delta / retention
helpers. It never imports stable-baselines3; the policy is only ever
queried through ``model.predict(obs, deterministic=True)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray

from gold_miner_sim.env import ANCHOR, FIRE, MAX_ANGLE, RANDOM_SPAWN_POINTS
from gold_miner_sim.oracle import (
    OBJECT_SLOTS,
    object_geometry_from_observation,
    predicted_first_hit_slot,
    target_angle_deg,
)

GEOMETRY_MODES = ("id", "rot3", "rot3_scale105")

# Per-mode (rotation_deg, radial_scale) applied around ANCHOR.
_GEOMETRY_TRANSFORMS: dict[str, tuple[float, float]] = {
    "id": (0.0, 1.0),
    "rot3": (3.0, 1.0),
    "rot3_scale105": (3.0, 1.05),
}


def transform_point(
    x: float, y: float, rotation_deg: float, radial_scale: float
) -> tuple[float, float]:
    """Rotate and radially scale one point around ``ANCHOR``.

    The angle convention is the environment's hook convention:
    ``a = atan2(dx, dy)`` with ``dx = x - ANCHOR[0]``, ``dy = y - ANCHOR[1]``
    (0 deg = straight down). The point is rotated by ``rotation_deg``
    (added to ``a``) and its anchor distance multiplied by ``radial_scale``.
    No rounding is applied.
    """
    dx = x - ANCHOR[0]
    dy = y - ANCHOR[1]
    r = math.hypot(dx, dy)
    a = math.atan2(dx, dy)  # Project convention: 0 deg = straight down.
    a2 = a + math.radians(rotation_deg)
    r2 = radial_scale * r
    x2 = ANCHOR[0] + r2 * math.sin(a2)
    y2 = ANCHOR[1] + r2 * math.cos(a2)
    return (x2, y2)


def geometry_spawn_pool(geometry_mode: str) -> tuple[tuple[float, float], ...]:
    """Return the frozen 12-point spawn pool for ``geometry_mode``.

    ``"id"`` returns ``RANDOM_SPAWN_POINTS`` itself (identity: no copy, no
    reordering); ``"rot3"`` and ``"rot3_scale105"`` transform every point
    with :func:`transform_point`, keeping the pool's point order one-to-one.
    Raises ``ValueError`` for unknown modes.
    """
    if geometry_mode not in _GEOMETRY_TRANSFORMS:
        raise ValueError(
            f"geometry_mode must be one of {GEOMETRY_MODES}, got {geometry_mode!r}"
        )
    if geometry_mode == "id":
        return RANDOM_SPAWN_POINTS
    rotation_deg, radial_scale = _GEOMETRY_TRANSFORMS[geometry_mode]
    return tuple(
        transform_point(x, y, rotation_deg, radial_scale)
        for x, y in RANDOM_SPAWN_POINTS
    )


# ---------------------------------------------------------------------------
# Issue #21 §10-§13: frozen-policy episode runner, metrics and paired plots.
# ---------------------------------------------------------------------------

# Full-collection episode score (250 gold + 500 diamond + 50 rock).  Kept
# local because gold_miner_sim.strong_dqn imports gold_miner_sim.benchmark,
# which imports this module (importing the constant from there would be a
# circular import).
FULL_SCORE = 800.0

# FIRE classification labels (mutually exclusive; see classify_fire).
PRODUCTIVE_FIRE = "productive"
MISS_FIRE = "miss"
TIMEOUT_FIRE = "timeout_fire"

# Shared benchmark wrapper-chain type (SwingAdvanceDecisionWrapper /
# FireBudgetWrapper / geometry factory output).
_BenchmarkEnv = gymnasium.Wrapper[NDArray[np.float32], int, NDArray[np.float32], int]


@dataclass(frozen=True, slots=True)
class FireDiagnostics:
    """Observation-only diagnostics recorded for one FIRE decision.

    All geometry fields are computed from the pre-decision observation and
    the public static constants (same rule as the M8 oracle); the collected
    fields come from the active flags before/after the FIRE transition.
    """

    fire_number: int
    current_fire_angle: float
    active_object_angles: tuple[float, ...]
    predicted_first_hit_slot: int | None
    nearest_active_center_angle_error_deg: float | None
    collected_slot_after_step: int | None
    collected_object_type: str | None
    classification: str
    score_after_fire: float


@dataclass(frozen=True, slots=True)
class GeometryEpisodeResult:
    """Public metrics collected from one frozen-policy episode."""

    map_seed: int
    score: float
    terminated: bool
    truncated: bool
    decisions: int
    fires_used: int
    productive_fire_count: int
    miss_fire_count: int
    timeout_fire_count: int
    collected: tuple[bool, bool, bool]
    collected_object_types: tuple[str | None, ...]
    fire_diagnostics: tuple[FireDiagnostics, ...]

    @property
    def collected_object_type_tally(self) -> dict[str, int]:
        """Count collected objects per type over this episode's fires."""
        return {
            name: sum(1 for slot in self.collected_object_types if slot == name)
            for name in ("gold", "diamond", "rock")
        }


def classify_fire(
    before_active: Sequence[bool], after_active: Sequence[bool], truncated: bool
) -> str:
    """Classify one FIRE transition into exactly one of three classes.

    ``"productive"`` when any object active flag dropped 1 -> 0 across the
    transition (an object was collected, even if the episode also ended);
    ``"timeout_fire"`` when no flag changed and the transition itself ended
    the episode by timeout; ``"miss"`` when no flag changed and the episode
    continued.  A timeout is therefore never counted as a plain miss.
    """
    collected = any(
        was_active and not is_active
        for was_active, is_active in zip(before_active, after_active)
    )
    if collected:
        return PRODUCTIVE_FIRE
    if truncated:
        return TIMEOUT_FIRE
    return MISS_FIRE


def fire_observation_diagnostics(
    obs: NDArray[np.float32],
) -> tuple[float, tuple[float, ...], int | None, float | None]:
    """Return read-only FIRE geometry diagnostics from one observation.

    Returns ``(current_fire_angle, active_object_angles,
    predicted_first_hit_slot, nearest_active_center_angle_error_deg)``.
    Only the observation and the public static geometry constants are read
    (the M8 oracle rule); neither the observation array nor the environment
    is modified.  Angles use the project convention (0 deg = straight
    down); ``nearest_active_center_angle_error_deg`` is
    ``min(|current_fire_angle - center_angle|)`` over active objects, or
    ``None`` when no object is active.
    """
    current_angle = float(obs[0]) * MAX_ANGLE
    active_angles: list[float] = []
    nearest_error: float | None = None
    for slot in OBJECT_SLOTS:
        geometry = object_geometry_from_observation(obs, slot)
        if not geometry.active:
            continue
        center_angle = target_angle_deg(geometry.x_px, geometry.y_px)
        active_angles.append(center_angle)
        error = abs(current_angle - center_angle)
        if nearest_error is None or error < nearest_error:
            nearest_error = error
    return (
        current_angle,
        tuple(active_angles),
        predicted_first_hit_slot(obs, current_angle),
        nearest_error,
    )


def _active_flags(obs: NDArray[np.float32]) -> tuple[bool, bool, bool]:
    """Read the three public active flags from a benchmark observation."""
    flags = tuple(
        object_geometry_from_observation(obs, slot).active for slot in OBJECT_SLOTS
    )
    return flags[0], flags[1], flags[2]


def run_geometry_episode(
    env: _BenchmarkEnv,
    model: Any,
    map_seed: int,
) -> GeometryEpisodeResult:
    """Run one frozen-policy episode and record per-FIRE diagnostics.

    The policy is queried exactly once per decision with
    ``model.predict(obs, deterministic=True)`` (deterministic inference is
    mandatory; no epsilon-greedy or stochastic inference), ``score`` is the
    accumulated reward, and the collected triple is read from the final
    observation's active flags.  Diagnostics only read the pre-decision
    observation plus public constants; they never enter the policy
    observation and never touch environment internals.
    """
    obs, _info = env.reset(seed=map_seed)
    terminated = False
    truncated = False
    score = 0.0
    decisions = 0
    fires_used = 0
    productive_fires = 0
    miss_fires = 0
    timeout_fires = 0
    fire_records: list[FireDiagnostics] = []

    while not (terminated or truncated):
        before_obs = obs
        before_active = _active_flags(before_obs)
        action, _state = model.predict(before_obs, deterministic=True)
        action_int = int(action)
        obs, reward, terminated, truncated, _info = env.step(action_int)
        score += float(reward)
        decisions += 1

        if action_int != FIRE:
            continue
        fires_used += 1
        after_active = _active_flags(obs)
        classification = classify_fire(before_active, after_active, truncated)
        collected_slot: int | None = None
        for index, (was_active, is_active) in enumerate(
            zip(before_active, after_active)
        ):
            if was_active and not is_active:
                collected_slot = index
                break
        collected_type = (
            OBJECT_SLOTS[collected_slot][0] if collected_slot is not None else None
        )
        (current_angle, active_angles, predicted_slot, nearest_error) = (
            fire_observation_diagnostics(before_obs)
        )
        fire_records.append(
            FireDiagnostics(
                fire_number=fires_used,
                current_fire_angle=current_angle,
                active_object_angles=active_angles,
                predicted_first_hit_slot=predicted_slot,
                nearest_active_center_angle_error_deg=nearest_error,
                collected_slot_after_step=collected_slot,
                collected_object_type=collected_type,
                classification=classification,
                score_after_fire=score,
            )
        )
        if classification == PRODUCTIVE_FIRE:
            productive_fires += 1
        elif classification == TIMEOUT_FIRE:
            timeout_fires += 1
        else:
            miss_fires += 1

    final_active = _active_flags(obs)
    return GeometryEpisodeResult(
        map_seed=map_seed,
        score=score,
        terminated=bool(terminated),
        truncated=bool(truncated),
        decisions=decisions,
        fires_used=fires_used,
        productive_fire_count=productive_fires,
        miss_fire_count=miss_fires,
        timeout_fire_count=timeout_fires,
        collected=(not final_active[0], not final_active[1], not final_active[2]),
        collected_object_types=tuple(
            record.collected_object_type for record in fire_records
        ),
        fire_diagnostics=tuple(fire_records),
    )


def summarize_geometry_results(
    results: Sequence[GeometryEpisodeResult],
) -> dict[str, Any]:
    """Aggregate episode results into the Issue #21 §10 metric table.

    ``miss_fire_rate`` is ``miss_fire_count`` divided by the total number
    of fires (0.0 when no fire happened); timeouts are reported separately
    and never counted as misses.  ``collected_*_count/rate`` read the final
    episode active flags.  ``mean_nearest_active_center_angle_error_deg``
    averages the absolute per-fire nearest-angle errors over the fires
    that had at least one active object (``None`` when no fire qualifies),
    and ``episode_scores`` keeps the per-episode scores ordered by map seed
    for paired analysis.
    """
    if not results:
        raise ValueError("at least one episode is required")
    episodes = len(results)
    scores = np.asarray([result.score for result in results], dtype=np.float64)
    ordered = sorted(results, key=lambda result: result.map_seed)
    full_score_count = int(np.count_nonzero(scores == FULL_SCORE))
    productive_fires = sum(result.productive_fire_count for result in results)
    miss_fires = sum(result.miss_fire_count for result in results)
    timeout_fires = sum(result.timeout_fire_count for result in results)
    total_fires = sum(result.fires_used for result in results)
    tally: dict[str, int] = {"gold": 0, "diamond": 0, "rock": 0}
    for result in results:
        for object_type in result.collected_object_types:
            if object_type is not None:
                tally[object_type] += 1
    angle_errors = [
        abs(record.nearest_active_center_angle_error_deg)
        for result in results
        for record in result.fire_diagnostics
        if record.nearest_active_center_angle_error_deg is not None
    ]
    return {
        "episodes": episodes,
        "mean_score": float(np.mean(scores)),
        "std_episode": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "full_score_count": full_score_count,
        "full_score_rate": full_score_count / episodes,
        "mean_episode_decisions": float(
            np.mean([result.decisions for result in results])
        ),
        "mean_fires_used": float(np.mean([result.fires_used for result in results])),
        "gold_collection_count": sum(result.collected[0] for result in results),
        "gold_collection_rate": sum(result.collected[0] for result in results)
        / episodes,
        "diamond_collection_count": sum(result.collected[1] for result in results),
        "diamond_collection_rate": sum(result.collected[1] for result in results)
        / episodes,
        "rock_collection_count": sum(result.collected[2] for result in results),
        "rock_collection_rate": sum(result.collected[2] for result in results)
        / episodes,
        "productive_fire_count": productive_fires,
        "miss_fire_count": miss_fires,
        "miss_fire_rate": miss_fires / total_fires if total_fires else 0.0,
        "timeout_fire_count": timeout_fires,
        "collected_object_type_per_fire": tally,
        "mean_nearest_active_center_angle_error_deg": (
            float(np.mean(angle_errors)) if angle_errors else None
        ),
        "fires_with_no_predicted_hit_count": sum(
            1
            for result in results
            for record in result.fire_diagnostics
            if record.predicted_first_hit_slot is None
        ),
        "episode_scores": [result.score for result in ordered],
    }


def paired_analysis(
    id_scores: Sequence[float], ood_scores: Sequence[float]
) -> dict[str, float | int]:
    """Paired per-map analysis of two equal-length per-map score lists.

    Both lists must describe the same map seeds in the same order.  The
    paired delta is ``ood - id`` per element; ``improved_count`` counts
    strictly positive deltas, ``unchanged_count`` exact zeros and
    ``degraded_count`` strictly negative deltas.
    """
    if len(id_scores) != len(ood_scores):
        raise ValueError("id_scores and ood_scores must have the same length")
    if not id_scores:
        raise ValueError("at least one paired score is required")
    deltas = [
        float(ood) - float(id_score) for id_score, ood in zip(id_scores, ood_scores)
    ]
    return {
        "paired_count": len(deltas),
        "mean_paired_delta": float(np.mean(deltas)),
        "median_paired_delta": float(np.median(deltas)),
        "improved_count": sum(delta > 0.0 for delta in deltas),
        "unchanged_count": sum(delta == 0.0 for delta in deltas),
        "degraded_count": sum(delta < 0.0 for delta in deltas),
    }


def across_seed_paired_analysis(
    id_means: Sequence[float], ood_means: Sequence[float]
) -> dict[str, Any]:
    """Across-training-seed paired drop / retention aggregate (Issue #21 §12).

    Per training seed ``drop = ood_mean - id_mean`` and
    ``retention = ood_mean / id_mean``.  When ``id_mean == 0.0`` the
    retention ratio is undefined and recorded as ``None``: ``None`` entries
    are excluded from the across-seed retention mean/std and from both
    retention counts (the drops remain defined and always counted).
    ``seeds_retention_ge_090`` / ``seeds_retention_lt_060`` are the counts
    of defined retentions with ``>= 0.90`` / ``< 0.60``.
    """
    if len(id_means) != len(ood_means):
        raise ValueError("id_means and ood_means must have the same length")
    if not id_means:
        raise ValueError("at least one paired seed mean is required")
    drops = [float(ood) - float(id_mean) for id_mean, ood in zip(id_means, ood_means)]
    retentions: list[float | None] = [
        float(ood) / float(id_mean) if float(id_mean) != 0.0 else None
        for id_mean, ood in zip(id_means, ood_means)
    ]
    defined_retentions = [value for value in retentions if value is not None]
    return {
        "seed_count": len(drops),
        "per_seed_drop": drops,
        "per_seed_retention": retentions,
        "mean_drop": float(np.mean(drops)),
        "std_drop": float(np.std(drops)),
        "mean_retention": (
            float(np.mean(defined_retentions)) if defined_retentions else None
        ),
        "std_retention": (
            float(np.std(defined_retentions)) if defined_retentions else None
        ),
        "seeds_retention_ge_090": sum(
            value is not None and value >= 0.90 for value in retentions
        ),
        "seeds_retention_lt_060": sum(
            value is not None and value < 0.60 for value in retentions
        ),
    }
