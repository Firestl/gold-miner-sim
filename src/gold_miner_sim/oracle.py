"""Observation-only geometry oracle for the benchmark environment.

The oracle deliberately reads only the normalized observation and the static
geometry constants exported by :mod:`gold_miner_sim.env`.  It predicts the
first active object intersected by the legal hook ray and fires only when that
object is the first active target in the fixed GOLD/DIAMOND/ROCK priority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
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
)

type ObjectSlot = tuple[str, int, int, int, int]

# (name, x index, y index, radius index, active index) in fixed observation
# order.  The final FIRE-budget slot is intentionally not part of this table.
OBJECT_SLOTS: tuple[ObjectSlot, ...] = (
    ("gold", 8, 9, 10, 13),
    ("diamond", 14, 15, 16, 19),
    ("rock", 20, 21, 22, 25),
)


@dataclass(frozen=True, slots=True)
class ObjectGeometry:
    """One object's geometry decoded from an agent observation."""

    name: str
    x_px: float
    y_px: float
    radius_px: float
    active: bool


def object_geometry_from_observation(
    obs: NDArray[np.float32], slot: ObjectSlot
) -> ObjectGeometry:
    """Decode one object slot from the normalized benchmark observation."""
    name, x_index, y_index, radius_index, active_index = slot
    return ObjectGeometry(
        name=name,
        x_px=float(obs[x_index]) * WIDTH,
        y_px=float(obs[y_index]) * HEIGHT,
        radius_px=float(obs[radius_index]) * 100.0,
        active=float(obs[active_index]) > 0.5,
    )


def target_angle_deg(x_px: float, y_px: float) -> float:
    """Return the target center angle in the environment's angle convention."""
    dx = x_px - ANCHOR[0]
    dy = y_px - ANCHOR[1]
    return math.degrees(math.atan2(dx, dy))


def ray_circle_entry_distance(
    x_px: float, y_px: float, radius_px: float, angle_deg: float
) -> float | None:
    """Return the first legal rope distance where the ray hits an object.

    The object's circle is expanded by ``HOOK_RADIUS`` and the ray is clipped
    to the legal hook segment ``[MIN_ROPE_LENGTH, MAX_ROPE_LENGTH]``.  ``None``
    means that the ray and legal segment do not intersect the expanded circle.
    """
    center_x = x_px - ANCHOR[0]
    center_y = y_px - ANCHOR[1]
    radians = math.radians(angle_deg)
    unit_x = math.sin(radians)
    unit_y = math.cos(radians)

    projection = center_x * unit_x + center_y * unit_y
    center_distance_squared = center_x * center_x + center_y * center_y
    perpendicular_distance_squared = center_distance_squared - projection * projection
    expanded_radius = radius_px + HOOK_RADIUS
    expanded_radius_squared = expanded_radius * expanded_radius

    # Roundoff can make a tangent's perpendicular distance very slightly
    # negative after subtraction; it should still be treated as a tangent hit.
    perpendicular_distance_squared = max(0.0, perpendicular_distance_squared)
    if perpendicular_distance_squared > expanded_radius_squared:
        return None

    half_chord = math.sqrt(
        max(0.0, expanded_radius_squared - perpendicular_distance_squared)
    )
    entry = projection - half_chord
    exit = projection + half_chord
    if exit < MIN_ROPE_LENGTH or entry > MAX_ROPE_LENGTH:
        return None
    return max(MIN_ROPE_LENGTH, entry)


def first_active_slot(obs: NDArray[np.float32]) -> int | None:
    """Return the first active object index in GOLD/DIAMOND/ROCK order."""
    for index, slot in enumerate(OBJECT_SLOTS):
        if object_geometry_from_observation(obs, slot).active:
            return index
    return None


def predicted_first_hit_slot(obs: NDArray[np.float32], angle_deg: float) -> int | None:
    """Predict which active object the current legal ray hits first."""
    best_slot: int | None = None
    best_distance = math.inf
    for index, slot in enumerate(OBJECT_SLOTS):
        geometry = object_geometry_from_observation(obs, slot)
        if not geometry.active:
            continue
        entry_distance = ray_circle_entry_distance(
            geometry.x_px,
            geometry.y_px,
            geometry.radius_px,
            angle_deg,
        )
        if entry_distance is not None and entry_distance < best_distance:
            best_slot = index
            best_distance = entry_distance
    return best_slot


def oracle_action(obs: NDArray[np.float32]) -> int:
    """Return ``FIRE`` only when the current ray first hits the target."""
    target = first_active_slot(obs)
    if target is None:
        return WAIT

    current_angle = float(obs[0]) * MAX_ANGLE
    predicted_hit = predicted_first_hit_slot(obs, current_angle)
    if predicted_hit == target:
        return FIRE
    return WAIT
