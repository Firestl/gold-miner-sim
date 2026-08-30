"""Evaluate the observation-only Geometry Oracle on the random-map benchmark.

Usage:
    uv run python scripts/oracle_baseline.py --episodes 100 --seed 1000
    uv run python scripts/oracle_baseline.py --episodes 1 --seed 1000 --trace
    uv run python scripts/oracle_baseline.py --episodes 1 --seed 1000 --render
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import gymnasium
import numpy as np
from numpy.typing import NDArray

from gold_miner_sim.benchmark import make_benchmark_env
from gold_miner_sim.env import FIRE, MAX_ANGLE
from gold_miner_sim.oracle import (
    OBJECT_SLOTS,
    first_active_slot,
    object_geometry_from_observation,
    oracle_action,
    target_angle_deg,
)

FULL_SCORE = 800.0
_BenchmarkEnv = gymnasium.Wrapper[NDArray[np.float32], int, NDArray[np.float32], int]


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Public metrics collected from one oracle episode."""

    score: float
    terminated: bool
    truncated: bool
    decision_count: int
    fires_used: int
    target_hit_count: int
    wrong_object_hit_count: int
    miss_count: int
    collected: tuple[bool, bool, bool]


def _active_flags(obs: NDArray[np.float32]) -> tuple[bool, bool, bool]:
    """Read the three public active flags from an observation."""
    flags = tuple(
        object_geometry_from_observation(obs, slot).active for slot in OBJECT_SLOTS
    )
    return flags[0], flags[1], flags[2]


def run_episode(
    env: _BenchmarkEnv, map_seed: int, trace: bool = False
) -> EpisodeResult:
    """Run one deterministic oracle episode using only observations.

    The policy and outcome diagnostics use observations before and after each
    public step; no environment internals are read.
    """
    obs, _info = env.reset(seed=map_seed)
    score = 0.0
    decision_count = 0
    fires_used = 0
    target_hit_count = 0
    wrong_object_hit_count = 0
    miss_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        before_obs = obs
        before_active = _active_flags(before_obs)
        target = first_active_slot(before_obs)
        action = oracle_action(before_obs)
        if action == FIRE:
            fires_used += 1
            if trace and target is not None:
                target_geometry = object_geometry_from_observation(
                    before_obs, OBJECT_SLOTS[target]
                )
                current_angle = float(before_obs[0]) * MAX_ANGLE
                print(
                    f"map_seed={map_seed} fire={fires_used} "
                    f"target={target_geometry.name.upper()} "
                    f"current_angle={current_angle:.1f} "
                    f"target_center_angle="
                    f"{target_angle_deg(target_geometry.x_px, target_geometry.y_px):.1f}"
                )

        obs, reward, terminated, truncated, _info = env.step(action)
        score += float(reward)
        decision_count += 1

        if action == FIRE:
            after_active = _active_flags(obs)
            changed_slots = [
                index
                for index, (was_active, is_active) in enumerate(
                    zip(before_active, after_active)
                )
                if was_active and not is_active
            ]
            if target is not None and target in changed_slots:
                target_hit_count += 1
            elif changed_slots:
                wrong_object_hit_count += 1
            else:
                miss_count += 1

    if trace:
        print(f"final_score={score:g}")

    final_active = _active_flags(obs)
    return EpisodeResult(
        score=score,
        terminated=bool(terminated),
        truncated=bool(truncated),
        decision_count=decision_count,
        fires_used=fires_used,
        target_hit_count=target_hit_count,
        wrong_object_hit_count=wrong_object_hit_count,
        miss_count=miss_count,
        collected=(
            not final_active[0],
            not final_active[1],
            not final_active[2],
        ),
    )


def summarize_results(
    results: list[EpisodeResult],
) -> dict[str, float | int]:
    """Return aggregate benchmark metrics for a non-empty result list."""
    if not results:
        raise ValueError("at least one episode is required")
    scores = np.asarray([result.score for result in results], dtype=np.float64)
    full_score_count = int(np.count_nonzero(scores == FULL_SCORE))
    return {
        "episodes": len(results),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "full_score_count": full_score_count,
        "full_score_rate": full_score_count / len(results),
        "timeout_count": sum(result.truncated for result in results),
        "mean_decisions_per_episode": float(
            np.mean([result.decision_count for result in results])
        ),
        "mean_fires_used": float(np.mean([result.fires_used for result in results])),
        "target_hit_count": sum(result.target_hit_count for result in results),
        "wrong_object_hit_count": sum(
            result.wrong_object_hit_count for result in results
        ),
        "miss_count": sum(result.miss_count for result in results),
        "gold_collected_rate": sum(result.collected[0] for result in results)
        / len(results),
        "diamond_collected_rate": sum(result.collected[1] for result in results)
        / len(results),
        "rock_collected_rate": sum(result.collected[2] for result in results)
        / len(results),
    }


def print_metrics(metrics: dict[str, float | int]) -> None:
    """Print aggregate metrics with stable machine-readable field names."""
    for key in (
        "episodes",
        "mean",
        "std",
        "min",
        "max",
        "full_score_count",
        "full_score_rate",
        "timeout_count",
        "mean_decisions_per_episode",
        "mean_fires_used",
        "target_hit_count",
        "wrong_object_hit_count",
        "miss_count",
        "gold_collected_rate",
        "diamond_collected_rate",
        "rock_collected_rate",
    ):
        value = metrics[key]
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


def main() -> None:
    """Parse CLI arguments and run the requested benchmark episodes."""
    parser = argparse.ArgumentParser(
        description="Evaluate the observation-only Geometry Oracle baseline."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="number of random-map episodes (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help=("starting map seed; episode i uses seed + i (default: 1000)"),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render the episodes in a human window",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="print every FIRE decision and target center angle",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    env = make_benchmark_env(
        observation_mode="full",
        render_mode="human" if args.render else None,
    )
    try:
        results = [
            run_episode(env, map_seed=args.seed + index, trace=args.trace)
            for index in range(args.episodes)
        ]
    finally:
        env.close()

    print(f"seed_range: {args.seed}-{args.seed + args.episodes - 1}")
    print_metrics(summarize_results(results))


if __name__ == "__main__":
    main()
