"""Random-policy baseline for the Gold Miner environment.

Samples a uniformly random action (WAIT/FIRE) at every decision step and
reports the score distribution over 100 headless episodes on random maps.
The ``SwingAdvanceDecisionWrapper`` still chooses only at real decision
points, and ``FireBudgetWrapper(max_fires=3)`` limits each episode to three
FIRE actions. Episode ``i`` is played on the map drawn with
``map_seed = seed + i``, so the default ``--episodes 100 --seed 1000``
covers map seeds 1000-1099 — the same map set ``eval_dqn.py`` uses. The
action space RNG is seeded once per run (not per episode), so different maps
see different action sequences.

Usage:
    uv run --group train python scripts/random_baseline.py
    uv run --group train python scripts/random_baseline.py --episodes 50 --seed 2000
"""

from __future__ import annotations

import argparse

import numpy as np

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.wrappers import FireBudgetWrapper, SwingAdvanceDecisionWrapper

FULL_SCORE = 800.0


def run_episode(env: FireBudgetWrapper, map_seed: int) -> float:
    """Run one headless episode with a uniformly random policy.

    Resets the environment with ``map_seed`` (which fixes the random map
    layout), then samples an action from the pre-seeded action space at
    every decision step until the three-FIRE budget or the episode ends.

    Returns the final score (accumulated decision-step reward).
    """
    env.reset(seed=map_seed)
    episode_score = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = int(env.action_space.sample())
        _obs, reward, terminated, truncated, _info = env.step(action)
        episode_score += float(reward)
    return episode_score


def summarize_scores(scores: list[float]) -> dict[str, float | int]:
    """Return the benchmark metrics for a non-empty score list."""
    if not scores:
        raise ValueError("at least one episode is required")
    values = np.asarray(scores, dtype=np.float64)
    full_score_count = int(np.count_nonzero(values == FULL_SCORE))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "full_score_count": full_score_count,
        "full_score_rate": full_score_count / len(scores),
    }


def print_metrics(metrics: dict[str, float | int]) -> None:
    """Print benchmark metrics using stable, machine-readable field names."""
    print(f"mean: {metrics['mean']:.2f}")
    print(f"std: {metrics['std']:.2f}")
    print(f"min: {metrics['min']:.2f}")
    print(f"max: {metrics['max']:.2f}")
    print(f"full_score_count: {metrics['full_score_count']}")
    print(f"full_score_rate: {metrics['full_score_rate']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score distribution of a random policy on random-map GoldMinerEnv."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="number of episodes to run (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1000,
        help=(
            "starting map seed for the evaluation maps: episode i uses "
            "seed + i (default: 1000)"
        ),
    )
    args = parser.parse_args()

    env = FireBudgetWrapper(
        SwingAdvanceDecisionWrapper(
            GoldMinerEnv(render_mode=None, map_mode="random")
        ),
        max_fires=3,
    )
    # Seed the action RNG once for the whole run; re-seeding per episode
    # would replay the same action sequence on every map.
    env.action_space.seed(args.seed)

    scores: list[float] = [
        run_episode(env, map_seed=args.seed + i) for i in range(args.episodes)
    ]
    env.close()

    print(f"episodes: {len(scores)}")
    print(f"seed_range: {args.seed}-{args.seed + args.episodes - 1}")
    print_metrics(summarize_scores(scores))


if __name__ == "__main__":
    main()
