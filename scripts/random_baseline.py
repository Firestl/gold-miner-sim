"""Random-policy baseline for the Gold Miner environment.

Samples a uniformly random action (WAIT/FIRE) at every decision step and
reports the score distribution over a number of headless episodes on
random maps. With ``SwingAdvanceDecisionWrapper`` the random policy only
chooses at real decision points -- while the hook is SWINGING, never at
the angle a FIRE just launched from (post-FIRE advance) -- so its numbers
are not directly comparable to the Milestone 4 ``SwingDecisionWrapper``
baseline and must be re-measured. Episode ``i`` is played on the map
drawn with ``map_seed = seed + i``, so the default
``--episodes 100 --seed 1000`` covers map seeds 1000-1099 — the same map
set ``eval_dqn.py`` uses by default. The action space RNG is seeded once
per run (not per episode), so different maps see different action
sequences.

Usage:
    uv run --group train python scripts/random_baseline.py
    uv run --group train python scripts/random_baseline.py --episodes 50 --seed 2000
"""

from __future__ import annotations

import argparse

import numpy as np

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.wrappers import SwingAdvanceDecisionWrapper


def run_episode(env: SwingAdvanceDecisionWrapper, map_seed: int) -> float:
    """Run one headless episode with a uniformly random policy.

    Resets the environment with ``map_seed`` (which fixes the random map
    layout), then samples an action from the pre-seeded action space at
    every decision step -- each time the hook is back SWINGING -- until
    the episode ends.

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

    env: SwingAdvanceDecisionWrapper = SwingAdvanceDecisionWrapper(
        GoldMinerEnv(render_mode=None, map_mode="random")
    )
    # Seed the action RNG once for the whole run; re-seeding per episode
    # would replay the same action sequence on every map.
    env.action_space.seed(args.seed)

    scores: list[float] = [
        run_episode(env, map_seed=args.seed + i) for i in range(args.episodes)
    ]
    env.close()

    print(f"Episodes: {len(scores)}")
    print(f"Seed range: {args.seed}-{args.seed + args.episodes - 1}")
    print(f"Mean score: {sum(scores) / len(scores):.2f}")
    print(f"Std score: {float(np.std(scores)):.2f}")
    print(f"Min score: {min(scores):.2f}")
    print(f"Max score: {max(scores):.2f}")


if __name__ == "__main__":
    main()
