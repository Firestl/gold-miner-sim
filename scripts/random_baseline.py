"""Random-policy baseline for the Gold Miner environment.

Samples a uniformly random action (WAIT/FIRE) at every decision step and
reports the score distribution over a number of headless episodes. Runs at
full speed: no rendering, no sleeping.

Usage:
    uv run python scripts/random_baseline.py
    uv run python scripts/random_baseline.py --episodes 50 --seed 42
"""

from __future__ import annotations

import argparse

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.wrappers import DecisionIntervalWrapper


def run_episode(env: DecisionIntervalWrapper, seed: int) -> float:
    """Run one headless episode with a uniformly random policy.

    Resets the environment with ``seed``, then samples an action from the
    action space at every decision step until the episode ends.

    Returns the final score (accumulated decision-step reward).
    """
    env.reset(seed=seed)
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
        description="Score distribution of a random policy on GoldMinerEnv."
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
        default=0,
        help="seed for the action space and every episode reset (default: 0)",
    )
    args = parser.parse_args()

    env: DecisionIntervalWrapper = DecisionIntervalWrapper(
        GoldMinerEnv(render_mode=None)
    )
    env.action_space.seed(args.seed)

    scores: list[float] = [run_episode(env, args.seed) for _ in range(args.episodes)]
    env.close()

    print(f"Episodes: {len(scores)}")
    print(f"Mean score: {sum(scores) / len(scores):.2f}")
    print(f"Min score: {min(scores):.2f}")
    print(f"Max score: {max(scores):.2f}")


if __name__ == "__main__":
    main()
