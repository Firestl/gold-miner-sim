"""Evaluate a trained DQN agent over multiple random maps of GoldMinerEnv.

Generalization evaluation: episode ``i`` is played on the map drawn with
``map_seed = seed + i``, so a given seed range always scores the same map
set (the default ``--episodes 100 --seed 1000`` matches the baseline's
map seeds 1000-1099). Uses the same wrapper stack as training
(DecisionIntervalWrapper) and the deterministic policy. Without
``--render`` it runs headless at full speed; with ``--render`` the env
auto-renders every physics tick and the renderer's internal
``Clock.tick(60)`` already paces playback to real time, so no extra sleep
is needed. Closing the window ends the current episode early.

Usage:
    uv run --group train python scripts/eval_dqn.py --episodes 100 --seed 1000
    uv run --group train python scripts/eval_dqn.py --seed 1007 --render
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import DQN

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.wrappers import DecisionIntervalWrapper


def run_episode(
    env: DecisionIntervalWrapper, model: DQN, map_seed: int, render: bool
) -> float:
    """Run one episode on the map drawn from ``map_seed``.

    Resets the environment with ``map_seed`` (which fixes the random map
    layout), then steps with ``model.predict(obs, deterministic=True)``
    until the episode is terminated or truncated; in render mode it also
    stops as soon as the window is closed. In human mode each ``step()``
    renders automatically.

    Returns the final score of the episode.
    """
    obs, _info = env.reset(seed=map_seed)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        if render and env.unwrapped.window_closed:  # always False in headless mode
            break
    score: float = env.unwrapped.score
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent over random GoldMinerEnv maps."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/dqn_gold_miner_random.zip",
        help="path to the saved model zip (default: models/dqn_gold_miner_random.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=(
            "number of maps/episodes to evaluate "
            "(default: 100 headless, 1 with --render)"
        ),
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
    parser.add_argument(
        "--render",
        action="store_true",
        help="show the pygame window (default: headless)",
    )
    args = parser.parse_args()

    episodes = (
        args.episodes if args.episodes is not None else (1 if args.render else 100)
    )

    env: DecisionIntervalWrapper = DecisionIntervalWrapper(
        GoldMinerEnv(
            render_mode="human" if args.render else None, map_mode="random"
        )
    )
    model = DQN.load(args.model)
    try:
        scores: list[float] = [
            run_episode(env, model, args.seed + i, args.render)
            for i in range(episodes)
        ]
    finally:
        env.close()

    print(f"Episodes: {len(scores)}")
    print(f"Seed range: {args.seed}-{args.seed + episodes - 1}")
    print(f"Mean score: {sum(scores) / len(scores):.2f}")
    print(f"Std score: {float(np.std(scores)):.2f}")
    print(f"Min score: {min(scores):.2f}")
    print(f"Max score: {max(scores):.2f}")


if __name__ == "__main__":
    main()
