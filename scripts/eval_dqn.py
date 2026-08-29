"""Evaluate a trained DQN agent over multiple random maps of GoldMinerEnv.

Generalization evaluation: episode ``i`` is played on the map drawn with
``map_seed = seed + i``, so a given seed range always scores the same map
set (the default headless run matches the baseline's map seeds 1000-1099).
Uses the same wrapper stack as training (``SwingAdvanceDecisionWrapper``
followed by ``FireBudgetWrapper(max_fires=3)``), so a policy can FIRE at
most three times per episode. The deterministic policy is queried only at
real decision points. Without ``--render`` it runs headless at full speed;
with ``--render`` the env auto-renders every physics tick and the
renderer's internal ``Clock.tick(60)`` already paces playback to real time,
so no extra sleep is needed. Closing the window ends the current episode
early.

The same random-policy baseline is measured in this script on the same map
seeds, making ``delta`` directly comparable to the DQN mean.

Usage:
    uv run --group train python scripts/eval_dqn.py --episodes 100 --seed 1000
    uv run --group train python scripts/eval_dqn.py --episodes 1 --seed 1007 --render
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import DQN

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.wrappers import FireBudgetWrapper, SwingAdvanceDecisionWrapper

FULL_SCORE = 800.0


def run_episode(
    env: FireBudgetWrapper,
    inner_env: GoldMinerEnv,
    model: DQN,
    map_seed: int,
    render: bool,
) -> float:
    """Run one deterministic DQN episode on the map for ``map_seed``.

    In render mode the episode also stops as soon as the window is closed.
    In human mode each ``step()`` renders automatically.
    """
    obs, _info = env.reset(seed=map_seed)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        if render and inner_env.window_closed:  # always False in headless mode
            break
    score: float = inner_env.score
    return score


def run_random_episode(env: FireBudgetWrapper, map_seed: int) -> float:
    """Run one uniformly random episode on the map for ``map_seed``."""
    env.reset(seed=map_seed)
    episode_score = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        _obs, reward, terminated, truncated, _info = env.step(
            int(env.action_space.sample())
        )
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
    """Print DQN benchmark metrics using stable field names."""
    print(f"mean: {metrics['mean']:.2f}")
    print(f"std: {metrics['std']:.2f}")
    print(f"min: {metrics['min']:.2f}")
    print(f"max: {metrics['max']:.2f}")
    print(f"full_score_count: {metrics['full_score_count']}")
    print(f"full_score_rate: {metrics['full_score_rate']:.2f}")


def make_env(render_mode: str | None = None) -> tuple[FireBudgetWrapper, GoldMinerEnv]:
    """Create the M6 evaluation chain and return it with its base env."""
    inner_env = GoldMinerEnv(render_mode=render_mode, map_mode="random")
    env = FireBudgetWrapper(
        SwingAdvanceDecisionWrapper(inner_env),
        max_fires=3,
    )
    return env, inner_env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent over random GoldMinerEnv maps."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/dqn_gold_miner_fire_budget.zip",
        help=(
            "path to the saved model zip "
            "(default: models/dqn_gold_miner_fire_budget.zip)"
        ),
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

    random_env, _random_inner_env = make_env()
    # Keep action sequences reproducible while using a fresh map seed for
    # every episode. This is the same map set used by the DQN run below.
    random_env.action_space.seed(args.seed)
    try:
        random_scores = [
            run_random_episode(random_env, args.seed + i)
            for i in range(episodes)
        ]
    finally:
        random_env.close()

    inner_env_mode = "human" if args.render else None
    env, inner_env = make_env(render_mode=inner_env_mode)
    model = DQN.load(args.model)
    try:
        scores: list[float] = [
            run_episode(env, inner_env, model, args.seed + i, args.render)
            for i in range(episodes)
        ]
    finally:
        env.close()

    random_metrics = summarize_scores(random_scores)
    dqn_metrics = summarize_scores(scores)
    delta = float(dqn_metrics["mean"]) - float(random_metrics["mean"])

    print(f"episodes: {len(scores)}")
    print(f"seed_range: {args.seed}-{args.seed + episodes - 1}")
    print_metrics(dqn_metrics)
    print(f"mean_random: {random_metrics['mean']:.2f}")
    print(f"mean_dqn: {dqn_metrics['mean']:.2f}")
    print(f"delta: {delta:.2f}")


if __name__ == "__main__":
    main()
