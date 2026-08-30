"""Evaluate a trained DQN agent over multiple random maps of GoldMinerEnv.

Generalization evaluation: episode ``i`` is played on the map drawn with
``map_seed = seed + i``, so a given seed range always scores the same map
set (the default headless run matches the baseline's map seeds 1000-1099).
Uses the shared benchmark chain from
``gold_miner_sim.benchmark.make_benchmark_env`` (``SwingAdvanceDecisionWrapper``
followed by ``FireBudgetWrapper(max_fires=3)``), so a policy can FIRE at
most three times per episode. ``--observation`` selects the chain: a model
trained with ``--observation blind`` MUST be evaluated with
``--observation blind`` (its object x/y inputs are masked to 0), and a
model trained with ``--observation polar`` MUST be evaluated with
``--observation polar`` (its position inputs are Polar angle/distance).
Without ``--model`` the full condition falls back to the Milestone 6
artifact ``models/dqn_gold_miner_fire_budget.zip``; blind and polar modes
require an explicit ``--model`` (argparse exits with code 2 before any
episode runs).
The deterministic policy is queried only at real decision points. Without
``--render`` it runs headless at full speed; with ``--render`` the env
auto-renders every physics tick and the renderer's internal
``Clock.tick(60)`` already paces playback to real time, so no extra sleep
is needed. Closing the window ends the current episode early.

The same random-policy baseline is measured in this script on the same map
seeds, making ``delta`` directly comparable to the DQN mean. With
``--json-output PATH`` the run is additionally written as a JSON object
(observation, model, episodes, seed_range, dqn/random metrics with
``std_episode`` instead of the stdout ``std``, and delta).

Usage:
    uv run --group train python scripts/eval_dqn.py --episodes 100 --seed 1000
    uv run --group train python scripts/eval_dqn.py --episodes 1 --seed 1007 --render
    uv run --group train python scripts/eval_dqn.py --model models/ablation/blind/seed_0.zip --observation blind --json-output runs/ablation/blind/seed_0/eval.json
    uv run --group train python scripts/eval_dqn.py --model models/representation/polar/seed_0.zip --observation polar --json-output runs/representation/polar/seed_0/eval.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import cast

import gymnasium
import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import DQN

from gold_miner_sim.benchmark import make_benchmark_env
from gold_miner_sim.env import GoldMinerEnv

FULL_SCORE = 800.0


def run_episode(
    env: gymnasium.Wrapper[NDArray[np.float32], int, NDArray[np.float32], int],
    model: DQN,
    map_seed: int,
    render: bool,
) -> float:
    """Run one deterministic DQN episode on the map for ``map_seed``.

    In render mode the episode also stops as soon as the window is closed.
    In human mode each ``step()`` renders automatically.
    """
    inner_env = cast(GoldMinerEnv, env.unwrapped)
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


def run_random_episode(
    env: gymnasium.Wrapper[NDArray[np.float32], int, NDArray[np.float32], int],
    map_seed: int,
) -> float:
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


def to_json_metrics(
    metrics: dict[str, float | int],
) -> dict[str, float | int]:
    """Rename the stdout ``std`` key to ``std_episode`` for the JSON schema."""
    return {
        "mean": metrics["mean"],
        "std_episode": metrics["std"],
        "min": metrics["min"],
        "max": metrics["max"],
        "full_score_count": metrics["full_score_count"],
        "full_score_rate": metrics["full_score_rate"],
    }


def print_metrics(metrics: dict[str, float | int]) -> None:
    """Print DQN benchmark metrics using stable field names."""
    print(f"mean: {metrics['mean']:.2f}")
    print(f"std: {metrics['std']:.2f}")
    print(f"min: {metrics['min']:.2f}")
    print(f"max: {metrics['max']:.2f}")
    print(f"full_score_count: {metrics['full_score_count']}")
    print(f"full_score_rate: {metrics['full_score_rate']:.2f}")


def write_json_output(
    path: str,
    observation: str,
    model: str,
    episodes: int,
    seed: int,
    dqn_metrics: dict[str, float | int],
    random_metrics: dict[str, float | int],
    delta: float,
) -> None:
    """Write the evaluation summary JSON payload to ``path``."""
    payload = {
        "observation": observation,
        "model": model,
        "episodes": episodes,
        "seed_range": f"{seed}-{seed + episodes - 1}",
        "dqn": to_json_metrics(dqn_metrics),
        "random": to_json_metrics(random_metrics),
        "delta": delta,
    }
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file_handler:
        json.dump(payload, file_handler, indent=2)
        file_handler.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent over random GoldMinerEnv maps."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "path to the saved model zip (default: "
            "models/dqn_gold_miner_fire_budget.zip with --observation full; "
            "required with --observation blind or polar)"
        ),
    )
    parser.add_argument(
        "--observation",
        type=str,
        choices=["full", "blind", "polar"],
        default="full",
        help=(
            "evaluation env chain; a blind-trained model MUST be evaluated "
            "with 'blind' and a polar-trained model with 'polar' "
            "(default: full)"
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
        "--json-output",
        type=str,
        default=None,
        help="optional path to write the evaluation summary as JSON",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="show the pygame window (default: headless)",
    )
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        if args.observation == "blind":
            parser.error("--model is required when --observation blind")
        if args.observation == "polar":
            parser.error("--model is required when --observation polar")
        # Backward-compatible Milestone 6 default for the full condition.
        model_path = "models/dqn_gold_miner_fire_budget.zip"

    episodes = (
        args.episodes if args.episodes is not None else (1 if args.render else 100)
    )

    random_env = make_benchmark_env(observation_mode=args.observation)
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
    env = make_benchmark_env(
        observation_mode=args.observation,
        render_mode=inner_env_mode,
    )
    model = DQN.load(model_path)
    try:
        scores: list[float] = [
            run_episode(env, model, args.seed + i, args.render)
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

    if args.json_output:
        write_json_output(
            args.json_output,
            observation=args.observation,
            model=model_path,
            episodes=episodes,
            seed=args.seed,
            dqn_metrics=dqn_metrics,
            random_metrics=random_metrics,
            delta=delta,
        )


if __name__ == "__main__":
    main()
