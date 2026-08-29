"""Headless FIRE-angle replay comparing the Full and Blind ablation models.

For selected benchmark map seeds this script runs one deterministic episode
per condition (``full`` / ``blind`` model, on the matching observation
chain) and records the hook angle in degrees of every FIRE decision,
captured from the observation BEFORE the FIRE step (``obs[0] * MAX_ANGLE``),
plus the final score. The FIRE-angle lists make the Issue #13 behavioral
question directly visible: does the blind agent (no object x/y inputs)
fire at systematically different angles than the full agent?

Usage:
    uv run --group train python scripts/replay_ablation.py --full-model models/ablation/full/seed_0.zip --blind-model models/ablation/blind/seed_0.zip
    uv run --group train python scripts/replay_ablation.py --full-model A.zip --blind-model B.zip --maps 1000 1007
"""

from __future__ import annotations

import argparse
from typing import cast

import gymnasium
import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import DQN

from gold_miner_sim.benchmark import make_benchmark_env
from gold_miner_sim.env import FIRE, MAX_ANGLE, GoldMinerEnv

_BenchmarkWrapper = gymnasium.Wrapper[
    NDArray[np.float32], int, NDArray[np.float32], int
]


def run_replay_episode(
    env: _BenchmarkWrapper, model: DQN, map_seed: int
) -> tuple[float, list[float]]:
    """Run one deterministic episode, recording every FIRE decision angle.

    The angle is read from the observation received BEFORE stepping with
    FIRE (the agent's own view, already masked in the blind condition).
    Returns ``(score, fire_angles_degrees)``.
    """
    inner_env = cast(GoldMinerEnv, env.unwrapped)
    obs, _info = env.reset(seed=map_seed)
    fire_angles: list[float] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=True)
        action_int = int(action)
        if action_int == FIRE:
            fire_angles.append(float(obs[0]) * MAX_ANGLE)
        obs, _reward, terminated, truncated, _info = env.step(action_int)
    return float(inner_env.score), fire_angles


def replay_condition(
    condition: str, model_path: str, map_seed: int
) -> tuple[float, list[float]]:
    """Run one deterministic replay episode for one condition and map."""
    env = make_benchmark_env(observation_mode=condition)
    model = DQN.load(model_path)
    try:
        return run_replay_episode(env, model, map_seed)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare FIRE angles and scores of the Full/Blind ablation "
            "models on individual benchmark maps (headless)."
        )
    )
    parser.add_argument(
        "--full-model",
        type=str,
        required=True,
        help="path to the model trained with --observation full",
    )
    parser.add_argument(
        "--blind-model",
        type=str,
        required=True,
        help="path to the model trained with --observation blind",
    )
    parser.add_argument(
        "--maps",
        type=int,
        nargs="+",
        default=[1000, 1007, 1042],
        help="map seeds to replay (default: 1000 1007 1042)",
    )
    args = parser.parse_args()

    model_paths = {"full": args.full_model, "blind": args.blind_model}
    for map_seed in args.maps:
        summary_parts: list[str] = []
        for condition in ("full", "blind"):
            score, fire_angles = replay_condition(
                condition, model_paths[condition], map_seed
            )
            angles_text = ", ".join(f"{angle:.1f}" for angle in fire_angles)
            summary_parts.append(
                f"{condition}: score={score:.1f} fire_angles=[{angles_text}]"
            )
        print(f"map {map_seed} | " + " | ".join(summary_parts))


if __name__ == "__main__":
    main()
