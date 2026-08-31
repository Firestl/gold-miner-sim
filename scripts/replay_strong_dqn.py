"""Replay a Strong DQN checkpoint on selected maps, with optional rendering.

Runs one deterministic episode per map on the unchanged Full Cartesian
benchmark and reports the Milestone 10 replay-inspection statistics: fires
per episode, the number of WAIT decisions before each FIRE, the hook angle
of every FIRE, and the final score. ``--render`` watches the episodes in a
human window instead of running headless at full speed.

Usage:
    uv run --group train python scripts/replay_strong_dqn.py
    uv run --group train python scripts/replay_strong_dqn.py --maps 1000 1007 1042
    uv run --group train python scripts/replay_strong_dqn.py --render --maps 1000
"""

from __future__ import annotations

import argparse
from typing import cast

from stable_baselines3 import DQN

from gold_miner_sim.benchmark import make_benchmark_env
from gold_miner_sim.env import FIRE, MAX_ANGLE, GoldMinerEnv


def replay_one_episode(
    model: DQN, map_seed: int, render: bool
) -> dict[str, float | list[float]]:
    """Run one deterministic episode and return its behavior statistics."""
    env = make_benchmark_env(
        observation_mode="full", render_mode="human" if render else None
    )
    try:
        obs, _info = env.reset(seed=map_seed)
        waits_before_fire: list[int] = []
        fire_angles: list[float] = []
        decisions = 0
        current_waits = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            action_int = int(action)
            if action_int == FIRE:
                waits_before_fire.append(current_waits)
                fire_angles.append(float(obs[0]) * MAX_ANGLE)
                current_waits = 0
            else:
                current_waits += 1
            obs, _reward, terminated, truncated, _info = env.step(action_int)
            decisions += 1
        score = float(cast(GoldMinerEnv, env.unwrapped).score)
    finally:
        env.close()
    return {
        "score": score,
        "fires": len(waits_before_fire),
        "waits_before_fire": [float(count) for count in waits_before_fire],
        "fire_angles": fire_angles,
        "decisions": decisions,
    }


def parse_args() -> argparse.Namespace:
    """Parse Strong DQN replay options."""
    parser = argparse.ArgumentParser(
        description=(
            "Replay a Strong DQN checkpoint on selected benchmark maps and "
            "report its per-episode firing behavior."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/strong_dqn/seed_0/best_model.zip",
        help="path to a Strong DQN checkpoint",
    )
    parser.add_argument(
        "--maps",
        type=int,
        nargs="+",
        default=[1000, 1007, 1042],
        help="map seeds to replay (default: 1000 1007 1042)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="render the episodes in a human window",
    )
    return parser.parse_args()


def main() -> None:
    """Replay the checkpoint on every requested map and print behavior."""
    args = parse_args()
    model = DQN.load(args.model)
    for map_seed in args.maps:
        stats = replay_one_episode(model, map_seed, render=args.render)
        waits = ", ".join(f"{count:.0f}" for count in stats["waits_before_fire"])
        angles = ", ".join(f"{angle:.1f}" for angle in stats["fire_angles"])
        print(
            f"map {map_seed} | score={stats['score']:.1f} "
            f"| fires={stats['fires']} "
            f"| waits_before_fire=[{waits}] "
            f"| fire_angles=[{angles}] "
            f"| decisions={stats['decisions']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
