"""Train a DQN agent on the Gold Miner environment (headless).

Trains on ``map_mode="random"`` so the policy generalizes across map
layouts: every reset draws the three spawn points from RANDOM_SPAWN_POINTS
through the Gymnasium-seeded RNG. The environment chain comes from
``gold_miner_sim.benchmark.make_benchmark_env``:
``GoldMinerEnv -> SwingAdvanceDecisionWrapper -> FireBudgetWrapper ->
Monitor``: like Milestone 4's SwingDecisionWrapper the agent decides only
while the hook is SWINGING (WAIT advances up to 10 physics ticks, FIRE
auto-plays the whole extend/retract round trip), but after a completed
FIRE cycle the wrapper swings on for another 10 WAIT ticks before
returning, so the next decision is never taken at the original firing
angle (anti angle-pinning). The outer wrapper limits each episode to three
FIRE actions and appends the normalized FIRE budget to the observation.

Observation modes (Issue #13 ablation): ``--observation full`` (default)
uses the complete 27-dim observation; ``--observation blind`` zeroes the
GOLD/DIAMOND/ROCK x/y slots (indices 8, 9, 14, 15, 20, 21) so the agent
cannot localize the objects.

Trains SB3's DQN with default hyperparameters except the few set below,
and saves the policy. The Monitor log path is derived from ``--output``:
``models/ablation/blind/seed_0.zip`` logs to
``runs/ablation/blind/seed_0/monitor``, so different seeds/modes never
overwrite each other.

Usage:
    uv run --group train python scripts/train_dqn.py
    uv run --group train python scripts/train_dqn.py --timesteps 200000 --seed 1
    uv run --group train python scripts/train_dqn.py --observation blind --timesteps 200000 --seed 0 --output models/ablation/blind/seed_0.zip
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from gold_miner_sim.benchmark import make_benchmark_env


def monitor_log_filename(output_path: str) -> str:
    """Derive the Monitor log path from the model output path.

    Strips the extension from ``output_path``; a ``models/`` prefix is
    mapped to ``runs/``, any other path contributes only its basename, and
    ``/monitor`` is appended (SB3 writes ``<path>.monitor.csv``).
    """
    stem = os.path.splitext(output_path)[0]
    parts = os.path.normpath(stem).split(os.sep)
    if parts and parts[0] == "models":
        log_dir = os.path.join("runs", *parts[1:])
    else:
        log_dir = os.path.join("runs", parts[-1])
    return os.path.join(log_dir, "monitor")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on random-map GoldMinerEnv (headless)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="total training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "training experiment seed: seeds both SB3 and the "
            "environment / map RNG for reproducibility (default: 0)"
        ),
    )
    parser.add_argument(
        "--observation",
        type=str,
        choices=["full", "blind"],
        default="full",
        help=(
            "observation mode: 'full' uses the complete 27-dim "
            "observation, 'blind' masks the object x/y slots "
            "(indices 8, 9, 14, 15, 20, 21) to 0 (default: full)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/dqn_gold_miner_fire_budget.zip",
        help=(
            "path to save the trained model zip "
            "(default: models/dqn_gold_miner_fire_budget.zip)"
        ),
    )
    args = parser.parse_args()

    monitor_log_path = monitor_log_filename(args.output)
    os.makedirs(os.path.dirname(monitor_log_path), exist_ok=True)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    env = Monitor(
        make_benchmark_env(args.observation),
        filename=monitor_log_path,
    )
    model = DQN(
        policy="MlpPolicy",
        env=env,
        buffer_size=100_000,
        learning_starts=1_000,
        exploration_fraction=0.3,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.output)
    env.close()


if __name__ == "__main__":
    main()
