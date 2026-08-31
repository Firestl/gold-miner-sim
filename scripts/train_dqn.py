"""Train a DQN agent on the Gold Miner environment (headless).

Trains on ``map_mode="random"`` so the policy generalizes across map
layouts: every reset draws the three spawn points from RANDOM_SPAWN_POINTS
through the Gymnasium-seeded RNG. The environment chain comes from
``gold_miner_sim.benchmark.make_benchmark_env``:
``GoldMinerEnv -> SwingAdvanceDecisionWrapper -> FireBudgetWrapper ->
Monitor``: ``SwingAdvanceDecisionWrapper`` lets the agent decide only
while the hook is SWINGING (WAIT advances up to 10 physics ticks, FIRE
auto-plays the whole extend/retract round trip), and after a completed
FIRE cycle the wrapper swings on for another 10 WAIT ticks before
returning, so the next decision is never taken at the original firing
angle (anti angle-pinning). The outer wrapper limits each episode to three
FIRE actions and appends the normalized FIRE budget to the observation.

Observation modes (Issue #13 / #17 ablations): ``--observation full``
(default) uses the complete 27-dim observation; ``--observation blind``
zeroes the GOLD/DIAMOND/ROCK x/y slots (indices 8, 9, 14, 15, 20, 21) so
the agent cannot localize the objects; ``--observation polar`` rewrites
those slots from Cartesian (x, y) into Polar (target_angle, distance)
relative to the hook anchor while keeping every other slot unchanged.

Trains SB3's DQN with default hyperparameters except the few set below,
and saves the policy. Without ``--output`` the model is saved to the
mode/seed-scoped default: ``models/representation/polar/seed_<seed>.zip``
for the polar condition (Issue #17 representation experiment) and
``models/ablation/<observation>/seed_<seed>.zip`` otherwise, so different
observation modes and training seeds never overwrite each other and
Milestone 6 artifacts stay untouched; reproducing the M6 artifact
(``models/dqn_gold_miner_fire_budget.zip``) requires an explicit
``--output``. The Monitor log path is derived from the output path:
``models/ablation/blind/seed_0.zip`` logs to
``runs/ablation/blind/seed_0/monitor``.

Usage:
    uv run --group train python scripts/train_dqn.py
    uv run --group train python scripts/train_dqn.py --timesteps 200000 --seed 1
    uv run --group train python scripts/train_dqn.py --observation blind --timesteps 200000 --seed 0 --output models/ablation/blind/seed_0.zip
    uv run --group train python scripts/train_dqn.py --observation polar --timesteps 200000 --seed 0 --output models/representation/polar/seed_0.zip
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from gold_miner_sim.benchmark import make_benchmark_env


def default_output_path(observation: str, seed: int) -> str:
    """Default model path scoped by observation mode and training seed.

    The Issue #17 polar representation experiment keeps its own artifact
    directory, separate from the Milestone 7 ablation layout.
    """
    if observation == "polar":
        return f"models/representation/polar/seed_{seed}.zip"
    return f"models/ablation/{observation}/seed_{seed}.zip"


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
        choices=["full", "blind", "polar"],
        default="full",
        help=(
            "observation mode: 'full' uses the complete 27-dim "
            "observation, 'blind' masks the object x/y slots "
            "(indices 8, 9, 14, 15, 20, 21) to 0, 'polar' rewrites "
            "those slots into (target_angle, distance) relative to the "
            "hook anchor (default: full)"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "path to save the trained model zip (default: "
            "models/representation/polar/seed_<seed>.zip for polar, "
            "models/ablation/<observation>/seed_<seed>.zip otherwise); an "
            "explicit path (e.g. Milestone 6's "
            "models/dqn_gold_miner_fire_budget.zip) overrides it"
        ),
    )
    args = parser.parse_args()

    output_path = (
        args.output
        if args.output is not None
        else default_output_path(args.observation, args.seed)
    )

    monitor_log_path = monitor_log_filename(output_path)
    os.makedirs(os.path.dirname(monitor_log_path), exist_ok=True)
    output_dir = os.path.dirname(output_path)
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
    model.save(output_path)
    env.close()


if __name__ == "__main__":
    main()
