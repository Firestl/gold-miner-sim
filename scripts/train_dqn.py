"""Train a DQN agent on the Gold Miner environment (headless).

Trains on ``map_mode="random"`` so the policy generalizes across map
layouts: every reset draws the three spawn points from RANDOM_SPAWN_POINTS
through the Gymnasium-seeded RNG. Builds the environment chain
``GoldMinerEnv -> SwingAdvanceDecisionWrapper -> Monitor``: like
Milestone 4's SwingDecisionWrapper the agent decides only while the hook
is SWINGING (WAIT advances up to 10 physics ticks, FIRE auto-plays the
whole extend/retract round trip), but after a completed FIRE cycle the
wrapper swings on for another 10 WAIT ticks before returning, so the next
decision is never taken at the original firing angle (anti angle-pinning).
Trains SB3's DQN with default hyperparameters except the few set below,
and saves the policy. Monitor logs go to ``runs/dqn_advance/``.

Usage:
    uv run --group train python scripts/train_dqn.py
    uv run --group train python scripts/train_dqn.py --timesteps 50000 --seed 1
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.wrappers import SwingAdvanceDecisionWrapper

MONITOR_LOG_DIR = "runs/dqn_advance"
MONITOR_LOG_FILENAME = os.path.join(MONITOR_LOG_DIR, "monitor")


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
        "--output",
        type=str,
        default="models/dqn_gold_miner_advance.zip",
        help="path to save the trained model zip (default: models/dqn_gold_miner_advance.zip)",
    )
    args = parser.parse_args()

    os.makedirs(MONITOR_LOG_DIR, exist_ok=True)
    env = Monitor(
        SwingAdvanceDecisionWrapper(
            GoldMinerEnv(render_mode=None, map_mode="random")
        ),
        filename=MONITOR_LOG_FILENAME,
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
