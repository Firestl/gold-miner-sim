"""Train a DQN agent on the Gold Miner environment (headless).

Builds the fixed environment chain
``GoldMinerEnv -> DecisionIntervalWrapper -> Monitor``, trains SB3's DQN
with default hyperparameters except the few set below, and saves the
policy. Monitor logs go to ``runs/dqn/``.

Usage:
    uv run python scripts/train_dqn.py
    uv run python scripts/train_dqn.py --timesteps 50000 --seed 1
"""

from __future__ import annotations

import argparse
import os

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.wrappers import DecisionIntervalWrapper
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

MONITOR_LOG_DIR = "runs/dqn"
MONITOR_LOG_FILENAME = os.path.join(MONITOR_LOG_DIR, "monitor")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent on GoldMinerEnv (headless)."
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
        help="seed for the DQN model (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/dqn_gold_miner.zip",
        help="path to save the trained model zip (default: models/dqn_gold_miner.zip)",
    )
    args = parser.parse_args()

    os.makedirs(MONITOR_LOG_DIR, exist_ok=True)
    env = Monitor(
        DecisionIntervalWrapper(GoldMinerEnv(render_mode=None)),
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
