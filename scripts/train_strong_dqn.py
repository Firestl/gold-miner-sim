"""Train the Milestone 10 Strong DQN v2 recipe.

The script keeps the historical ``train_dqn.py`` unchanged.  It trains only
the Full Cartesian benchmark, evaluates deterministic checkpoints on maps
1000..1019 every 25,000 transitions, and stores the best and final models
separately.

Usage:
    uv run --group train python scripts/train_strong_dqn.py
    uv run --group train python scripts/train_strong_dqn.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import gymnasium
from numpy.typing import NDArray
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from gold_miner_sim.benchmark import make_benchmark_env
from gold_miner_sim.strong_dqn import (
    DEFAULT_STRONG_DQN_CONFIG,
    StrongDQNConfig,
    evaluate_dqn,
)

DEFAULT_TIMESTEPS = DEFAULT_STRONG_DQN_CONFIG.max_timesteps
DEFAULT_EVAL_INTERVAL = DEFAULT_STRONG_DQN_CONFIG.evaluation_interval
DEFAULT_EVAL_SEED = DEFAULT_STRONG_DQN_CONFIG.selection_seed
DEFAULT_EVAL_EPISODES = DEFAULT_STRONG_DQN_CONFIG.selection_episodes


def seed_paths(
    seed: int,
    model_root: str = "models/strong_dqn",
    run_root: str = "runs/strong_dqn",
) -> dict[str, str]:
    """Return the model and run paths for one training seed."""
    model_dir = os.path.join(model_root, f"seed_{seed}")
    run_dir = os.path.join(run_root, f"seed_{seed}")
    return {
        "model_dir": model_dir,
        "run_dir": run_dir,
        "best_model": os.path.join(model_dir, "best_model.zip"),
        "final_model": os.path.join(model_dir, "final_model.zip"),
        "monitor": os.path.join(run_dir, "monitor"),
        "evaluations": os.path.join(run_dir, "evaluations.json"),
    }


def build_model(
    env: gymnasium.Env[NDArray[Any], int],
    seed: int,
    config: StrongDQNConfig = DEFAULT_STRONG_DQN_CONFIG,
) -> DQN:
    """Build a DQN using the fixed Strong DQN v2 recipe."""
    return DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        n_steps=config.n_steps,
        policy_kwargs={"net_arch": list(config.net_arch)},
        seed=seed,
        verbose=1,
    )


class StrongDQNEvaluationCallback(BaseCallback):
    """Periodically evaluate and save the best Strong DQN checkpoint."""

    def __init__(
        self,
        *,
        eval_seed: int,
        eval_episodes: int,
        eval_interval: int,
        best_model_path: str,
        history_path: str,
        config: StrongDQNConfig = DEFAULT_STRONG_DQN_CONFIG,
        training_timesteps: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if eval_episodes <= 0:
            raise ValueError("eval_episodes must be positive")
        if eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        self.eval_seed = eval_seed
        self.eval_episodes = eval_episodes
        self.eval_interval = eval_interval
        self.best_model_path = best_model_path
        self.history_path = history_path
        self.config = config
        self.training_timesteps = training_timesteps
        self.history: list[dict[str, Any]] = []
        self.best_mean_score: float | None = None
        self.best_checkpoint_timestep: int | None = None
        self._next_eval_timestep = eval_interval

    def _on_step(self) -> bool:
        """Evaluate whenever the next configured transition boundary is met."""
        if self.num_timesteps >= self._next_eval_timestep:
            self._evaluate()
            while self._next_eval_timestep <= self.num_timesteps:
                self._next_eval_timestep += self.eval_interval
        return True

    def _on_training_end(self) -> None:
        """Evaluate a non-boundary final timestep before training exits."""
        if not self.history or self.history[-1]["step"] != self.num_timesteps:
            self._evaluate()

    def _evaluate(self) -> None:
        """Run one validation pass, update the best model, and persist history."""
        metrics = evaluate_dqn(
            self.model,
            seed_start=self.eval_seed,
            episodes=self.eval_episodes,
        )
        mean_score = float(metrics["mean_score"])
        is_best = self.best_mean_score is None or mean_score > self.best_mean_score
        if is_best:
            self.best_mean_score = mean_score
            self.best_checkpoint_timestep = int(self.num_timesteps)
            output_dir = os.path.dirname(self.best_model_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            self.model.save(self.best_model_path)

        self.history.append(
            {
                "step": int(self.num_timesteps),
                **metrics,
                "is_best": is_best,
            }
        )
        self._write_history()
        if self.verbose:
            print(
                f"validation step={self.num_timesteps} "
                f"mean_score={mean_score:.2f} "
                f"best={self.best_checkpoint_timestep}"
            )

    def _write_history(self) -> None:
        """Write the recipe and all completed validation records to JSON."""
        output_dir = os.path.dirname(self.history_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload = {
            "observation": "full",
            "evaluation_seed_start": self.eval_seed,
            "evaluation_episodes": self.eval_episodes,
            "evaluation_seed_range": (
                f"{self.eval_seed}-{self.eval_seed + self.eval_episodes - 1}"
            ),
            "evaluation_interval": self.eval_interval,
            "training_timesteps": self.training_timesteps,
            "best_checkpoint_timestep": self.best_checkpoint_timestep,
            "best_mean_score": self.best_mean_score,
            "recipe": self.config.as_dict(),
            "evaluations": self.history,
        }
        with open(self.history_path, "w", encoding="utf-8") as file_handler:
            json.dump(payload, file_handler, indent=2)
            file_handler.write("\n")


def parse_args() -> argparse.Namespace:
    """Parse Strong DQN training options."""
    parser = argparse.ArgumentParser(
        description="Train Strong DQN v2 on the Full Cartesian benchmark."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"maximum RL transitions (default: {DEFAULT_TIMESTEPS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="training seed for SB3 and the environment (default: 0)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=DEFAULT_EVAL_INTERVAL,
        help=(
            "transitions between checkpoint evaluations "
            f"(default: {DEFAULT_EVAL_INTERVAL})"
        ),
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=f"first map seed for checkpoint selection (default: {DEFAULT_EVAL_SEED})",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=(
            f"number of checkpoint-selection maps (default: {DEFAULT_EVAL_EPISODES})"
        ),
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="models/strong_dqn",
        help="root directory for best/final models",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs/strong_dqn",
        help="root directory for Monitor and evaluation history",
    )
    return parser.parse_args()


def main() -> None:
    """Train one seed and write its best/final artifacts."""
    args = parse_args()
    if args.timesteps <= 0:
        raise SystemExit("--timesteps must be positive")
    paths = seed_paths(args.seed, args.model_root, args.run_root)
    os.makedirs(paths["model_dir"], exist_ok=True)
    os.makedirs(paths["run_dir"], exist_ok=True)

    env = Monitor(
        make_benchmark_env(observation_mode="full"),
        filename=paths["monitor"],
    )
    callback = StrongDQNEvaluationCallback(
        eval_seed=args.eval_seed,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        best_model_path=paths["best_model"],
        history_path=paths["evaluations"],
        training_timesteps=args.timesteps,
    )
    model = build_model(env, seed=args.seed)
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(paths["final_model"])
    finally:
        env.close()


if __name__ == "__main__":
    main()
