"""Evaluate Strong DQN checkpoints on validation or held-out maps.

The default ``validation`` split is maps 1000..1099.  The ``test`` split is
held out until the recipe is frozen and uses maps 2000..2099.

Usage:
    uv run --group train python scripts/eval_strong_dqn.py --seeds 0
    uv run --group train python scripts/eval_strong_dqn.py --split test
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
from stable_baselines3 import DQN

from gold_miner_sim.strong_dqn import DEFAULT_STRONG_DQN_CONFIG, evaluate_dqn


def parse_seeds(spec: str) -> list[int]:
    """Parse a comma-separated list of training seeds."""
    tokens = [token.strip() for token in spec.split(",")]
    if not tokens or not all(tokens):
        raise ValueError("--seeds must be a comma-separated list of integers")
    return [int(token) for token in tokens]


def model_path_for(seed: int, model_root: str) -> str:
    """Return the Strong DQN best-checkpoint path for ``seed``."""
    return os.path.join(model_root, f"seed_{seed}", "best_model.zip")


def evaluation_history_path_for(seed: int, run_root: str) -> str:
    """Return the checkpoint history path for ``seed``."""
    return os.path.join(run_root, f"seed_{seed}", "evaluations.json")


def monitor_csv_path_for(seed: int, run_root: str) -> str:
    """Return the SB3 Monitor CSV path for ``seed``."""
    return os.path.join(run_root, f"seed_{seed}", "monitor.monitor.csv")


def count_monitor_episodes(path: str) -> int | None:
    """Count Monitor rows, returning ``None`` when no log exists."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as file_handler:
        data_lines = [line for line in file_handler.readlines()[2:] if line.strip()]
    return len(data_lines)


def best_checkpoint_timestep(seed: int, run_root: str) -> int | None:
    """Read the selected checkpoint timestep when a history file exists."""
    path = evaluation_history_path_for(seed, run_root)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as file_handler:
        payload: dict[str, Any] = json.load(file_handler)
    value = payload.get("best_checkpoint_timestep")
    return int(value) if value is not None else None


def split_settings(split: str) -> tuple[int, int]:
    """Return ``(map_seed_start, default_episode_count)`` for a split."""
    if split == "validation":
        return (
            DEFAULT_STRONG_DQN_CONFIG.validation_seed,
            DEFAULT_STRONG_DQN_CONFIG.validation_episodes,
        )
    if split == "test":
        return (
            DEFAULT_STRONG_DQN_CONFIG.heldout_seed,
            DEFAULT_STRONG_DQN_CONFIG.heldout_episodes,
        )
    raise ValueError("split must be 'validation' or 'test'")


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate per-training-seed Strong DQN means."""
    if not runs:
        raise ValueError("at least one run is required")
    means = np.asarray([float(run["mean_score"]) for run in runs], dtype=np.float64)
    return {
        "mean_score_across_training_seeds": float(np.mean(means)),
        "std_score_across_training_seeds": float(np.std(means)),
        "seeds_above_historical_baseline": int(np.count_nonzero(means > 207.70)),
        "seeds_at_least_300": int(np.count_nonzero(means >= 300.0)),
        "seeds_at_least_400": int(np.count_nonzero(means >= 400.0)),
        "number_of_seeds": len(runs),
    }


def evaluate_checkpoints(
    seeds: list[int],
    split: str,
    episodes: int,
    seed_start: int,
    model_root: str,
    run_root: str,
) -> dict[str, Any]:
    """Evaluate each seed's best checkpoint and return a JSON-ready payload."""
    runs: list[dict[str, Any]] = []
    for training_seed in seeds:
        model_path = model_path_for(training_seed, model_root)
        model = DQN.load(model_path)
        metrics = evaluate_dqn(model, seed_start=seed_start, episodes=episodes)
        history_path = evaluation_history_path_for(training_seed, run_root)
        training_timesteps: int | None = None
        if os.path.isfile(history_path):
            with open(history_path, encoding="utf-8") as file_handler:
                history_payload: dict[str, Any] = json.load(file_handler)
            raw_training_timesteps = history_payload.get("training_timesteps")
            if raw_training_timesteps is not None:
                training_timesteps = int(raw_training_timesteps)
        monitor_episodes = count_monitor_episodes(
            monitor_csv_path_for(training_seed, run_root)
        )
        runs.append(
            {
                "training_seed": training_seed,
                "model_path": model_path,
                "training_timesteps": training_timesteps,
                "best_checkpoint_timestep": best_checkpoint_timestep(
                    training_seed, run_root
                ),
                "monitor_episode_count": monitor_episodes,
                "approx_decisions_per_episode": (
                    training_timesteps / monitor_episodes
                    if training_timesteps is not None and monitor_episodes
                    else None
                ),
                **metrics,
            }
        )

    return {
        "split": split,
        "observation": "full",
        "map_seed_range": f"{seed_start}-{seed_start + episodes - 1}",
        "episodes": episodes,
        "recipe": DEFAULT_STRONG_DQN_CONFIG.as_dict(),
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }


def main() -> None:
    """Evaluate selected Strong DQN checkpoints and write the split result."""
    parser = argparse.ArgumentParser(
        description="Evaluate Strong DQN best checkpoints on a fixed map split."
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="evaluation split (default: validation; test is held out)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="comma-separated training seeds (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="episodes per checkpoint (split default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="first map seed (split default: 1000 validation, 2000 test)",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="models/strong_dqn",
        help="root containing seed_<n>/best_model.zip",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs/strong_dqn",
        help="root containing per-seed evaluations.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON output path (default: runs/strong_dqn/results_<split>.json)",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    default_seed, default_episodes = split_settings(args.split)
    seed_start = default_seed if args.seed is None else args.seed
    episodes = default_episodes if args.episodes is None else args.episodes
    if episodes <= 0:
        raise SystemExit("--episodes must be positive")

    payload = evaluate_checkpoints(
        seeds=seeds,
        split=args.split,
        episodes=episodes,
        seed_start=seed_start,
        model_root=args.model_root,
        run_root=args.run_root,
    )
    output_path = args.output or os.path.join(
        args.run_root, f"results_{args.split}.json"
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_handler:
        json.dump(payload, file_handler, indent=2)
        file_handler.write("\n")

    print(f"split: {args.split}")
    print(f"map_seed_range: {payload['map_seed_range']}")
    print(f"results written to {output_path}")
    aggregate = payload["aggregate"]
    print(
        f"mean_score_across_training_seeds: "
        f"{aggregate['mean_score_across_training_seeds']:.2f}"
    )


if __name__ == "__main__":
    main()
