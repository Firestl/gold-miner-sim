"""Run the frozen Strong DQN recipe for several training seeds.

Training is intentionally sequential so each seed has an isolated model and
Monitor directory.  This runner evaluates the resulting best checkpoints on
the development validation maps.  Use ``eval_strong_dqn.py --split test``
separately once the recipe and all best checkpoints are frozen.

Usage:
    uv run --group train python scripts/run_strong_dqn.py
    uv run --group train python scripts/run_strong_dqn.py --seeds 0,1 --timesteps 50000
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from gold_miner_sim.strong_dqn import DEFAULT_STRONG_DQN_CONFIG


def parse_seeds(spec: str) -> list[int]:
    """Parse a comma-separated list of integer training seeds."""
    tokens = [token.strip() for token in spec.split(",")]
    if not tokens or not all(tokens):
        raise ValueError("--seeds must be a comma-separated list of integers")
    return [int(token) for token in tokens]


def train_one(
    seed: int,
    timesteps: int,
    model_root: str,
    run_root: str,
) -> None:
    """Train one seed with the independent Strong DQN entry point."""
    subprocess.run(
        [
            sys.executable,
            "scripts/train_strong_dqn.py",
            "--seed",
            str(seed),
            "--timesteps",
            str(timesteps),
            "--model-root",
            model_root,
            "--run-root",
            run_root,
        ],
        check=True,
    )


def evaluate_validation(
    seeds: str,
    episodes: int,
    seed_start: int,
    model_root: str,
    run_root: str,
    output: str | None,
) -> None:
    """Evaluate all trained best checkpoints on the validation split."""
    command = [
        sys.executable,
        "scripts/eval_strong_dqn.py",
        "--split",
        "validation",
        "--seeds",
        seeds,
        "--episodes",
        str(episodes),
        "--seed",
        str(seed_start),
        "--model-root",
        model_root,
        "--run-root",
        run_root,
    ]
    if output is not None:
        command.extend(["--output", output])
    subprocess.run(command, check=True)


def main() -> None:
    """Train all requested seeds and write validation aggregate results."""
    parser = argparse.ArgumentParser(
        description="Sequentially train and validate Strong DQN checkpoints."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="comma-separated training seeds (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_STRONG_DQN_CONFIG.max_timesteps,
        help="training transitions per seed",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_STRONG_DQN_CONFIG.validation_episodes,
        help="validation maps per seed (default: 100)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_STRONG_DQN_CONFIG.validation_seed,
        help="first validation map seed (default: 1000)",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default="models/strong_dqn",
        help="root directory for Strong DQN models",
    )
    parser.add_argument(
        "--run-root",
        type=str,
        default="runs/strong_dqn",
        help="root directory for Strong DQN runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="optional validation JSON path",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if args.timesteps <= 0:
        raise SystemExit("--timesteps must be positive")
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")

    for seed in seeds:
        print(f"=== Strong DQN seed={seed} ===", flush=True)
        train_one(seed, args.timesteps, args.model_root, args.run_root)

    evaluate_validation(
        args.seeds,
        args.episodes,
        args.eval_seed,
        args.model_root,
        args.run_root,
        args.output,
    )


if __name__ == "__main__":
    main()
