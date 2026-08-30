"""Run the Issue #13 paired Full/Blind observation ablation sequentially.

For every training seed this script trains one DQN agent per observation
mode (``full`` before ``blind``), evaluates it on the benchmark maps, and
collects the results into ``runs/ablation/results.json``. Runs are strictly
sequential (Issue #13 forbids parallel training frameworks): each child
process inherits stdout and writes its own model under
``models/ablation/{mode}/seed_{seed}.zip``, its Monitor CSV under
``runs/ablation/{mode}/seed_{seed}/``, and an ``eval.json`` summary read
back by this script.

The paired statistic is ``paired_delta = full_mean - blind_mean`` per
training seed; across seeds the script reports population mean/std for
both conditions and for the paired deltas.

Usage:
    uv run --group train python scripts/run_ablation.py
    uv run --group train python scripts/run_ablation.py --seeds 0,1 --timesteps 50000 --episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np

OBSERVATION_MODES = ("full", "blind")
RESULTS_JSON_PATH = "runs/ablation/results.json"


def parse_seeds(spec: str) -> list[int]:
    """Parse a comma-separated list of integer training seeds."""
    tokens = [token.strip() for token in spec.split(",")]
    if not all(tokens):
        raise ValueError("--seeds must be a comma-separated list of integers")
    return [int(token) for token in tokens]


def model_path_for(mode: str, seed: int) -> str:
    """Return the model zip path for one (mode, seed) run."""
    return f"models/ablation/{mode}/seed_{seed}.zip"


def monitor_csv_path_for(mode: str, seed: int) -> str:
    """Return the SB3 Monitor CSV path for one (mode, seed) run."""
    return f"runs/ablation/{mode}/seed_{seed}/monitor.monitor.csv"


def eval_json_path_for(mode: str, seed: int) -> str:
    """Return the eval JSON output path for one (mode, seed) run."""
    return f"runs/ablation/{mode}/seed_{seed}/eval.json"


def count_monitor_episodes(path: str) -> int:
    """Count episodes in an SB3 Monitor CSV (exactly two header lines)."""
    with open(path, encoding="utf-8") as file_handler:
        lines = file_handler.readlines()
    data_lines = [line for line in lines[2:] if line.strip()]
    if not data_lines:
        raise ValueError(f"Monitor CSV has no episode rows: {path}")
    return len(data_lines)


def train_one(mode: str, seed: int, timesteps: int) -> None:
    """Train one agent via scripts/train_dqn.py (inherits stdout)."""
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn.py",
            "--observation",
            mode,
            "--timesteps",
            str(timesteps),
            "--seed",
            str(seed),
            "--output",
            model_path_for(mode, seed),
        ],
        check=True,
    )


def evaluate_one(mode: str, seed: int, episodes: int, eval_seed: int) -> None:
    """Evaluate one agent via scripts/eval_dqn.py (inherits stdout)."""
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_dqn.py",
            "--model",
            model_path_for(mode, seed),
            "--observation",
            mode,
            "--episodes",
            str(episodes),
            "--seed",
            str(eval_seed),
            "--json-output",
            eval_json_path_for(mode, seed),
        ],
        check=True,
    )


def run_one(
    mode: str, seed: int, timesteps: int, episodes: int, eval_seed: int
) -> dict[str, Any]:
    """Train, evaluate and summarize one (mode, seed) run."""
    train_one(mode, seed, timesteps)
    evaluate_one(mode, seed, episodes, eval_seed)

    monitor_csv = monitor_csv_path_for(mode, seed)
    eval_json = eval_json_path_for(mode, seed)
    if not os.path.isfile(monitor_csv):
        raise FileNotFoundError(f"missing Monitor CSV after training: {monitor_csv}")
    if not os.path.isfile(eval_json):
        raise FileNotFoundError(f"missing eval JSON after evaluation: {eval_json}")

    episode_count = count_monitor_episodes(monitor_csv)
    with open(eval_json, encoding="utf-8") as file_handler:
        eval_payload: dict[str, Any] = json.load(file_handler)

    return {
        "condition": mode,
        "training_seed": seed,
        "timesteps": timesteps,
        "monitor_episode_count": episode_count,
        "approx_decisions_per_episode": timesteps / episode_count,
        "model_path": model_path_for(mode, seed),
        "eval_json_path": eval_json,
        "dqn": eval_payload["dqn"],
        "random": eval_payload["random"],
        "delta": eval_payload["delta"],
    }


def paired_rows(runs: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    """Pair full/blind DQN means per training seed (paired_delta = full - blind)."""
    by_condition_seed = {(run["condition"], run["training_seed"]): run for run in runs}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        full_run = by_condition_seed[("full", seed)]
        blind_run = by_condition_seed[("blind", seed)]
        full_mean = float(full_run["dqn"]["mean"])
        blind_mean = float(blind_run["dqn"]["mean"])
        rows.append(
            {
                "training_seed": seed,
                "full_mean": full_mean,
                "blind_mean": blind_mean,
                "paired_delta": full_mean - blind_mean,
            }
        )
    return rows


def aggregate_stats(paired: list[dict[str, Any]]) -> dict[str, float]:
    """Return population mean/std across training seeds."""
    full = np.asarray([row["full_mean"] for row in paired], dtype=np.float64)
    blind = np.asarray([row["blind_mean"] for row in paired], dtype=np.float64)
    deltas = np.asarray([row["paired_delta"] for row in paired], dtype=np.float64)
    return {
        "mean_full": float(np.mean(full)),
        "std_full": float(np.std(full)),
        "mean_blind": float(np.mean(blind)),
        "std_blind": float(np.std(blind)),
        "mean_paired_delta": float(np.mean(deltas)),
        "std_paired_delta": float(np.std(deltas)),
    }


def print_run_table(runs: list[dict[str, Any]]) -> None:
    """Print one line per (condition, seed) run."""
    print("per-run results:")
    print(
        f"{'condition':>9} | {'seed':>4} | {'train_eps':>9} | "
        f"{'decisions/ep':>12} | {'dqn_mean':>9} | {'delta':>8}"
    )
    for run in runs:
        print(
            f"{run['condition']:>9} | {run['training_seed']:>4} | "
            f"{run['monitor_episode_count']:>9} | "
            f"{run['approx_decisions_per_episode']:>12.1f} | "
            f"{float(run['dqn']['mean']):>9.2f} | {float(run['delta']):>8.2f}"
        )


def print_paired_table(paired: list[dict[str, Any]]) -> None:
    """Print the per-seed paired full/blind comparison."""
    print("paired per training seed:")
    print(f"{'seed':>4} | {'full_mean':>9} | {'blind_mean':>10} | {'delta':>8}")
    for row in paired:
        print(
            f"{row['training_seed']:>4} | {row['full_mean']:>9.2f} | "
            f"{row['blind_mean']:>10.2f} | {row['paired_delta']:>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run the Issue #13 paired Full/Blind observation "
            "ablation (train + evaluate per seed and mode)."
        )
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
        default=200_000,
        help="training timesteps per run (default: 200000)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="evaluation episodes per run (default: 100)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1000,
        help="starting evaluation map seed (default: 1000)",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        for mode in OBSERVATION_MODES:
            print(f"=== run: seed={seed} observation={mode} ===", flush=True)
            runs.append(
                run_one(mode, seed, args.timesteps, args.episodes, args.eval_seed)
            )

    paired = paired_rows(runs, seeds)
    aggregate = aggregate_stats(paired)

    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as file_handler:
        json.dump(
            {
                "timesteps": args.timesteps,
                "episodes": args.episodes,
                "eval_seed": args.eval_seed,
                "runs": runs,
                "paired": paired,
                "aggregate": aggregate,
            },
            file_handler,
            indent=2,
        )
        file_handler.write("\n")

    print_run_table(runs)
    print_paired_table(paired)
    print(
        f"aggregate: mean_full {aggregate['mean_full']:.2f} "
        f"± {aggregate['std_full']:.2f} | "
        f"mean_blind {aggregate['mean_blind']:.2f} "
        f"± {aggregate['std_blind']:.2f} | "
        f"paired_delta {aggregate['mean_paired_delta']:.2f} "
        f"± {aggregate['std_paired_delta']:.2f}"
    )
    print(f"results written to {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
