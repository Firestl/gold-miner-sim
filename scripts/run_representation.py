"""Run the Issue #17 paired Cartesian/Polar representation experiment.

For every training seed this script trains one fresh DQN agent per
representation condition -- ``cartesian`` (``--observation full``) before
``polar`` (``--observation polar``) -- evaluates it on the benchmark maps,
and collects the results into ``runs/representation/results.json``. Runs
are strictly sequential (Issue #17 forbids parallel training frameworks):
each child process inherits stdout and writes its own model under
``models/representation/{condition}/seed_{seed}.zip``, its Monitor CSV
under ``runs/representation/{condition}/seed_{seed}/``, and an ``eval.json``
summary read back by this script. Neither path collides with the
Milestone 7 ``models/ablation`` / ``runs/ablation`` artifacts.

Both conditions always use the same code version and the same evaluation
map seeds; the only experimental variable is the object position
representation. The paired statistic is ``paired_delta = polar_mean -
cartesian_mean`` per training seed; across seeds the script reports
population mean/std for both conditions and for the paired deltas, plus
the number of seeds where Polar beats Cartesian.

Note on std fields: ``std_episode`` is the score std of one model across
the evaluation maps, while ``std_*_across_training_seeds`` is the std of
per-seed means across training seeds; the two must never be conflated.

Usage:
    uv run --group train python scripts/run_representation.py
    uv run --group train python scripts/run_representation.py --seeds 0,1 --timesteps 50000 --episodes 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

import numpy as np

CONDITIONS = ("cartesian", "polar")
# Representation condition -> scripts/train_dqn.py --observation value.
CONDITION_OBSERVATION = {"cartesian": "full", "polar": "polar"}
RESULTS_JSON_PATH = "runs/representation/results.json"


def parse_seeds(spec: str) -> list[int]:
    """Parse a comma-separated list of integer training seeds."""
    tokens = [token.strip() for token in spec.split(",")]
    if not all(tokens):
        raise ValueError("--seeds must be a comma-separated list of integers")
    return [int(token) for token in tokens]


def model_path_for(condition: str, seed: int) -> str:
    """Return the model zip path for one (condition, seed) run."""
    return f"models/representation/{condition}/seed_{seed}.zip"


def monitor_csv_path_for(condition: str, seed: int) -> str:
    """Return the SB3 Monitor CSV path for one (condition, seed) run."""
    return f"runs/representation/{condition}/seed_{seed}/monitor.monitor.csv"


def eval_json_path_for(condition: str, seed: int) -> str:
    """Return the eval JSON output path for one (condition, seed) run."""
    return f"runs/representation/{condition}/seed_{seed}/eval.json"


def count_monitor_episodes(path: str) -> int:
    """Count episodes in an SB3 Monitor CSV (exactly two header lines)."""
    with open(path, encoding="utf-8") as file_handler:
        lines = file_handler.readlines()
    data_lines = [line for line in lines[2:] if line.strip()]
    if not data_lines:
        raise ValueError(f"Monitor CSV has no episode rows: {path}")
    return len(data_lines)


def train_one(condition: str, seed: int, timesteps: int) -> None:
    """Train one agent via scripts/train_dqn.py (inherits stdout)."""
    subprocess.run(
        [
            sys.executable,
            "scripts/train_dqn.py",
            "--observation",
            CONDITION_OBSERVATION[condition],
            "--timesteps",
            str(timesteps),
            "--seed",
            str(seed),
            "--output",
            model_path_for(condition, seed),
        ],
        check=True,
    )


def evaluate_one(condition: str, seed: int, episodes: int, eval_seed: int) -> None:
    """Evaluate one agent via scripts/eval_dqn.py (inherits stdout)."""
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_dqn.py",
            "--model",
            model_path_for(condition, seed),
            "--observation",
            CONDITION_OBSERVATION[condition],
            "--episodes",
            str(episodes),
            "--seed",
            str(eval_seed),
            "--json-output",
            eval_json_path_for(condition, seed),
        ],
        check=True,
    )


def run_one(
    condition: str, seed: int, timesteps: int, episodes: int, eval_seed: int
) -> dict[str, Any]:
    """Train, evaluate and summarize one (condition, seed) run."""
    train_one(condition, seed, timesteps)
    evaluate_one(condition, seed, episodes, eval_seed)

    monitor_csv = monitor_csv_path_for(condition, seed)
    eval_json = eval_json_path_for(condition, seed)
    if not os.path.isfile(monitor_csv):
        raise FileNotFoundError(f"missing Monitor CSV after training: {monitor_csv}")
    if not os.path.isfile(eval_json):
        raise FileNotFoundError(f"missing eval JSON after evaluation: {eval_json}")

    episode_count = count_monitor_episodes(monitor_csv)
    with open(eval_json, encoding="utf-8") as file_handler:
        eval_payload: dict[str, Any] = json.load(file_handler)

    return {
        "condition": condition,
        "training_seed": seed,
        "timesteps": timesteps,
        "monitor_episode_count": episode_count,
        "approx_decisions_per_episode": timesteps / episode_count,
        "model_path": model_path_for(condition, seed),
        "eval_json_path": eval_json,
        "dqn": eval_payload["dqn"],
        "random": eval_payload["random"],
        "delta": eval_payload["delta"],
    }


def paired_rows(
    runs: list[dict[str, Any]], seeds: list[int]
) -> list[dict[str, Any]]:
    """Pair Cartesian/Polar DQN means per training seed.

    ``paired_delta = polar_mean - cartesian_mean``: positive values mean
    the Polar representation scored higher for that seed.
    """
    by_condition_seed = {
        (run["condition"], run["training_seed"]): run for run in runs
    }
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        cartesian_run = by_condition_seed[("cartesian", seed)]
        polar_run = by_condition_seed[("polar", seed)]
        cartesian_mean = float(cartesian_run["dqn"]["mean"])
        polar_mean = float(polar_run["dqn"]["mean"])
        rows.append(
            {
                "training_seed": seed,
                "cartesian_mean": cartesian_mean,
                "polar_mean": polar_mean,
                "paired_delta": polar_mean - cartesian_mean,
            }
        )
    return rows


def aggregate_stats(paired: list[dict[str, Any]]) -> dict[str, float | int]:
    """Return population statistics across training seeds."""
    cartesian = np.asarray(
        [row["cartesian_mean"] for row in paired], dtype=np.float64
    )
    polar = np.asarray([row["polar_mean"] for row in paired], dtype=np.float64)
    deltas = np.asarray([row["paired_delta"] for row in paired], dtype=np.float64)
    return {
        "mean_cartesian": float(np.mean(cartesian)),
        "std_cartesian_across_training_seeds": float(np.std(cartesian)),
        "mean_polar": float(np.mean(polar)),
        "std_polar_across_training_seeds": float(np.std(polar)),
        "mean_paired_delta": float(np.mean(deltas)),
        "std_paired_delta": float(np.std(deltas)),
        "seeds_where_polar_gt_cartesian": int(
            np.count_nonzero(deltas > 0.0)
        ),
        "number_of_seeds": len(paired),
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
    """Print the per-seed paired Cartesian/Polar comparison."""
    print("paired per training seed:")
    print(
        f"{'seed':>4} | {'cartesian_mean':>14} | {'polar_mean':>10} | {'delta':>8}"
    )
    for row in paired:
        print(
            f"{row['training_seed']:>4} | {row['cartesian_mean']:>14.2f} | "
            f"{row['polar_mean']:>10.2f} | {row['paired_delta']:>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially run the Issue #17 paired Cartesian/Polar "
            "representation experiment (train + evaluate per seed and "
            "condition)."
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
        for condition in CONDITIONS:
            print(
                f"=== run: seed={seed} condition={condition} ===", flush=True
            )
            runs.append(
                run_one(condition, seed, args.timesteps, args.episodes, args.eval_seed)
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
        f"aggregate: mean_cartesian {aggregate['mean_cartesian']:.2f} "
        f"± {aggregate['std_cartesian_across_training_seeds']:.2f} | "
        f"mean_polar {aggregate['mean_polar']:.2f} "
        f"± {aggregate['std_polar_across_training_seeds']:.2f} | "
        f"paired_delta {aggregate['mean_paired_delta']:.2f} "
        f"± {aggregate['std_paired_delta']:.2f} | "
        f"polar > cartesian: "
        f"{aggregate['seeds_where_polar_gt_cartesian']}"
        f"/{aggregate['number_of_seeds']} seeds"
    )
    print(f"results written to {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
