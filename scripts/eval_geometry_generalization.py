"""Evaluate frozen Strong DQN checkpoints across geometry modes (M11).

Issue #21 frozen-policy geometry generalization stress test: the M10 best
checkpoints are loaded as-is and only ever queried with
``predict(obs, deterministic=True)``.  This script never calls
``model.learn()`` and never retrains / fine-tunes / skips a checkpoint.
Every requested ``best_model.zip`` is checked for existence before any
episode runs; missing files are all listed and the script exits non-zero.

Each checkpoint is evaluated on the same paired map seeds (default
3000..3099) under the three geometry modes ``id`` / ``rot3`` /
``rot3_scale105`` via ``gold_miner_sim.benchmark.make_geometry_eval_env``,
and the results are written to
``runs/geometry_generalization/results.json`` (never to
``runs/strong_dqn/``).

Usage:
    uv run --group train python scripts/eval_geometry_generalization.py
    uv run --group train python scripts/eval_geometry_generalization.py --seeds 0 --episodes 2 --map-seed-start 3000 --output /tmp/m11_smoke.json
    uv run --group train python scripts/eval_geometry_generalization.py --trace --trace-seeds 0,3 --trace-maps 3000,3007,3042
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from stable_baselines3 import DQN

from gold_miner_sim.benchmark import make_geometry_eval_env
from gold_miner_sim.geometry_eval import (
    GEOMETRY_MODES,
    across_seed_paired_analysis,
    paired_analysis,
    run_geometry_episode,
    summarize_geometry_results,
)

DEFAULT_MODEL_ROOT = "models/strong_dqn"
DEFAULT_OUTPUT = os.path.join("runs", "geometry_generalization", "results.json")
DEFAULT_TRACE_SEEDS = "0,3"
DEFAULT_TRACE_MAPS = "3000,3007,3042"
OOD_GEOMETRY_MODES = tuple(mode for mode in GEOMETRY_MODES if mode != "id")


def parse_seeds(spec: str) -> list[int]:
    """Parse a comma-separated list of integers (seeds or map seeds)."""
    tokens = [token.strip() for token in spec.split(",")]
    if not tokens or not all(tokens):
        raise ValueError("--seeds must be a comma-separated list of integers")
    return [int(token) for token in tokens]


def model_path_for(seed: int, model_root: str) -> str:
    """Return the Strong DQN best-checkpoint path for ``seed``."""
    return os.path.join(model_root, f"seed_{seed}", "best_model.zip")


def require_checkpoints(model_paths: Sequence[str]) -> None:
    """Exit non-zero when any requested frozen checkpoint is missing.

    Every missing path is printed; the script never trains a replacement
    and never skips a missing checkpoint (Issue #21 checkpoint freeze).
    """
    missing = [path for path in model_paths if not os.path.isfile(path)]
    if not missing:
        return
    print("error: missing frozen Strong DQN checkpoints:")
    for path in missing:
        print(f"  {path}")
    print(
        "M11 never trains or fine-tunes; produce the M10 artifacts first "
        "(uv run --group train python scripts/run_strong_dqn.py) or adjust "
        "--model-root / --seeds / --trace-seeds."
    )
    raise SystemExit(1)


def evaluate_runs(
    seeds: Sequence[int],
    episodes: int,
    map_seed_start: int,
    model_root: str,
) -> list[dict[str, Any]]:
    """Evaluate every seed's checkpoint on every geometry mode.

    Returns one JSON-ready summary row per (training_seed x geometry_mode),
    each holding the Issue #21 §10 metrics plus the per-map
    ``episode_scores`` (ordered by map seed) for paired analysis.
    """
    runs: list[dict[str, Any]] = []
    map_seed_range = f"{map_seed_start}-{map_seed_start + episodes - 1}"
    for training_seed in seeds:
        model = DQN.load(model_path_for(training_seed, model_root))
        for geometry_mode in GEOMETRY_MODES:
            env = make_geometry_eval_env(geometry_mode)
            try:
                results = [
                    run_geometry_episode(env, model, map_seed_start + index)
                    for index in range(episodes)
                ]
            finally:
                env.close()
            summary = summarize_geometry_results(results)
            runs.append(
                {
                    "training_seed": training_seed,
                    "geometry_mode": geometry_mode,
                    "map_seed_range": map_seed_range,
                    **summary,
                }
            )
            print(
                f"training_seed={training_seed} geometry_mode={geometry_mode} "
                f"mean_score={summary['mean_score']:.2f}"
            )
    return runs


def _per_seed_row_values(
    runs: Sequence[dict[str, Any]], seeds: Sequence[int]
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], list[float]]]:
    """Index mean scores and episode-score lists by (seed, geometry mode)."""
    means: dict[tuple[int, str], float] = {}
    scores: dict[tuple[int, str], list[float]] = {}
    for run in runs:
        key = (int(run["training_seed"]), str(run["geometry_mode"]))
        means[key] = float(run["mean_score"])
        scores[key] = [float(value) for value in run["episode_scores"]]
    return means, scores


def paired_tables(
    seeds: Sequence[int], runs: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Build the §12 per-seed / across-seed and §13 per-map paired tables."""
    means, scores = _per_seed_row_values(runs, seeds)
    id_means = [means[(seed, "id")] for seed in seeds]
    analyses = {
        mode: across_seed_paired_analysis(
            id_means, [means[(seed, mode)] for seed in seeds]
        )
        for mode in OOD_GEOMETRY_MODES
    }

    per_seed_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        row: dict[str, Any] = {
            "training_seed": seed,
            "id_mean_score": id_means[index],
        }
        for mode in OOD_GEOMETRY_MODES:
            row[f"{mode}_mean_score"] = means[(seed, mode)]
            row[f"{mode}_drop"] = analyses[mode]["per_seed_drop"][index]
            row[f"{mode}_retention"] = analyses[mode]["per_seed_retention"][index]
        per_seed_rows.append(row)

    across_seed_aggregate: dict[str, Any] = {
        "training_seed_count": len(seeds),
        "id": {
            "mean_score_across_training_seeds": float(np.mean(id_means)),
            "std_score_across_training_seeds": float(np.std(id_means)),
        },
    }
    for mode in OOD_GEOMETRY_MODES:
        mode_means = [means[(seed, mode)] for seed in seeds]
        across_seed_aggregate[mode] = {
            "mean_score_across_training_seeds": float(np.mean(mode_means)),
            "std_score_across_training_seeds": float(np.std(mode_means)),
            **analyses[mode],
        }

    per_map_paired = {
        mode: paired_analysis(
            [score for seed in seeds for score in scores[(seed, "id")]],
            [score for seed in seeds for score in scores[(seed, mode)]],
        )
        for mode in OOD_GEOMETRY_MODES
    }
    return {
        "per_seed_paired": per_seed_rows,
        "across_seed_aggregate": across_seed_aggregate,
        "per_map_paired": per_map_paired,
    }


def print_summary(seeds: Sequence[int], runs: Sequence[dict[str, Any]]) -> None:
    """Print the per-seed paired means and the across-seed retention."""
    means, _scores = _per_seed_row_values(runs, seeds)
    id_means = [means[(seed, "id")] for seed in seeds]
    for index, seed in enumerate(seeds):
        parts = [f"training_seed={seed} id={id_means[index]:.2f}"]
        for mode in OOD_GEOMETRY_MODES:
            mode_mean = means[(seed, mode)]
            retention = (
                f"{mode_mean / id_means[index]:.3f}" if id_means[index] else "n/a"
            )
            parts.append(f"{mode}={mode_mean:.2f} (retention {retention})")
        print(" ".join(parts))
    aggregate_parts = [
        f"id={float(np.mean(id_means)):.2f}",
    ]
    for mode in OOD_GEOMETRY_MODES:
        analysis = across_seed_paired_analysis(
            id_means, [means[(seed, mode)] for seed in seeds]
        )
        mean_retention = analysis["mean_retention"]
        retention_text = (
            f"{float(mean_retention):.3f}" if mean_retention is not None else "n/a"
        )
        mode_means = [means[(seed, mode)] for seed in seeds]
        aggregate_parts.append(
            f"{mode}={float(np.mean(mode_means)):.2f} (retention {retention_text})"
        )
    print("across seeds: " + " ".join(aggregate_parts))


def run_trace(seeds: Sequence[int], map_seeds: Sequence[int], model_root: str) -> None:
    """Run the behavior replay and print one line per FIRE plus final score."""
    for training_seed in seeds:
        model = DQN.load(model_path_for(training_seed, model_root))
        for map_seed in map_seeds:
            for geometry_mode in GEOMETRY_MODES:
                env = make_geometry_eval_env(geometry_mode)
                try:
                    result = run_geometry_episode(env, model, map_seed)
                finally:
                    env.close()
                for record in result.fire_diagnostics:
                    active_angles = (
                        "["
                        + ", ".join(
                            f"{angle:.1f}" for angle in record.active_object_angles
                        )
                        + "]"
                    )
                    print(
                        f"training_seed={training_seed} map_seed={map_seed} "
                        f"geometry_mode={geometry_mode} "
                        f"fire={record.fire_number} "
                        f"current_angle={record.current_fire_angle:.1f} "
                        f"active_angles={active_angles} "
                        f"predicted_first_hit_slot="
                        f"{record.predicted_first_hit_slot} "
                        f"collected_slot={record.collected_slot_after_step} "
                        f"score_after_fire={record.score_after_fire:g}"
                    )
                print(
                    f"training_seed={training_seed} map_seed={map_seed} "
                    f"geometry_mode={geometry_mode} "
                    f"final_score={result.score:g}"
                )


def main() -> None:
    """Parse CLI arguments, verify checkpoints, evaluate or trace."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Strong DQN checkpoints on paired map seeds "
            "under the id / rot3 / rot3_scale105 geometry modes."
        )
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
        default=100,
        help="episodes per checkpoint and geometry mode (default: 100)",
    )
    parser.add_argument(
        "--map-seed-start",
        type=int,
        default=3000,
        help="first paired evaluation map seed (default: 3000)",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default=DEFAULT_MODEL_ROOT,
        help="root containing seed_<n>/best_model.zip",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="JSON output path (default: runs/geometry_generalization/results.json)",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "skip the full evaluation and run a per-FIRE behavior replay "
            "on --trace-seeds x --trace-maps x all geometry modes"
        ),
    )
    parser.add_argument(
        "--trace-seeds",
        type=str,
        default=DEFAULT_TRACE_SEEDS,
        help="comma-separated training seeds for --trace (default: 0,3)",
    )
    parser.add_argument(
        "--trace-maps",
        type=str,
        default=DEFAULT_TRACE_MAPS,
        help="comma-separated map seeds for --trace (default: 3000,3007,3042)",
    )
    args = parser.parse_args()

    if args.trace:
        trace_seeds = parse_seeds(args.trace_seeds)
        trace_maps = parse_seeds(args.trace_maps)
        require_checkpoints(
            [model_path_for(seed, args.model_root) for seed in trace_seeds]
        )
        run_trace(trace_seeds, trace_maps, args.model_root)
        return

    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    seeds = parse_seeds(args.seeds)
    require_checkpoints([model_path_for(seed, args.model_root) for seed in seeds])

    runs = evaluate_runs(seeds, args.episodes, args.map_seed_start, args.model_root)
    payload: dict[str, Any] = {
        "milestone": "M11",
        "issue": 21,
        "policy": (
            "frozen Strong DQN best checkpoints; deterministic=True inference "
            "only; never trained or fine-tuned by this evaluation"
        ),
        "geometry_modes": list(GEOMETRY_MODES),
        "map_seed_range": f"{args.map_seed_start}-{args.map_seed_start + args.episodes - 1}",
        "episodes": args.episodes,
        "runs": runs,
        **paired_tables(seeds, runs),
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file_handler:
        json.dump(payload, file_handler, indent=2)
        file_handler.write("\n")

    print(f"map_seed_range: {payload['map_seed_range']}")
    print(f"results written to {args.output}")
    print_summary(seeds, runs)


if __name__ == "__main__":
    main()
