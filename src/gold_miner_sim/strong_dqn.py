"""Small shared helpers for the Milestone 10 Strong DQN recipe.

The environment and its benchmark wrapper chain deliberately live elsewhere.
This module only keeps the recipe constants and deterministic evaluation logic
shared by the Strong DQN training and evaluation scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray

from gold_miner_sim.benchmark import make_benchmark_env

FULL_SCORE = 800.0


@dataclass(frozen=True, slots=True)
class StrongDQNConfig:
    """Default Strong DQN v2 recipe and evaluation schedule."""

    learning_rate: float = 1e-4
    buffer_size: int = 200_000
    learning_starts: int = 5_000
    batch_size: int = 256
    gamma: float = 1.0
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.6
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    n_steps: int = 5
    net_arch: tuple[int, int] = (256, 256)
    max_timesteps: int = 1_000_000
    evaluation_interval: int = 25_000
    selection_seed: int = 1_000
    selection_episodes: int = 20
    validation_seed: int = 1_000
    validation_episodes: int = 100
    heldout_seed: int = 2_000
    heldout_episodes: int = 100

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable copy of the recipe."""
        return asdict(self)


DEFAULT_STRONG_DQN_CONFIG = StrongDQNConfig()

_BenchmarkEnv = gymnasium.Wrapper[NDArray[np.float32], int, NDArray[np.float32], int]


def run_dqn_episode(
    env: _BenchmarkEnv,
    model: Any,
    map_seed: int,
) -> tuple[float, int]:
    """Run one deterministic episode and return ``(score, decisions)``."""
    observation, _info = env.reset(seed=map_seed)
    terminated = False
    truncated = False
    score = 0.0
    decisions = 0

    while not (terminated or truncated):
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _info = env.step(int(action))
        score += float(reward)
        decisions += 1

    return score, decisions


def summarize_evaluation(
    scores: list[float], decisions: list[int]
) -> dict[str, float | int]:
    """Return the metrics required for Strong DQN checkpoint selection."""
    if not scores:
        raise ValueError("at least one episode is required")
    if len(scores) != len(decisions):
        raise ValueError("scores and decisions must have the same length")

    values = np.asarray(scores, dtype=np.float64)
    decision_values = np.asarray(decisions, dtype=np.float64)
    full_score_count = int(np.count_nonzero(values == FULL_SCORE))
    return {
        "mean_score": float(np.mean(values)),
        "std_episode": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "full_score_count": full_score_count,
        "full_score_rate": full_score_count / len(scores),
        "mean_episode_decisions": float(np.mean(decision_values)),
    }


def evaluate_dqn(
    model: Any,
    seed_start: int,
    episodes: int,
) -> dict[str, float | int]:
    """Evaluate a model on deterministic random-map benchmark episodes.

    The model is queried with ``deterministic=True`` and the benchmark chain
    is always the unchanged Full Cartesian condition.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    env = make_benchmark_env(observation_mode="full")
    try:
        results = [
            run_dqn_episode(env, model, seed_start + index) for index in range(episodes)
        ]
    finally:
        env.close()

    scores = [score for score, _decisions in results]
    decisions = [decision_count for _score, decision_count in results]
    return summarize_evaluation(scores, decisions)
