"""Tests for the Milestone 10 Strong DQN configuration and bookkeeping."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from gold_miner_sim.strong_dqn import (
    DEFAULT_STRONG_DQN_CONFIG,
    StrongDQNConfig,
    summarize_evaluation,
)


def test_strong_dqn_default_recipe_matches_issue_19() -> None:
    """The default recipe is explicit and keeps the Full benchmark schedule."""
    config = DEFAULT_STRONG_DQN_CONFIG

    assert isinstance(config, StrongDQNConfig)
    assert config.learning_rate == 1e-4
    assert config.buffer_size == 200_000
    assert config.learning_starts == 5_000
    assert config.batch_size == 256
    assert config.gamma == 1.0
    assert config.train_freq == 1
    assert config.gradient_steps == 1
    assert config.target_update_interval == 10_000
    assert config.exploration_fraction == 0.6
    assert config.exploration_initial_eps == 1.0
    assert config.exploration_final_eps == 0.05
    assert config.n_steps == 5
    assert config.net_arch == (256, 256)
    assert config.max_timesteps == 1_000_000
    assert config.evaluation_interval == 25_000
    assert config.selection_seed == 1_000
    assert config.selection_episodes == 20
    assert config.validation_seed == 1_000
    assert config.validation_episodes == 100
    assert config.heldout_seed == 2_000
    assert config.heldout_episodes == 100


def test_summarize_evaluation_reports_episode_and_decision_metrics() -> None:
    """Checkpoint selection receives stable score and timing metrics."""
    metrics = summarize_evaluation([0.0, 800.0, 400.0], [7, 9, 11])

    assert metrics["mean_score"] == pytest.approx(400.0)
    assert metrics["std_episode"] == pytest.approx(((400.0**2 + 400.0**2) / 3.0) ** 0.5)
    assert metrics["min"] == 0.0
    assert metrics["max"] == 800.0
    assert metrics["full_score_count"] == 1
    assert metrics["full_score_rate"] == pytest.approx(1 / 3)
    assert metrics["mean_episode_decisions"] == pytest.approx(9.0)


def test_strong_dqn_callback_selects_earliest_equal_best(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation ties keep the earlier checkpoint deterministically."""
    pytest.importorskip("stable_baselines3")
    script_path = Path(__file__).parents[1] / "scripts" / "train_strong_dqn.py"
    module_spec = importlib.util.spec_from_file_location(
        "train_strong_dqn_for_test", script_path
    )
    assert module_spec is not None and module_spec.loader is not None
    train_strong_dqn = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(train_strong_dqn)

    validation_calls = 0

    def fake_evaluate(
        model: Any, seed_start: int, episodes: int
    ) -> dict[str, float | int]:
        nonlocal validation_calls
        validation_calls += 1
        assert seed_start == 1_000
        assert episodes == 2
        return {
            "mean_score": 300.0,
            "std_episode": 0.0,
            "min": 300.0,
            "max": 300.0,
            "full_score_count": 0,
            "full_score_rate": 0.0,
            "mean_episode_decisions": 8.0,
        }

    class FakeModel:
        """Small save-only stand-in for the SB3 model."""

        def __init__(self) -> None:
            self.saved: list[str] = []

        def save(self, path: str) -> None:
            self.saved.append(path)

    monkeypatch.setattr(train_strong_dqn, "evaluate_dqn", fake_evaluate)
    model = FakeModel()
    history_path = tmp_path / "runs" / "evaluations.json"
    best_path = tmp_path / "models" / "best_model.zip"
    callback = train_strong_dqn.StrongDQNEvaluationCallback(
        eval_seed=1_000,
        eval_episodes=2,
        eval_interval=25_000,
        best_model_path=str(best_path),
        history_path=str(history_path),
    )
    callback.model = model  # type: ignore[assignment]

    callback.num_timesteps = 25_000
    assert callback._on_step() is True
    callback.num_timesteps = 50_000
    assert callback._on_step() is True
    callback._on_training_end()

    assert validation_calls == 2
    assert callback.best_checkpoint_timestep == 25_000
    assert model.saved == [str(best_path)]
    with open(history_path, encoding="utf-8") as file_handler:
        payload = json.load(file_handler)
    assert payload["observation"] == "full"
    assert payload["best_checkpoint_timestep"] == 25_000
    assert [row["step"] for row in payload["evaluations"]] == [25_000, 50_000]
    assert [row["is_best"] for row in payload["evaluations"]] == [True, False]
