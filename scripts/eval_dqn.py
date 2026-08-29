"""Evaluate a trained DQN agent for one episode of GoldMinerEnv.

Uses the same wrapper stack as training (DecisionIntervalWrapper) and the
deterministic policy. Without ``--render`` it runs headless at full speed;
with ``--render`` the env auto-renders every physics tick and the renderer's
internal ``Clock.tick(60)`` already paces playback to real time, so no extra
sleep is needed. Closing the window ends the episode early.

Usage:
    uv run python scripts/eval_dqn.py
    uv run python scripts/eval_dqn.py --render
    uv run python scripts/eval_dqn.py --model models/my_model.zip --render
"""

from __future__ import annotations

import argparse

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.wrappers import DecisionIntervalWrapper
from stable_baselines3 import DQN


def run_episode(
    env: DecisionIntervalWrapper, model: DQN, render: bool
) -> tuple[float, int]:
    """Run one episode with the deterministic policy.

    Steps with ``model.predict(obs, deterministic=True)`` until the episode
    is terminated or truncated; in render mode it also stops as soon as the
    window is closed. In human mode each ``step()`` renders automatically.

    Returns ``(final_score, decision_steps)``.
    """
    obs, _info = env.reset()
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        steps += 1
        if render and env.unwrapped.window_closed:  # always False in headless mode
            break
    score: float = env.unwrapped.score
    return score, steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DQN agent for one GoldMinerEnv episode."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/dqn_gold_miner.zip",
        help="path to the saved model zip (default: models/dqn_gold_miner.zip)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="show the pygame window (default: headless)",
    )
    args = parser.parse_args()

    env: DecisionIntervalWrapper = DecisionIntervalWrapper(
        GoldMinerEnv(render_mode="human" if args.render else None)
    )
    model = DQN.load(args.model)
    try:
        score, steps = run_episode(env, model, args.render)
    finally:
        env.close()

    print(f"Final score: {score:.2f}")
    print(f"Decision steps: {steps}")


if __name__ == "__main__":
    main()
