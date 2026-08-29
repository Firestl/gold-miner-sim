"""Fixed-policy demo episode for the Gold Miner environment.

Fires the hook at three fixed target angles (-30 deg -> GOLD, +2 deg ->
DIAMOND, +31 deg -> ROCK), then waits until the 60 s episode times out.
The point is to exercise the whole hook state machine (SWINGING ->
EXTENDING -> RETRACT_LOADED -> SWINGING, three times), not to play well.

Usage:
    uv run python scripts/demo_episode.py             # watch in a window
    uv run python scripts/demo_episode.py --headless  # no window, full speed
"""

from __future__ import annotations

import argparse

from gold_miner_sim.env import (
    EPISODE_TIME,
    FIRE,
    WAIT,
    GoldMinerEnv,
    HookState,
)

# Fixed launch targets; the fixed map geometry guarantees hits near these.
TARGET_ANGLES: tuple[float, ...] = (-30.0, 2.0, 31.0)
# The hook swings exactly 1 deg per tick, so this small fixed tolerance
# fires within one tick of reaching the target angle.
FIRE_TOLERANCE_DEG = 0.6


def run_episode(env: GoldMinerEnv) -> tuple[float, int, int, float]:
    """Run one episode with the fixed fire-angle policy.

    Fires once per entry in ``TARGET_ANGLES`` (only while SWINGING and only
    when the current angle is within tolerance of the target), then waits
    until the episode ends or the render window is closed. In human mode
    ``step()`` renders automatically, so no explicit per-step ``render()``
    is needed.

    Returns ``(score, collected_count, steps, elapsed_seconds)``.
    """
    env.reset()
    target_index = 0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = WAIT
        if (
            target_index < len(TARGET_ANGLES)
            and env.hook_state is HookState.SWINGING
            and abs(env.angle - TARGET_ANGLES[target_index]) <= FIRE_TOLERANCE_DEG
        ):
            action = FIRE
            # One launch per target; the next target is only considered once
            # the hook is back in SWINGING (required by the check above).
            target_index += 1

        _obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1

        if env.window_closed:  # always False in headless mode
            break

    collected = sum(1 for obj in env.objects if not obj.active)
    return env.score, collected, steps, EPISODE_TIME - env.remaining_time


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one fixed-policy demo episode of GoldMinerEnv."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without a window at full speed (render_mode=None)",
    )
    args = parser.parse_args()

    env = GoldMinerEnv(render_mode=None if args.headless else "human")
    score, collected, steps, elapsed = run_episode(env)
    env.close()

    print(f"Final score: {score:g}")
    print(f"Collected objects: {collected}")
    print(f"Episode steps: {steps}")
    print(f"Simulated elapsed time: {elapsed:g}s")


if __name__ == "__main__":
    main()
