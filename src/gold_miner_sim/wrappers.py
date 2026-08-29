"""Minimal Gymnasium wrappers batching physics ticks per agent decision.

``DecisionIntervalWrapper`` advances the underlying environment by exactly
``DECISION_INTERVAL`` physics ticks per ``step()``: the first tick uses the
agent's action, the remaining ticks use WAIT (0). Rewards of all executed
ticks are summed; the loop stops early as soon as any underlying tick ends.

``SwingDecisionWrapper`` is a Gold Miner specific wrapper whose ``WAIT``
decision behaves like the above, while its ``FIRE`` decision automatically
runs until the hook returns to the swinging phase (variable-length
transition; see the class docstring).
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray

from gold_miner_sim.env import FIRE, WAIT, HookState

DECISION_INTERVAL = 10  # Physics ticks executed per agent decision (1/6 s).
SWING_WAIT_INTERVAL = 10  # Max physics ticks executed per WAIT decision.


class DecisionIntervalWrapper(gymnasium.Wrapper):
    """Repeat each agent action for ``DECISION_INTERVAL`` physics ticks.

    ``step(action)`` runs one underlying tick with the given action, then
    up to ``DECISION_INTERVAL - 1`` further ticks with WAIT. Rewards of all
    executed ticks are summed; execution stops immediately once an
    underlying tick reports ``terminated`` or ``truncated``. The returned
    observation, info and end flags come from the last executed tick.
    Action and observation spaces are inherited from the wrapped env.
    """

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        for i in range(DECISION_INTERVAL):
            tick_action = action if i == 0 else WAIT
            obs, reward, terminated, truncated, info = self.env.step(tick_action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)


class SwingDecisionWrapper(gymnasium.Wrapper):
    """Gold Miner wrapper: the agent only decides while the hook swings.

    The underlying hook state machine accepts FIRE only in ``SWINGING``.
    This wrapper lifts that invariant to the decision level: except for
    the tick that ends the episode, every ``step()`` returns with the
    underlying hook back in ``HookState.SWINGING``. The agent therefore
    never has to reason about EXTENDING / RETRACT_* phases.

    ``step(WAIT)`` executes at most ``SWING_WAIT_INTERVAL`` underlying
    WAIT ticks, summing their rewards and stopping at the first tick that
    reports ``terminated`` or ``truncated``. While waiting, the hook stays
    SWINGING, so this is equivalent to ``DecisionIntervalWrapper.step(WAIT)``.

    ``step(FIRE)`` executes one underlying FIRE tick (SWINGING ->
    EXTENDING) and then automatically keeps issuing WAIT ticks until the
    hook re-enters ``SWINGING``. It returns at the exact tick the hook
    becomes SWINGING again -- never one tick later -- or at the first tick
    that ends the episode, whichever comes first. Rewards of all executed
    ticks are summed; the observation, info and end flags come from the
    last executed tick (e.g. a fully recovered DIAMOND yields reward 500
    from a single ``step(FIRE)`` call, an empty round trip yields 0, and a
    timeout before recovery yields whatever was scored so far).

    Transitions are therefore variable-length by design; no duration-aware
    discounting or tick-count correction is applied. Actions other than
    WAIT (0) or FIRE (1) raise ``ValueError`` instead of silently acting
    as WAIT. Action and observation spaces are inherited from the env.
    """

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if action != WAIT and action != FIRE:
            raise ValueError(
                f"invalid action {action!r}, expected 0 (WAIT) or 1 (FIRE)"
            )

        total_reward = 0.0
        if action == FIRE:
            # First tick: SWINGING -> EXTENDING (angle frozen from here on).
            obs, reward, terminated, truncated, info = self.env.step(FIRE)
            total_reward += reward
            # Auto-advance until the hook is back at the top; stop at the
            # exact SWINGING tick or at the first episode-ending tick.
            hook_env = self.env.unwrapped
            while not (terminated or truncated) and (
                hook_env.hook_state is not HookState.SWINGING
            ):
                obs, reward, terminated, truncated, info = self.env.step(WAIT)
                total_reward += reward
            return obs, total_reward, terminated, truncated, info

        for _ in range(SWING_WAIT_INTERVAL):
            obs, reward, terminated, truncated, info = self.env.step(WAIT)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)
