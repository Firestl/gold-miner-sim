"""Minimal Gymnasium wrapper batching physics ticks per agent decision.

Each ``step()`` advances the underlying environment by exactly
``DECISION_INTERVAL`` physics ticks: the first tick uses the agent's
action, the remaining ticks use WAIT (0). Rewards of all executed ticks
are summed; the loop stops early as soon as any underlying tick ends.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from numpy.typing import NDArray

from gold_miner_sim.env import WAIT

DECISION_INTERVAL = 10  # Physics ticks executed per agent decision (1/6 s).


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
