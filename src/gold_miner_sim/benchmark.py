"""Random-map benchmark environment chain used by the DQN scripts.

Single source of truth for the Milestone 6 chain
``GoldMinerEnv(map_mode="random") -> SwingAdvanceDecisionWrapper ->
FireBudgetWrapper(max_fires=3)``, plus the Issue #13 observation ablation:
``observation_mode="blind"`` additionally wraps
``ObjectPositionMaskWrapper`` around the chain so the GOLD / DIAMOND /
ROCK x/y observation slots are zeroed.
"""

from __future__ import annotations

import gymnasium
import numpy as np
from numpy.typing import NDArray

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.wrappers import (
    FireBudgetWrapper,
    ObjectPositionMaskWrapper,
    SwingAdvanceDecisionWrapper,
)

OBSERVATION_MODES = ("full", "blind")


def make_benchmark_env(
    observation_mode: str = "full",
    render_mode: str | None = None,
) -> gymnasium.Wrapper[NDArray[np.float32], int, NDArray[np.float32], int]:
    """Build the random-map benchmark chain for ``observation_mode``.

    ``"full"`` returns the unmodified 27-dim benchmark chain; ``"blind"``
    additionally masks the object position slots to 0 (Issue #13). Raises
    ``ValueError`` for unknown modes.
    """
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(
            f"observation_mode must be one of {OBSERVATION_MODES}, "
            f"got {observation_mode!r}"
        )
    env = FireBudgetWrapper(
        SwingAdvanceDecisionWrapper(
            GoldMinerEnv(render_mode=render_mode, map_mode="random")
        ),
        max_fires=3,
    )
    if observation_mode == "blind":
        return ObjectPositionMaskWrapper(env)
    return env
