"""Random-map benchmark environment chain used by the DQN scripts.

Single source of truth for the Milestone 6 chain
``GoldMinerEnv(map_mode="random") -> SwingAdvanceDecisionWrapper ->
FireBudgetWrapper(max_fires=3)``, plus the observation ablations: the
Issue #13 ``observation_mode="blind"`` additionally wraps
``ObjectPositionMaskWrapper`` around the chain so the GOLD / DIAMOND /
ROCK x/y observation slots are zeroed, and the Issue #17
``observation_mode="polar"`` instead wraps
``ObjectPolarRepresentationWrapper`` so those slots are rewritten from
Cartesian (x, y) into Polar (target_angle, distance).

``make_geometry_eval_env`` (Issue #21) builds the same wrapper chain over
``GoldMinerEnv(map_mode="random", spawn_points=<geometry pool>)`` for the
frozen-policy geometry generalization stress test; only the spawn pool
changes, never the wrappers / reward / observation.
"""

from __future__ import annotations

import gymnasium
import numpy as np
from numpy.typing import NDArray

from gold_miner_sim.env import GoldMinerEnv
from gold_miner_sim.geometry_eval import GEOMETRY_MODES, geometry_spawn_pool
from gold_miner_sim.wrappers import (
    FireBudgetWrapper,
    ObjectPolarRepresentationWrapper,
    ObjectPositionMaskWrapper,
    SwingAdvanceDecisionWrapper,
)

OBSERVATION_MODES = ("full", "blind", "polar")


def make_benchmark_env(
    observation_mode: str = "full",
    render_mode: str | None = None,
) -> gymnasium.Wrapper[NDArray[np.float32], int, NDArray[np.float32], int]:
    """Build the random-map benchmark chain for ``observation_mode``.

    ``"full"`` returns the unmodified 27-dim benchmark chain; ``"blind"``
    additionally masks the object position slots to 0 (Issue #13);
    ``"polar"`` additionally rewrites those slots from Cartesian (x, y)
    into Polar (target_angle, distance) relative to the hook anchor
    (Issue #17). Raises ``ValueError`` for unknown modes.
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
    if observation_mode == "polar":
        return ObjectPolarRepresentationWrapper(env)
    return env


def make_geometry_eval_env(
    geometry_mode: str,
    render_mode: str | None = None,
) -> gymnasium.Wrapper[NDArray[np.float32], int, NDArray[np.float32], int]:
    """Build the Issue #21 geometry evaluation chain for ``geometry_mode``.

    Same wrapper chain as :func:`make_benchmark_env`, but the underlying
    ``GoldMinerEnv(map_mode="random")`` receives the spawn pool of
    ``geometry_mode`` (see ``gold_miner_sim.geometry_eval``): ``"id"`` is
    the identity control, ``"rot3"`` rotates the pool by +3 degrees around
    the anchor, ``"rot3_scale105"`` additionally scales radial distances by
    1.05. Wrappers, reward and observation are identical for every mode.
    Raises ``ValueError`` for unknown modes.
    """
    if geometry_mode not in GEOMETRY_MODES:
        raise ValueError(
            f"geometry_mode must be one of {GEOMETRY_MODES}, got {geometry_mode!r}"
        )
    return FireBudgetWrapper(
        SwingAdvanceDecisionWrapper(
            GoldMinerEnv(
                render_mode=render_mode,
                map_mode="random",
                spawn_points=geometry_spawn_pool(geometry_mode),
            )
        ),
        max_fires=3,
    )
