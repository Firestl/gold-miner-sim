"""Minimal Gymnasium wrappers batching physics ticks per agent decision.

``DecisionIntervalWrapper`` advances the underlying environment by exactly
``DECISION_INTERVAL`` physics ticks per ``step()``: the first tick uses the
agent's action, the remaining ticks use WAIT (0). Rewards of all executed
ticks are summed; the loop stops early as soon as any underlying tick ends.

``SwingDecisionWrapper`` is a Gold Miner specific wrapper whose ``WAIT``
decision behaves like the above, while its ``FIRE`` decision automatically
runs until the hook returns to the swinging phase (variable-length
transition; see the class docstring).

``SwingAdvanceDecisionWrapper`` keeps those semantics but, after a
completed FIRE cycle, swings on for another ``ADVANCE_INTERVAL`` WAIT
ticks before returning, so the next decision is never taken at the
original firing angle (anti angle-pinning; see the class docstring).

``ObjectPositionMaskWrapper`` is a pure observation post-processor: it
zeros the GOLD/DIAMOND/ROCK x/y slots of the 27-dimensional benchmark
observation (the Issue #13 "blind" ablation) and passes everything else
through unchanged.
"""

from __future__ import annotations

from typing import Any, cast

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from gold_miner_sim.env import FIRE, WAIT, GoldMinerEnv, HookState

DECISION_INTERVAL = 10  # Physics ticks executed per agent decision (1/6 s).
SWING_WAIT_INTERVAL = 10  # Max physics ticks executed per WAIT decision.
ADVANCE_INTERVAL = 10  # Post-FIRE SWINGING WAIT ticks before returning.

# Slots of the 27-dim benchmark observation holding the GOLD / DIAMOND /
# ROCK x, y coordinates (within each 6-value object block: x, y, radius,
# value, retract_speed, active).
OBJECT_POSITION_INDICES = (8, 9, 14, 15, 20, 21)

# Shared wrapper type parameters: observation space, action space and the
# wrapped env's own obs/act types all match ``GoldMinerEnv`` (the wrappers
# inherit both spaces unchanged).
_WrapperT = gymnasium.Wrapper[NDArray[np.float32], int, NDArray[np.float32], int]


class DecisionIntervalWrapper(_WrapperT):
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
        if not self.action_space.contains(action):
            raise ValueError(
                f"invalid action {action!r}, expected 0 (WAIT) or 1 (FIRE)"
            )

        total_reward = 0.0
        for i in range(DECISION_INTERVAL):
            tick_action = int(action) if i == 0 else WAIT
            obs, reward, terminated, truncated, info = self.env.step(tick_action)
            total_reward += float(reward)
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
    discounting or tick-count correction is applied. Actions are validated
    with ``action_space.contains`` — the same strict ``Discrete(2)``
    contract as the underlying env, which rejects e.g. float actions that
    compare equal to WAIT/FIRE — and never silently act as WAIT.
    Action and observation spaces are inherited from the env.
    """

    # This wrapper only ever wraps GoldMinerEnv and reads its game state
    # (hook_state) directly; the annotation narrows the gymnasium base
    # attribute for static typing.
    env: GoldMinerEnv

    def __init__(self, env: GoldMinerEnv) -> None:
        super().__init__(env)

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # contains() matches the underlying env: ``action == WAIT`` would
        # also accept floats like 0.0/1.0 that Discrete(2) does not contain.
        if not self.action_space.contains(action):
            raise ValueError(
                f"invalid action {action!r}, expected 0 (WAIT) or 1 (FIRE)"
            )

        total_reward = 0.0
        if action == FIRE:
            # First tick: SWINGING -> EXTENDING (angle frozen from here on).
            obs, reward, terminated, truncated, info = self.env.step(FIRE)
            total_reward += reward
            hook_env = cast(GoldMinerEnv, self.env.unwrapped)
            # Auto-advance until the hook is back at the top; stop at the
            # exact SWINGING tick or at the first episode-ending tick.
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


class SwingAdvanceDecisionWrapper(gymnasium.Wrapper):
    """Gold Miner wrapper: after a FIRE cycle, swing on before deciding.

    Same decision semantics as :class:`SwingDecisionWrapper`, plus one
    rule: once a FIRE decision's extend/retract cycle has completed and the
    hook is back in ``HookState.SWINGING``, this wrapper keeps issuing WAIT
    for another ``ADVANCE_INTERVAL`` physics ticks before returning. The
    next decision observation therefore sits at a swung-on angle instead of
    the original firing angle, which removes the structural angle-pinning
    loop observed in Milestone 4 (FIRE -> identical observation -> FIRE
    again at the same angle). The advance reuses the underlying swing
    physics, boundary reflection at -70/+70 deg included; the wrapper never
    computes angles itself.

    ``step(WAIT)`` executes at most ``SWING_WAIT_INTERVAL`` underlying WAIT
    ticks, exactly as ``SwingDecisionWrapper.step(WAIT)``. ``step(FIRE)``
    returns immediately when the episode ends during the FIRE cycle (no
    advance is executed) or, otherwise, after the ``ADVANCE_INTERVAL``
    advance ticks, cutting the advance short at the first episode-ending
    tick. Rewards of all executed ticks are summed; the observation, info
    and end flags come from the last executed tick.

    Transitions are variable-length by design; no duration-aware
    discounting or tick-count correction is applied. Actions are validated
    with ``action_space.contains`` — the same strict ``Discrete(2)``
    contract as the underlying env, which rejects e.g. float actions that
    compare equal to WAIT/FIRE — and never silently act as WAIT.
    Action and observation spaces are inherited from the env.
    """

    # This wrapper only ever wraps GoldMinerEnv and reads its game state
    # (hook_state) directly; the annotation narrows the gymnasium base
    # attribute for static typing.
    env: GoldMinerEnv

    def __init__(self, env: GoldMinerEnv) -> None:
        super().__init__(env)

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # contains() matches the underlying env: ``action == WAIT`` would
        # also accept floats like 0.0/1.0 that Discrete(2) does not contain.
        if not self.action_space.contains(action):
            raise ValueError(
                f"invalid action {action!r}, expected 0 (WAIT) or 1 (FIRE)"
            )

        total_reward = 0.0
        if action == FIRE:
            # Phase A -- the FIRE cycle, identical to SwingDecisionWrapper:
            # one FIRE tick, then WAIT ticks until the exact tick the hook is
            # back SWINGING or the first episode-ending tick.
            obs, reward, terminated, truncated, info = self.env.step(FIRE)
            total_reward += reward
            hook_env = cast(GoldMinerEnv, self.env.unwrapped)
            while not (terminated or truncated) and (
                hook_env.hook_state is not HookState.SWINGING
            ):
                obs, reward, terminated, truncated, info = self.env.step(WAIT)
                total_reward += reward
            if terminated or truncated:
                # Episode ended mid-cycle: never run the post-FIRE advance.
                return obs, total_reward, terminated, truncated, info
            # Phase B -- post-FIRE advance: swing on by ADVANCE_INTERVAL WAIT
            # ticks so the returned angle differs from the firing angle.
            for _ in range(ADVANCE_INTERVAL):
                obs, reward, terminated, truncated, info = self.env.step(WAIT)
                total_reward += reward
                if terminated or truncated:
                    break
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


class FireBudgetWrapper(_WrapperT):
    """Limit the number of FIRE decisions available in each episode.

    The wrapped environment performs the complete transition for every
    action.  FIRE consumes one budget unit immediately, including when the
    hook misses or the wrapped transition times out.  Once the final FIRE
    transition completes without an inner episode end, this wrapper marks the
    episode terminated.  An inner ``truncated`` or ``terminated`` result is
    always propagated unchanged, so a timeout cannot be replaced by the
    budget termination.

    One normalized ``fires_remaining / max_fires`` value is appended to the
    wrapped 26-dimensional observation, yielding a 27-dimensional output.
    ``reset()`` starts a fresh budget and appends the same value to the reset
    observation.  Budget fields are added to every returned ``info`` mapping.
    """

    env: gymnasium.Env[NDArray[np.float32], int]

    def __init__(
        self,
        env: gymnasium.Env[NDArray[np.float32], int],
        max_fires: int = 3,
    ) -> None:
        if isinstance(max_fires, bool) or not isinstance(max_fires, (int, np.integer)):
            raise TypeError("max_fires must be a positive integer")
        if max_fires <= 0:
            raise ValueError("max_fires must be a positive integer")

        super().__init__(env)
        inner_observation_space = env.observation_space
        if not isinstance(inner_observation_space, spaces.Box):
            raise TypeError("FireBudgetWrapper requires a Box observation space")
        if inner_observation_space.shape != (26,):
            raise ValueError(
                "FireBudgetWrapper requires a 26-dimensional observation space"
            )

        self.max_fires: int = int(max_fires)
        self.fires_used: int = 0
        self.fires_remaining: int = self.max_fires
        self.observation_space = spaces.Box(
            low=np.concatenate(
                (inner_observation_space.low, np.asarray([0.0], dtype=np.float32))
            ),
            high=np.concatenate(
                (inner_observation_space.high, np.asarray([1.0], dtype=np.float32))
            ),
            dtype=np.float32,
        )

    def _augment_observation(
        self, observation: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        budget = np.asarray(
            [self.fires_remaining / self.max_fires], dtype=np.float32
        )
        return np.concatenate((np.asarray(observation, dtype=np.float32), budget))

    def _augment_info(self, info: dict[str, Any]) -> dict[str, Any]:
        augmented_info = dict(info)
        augmented_info.update(
            fires_used=self.fires_used,
            fires_remaining=self.fires_remaining,
        )
        return augmented_info

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(
                f"invalid action {action!r}, expected 0 (WAIT) or 1 (FIRE)"
            )

        action_int = int(action)
        if action_int == FIRE:
            if self.fires_remaining == 0:
                raise ValueError("FIRE budget exhausted")
            self.fires_used += 1
            self.fires_remaining -= 1

        observation, reward, terminated, truncated, info = self.env.step(action_int)
        if (
            action_int == FIRE
            and self.fires_remaining == 0
            and not (terminated or truncated)
        ):
            terminated = True

        return (
            self._augment_observation(observation),
            float(reward),
            terminated,
            truncated,
            self._augment_info(info),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        self.fires_used = 0
        self.fires_remaining = self.max_fires
        observation, info = self.env.reset(seed=seed, options=options)
        return self._augment_observation(observation), self._augment_info(info)


class ObjectPositionMaskWrapper(_WrapperT):
    """Zero the object position slots of the 27-dim benchmark observation.

    The Issue #13 "blind" observation ablation: the six slots holding the
    GOLD / DIAMOND / ROCK x, y coordinates (``OBJECT_POSITION_INDICES``)
    are set to 0.0 in every returned observation, so the agent can no
    longer localize the objects while all other channels (hook state,
    radii, values, retract speeds, active flags, FIRE budget) stay
    identical to the "full" condition.

    This wrapper is a pure observation post-processor: rewards, episode-end
    flags, info mappings, action space and observation space bounds are
    propagated unchanged (0.0 is already a legal value because the original
    normalized x/y coordinates are non-negative). The inner environment's
    arrays are never modified in place; every returned observation is a
    fresh copy. The wrapped environment must already expose the
    27-dimensional FireBudgetWrapper observation.
    """

    env: gymnasium.Env[NDArray[np.float32], int]

    def __init__(self, env: gymnasium.Env[NDArray[np.float32], int]) -> None:
        super().__init__(env)
        inner_observation_space = env.observation_space
        if not isinstance(inner_observation_space, spaces.Box):
            raise TypeError(
                "ObjectPositionMaskWrapper requires a Box observation space"
            )
        if inner_observation_space.shape != (27,):
            raise ValueError(
                "ObjectPositionMaskWrapper requires a 27-dimensional "
                "observation space"
            )

        # Bounds are inherited unchanged: masking only writes 0.0, which the
        # inner space already allows for the non-negative x/y slots.
        self.observation_space = inner_observation_space

    def _mask_observation(
        self, observation: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        masked = np.array(observation, dtype=np.float32, copy=True)
        masked[list(OBJECT_POSITION_INDICES)] = 0.0
        return masked

    def step(
        self, action: int | np.integer[Any]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # No action validation here: the inner FireBudgetWrapper already
        # enforces the Discrete(2) contract for the whole chain.
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._mask_observation(observation), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        return self._mask_observation(observation), info
