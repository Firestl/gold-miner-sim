"""Lightweight public-interface tests (PR #2 review round 3).

Interface surface only, no pixel assertions:

- ``step()`` rejects actions outside ``action_space`` with ``ValueError``
  and accepts both Python ints and NumPy integers;
- ``render_mode="human"`` auto-renders on ``reset()`` and ``step()``
  (verified via a fake renderer, so pygame/SDL is never initialized);
- ``WAIT``/``FIRE`` are importable from the package root;
- the public ``window_closed`` property;
- headless mode never instantiates a renderer.
"""

from __future__ import annotations

import gold_miner_sim.renderer
import numpy as np
import pytest
from gold_miner_sim import FIRE, WAIT, GoldMinerEnv, HookState


class FakeRenderer:
    """Drop-in stand-in for HumanRenderer that only counts render calls."""

    instances: list["FakeRenderer"] = []

    def __init__(self, env: GoldMinerEnv) -> None:
        self.env: GoldMinerEnv = env
        self.render_calls: int = 0
        self.closed: bool = False
        FakeRenderer.instances.append(self)

    def render(self) -> None:
        self.render_calls += 1

    def close(self) -> None:
        pass


@pytest.fixture
def fake_renderer(monkeypatch: pytest.MonkeyPatch) -> type[FakeRenderer]:
    """Patch HumanRenderer so human mode never touches real pygame/SDL.

    ``GoldMinerEnv.render()`` imports the renderer lazily inside the
    function body, so patching the module attribute takes effect.
    """
    FakeRenderer.instances = []
    monkeypatch.setattr(gold_miner_sim.renderer, "HumanRenderer", FakeRenderer)
    return FakeRenderer


# ---------------------------------------------------------------------------
# step() action validation
# ---------------------------------------------------------------------------
def test_step_rejects_invalid_actions() -> None:
    env = GoldMinerEnv()
    env.reset()

    for bad_action in (-1, 2, 100):
        with pytest.raises(ValueError):
            env.step(bad_action)

    # The failed steps must not have advanced the simulation (validation
    # happens at the top of step(), before any physics).
    assert env.angle == pytest.approx(-70.0)
    assert env.hook_state is HookState.SWINGING


def test_step_accepts_python_and_numpy_int_actions() -> None:
    env = GoldMinerEnv()
    env.reset()

    for action in (0, 1, np.int64(0), np.int64(1)):
        result = env.step(action)
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (26,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool) and isinstance(truncated, bool)
        assert "hook_state" in info


# ---------------------------------------------------------------------------
# human mode auto-render (fake renderer)
# ---------------------------------------------------------------------------
def test_human_mode_auto_renders_on_reset_and_step(
    fake_renderer: type[FakeRenderer],
) -> None:
    env = GoldMinerEnv(render_mode="human")

    env.reset()
    assert len(fake_renderer.instances) == 1
    renderer = fake_renderer.instances[0]
    assert renderer.render_calls >= 1  # first frame on reset

    calls_before = renderer.render_calls
    env.step(WAIT)
    assert renderer.render_calls == calls_before + 1  # auto frame on step

    env.close()


# ---------------------------------------------------------------------------
# package root exports
# ---------------------------------------------------------------------------
def test_package_root_exports_actions() -> None:
    assert WAIT == 0
    assert FIRE == 1
    assert GoldMinerEnv is not None
    assert HookState is not None


# ---------------------------------------------------------------------------
# window_closed property
# ---------------------------------------------------------------------------
def test_window_closed_headless_is_always_false() -> None:
    env = GoldMinerEnv()
    env.reset()
    for _ in range(10):
        env.step(WAIT)
        assert env.window_closed is False


def test_window_closed_reflects_renderer_state(
    fake_renderer: type[FakeRenderer],
) -> None:
    env = GoldMinerEnv(render_mode="human")
    env.reset()
    renderer = fake_renderer.instances[0]
    assert env.window_closed is False
    renderer.closed = True
    assert env.window_closed is True
    env.close()


# ---------------------------------------------------------------------------
# headless never touches the renderer
# ---------------------------------------------------------------------------
def test_headless_step_never_creates_renderer() -> None:
    env = GoldMinerEnv()
    env.reset()
    for _ in range(100):
        _, _, _, truncated, _ = env.step(WAIT)
        assert env._renderer is None
    assert truncated is False
