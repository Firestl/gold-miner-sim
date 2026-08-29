"""公开接口测试（仅测 API 表层，不校验像素）。

覆盖：step 非法动作校验及 Python/NumPy 整型兼容、human 模式自动渲染、
包根导出、window_closed 属性、无头模式不创建渲染器。
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

import gold_miner_sim.renderer
from gold_miner_sim import FIRE, WAIT, GoldMinerEnv, HookState


class FakeRenderer:
    """HumanRenderer 替身，仅计数 render 调用次数。"""

    instances: ClassVar[list[FakeRenderer]] = []

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
    """用 FakeRenderer 替换真实 HumanRenderer，避免初始化 pygame/SDL。

    目的：验证 human 模式自动渲染逻辑。
    输入：无
    输出：返回 FakeRenderer 类型并完成 monkeypatch。
    """

    FakeRenderer.instances = []
    monkeypatch.setattr(gold_miner_sim.renderer, "HumanRenderer", FakeRenderer)
    return FakeRenderer


def test_step_rejects_invalid_actions() -> None:
    """目的：非法动作应抛 ValueError 且不推进仿真。

    输入：-1、2、100 三个越界动作。
    输出：均抛 ValueError；angle 保持 -70°，状态保持 SWINGING。
    """
    env = GoldMinerEnv()
    env.reset()

    for bad_action in (-1, 2, 100):
        with pytest.raises(ValueError):
            env.step(bad_action)

    assert env.angle == pytest.approx(-70.0)
    assert env.hook_state is HookState.SWINGING


def test_step_accepts_python_and_numpy_int_actions() -> None:
    """目的：验证兼容 Python int 与 NumPy 整型动作。

    输入：0、1、np.int64(0)、np.int64(1)。
    输出：均返回五元组 (obs, reward, terminated, truncated, info)，obs.shape=(26,)。
    """
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
# human 模式自动渲染
# ---------------------------------------------------------------------------
def test_human_mode_auto_renders_on_reset_and_step(
    fake_renderer: type[FakeRenderer],
) -> None:
    """目的：human 模式下 reset/step 自动触发渲染。

    输入：render_mode="human" 的环境，依次调用 reset、step(WAIT)。
    输出：reset 创建 1 个渲染器且 render_calls≥1，step 后 render_calls +1。
    """
    env = GoldMinerEnv(render_mode="human")

    env.reset()
    assert len(fake_renderer.instances) == 1
    renderer = fake_renderer.instances[0]
    assert renderer.render_calls >= 1  # reset 时渲染首帧

    calls_before = renderer.render_calls
    env.step(WAIT)
    assert renderer.render_calls == calls_before + 1  # step 时自动补帧

    env.close()


# ---------------------------------------------------------------------------
# 包根导出
# ---------------------------------------------------------------------------
def test_package_root_exports_actions() -> None:
    """目的：校验包根正确导出常量与类。

    输入：无。
    输出：WAIT==0、FIRE==1，GoldMinerEnv / HookState 可导入。
    """
    assert WAIT == 0
    assert FIRE == 1
    assert GoldMinerEnv is not None
    assert HookState is not None


# ---------------------------------------------------------------------------
# window_closed 属性
# ---------------------------------------------------------------------------
def test_window_closed_headless_is_always_false() -> None:
    """目的：无头模式下 window_closed 恒为 False。

    输入：默认环境，连续 step(WAIT) 10 次。
    输出：每次 window_closed 均为 False。
    """
    env = GoldMinerEnv()
    env.reset()
    for _ in range(10):
        env.step(WAIT)
        assert env.window_closed is False


def test_window_closed_reflects_renderer_state(
    fake_renderer: type[FakeRenderer],
) -> None:
    """目的：human 模式下 window_closed 跟随渲染器 closed 状态。

    输入：human 环境，手动置 renderer.closed=True。
    输出：置前为 False，置后为 True。
    """
    env = GoldMinerEnv(render_mode="human")
    env.reset()
    renderer = fake_renderer.instances[0]
    assert env.window_closed is False
    renderer.closed = True
    assert env.window_closed is True
    env.close()


# ---------------------------------------------------------------------------
# 无头模式不触及渲染器
# ---------------------------------------------------------------------------
def test_headless_step_never_creates_renderer() -> None:
    """目的：无头模式全程不实例化渲染器。

    输入：默认环境，连续 step(WAIT) 100 次。
    输出：env._renderer 始终为 None，truncated 保持 False。
    """
    env = GoldMinerEnv()
    env.reset()
    for _ in range(100):
        _, _, _, truncated, _ = env.step(WAIT)
        assert env._renderer is None
    assert truncated is False
