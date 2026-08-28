"""Pygame human renderer for the Gold Miner environment.

The renderer is a strict read-only view of :class:`GoldMinerEnv`: it only
reads public state (score, remaining time, hook, objects) and draws it.
It never mutates the environment and takes no part in collision, scoring
or timing. ``Clock.tick(60)`` only paces the on-screen playback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from gold_miner_sim.env import ANCHOR, HEIGHT, HOOK_RADIUS, WIDTH, ObjectType

if TYPE_CHECKING:
    from gold_miner_sim.env import GoldMinerEnv

_BACKGROUND_COLOR = (18, 18, 24)
_ROPE_COLOR = (210, 210, 210)
_HOOK_COLOR = (255, 255, 255)
_ANCHOR_COLOR = (255, 110, 110)
_TEXT_COLOR = (240, 240, 240)
_OBJECT_COLORS: dict[ObjectType, tuple[int, int, int]] = {
    ObjectType.GOLD: (255, 215, 0),  # yellow
    ObjectType.DIAMOND: (0, 255, 255),  # cyan
    ObjectType.ROCK: (130, 130, 130),  # gray
}


class HumanRenderer:
    """Draws the current environment state into a 900x600 pygame window."""

    def __init__(self, env: GoldMinerEnv) -> None:
        self.env: GoldMinerEnv = env
        self.closed: bool = False  # True after the user closes the window.
        self._pygame_quit: bool = False
        pygame.init()
        self._screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Gold Miner Sim")
        self._clock: pygame.time.Clock = pygame.time.Clock()
        self._font: pygame.font.Font = pygame.font.Font(None, 26)

    def render(self) -> None:
        """Draw one frame of the current env state.

        Pumps events; a QUIT event only sets :attr:`closed`, the caller
        decides when to stop stepping. Ends with ``Clock.tick(60)`` so a
        human demo plays back at roughly real-time speed.
        """
        if self.closed or self._pygame_quit:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed = True

        self._draw()
        pygame.display.flip()
        self._clock.tick(60)

    def close(self) -> None:
        """Quit pygame. Idempotent."""
        if self._pygame_quit:
            return
        self._pygame_quit = True
        pygame.display.quit()
        pygame.quit()

    def _draw(self) -> None:
        """Draw rope, hook, objects and HUD for the current tick."""
        self._screen.fill(_BACKGROUND_COLOR)
        tip_x, tip_y = self.env.hook_tip

        pygame.draw.line(self._screen, _ROPE_COLOR, ANCHOR, (tip_x, tip_y), 2)
        pygame.draw.circle(self._screen, _ANCHOR_COLOR, ANCHOR, 5)
        pygame.draw.circle(
            self._screen, _HOOK_COLOR, (tip_x, tip_y), int(HOOK_RADIUS)
        )
        # Active objects only. An attached object is still active and its
        # center already follows the hook tip, so it is drawn here too.
        for obj in self.env.objects:
            if obj.active:
                pygame.draw.circle(
                    self._screen,
                    _OBJECT_COLORS[obj.type],
                    (obj.x, obj.y),
                    int(obj.radius),
                )

        hud_lines = (
            f"Score: {self.env.score:g}",
            f"Time left: {self.env.remaining_time:.1f} s",
            f"State: {self.env.hook_state.name}",
            f"Angle: {self.env.angle:.1f} deg",
        )
        for row, text in enumerate(hud_lines):
            surface = self._font.render(text, True, _TEXT_COLOR)
            self._screen.blit(surface, (10, 10 + row * 26))
