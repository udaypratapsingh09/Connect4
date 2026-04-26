"""
main.py
=======
Entry point — opens the Pygame window and runs the menu/game loop.

Usage:
  python main.py

No CLI arguments needed. Everything is selected in the GUI.
"""

import sys
import pygame
from interface.pygame_ui import MenuScreen, PygameUI, WIN_W, WIN_H


def _build_fonts() -> dict:
    return {
        "xl": pygame.font.SysFont("segoeui", 46, bold=True),
        "lg": pygame.font.SysFont("segoeui", 34, bold=True),
        "md": pygame.font.SysFont("segoeui", 26),
        "sm": pygame.font.SysFont("segoeui", 20),
    }


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Connect Four — Minimax vs MCTS")
    fonts = _build_fonts()

    while True:
        # ── Menu ──────────────────────────────────────────────────────────────
        config = MenuScreen(screen, fonts).run()

        # ── Game ──────────────────────────────────────────────────────────────
        result = PygameUI(screen, fonts, config).run()

        if result == "quit":
            break
        # result == "menu"  →  loop back to MenuScreen

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()