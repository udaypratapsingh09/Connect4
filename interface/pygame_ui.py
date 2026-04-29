"""
interface/pygame_ui.py
======================
Pygame UI for Connect Four — includes a full GUI menu screen.

Flow:
  main.py  →  MenuScreen.run()  →  returns GameConfig
           →  PygameUI(config).run()  →  game loop
           →  on 'M' key / result screen → back to MenuScreen

No CLI arguments needed. Everything is selected in the window.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import pygame

from game import GameEngine, P1, P2, ROWS, COLS
from agents import MinimaxAgent, MCTSAgent


# ── Colours ───────────────────────────────────────────────────────────────────

C_BG          = ( 15,  30,  60)
C_BOARD       = ( 25,  55, 115)
C_EMPTY       = ( 18,  40,  85)
C_P1          = (255, 220,   0)   # yellow
C_P2          = (220,  50,  50)   # red
C_HOVER       = (255, 255, 255, 40)
C_TEXT        = (235, 235, 235)
C_MUTED       = (230, 150, 180)
C_BANNER_WIN  = ( 30, 160,  90)
C_BANNER_DRAW = ( 90,  90, 110)
C_BORDER      = (200, 200, 255)
C_BTN         = ( 35,  65, 130)
C_BTN_HOV     = ( 55,  95, 180)
C_BTN_SEL     = ( 60, 130, 220)
C_SLIDER_TRK  = ( 40,  60, 110)
C_SLIDER_FILL = ( 80, 140, 220)
C_SLIDER_KNB  = (200, 220, 255)


# ── Layout ────────────────────────────────────────────────────────────────────

CELL      = 100
RADIUS    = 40
HEADER_H  = 90
MARGIN    = 20
BOARD_W   = COLS * CELL
BOARD_H   = ROWS * CELL
WIN_W     = BOARD_W + 2 * MARGIN
WIN_H     = HEADER_H + BOARD_H + MARGIN
AI_DELAY  = 250   # ms delay before every AI move
FPS       = 60


# ── GameConfig dataclass ──────────────────────────────────────────────────────

@dataclass
class GameConfig:
    mode:         str    # 'human_vs_minimax' | 'human_vs_mcts' | 'ai_vs_ai'
    depth:        int    # minimax depth
    simulations:  int    # mcts iteration count

    def build_agents(self):
        """Return (agent1, agent2, label1, label2) based on config."""
        if self.mode == "human_vs_minimax":
            return (
                None,
                MinimaxAgent(player=P2, depth=self.depth),
                "You",
                f"Minimax  d={self.depth}",
            )
        elif self.mode == "human_vs_mcts":
            return (
                None,
                MCTSAgent(player=P2, iterations=self.simulations),
                "You",
                f"MCTS  {self.simulations} sims",
            )
        else:  # ai_vs_ai
            return (
                MinimaxAgent(player=P1, depth=self.depth),
                MCTSAgent(player=P2, iterations=self.simulations),
                f"Minimax  d={self.depth}",
                f"MCTS  {self.simulations} sims",
            )


# ── Utility helpers ───────────────────────────────────────────────────────────

def _cell_centre(row: int, col: int) -> tuple[int, int]:
    return MARGIN + col * CELL + CELL // 2, HEADER_H + row * CELL + CELL // 2

def _col_rect(col: int) -> pygame.Rect:
    return pygame.Rect(MARGIN + col * CELL, 0, CELL, WIN_H)

def _draw_rounded_rect(surf, colour, rect, radius=10, border=0, border_colour=None):
    pygame.draw.rect(surf, colour, rect, border_radius=radius)
    if border and border_colour:
        pygame.draw.rect(surf, border_colour, rect, width=border, border_radius=radius)


# ══════════════════════════════════════════════════════════════════════════════
#  MenuScreen
# ══════════════════════════════════════════════════════════════════════════════

class MenuScreen:
    """
    Full GUI menu for mode selection, depth, and simulation count.

    Sections
    ────────
    1. Title
    2. Mode buttons  (Human vs Minimax | Human vs MCTS | AI vs AI)
    3. Minimax depth slider        (shown when relevant mode selected)
    4. MCTS simulations slider     (shown when relevant mode selected)
    5. Play button
    """

    DEPTH_MIN, DEPTH_MAX   = 2, 7
    SIMS_OPTIONS           = [100, 200, 500, 1000, 2000, 3000]

    def __init__(self, screen: pygame.Surface, fonts: dict) -> None:
        self.screen = screen
        self.fonts  = fonts

        # State
        self._mode  = "human_vs_minimax"
        self._depth = 4
        self._sims  = 1000   # index into SIMS_OPTIONS
        self._sims_idx = self.SIMS_OPTIONS.index(self._sims)

        # Slider drag state
        self._dragging_depth = False
        self._dragging_sims  = False

        # Button rects — computed in _layout()
        self._btn_modes: dict[str, pygame.Rect] = {}
        self._slider_depth_rect: pygame.Rect    = pygame.Rect(0, 0, 0, 0)
        self._slider_sims_rect:  pygame.Rect    = pygame.Rect(0, 0, 0, 0)
        self._play_rect: pygame.Rect            = pygame.Rect(0, 0, 0, 0)
        self._layout()

    def _layout(self) -> None:
        cx = WIN_W // 2
        btn_w, btn_h = 220, 48
        gap          = 14

        modes = ["human_vs_minimax", "human_vs_mcts", "ai_vs_ai"]
        labels = ["Human  vs  Minimax", "Human  vs  MCTS", "AI  vs  AI"]
        total_w = len(modes) * btn_w + (len(modes) - 1) * gap
        start_x = cx - total_w // 2
        btn_y   = 210

        for i, (m, _) in enumerate(zip(modes, labels)):
            self._btn_modes[m] = pygame.Rect(
                start_x + i * (btn_w + gap), btn_y, btn_w, btn_h
            )

        # Sliders
        sl_w = 340
        sl_h = 8
        sl_x = cx - sl_w // 2

        self._slider_depth_rect = pygame.Rect(sl_x, 340, sl_w, sl_h)
        self._slider_sims_rect  = pygame.Rect(sl_x, 440, sl_w, sl_h)

        # Play button
        self._play_rect = pygame.Rect(cx - 110, WIN_H - 100, 220, 54)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> GameConfig:
        """Blocking loop — returns a GameConfig when the user clicks Play."""
        clock = pygame.time.Clock()
        while True:
            clock.tick(FPS)
            mx, my = pygame.mouse.get_pos()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    result = self._on_click(mx, my)
                    if result is not None:
                        return result

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self._dragging_depth = False
                    self._dragging_sims  = False

                if event.type == pygame.MOUSEMOTION:
                    if self._dragging_depth:
                        self._update_slider_depth(mx)
                    if self._dragging_sims:
                        self._update_slider_sims(mx)

            self._draw(mx, my)

    # ── Event helpers ─────────────────────────────────────────────────────────

    def _on_click(self, mx: int, my: int) -> Optional[GameConfig]:
        # Mode buttons
        for mode, rect in self._btn_modes.items():
            if rect.collidepoint(mx, my):
                self._mode = mode
                return None

        # Depth slider knob
        if self._depth_visible():
            kx, ky = self._depth_knob_pos()
            if abs(mx - kx) <= 14 and abs(my - ky) <= 14:
                self._dragging_depth = True
                return None

        # Sims slider knob
        if self._sims_visible():
            kx, ky = self._sims_knob_pos()
            if abs(mx - kx) <= 14 and abs(my - ky) <= 14:
                self._dragging_sims = True
                return None

        # Play button
        if self._play_rect.collidepoint(mx, my):
            return GameConfig(
                mode        = self._mode,
                depth       = self._depth,
                simulations = self.SIMS_OPTIONS[self._sims_idx],
            )

        return None

    def _update_slider_depth(self, mx: int) -> None:
        r = self._slider_depth_rect
        t = max(0.0, min(1.0, (mx - r.left) / r.width))
        self._depth = self.DEPTH_MIN + round(t * (self.DEPTH_MAX - self.DEPTH_MIN))

    def _update_slider_sims(self, mx: int) -> None:
        r   = self._slider_sims_rect
        t   = max(0.0, min(1.0, (mx - r.left) / r.width))
        idx = round(t * (len(self.SIMS_OPTIONS) - 1))
        self._sims_idx = idx

    # ── Visibility rules ──────────────────────────────────────────────────────

    def _depth_visible(self) -> bool:
        return self._mode in ("human_vs_minimax", "ai_vs_ai")

    def _sims_visible(self) -> bool:
        return self._mode in ("human_vs_mcts", "ai_vs_ai")

    # ── Knob positions ────────────────────────────────────────────────────────

    def _depth_knob_pos(self) -> tuple[int, int]:
        r = self._slider_depth_rect
        t = (self._depth - self.DEPTH_MIN) / (self.DEPTH_MAX - self.DEPTH_MIN)
        return int(r.left + t * r.width), r.centery

    def _sims_knob_pos(self) -> tuple[int, int]:
        r = self._slider_sims_rect
        t = self._sims_idx / (len(self.SIMS_OPTIONS) - 1)
        return int(r.left + t * r.width), r.centery

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self, mx: int, my: int) -> None:
        self.screen.fill(C_BG)
        self._draw_title()
        self._draw_mode_buttons(mx, my)
        if self._depth_visible():
            self._draw_slider(
                self._slider_depth_rect,
                self._depth_knob_pos(),
                f"Minimax depth:  {self._depth}",
                self._depth_visible(),
            )
        if self._sims_visible():
            self._draw_slider(
                self._slider_sims_rect,
                self._sims_knob_pos(),
                f"MCTS simulations:  {self.SIMS_OPTIONS[self._sims_idx]:,}",
                self._sims_visible(),
            )
        self._draw_play_button(mx, my)
        pygame.display.flip()

    def _draw_title(self) -> None:
        cx = WIN_W // 2

        title = self.fonts["xl"].render("Connect Four", True, C_TEXT)
        self.screen.blit(title, (cx - title.get_width() // 2, 40))

        sub = self.fonts["sm"].render("Minimax  ×  MCTS", True, C_MUTED)
        self.screen.blit(sub, (cx - sub.get_width() // 2, 100))

        # Yellow / red dot decorations beside title
        pygame.draw.circle(self.screen, C_P1, (cx - title.get_width() // 2 - 20, 62), 10)
        pygame.draw.circle(self.screen, C_P2, (cx + title.get_width() // 2 + 20, 62), 10)

        # Divider
        pygame.draw.line(self.screen, C_BTN, (MARGIN * 2, 145), (WIN_W - MARGIN * 2, 145), 1)

        mode_label = self.fonts["md"].render("Select mode", True, C_MUTED)
        self.screen.blit(mode_label, (cx - mode_label.get_width() // 2, 158))

    def _draw_mode_buttons(self, mx: int, my: int) -> None:
        labels = {
            "human_vs_minimax": "Human  vs  Minimax",
            "human_vs_mcts":    "Human  vs  MCTS",
            "ai_vs_ai":         "AI  vs  AI",
        }
        for mode, rect in self._btn_modes.items():
            selected = (mode == self._mode)
            hovered  = rect.collidepoint(mx, my) and not selected

            if selected:
                bg = C_BTN_SEL
            elif hovered:
                bg = C_BTN_HOV
            else:
                bg = C_BTN

            _draw_rounded_rect(self.screen, bg, rect, radius=8,
                               border=2 if selected else 1,
                               border_colour=C_BORDER if selected else C_BTN_HOV)

            txt = self.fonts["md"].render(labels[mode], True, C_TEXT)
            self.screen.blit(
                txt,
                (rect.centerx - txt.get_width() // 2,
                 rect.centery - txt.get_height() // 2),
            )

    def _draw_slider(
        self,
        track: pygame.Rect,
        knob_pos: tuple[int, int],
        label: str,
        visible: bool,
    ) -> None:
        if not visible:
            return

        cx  = WIN_W // 2
        kx, ky = knob_pos

        # Label above slider
        lbl = self.fonts["sm"].render(label, True, C_TEXT)
        self.screen.blit(lbl, (cx - lbl.get_width() // 2, track.top - 30))

        # Track background
        pygame.draw.rect(self.screen, C_SLIDER_TRK,
                         pygame.Rect(track.left, track.top - 2, track.width, track.height + 4),
                         border_radius=4)

        # Filled portion
        filled_w = max(0, kx - track.left)
        pygame.draw.rect(self.screen, C_SLIDER_FILL,
                         pygame.Rect(track.left, track.top - 2, filled_w, track.height + 4),
                         border_radius=4)

        # Knob
        pygame.draw.circle(self.screen, C_SLIDER_KNB, (kx, ky + track.height // 2), 11)
        pygame.draw.circle(self.screen, C_BORDER,     (kx, ky + track.height // 2), 11, width=2)

    def _draw_play_button(self, mx: int, my: int) -> None:
        hovered = self._play_rect.collidepoint(mx, my)
        bg      = C_BTN_SEL if hovered else C_BTN_HOV
        _draw_rounded_rect(self.screen, bg, self._play_rect, radius=10,
                           border=2, border_colour=C_BORDER)

        txt = self.fonts["lg"].render("▶  Play", True, C_TEXT)
        self.screen.blit(
            txt,
            (self._play_rect.centerx - txt.get_width() // 2,
             self._play_rect.centery - txt.get_height() // 2),
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PygameUI  (game screen)
# ══════════════════════════════════════════════════════════════════════════════

class PygameUI:
    """
    Manages the game window and loop for one game session.

    Returns 'menu' when the player wants to go back to the menu,
    'quit' when they close the window.
    """

    def __init__(self, screen: pygame.Surface, fonts: dict, config: GameConfig) -> None:
        self.screen = screen
        self.fonts  = fonts
        self.config = config

        agent1, agent2, label1, label2 = config.build_agents()
        self.agent1  = agent1
        self.agent2  = agent2
        self.label   = {P1: label1, P2: label2}
        self.colour  = {P1: C_P1, P2: C_P2}

        self._hover_surf = pygame.Surface((CELL, WIN_H), pygame.SRCALPHA)
        self._hover_surf.fill(C_HOVER)

        self._engine:    GameEngine    = GameEngine()
        self._hover_col: Optional[int] = None
        self._waiting:   bool          = False
        self._human_just_moved: bool   = False
        self._result_logged:    bool   = False
        self._clock = pygame.time.Clock()
        self._print_game_header()

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> str:
        """Game loop. Returns 'menu' or 'quit'."""
        while True:
            self._clock.tick(FPS)
            result = self._handle_events()
            if result in ("menu", "quit"):
                return result

            if not self._engine.is_terminal():
                self._maybe_trigger_ai()

            self._draw()

    # ── State ─────────────────────────────────────────────────────────────────

    def _current_agent(self):
        p = self._engine.get_current_player()
        return self.agent1 if p == P1 else self.agent2

    def _is_human_turn(self) -> bool:
        return self._current_agent() is None

    # ── Events ────────────────────────────────────────────────────────────────

    def _handle_events(self) -> Optional[str]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_r:
                    self._engine           = GameEngine()
                    self._hover_col        = None
                    self._waiting          = False
                    self._human_just_moved = False
                    self._result_logged    = False
                    self._print_game_header()
                    # Rebuild agents so player tokens reset correctly
                    a1, a2, _, _ = self.config.build_agents()
                    self.agent1, self.agent2 = a1, a2
                if event.key == pygame.K_m:
                    return "menu"

            if self._engine.is_terminal():
                continue

            if event.type == pygame.MOUSEMOTION and self._is_human_turn():
                col = self._x_to_col(event.pos[0])
                self._hover_col = col

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self._is_human_turn() and not self._waiting:
                    col = self._x_to_col(event.pos[0])
                    if col is not None and self._engine.is_valid_move(col):
                        self._engine.make_move(col)
                        move_num = self._engine.get_move_count()
                        player   = self._engine.get_opponent()
                        print(f"  Move {move_num:>2} | {self.label[player]:<20} | col {col}")
                        self._log_result_if_terminal()
                        self._hover_col        = None
                        self._human_just_moved = True

        return None

    # ── AI ────────────────────────────────────────────────────────────────────

    def _maybe_trigger_ai(self) -> None:
        if self._is_human_turn() or self._waiting:
            return

        # Let the board redraw with the human's piece before AI starts
        if self._human_just_moved:
            self._human_just_moved = False
            return

        self._draw_thinking_overlay()
        pygame.display.flip()

        t_start = pygame.time.get_ticks()
        col     = self._current_agent().choose_move(self._engine)
        elapsed = pygame.time.get_ticks() - t_start

        # Guarantee overlay stays visible for at least AI_DELAY ms
        leftover = AI_DELAY - elapsed
        if leftover > 0:
            pygame.time.wait(leftover)

        pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])
        self._engine.make_move(col)
        move_num = self._engine.get_move_count()
        player   = self._engine.get_opponent()  # who just moved
        print(f"  Move {move_num:>2} | {self.label[player]:<20} | col {col}  ({elapsed}ms)")
        self._log_result_if_terminal()

        # Brief pause after piece lands so the move registers visually
        self._draw()
        pygame.time.wait(150)
        pygame.event.clear([pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP])

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        self.screen.fill(C_BG)
        self._draw_header()
        self._draw_board()
        if self._hover_col is not None and self._is_human_turn():
            self._draw_hover(self._hover_col)
        self._draw_pieces()
        if self._engine.is_terminal():
            self._draw_result_banner()
        pygame.display.flip()

    def _draw_header(self) -> None:
        if self._engine.is_terminal():
            return

        player = self._engine.get_current_player()
        colour = self.colour[player]
        name   = self.label[player]

        pygame.draw.circle(self.screen, colour, (30, HEADER_H // 2), 14)
        lbl = self.fonts["md"].render(f"{name}'s turn", True, C_TEXT)
        self.screen.blit(lbl, (52, HEADER_H // 2 - lbl.get_height() // 2))

        move_txt = self.fonts["sm"].render(
            f"Move {self._engine.get_move_count() + 1}", True, C_MUTED
        )
        self.screen.blit(move_txt, (WIN_W - move_txt.get_width() - 16, 12))

        hint = self.fonts["sm"].render("R=restart  M=menu  ESC=quit", True, C_MUTED)
        self.screen.blit(hint, (WIN_W - hint.get_width() - 16, HEADER_H - 28))

    def _draw_board(self) -> None:
        _draw_rounded_rect(
            self.screen, C_BOARD,
            pygame.Rect(MARGIN, HEADER_H, BOARD_W, BOARD_H),
            radius=10,
        )
        for r in range(ROWS):
            for c in range(COLS):
                cx, cy = _cell_centre(r, c)
                pygame.draw.circle(self.screen, C_EMPTY, (cx, cy), RADIUS)

    def _draw_hover(self, col: int) -> None:
        self.screen.blit(self._hover_surf, _col_rect(col).topleft)
        cx = MARGIN + col * CELL + CELL // 2
        cy = HEADER_H // 2
        player = self._engine.get_current_player()
        pygame.draw.circle(self.screen, self.colour[player], (cx, cy), RADIUS // 2)

    def _draw_pieces(self) -> None:
        board = self._engine.get_board_state()
        for r in range(ROWS):
            for c in range(COLS):
                val = board[r, c]
                if val == 0:
                    continue
                cx, cy = _cell_centre(r, c)
                colour = C_P1 if val == P1 else C_P2
                pygame.draw.circle(self.screen, colour, (cx, cy), RADIUS)
                pygame.draw.circle(
                    self.screen,
                    tuple(min(255, v + 60) for v in colour),
                    (cx - RADIUS // 5, cy - RADIUS // 5),
                    RADIUS // 5,
                )

    def _draw_thinking_overlay(self) -> None:
        player = self._engine.get_current_player()
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))
        txt = self.fonts["lg"].render(f"{self.label[player]} is thinking…", True, self.colour[player])
        self.screen.blit(txt, (WIN_W // 2 - txt.get_width() // 2, WIN_H // 2 - txt.get_height() // 2))

    def _draw_result_banner(self) -> None:
        winner = self._engine.get_winner()
        if winner is not None:
            message = f"{self.label[winner]} wins!"
            colour  = self.colour[winner]
            bg      = C_BANNER_WIN
        else:
            message = "It's a draw!"
            colour  = C_TEXT
            bg      = C_BANNER_DRAW

        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        banner_h = 170
        banner_y = WIN_H // 2 - banner_h // 2
        banner   = pygame.Rect(MARGIN, banner_y, WIN_W - 2 * MARGIN, banner_h)
        _draw_rounded_rect(self.screen, bg, banner, radius=12, border=2, border_colour=C_BORDER)

        win_surf = self.fonts["lg"].render(message, True, colour)
        self.screen.blit(win_surf, (WIN_W // 2 - win_surf.get_width() // 2, banner_y + 28))

        move_surf = self.fonts["sm"].render(
            f"Moves played: {self._engine.get_move_count()}", True, C_TEXT
        )
        self.screen.blit(move_surf, (WIN_W // 2 - move_surf.get_width() // 2, banner_y + 80))

        sub = self.fonts["sm"].render("R = play again    M = menu    ESC = quit", True, C_MUTED)
        self.screen.blit(sub, (WIN_W // 2 - sub.get_width() // 2, banner_y + 115))

    def _log_result_if_terminal(self) -> None:
        if not self._engine.is_terminal() or self._result_logged:
            return
        winner = self._engine.get_winner()
        print(f"  {'─'*48}")
        if winner is not None:
            print(f"  Result : {self.label[winner]} wins!  (moves: {self._engine.get_move_count()})")
        else:
            print(f"  Result : Draw  (moves: {self._engine.get_move_count()})")
        print(f"  {'─'*48}")
        self._result_logged = True

    def _print_game_header(self) -> None:
        print(f"\n  {'─'*48}")
        print(f"  {self.label[P1]} (P1)  vs  {self.label[P2]} (P2)")
        print(f"  {'─'*48}")
        print(f"  {'Move':>6} | {'Player':<20} | {'Col'}  (think time)")
        print(f"  {'─'*6}-+-{'─'*20}-+-{'─'*18}")

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _x_to_col(x: int) -> Optional[int]:
        col = (x - MARGIN) // CELL
        return col if 0 <= col < COLS else None