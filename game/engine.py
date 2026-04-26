"""
game/engine.py
==============
GameEngine is the single source of truth for all Connect Four state.

Rules:
  - Only GameEngine can mutate the board.
  - Agents never hold a board reference. They query state and call make_move().
  - For lookahead, agents call get_next_state(col) which returns a brand-new
    GameEngine instance — the real game is never touched during search.

Board encoding:
  - 0  empty
  - 1  player 1  (human or agent 1)
  - 2  player 2  (human or agent 2)

Grid layout:
  - Row 0 is the TOP of the visual board.
  - Row 5 is the BOTTOM (pieces fall downward, so row 5 fills first).
  - Columns 0-6 left to right.
"""

import numpy as np
import copy
from typing import Optional


# ── Constants ────────────────────────────────────────────────────────────────

ROWS    = 6
COLS    = 7
EMPTY   = 0
P1      = 1
P2      = 2
WIN_LEN = 4   # pieces in a row needed to win


# ── GameEngine ───────────────────────────────────────────────────────────────

class GameEngine:
    """
    Encapsulates all Connect Four game state and rules.

    Public interface (for agents, UI, and experiment runner):
    ─────────────────────────────────────────────────────────
      State queries   → get_valid_moves, get_board_state, get_current_player,
                        is_terminal, get_winner, get_move_count
      Action          → make_move(col)          mutates THIS engine (real game)
      Simulation      → get_next_state(col)     returns a NEW engine (lookahead)
      Display helper  → __str__                 printable board for debugging
    """

    def __init__(self) -> None:
        self._board: np.ndarray = np.zeros((ROWS, COLS), dtype=np.int8)
        self._current_player: int = P1
        self._move_count: int = 0
        self._winner: Optional[int] = None
        self._terminal: bool = False

    # ── State queries ────────────────────────────────────────────────────────

    def get_valid_moves(self) -> list[int]:
        """
        Return list of column indices where a piece can be dropped.
        A column is valid when its top cell (row 0) is still empty.
        Returns empty list when the game is over.
        """
        if self._terminal:
            return []
        return [col for col in range(COLS) if self._board[0, col] == EMPTY]

    def get_board_state(self) -> np.ndarray:
        """
        Return a read-only copy of the board.
        Shape: (6, 7), dtype int8.  Values: 0=empty, 1=P1, 2=P2.
        """
        state = self._board.copy()
        state.flags.writeable = False
        return state

    def get_current_player(self) -> int:
        """Return the player whose turn it is (1 or 2)."""
        return self._current_player

    def get_opponent(self) -> int:
        """Return the player who is NOT currently moving."""
        return P2 if self._current_player == P1 else P1

    def is_terminal(self) -> bool:
        """True when the game is over (win or draw)."""
        return self._terminal

    def get_winner(self) -> Optional[int]:
        """
        Return winning player (1 or 2), or None if draw / game still ongoing.
        Always check is_terminal() first.
        """
        return self._winner

    def get_move_count(self) -> int:
        """Total number of moves made so far."""
        return self._move_count

    def is_valid_move(self, col: int) -> bool:
        """True when col is in bounds and not full."""
        return (0 <= col < COLS) and (self._board[0, col] == EMPTY) and (not self._terminal)

    # ── Action ───────────────────────────────────────────────────────────────

    def make_move(self, col: int) -> bool:
        """
        Drop the current player's piece into col.

        Returns True on success, False if the move is invalid (full column,
        out of bounds, or game already over).  On success the engine updates:
          - board array
          - current player (toggles)
          - move count
          - terminal flag and winner (if the move ends the game)
        """
        if not self.is_valid_move(col):
            return False

        row = self._find_landing_row(col)
        self._board[row, col] = self._current_player
        self._move_count += 1

        # Check win before toggling player so _winner reflects who just moved
        if self._check_win(self._current_player):
            self._winner = self._current_player
            self._terminal = True
        elif self._move_count == ROWS * COLS:
            # Board full, no winner → draw
            self._terminal = True

        self._current_player = P2 if self._current_player == P1 else P1
        return True

    # ── Simulation ───────────────────────────────────────────────────────────

    def get_next_state(self, col: int) -> "GameEngine":
        """
        Return a NEW GameEngine representing the board after dropping into col.

        This is the method agents use during search/rollout.  The current
        engine is never modified.  Raises ValueError on invalid col so that
        AI bugs surface immediately rather than silently producing wrong trees.
        """
        if not self.is_valid_move(col):
            raise ValueError(
                f"get_next_state called with invalid col={col} "
                f"(valid: {self.get_valid_moves()}, terminal: {self._terminal})"
            )
        next_engine = self._fast_copy()
        next_engine.make_move(col)
        return next_engine

    # ── Private helpers ──────────────────────────────────────────────────────

    def _find_landing_row(self, col: int) -> int:
        """
        Return the lowest empty row in col.
        Pieces 'fall' to the highest row index (row 5 = bottom of screen).
        """
        for row in range(ROWS - 1, -1, -1):
            if self._board[row, col] == EMPTY:
                return row
        # Should never reach here if is_valid_move() was checked first
        raise RuntimeError(f"Column {col} is full — this should not happen.")

    def _fast_copy(self) -> "GameEngine":
        """
        Lightweight copy for use in get_next_state().
        Copies only the fields that change during play.
        Much faster than copy.deepcopy() — critical for MCTS performance.
        """
        new = GameEngine.__new__(GameEngine)
        new._board          = self._board.copy()
        new._current_player = self._current_player
        new._move_count     = self._move_count
        new._winner         = self._winner
        new._terminal       = self._terminal
        return new

    def _check_win(self, player: int) -> bool:
        """
        Return True if `player` has four in a row anywhere on the board.
        Checks all four directions: horizontal, vertical, diagonal ↗, diagonal ↘.
        """
        b = self._board

        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                if b[r, c] == player and b[r, c+1] == player \
                        and b[r, c+2] == player and b[r, c+3] == player:
                    return True

        # Vertical
        for r in range(ROWS - 3):
            for c in range(COLS):
                if b[r, c] == player and b[r+1, c] == player \
                        and b[r+2, c] == player and b[r+3, c] == player:
                    return True

        # Diagonal ↘  (top-left to bottom-right)
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if b[r, c] == player and b[r+1, c+1] == player \
                        and b[r+2, c+2] == player and b[r+3, c+3] == player:
                    return True

        # Diagonal ↗  (bottom-left to top-right)
        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if b[r, c] == player and b[r-1, c+1] == player \
                        and b[r-2, c+2] == player and b[r-3, c+3] == player:
                    return True

        return False

    # ── Display ──────────────────────────────────────────────────────────────

    def __str__(self) -> str:
        """
        Printable board for CLI debugging.

        Example output:
          | . . . . . . . |
          | . . . . . . . |
          | . . . . . . . |
          | . . . 1 . . . |
          | . . . 2 1 . . |
          | . . 2 1 2 1 . |
            0 1 2 3 4 5 6
        """
        symbols = {EMPTY: ".", P1: "1", P2: "2"}
        rows = []
        for r in range(ROWS):
            row_str = " ".join(symbols[self._board[r, c]] for c in range(COLS))
            rows.append(f"  | {row_str} |")
        rows.append("    " + " ".join(str(c) for c in range(COLS)))

        status = ""
        if self._terminal:
            status = f"\n  Result: {'Draw' if self._winner is None else f'Player {self._winner} wins'}"
        else:
            status = f"\n  Next:   Player {self._current_player}  (move #{self._move_count + 1})"

        return "\n".join(rows) + status

    def __repr__(self) -> str:
        return (f"GameEngine(move={self._move_count}, "
                f"player={self._current_player}, "
                f"terminal={self._terminal}, "
                f"winner={self._winner})")