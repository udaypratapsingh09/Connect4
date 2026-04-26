"""
agents/minimax_agent.py
=======================
Minimax agent with alpha-beta pruning for Connect Four.

Design contract (matches GameEngine's agent interface):
  - Never holds a reference to the real board.
  - Only calls engine.get_valid_moves(), engine.get_board_state(),
    engine.is_terminal(), engine.get_winner(), engine.get_next_state().
  - choose_move(engine) → int  (the column to play)

Heuristic:
  Evaluates non-terminal positions by scoring all windows of 4 cells
  (horizontal, vertical, diagonal) on the board.  Center-column occupancy
  gets a bonus because the centre gives the most winning lines.

Constants you can tune:
  DEFAULT_DEPTH   – search depth (4–6 is practical; 7+ is slow without
                    further optimisation)
  WIN_SCORE       – value assigned to a terminal win
  CENTER_WEIGHT   – bonus per piece in the centre column
  THREE_WEIGHT    – bonus per 3-in-a-row window (opponent's 3s are penalised)
  TWO_WEIGHT      – bonus per 2-in-a-row window
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Import the engine constants so this file stays in sync automatically.
from game.engine import GameEngine, ROWS, COLS, EMPTY, P1, P2, WIN_LEN


# ── Tuneable constants ────────────────────────────────────────────────────────

DEFAULT_DEPTH  = 5       # ply depth for minimax search
WIN_SCORE      = 100_000 # effectively ±infinity for terminal states
CENTER_COL     = COLS // 2
CENTER_WEIGHT  = 3
THREE_WEIGHT   = 5
TWO_WEIGHT     = 2


# ── MinimaxAgent ─────────────────────────────────────────────────────────────

class MinimaxAgent:
    """
    Minimax agent with alpha-beta pruning.

    Parameters
    ----------
    player : int
        Which player this agent controls (P1=1 or P2=2).
    depth : int
        Search depth in plies.  Default is DEFAULT_DEPTH.
    """

    def __init__(self, player: int, depth: int = DEFAULT_DEPTH) -> None:
        if player not in (P1, P2):
            raise ValueError(f"player must be {P1} or {P2}, got {player!r}")
        self.player  = player
        self.opponent = P2 if player == P1 else P1
        self.depth   = depth

    # ── Public interface ──────────────────────────────────────────────────────

    def choose_move(self, engine: GameEngine) -> int:
        """
        Return the column index this agent wants to play.

        Raises RuntimeError if called on a terminal state (no moves available).
        """
        valid_moves = engine.get_valid_moves()
        if not valid_moves:
            raise RuntimeError("choose_move called on a terminal game state.")

        # Immediate win / block shortcuts (depth-0 speed wins)
        instant = self._instant_move(engine, valid_moves)
        if instant is not None:
            return instant

        best_col   = valid_moves[len(valid_moves) // 2]  # default: centre-ish
        best_score = -math.inf

        alpha, beta = -math.inf, math.inf

        for col in self._move_order(valid_moves):
            next_state = engine.get_next_state(col)
            score = self._minimax(
                next_state,
                depth    = self.depth - 1,
                alpha    = alpha,
                beta     = beta,
                maximise = False,       # opponent moves next
            )
            if score > best_score:
                best_score = score
                best_col   = col
            alpha = max(alpha, best_score)
            # No beta cut at root; we need to check every move.

        return best_col

    # ── Minimax (private) ─────────────────────────────────────────────────────

    def _minimax(
        self,
        engine:   GameEngine,
        depth:    int,
        alpha:    float,
        beta:     float,
        maximise: bool,
    ) -> float:
        """
        Recursive minimax with alpha-beta pruning.

        maximise=True  → it is this agent's turn (maximising player).
        maximise=False → it is the opponent's turn (minimising player).
        """
        if engine.is_terminal():
            winner = engine.get_winner()
            if winner == self.player:
                return WIN_SCORE + depth   # prefer faster wins
            elif winner == self.opponent:
                return -(WIN_SCORE + depth)
            else:
                return 0  # draw

        if depth == 0:
            return self._evaluate(engine)

        valid_moves = engine.get_valid_moves()

        if maximise:
            value = -math.inf
            for col in self._move_order(valid_moves):
                child = engine.get_next_state(col)
                value = max(value, self._minimax(child, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # β cut-off
            return value
        else:
            value = math.inf
            for col in self._move_order(valid_moves):
                child = engine.get_next_state(col)
                value = min(value, self._minimax(child, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if alpha >= beta:
                    break  # α cut-off
            return value

    # ── Evaluation heuristic ──────────────────────────────────────────────────

    def _evaluate(self, engine: GameEngine) -> float:
        """
        Static board evaluation for non-terminal positions.

        Positive → good for self.player.
        Negative → good for opponent.
        """
        board = engine.get_board_state()
        score = 0.0

        # Centre column bonus
        centre_col = [board[r, CENTER_COL] for r in range(ROWS)]
        score += centre_col.count(self.player) * CENTER_WEIGHT

        # Score every window of WIN_LEN cells
        score += self._score_all_windows(board)

        return score

    def _score_all_windows(self, board: np.ndarray) -> float:
        score = 0.0

        # Horizontal windows
        for r in range(ROWS):
            for c in range(COLS - WIN_LEN + 1):
                window = list(board[r, c:c + WIN_LEN])
                score += self._score_window(window)

        # Vertical windows
        for c in range(COLS):
            for r in range(ROWS - WIN_LEN + 1):
                window = [board[r + i, c] for i in range(WIN_LEN)]
                score += self._score_window(window)

        # Diagonal ↘
        for r in range(ROWS - WIN_LEN + 1):
            for c in range(COLS - WIN_LEN + 1):
                window = [board[r + i, c + i] for i in range(WIN_LEN)]
                score += self._score_window(window)

        # Diagonal ↗
        for r in range(WIN_LEN - 1, ROWS):
            for c in range(COLS - WIN_LEN + 1):
                window = [board[r - i, c + i] for i in range(WIN_LEN)]
                score += self._score_window(window)

        return score

    def _score_window(self, window: list[int]) -> float:
        """
        Score a single window of WIN_LEN cells.

        Rules:
          - 4 of our pieces               → WIN_SCORE  (shouldn't occur in
                                            practice; terminal check fires first)
          - 3 of ours + 1 empty           → +THREE_WEIGHT
          - 2 of ours + 2 empty           → +TWO_WEIGHT
          - 3 of opponent + 1 empty       → -THREE_WEIGHT  (block priority)
        """
        mine = window.count(self.player)
        opp  = window.count(self.opponent)
        empty = window.count(EMPTY)

        if mine == WIN_LEN:
            return WIN_SCORE
        if opp == WIN_LEN:
            return -WIN_SCORE

        # Mixed windows (both players present) are useless — score 0
        if mine > 0 and opp > 0:
            return 0.0

        if mine == 3 and empty == 1:
            return THREE_WEIGHT
        if mine == 2 and empty == 2:
            return TWO_WEIGHT
        if opp == 3 and empty == 1:
            return -THREE_WEIGHT

        return 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _move_order(moves: list[int]) -> list[int]:
        """
        Return moves sorted by distance from centre (centre first).
        Visiting centre columns first dramatically improves alpha-beta pruning.
        """
        centre = COLS // 2
        return sorted(moves, key=lambda c: abs(c - centre))

    def _instant_move(
        self, engine: GameEngine, valid_moves: list[int]
    ) -> Optional[int]:
        """
        Return a move immediately if:
          1. We can win right now, or
          2. We must block the opponent's immediate win.
        Returns None when no such move exists.
        """
        # Check for our own winning move first
        for col in valid_moves:
            state = engine.get_next_state(col)
            if state.is_terminal() and state.get_winner() == self.player:
                return col

        # Check for opponent's winning threat
        for col in valid_moves:
            # Simulate opponent playing in this col
            # We need a temporary engine with the opponent as current player
            # Use get_board_state + manual check to avoid mutating anything
            opp_wins = self._opponent_wins_in(engine, col)
            if opp_wins:
                return col

        return None

    def _opponent_wins_in(self, engine: GameEngine, col: int) -> bool:
        """
        Return True if the opponent would win by playing in `col` right now.
        We check by building a next_state from the *opponent's* perspective.
        Since we're the current player in `engine`, we peek one move ahead
        for the opponent by checking whether, after our (dummy) move somewhere,
        the opponent can play col and win.  A simpler approach: temporarily
        make_move on a copy isn't exposed, so we use get_next_state to give
        the turn to the opponent then check col.
        """
        # After we move (in any column), it becomes the opponent's turn.
        # To check "what happens if opponent plays col", we need to first
        # advance the game to the opponent's turn.
        # We do this by peeking one level: if valid_moves contains col,
        # make a dummy next_state (with our move), then check if the opponent
        # (now current player) wins by playing col.
        valid = engine.get_valid_moves()
        # Find a dummy column that is NOT col so we don't interfere
        dummy_cols = [c for c in valid if c != col]
        if not dummy_cols:
            # Only move available is col itself — just play it
            return False

        # Advance past our turn using the first dummy column
        after_our_move = engine.get_next_state(dummy_cols[0])

        # Now it's the opponent's turn; check if they can win in col
        if col not in after_our_move.get_valid_moves():
            return False
        after_opp_move = after_our_move.get_next_state(col)
        return (
            after_opp_move.is_terminal()
            and after_opp_move.get_winner() == self.opponent
        )

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MinimaxAgent(player={self.player}, depth={self.depth})"
        )