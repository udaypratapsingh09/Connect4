"""
agents/mcts_agent.py
====================
Monte Carlo Tree Search (MCTS) agent for Connect Four.

Algorithm: UCT (Upper Confidence bounds applied to Trees)
  1. Selection   – walk the tree choosing children by UCB1 until a node that
                   is not fully expanded or is terminal is reached.
  2. Expansion   – add one new child for an untried move.
  3. Simulation  – play out a random game (rollout) from that child.
  4. Backprop    – propagate the result back up to the root, updating visit
                   counts and win counts.

Design contract (mirrors MinimaxAgent):
  - Never holds a reference to the real board between calls.
  - Only uses the public GameEngine API.
  - choose_move(engine) → int

Parameters you can tune
───────────────────────
  iterations   : total MCTS simulations per move (default 1000)
  exploration  : UCB1 exploration constant C (default √2 ≈ 1.414)
                 Higher → more exploration; lower → more exploitation.
  rollout_depth: cap random rollouts at this many moves to keep them fast.
                 None means play to terminal (accurate but slower).

Usage
─────
  from game.engine import GameEngine, P2
  from agents.mcts_agent import MCTSAgent

  engine = GameEngine()
  agent  = MCTSAgent(player=P2, iterations=1000)
  col    = agent.choose_move(engine)
  engine.make_move(col)
"""

from __future__ import annotations

import math
import random
from typing import Optional

from game.engine import GameEngine, ROWS, COLS, EMPTY, P1, P2


# ── Tuneable constants ────────────────────────────────────────────────────────

DEFAULT_ITERATIONS   = 1000
DEFAULT_EXPLORATION  = math.sqrt(2)   # UCB1 C constant
DEFAULT_ROLLOUT_DEPTH: Optional[int] = None   # None = play to terminal


# ── _Node ─────────────────────────────────────────────────────────────────────

class _Node:
    """
    A single node in the MCTS tree.

    Attributes
    ----------
    engine       : game state AT this node (after the move that created it)
    move         : column that produced this node from its parent (None at root)
    parent       : parent _Node, or None at root
    children     : list of expanded child nodes
    untried_moves: moves not yet expanded from this node
    visits       : times this node has been visited
    wins         : accumulated win score for the player who JUST MOVED into
                   this node (i.e. the player whose move produced this state)
    """

    __slots__ = (
        "engine", "move", "parent",
        "children", "untried_moves",
        "visits", "wins",
        "_player_who_moved",
    )

    def __init__(
        self,
        engine: GameEngine,
        move:   Optional[int] = None,
        parent: Optional[_Node] = None,
    ) -> None:
        self.engine   = engine
        self.move     = move
        self.parent   = parent
        self.children: list[_Node] = []
        # Shuffle so expansion tries moves in random order → more tree diversity
        self.untried_moves: list[int] = engine.get_valid_moves()
        random.shuffle(self.untried_moves)
        self.visits: int   = 0
        self.wins:   float = 0.0
        # The player who JUST moved to reach this state is the opponent of
        # the current player.
        self._player_who_moved: int = engine.get_opponent()

    # ── UCB1 score ────────────────────────────────────────────────────────────

    def ucb1(self, exploration: float) -> float:
        """
        UCB1 = win_rate + C * sqrt(ln(parent.visits) / visits)

        Returns +inf for unvisited nodes so they are always explored first.
        """
        if self.visits == 0:
            return math.inf
        exploit = self.wins / self.visits
        explore = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    @property
    def is_terminal(self) -> bool:
        return self.engine.is_terminal()

    def best_child(self, exploration: float) -> "_Node":
        """Return the child with the highest UCB1 score."""
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def most_visited_child(self) -> "_Node":
        """Return the child visited most often (used for final move selection)."""
        return max(self.children, key=lambda c: c.visits)

    def expand(self) -> "_Node":
        """
        Pop one untried move, create its child node, and return it.
        Must only be called when untried_moves is non-empty.
        """
        col        = self.untried_moves.pop()
        new_engine = self.engine.get_next_state(col)
        child      = _Node(new_engine, move=col, parent=self)
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        return (
            f"_Node(move={self.move}, visits={self.visits}, "
            f"wins={self.wins:.1f}, untried={len(self.untried_moves)})"
        )


# ── MCTSAgent ─────────────────────────────────────────────────────────────────

class MCTSAgent:
    """
    Monte Carlo Tree Search agent (UCT variant).

    Parameters
    ----------
    player      : which player this agent controls (P1=1 or P2=2)
    iterations  : MCTS simulations budget per move
    exploration : UCB1 exploration constant (default √2)
    rollout_depth: max moves per random rollout (None = full game)
    """

    def __init__(
        self,
        player:        int,
        iterations:    int            = DEFAULT_ITERATIONS,
        exploration:   float          = DEFAULT_EXPLORATION,
        rollout_depth: Optional[int]  = DEFAULT_ROLLOUT_DEPTH,
    ) -> None:
        if player not in (P1, P2):
            raise ValueError(f"player must be {P1} or {P2}, got {player!r}")
        self.player        = player
        self.opponent      = P2 if player == P1 else P1
        self.iterations    = iterations
        self.exploration   = exploration
        self.rollout_depth = rollout_depth

    # ── Public interface ──────────────────────────────────────────────────────

    def choose_move(self, engine: GameEngine) -> int:
        """
        Run MCTS for `self.iterations` simulations and return the best column.

        Raises RuntimeError if called on a terminal state.
        """
        valid_moves = engine.get_valid_moves()
        if not valid_moves:
            raise RuntimeError("choose_move called on a terminal game state.")

        # Single-move shortcut: only one option available
        if len(valid_moves) == 1:
            return valid_moves[0]

        # Immediate win / forced block (free, before burning MCTS budget)
        instant = self._instant_move(engine, valid_moves)
        if instant is not None:
            return instant

        root = _Node(engine)

        for _ in range(self.iterations):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion (skip if terminal)
            if not node.is_terminal and not node.is_fully_expanded:
                node = node.expand()

            # 3. Simulation / rollout
            result = self._rollout(node.engine)

            # 4. Backpropagation
            self._backprop(node, result)

        # Choose the move with the most visits (robust to outliers)
        best = root.most_visited_child()
        return best.move

    # ── MCTS phases ──────────────────────────────────────────────────────────

    def _select(self, node: _Node) -> _Node:
        """
        Walk down the tree using UCB1 until we reach a node that is either:
          - not fully expanded (has untried moves), or
          - terminal.
        """
        while not node.is_terminal:
            if not node.is_fully_expanded:
                return node
            node = node.best_child(self.exploration)
        return node

    def _rollout(self, engine: GameEngine) -> Optional[int]:
        """
        Simulate a random game from `engine` and return the winner (1 or 2),
        or None for a draw.

        Uses a lightweight copy strategy: instead of calling get_next_state
        (which copies the engine each time), we use a single local engine and
        play random moves directly.  This is safe because rollouts are always
        thrown away after backpropagation.
        """
        # We need a mutable copy that we can run to completion.
        # get_next_state gives us a fresh engine; make_move mutates it.
        # To avoid excessive copying inside the loop, clone once here and
        # then mutate that clone.
        sim = engine.get_next_state(  # type: ignore[arg-type]
            # Trick: get_next_state requires a valid move; if the engine is
            # already terminal we never reach this point (caller checks).
            # We pass a dummy move and then re-use the resulting engine state.
            # Actually, to avoid the dummy, we clone differently:
            _noop := None  # placeholder — see actual implementation below
        ) if False else None

        # Real implementation: use Python's approach since we need the engine
        # as-is without making a move first.
        sim = _clone_engine(engine)

        depth = 0
        while not sim.is_terminal():
            if self.rollout_depth is not None and depth >= self.rollout_depth:
                break
            moves = sim.get_valid_moves()
            col   = _rollout_policy(sim, moves)
            sim.make_move(col)
            depth += 1

        return sim.get_winner()

    def _backprop(self, node: _Node, result: Optional[int]) -> None:
        """
        Walk from `node` back to the root, incrementing visit counts and
        crediting wins to the correct player.

        Win scoring:
          +1.0  → the player who moved INTO this node won
          +0.5  → draw (partial credit keeps draws preferred over losses)
          +0.0  → the player who moved INTO this node lost
        """
        current = node
        while current is not None:
            current.visits += 1
            if result is None:
                current.wins += 0.5          # draw
            elif result == current._player_who_moved:
                current.wins += 1.0          # win for the mover
            # else: loss → add 0 (no credit)
            current = current.parent

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _instant_move(
        self, engine: GameEngine, valid_moves: list[int]
    ) -> Optional[int]:
        """
        Return immediately if we can win or must block, without using MCTS.
        Checks our win first (always take a free win), then opponent's threat.
        """
        # Can we win right now?
        for col in valid_moves:
            state = engine.get_next_state(col)
            if state.is_terminal() and state.get_winner() == self.player:
                return col

        # Must we block?
        for col in valid_moves:
            dummy_cols = [c for c in valid_moves if c != col]
            if not dummy_cols:
                continue
            after_us  = engine.get_next_state(dummy_cols[0])
            if col not in after_us.get_valid_moves():
                continue
            after_opp = after_us.get_next_state(col)
            if after_opp.is_terminal() and after_opp.get_winner() == self.opponent:
                return col

        return None

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MCTSAgent(player={self.player}, "
            f"iterations={self.iterations}, "
            f"exploration={self.exploration:.3f})"
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _clone_engine(engine: GameEngine) -> GameEngine:
    """
    Lightweight clone via get_next_state is not possible without a move.
    We instead leverage the fact that GameEngine._fast_copy is private but
    the board state is fully captured by make_move calls on a fresh engine.

    Strategy: read board state + metadata via the public API, reconstruct
    by replaying — but that is O(moves²).  Instead we do a shallow attribute
    copy using __new__ + numpy copy, which is the same trick _fast_copy uses
    internally.  This keeps rollouts fast.

    If the GameEngine internals change, only this function needs updating.
    """
    import numpy as np

    new = GameEngine.__new__(GameEngine)
    # Access private attributes — acceptable here because this module is a
    # peer to engine.py in the same project, not a third-party consumer.
    new._board          = engine._board.copy()          # type: ignore[attr-defined]
    new._current_player = engine._current_player        # type: ignore[attr-defined]
    new._move_count     = engine._move_count            # type: ignore[attr-defined]
    new._winner         = engine._winner                # type: ignore[attr-defined]
    new._terminal       = engine._terminal              # type: ignore[attr-defined]
    return new


def _rollout_policy(engine: GameEngine, moves: list[int]) -> int:
    """
    Rollout move selection policy.

    Uses a light heuristic instead of pure random:
      1. If any move wins immediately → take it.
      2. Otherwise pick randomly (uniform).

    Pure random rollouts work but tend to underestimate the value of positions
    with an immediate threat.  This one-step look-ahead is cheap and lifts
    play quality noticeably without sacrificing speed.
    """
    # Check for immediate win in rollout (one-step look-ahead)
    current_player = engine.get_current_player()
    for col in moves:
        next_state = engine.get_next_state(col)
        if next_state.is_terminal() and next_state.get_winner() == current_player:
            return col

    # Prefer centre columns slightly (biased random)
    centre = COLS // 2
    weights = [centre - abs(col - centre) + 1 for col in moves]
    return random.choices(moves, weights=weights, k=1)[0]