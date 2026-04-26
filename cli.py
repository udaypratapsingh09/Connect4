"""
main.py
=======
CLI game runner for Connect Four.

Supported modes
───────────────
  1. Human vs Minimax
  2. Human vs MCTS
  3. Minimax vs MCTS

Run:
  python main.py
"""

import time

from game import GameEngine, P1, P2
from agents import MinimaxAgent, MCTSAgent


# ── Display helpers ───────────────────────────────────────────────────────────

DIVIDER = "─" * 40

def clear_screen() -> None:
    import os
    os.system("cls" if os.name == "nt" else "clear")

def print_header() -> None:
    print(DIVIDER)
    print("        CONNECT FOUR")
    print(DIVIDER)

def print_board(engine: GameEngine) -> None:
    print()
    print(engine)
    print()

def print_result(engine: GameEngine, names: dict[int, str]) -> None:
    print(DIVIDER)
    winner = engine.get_winner()
    if winner is None:
        print("  Result : Draw!")
    else:
        print(f"  Result : {names[winner]} wins! 🎉")
    print(DIVIDER)


# ── Human input ───────────────────────────────────────────────────────────────

def get_human_move(engine: GameEngine) -> int:
    valid = engine.get_valid_moves()
    while True:
        try:
            raw = input(f"  Your turn — enter column {valid}: ").strip()
            col = int(raw)
            if col in valid:
                return col
            print(f"  ✗ Column {col} is not available. Choose from {valid}.")
        except ValueError:
            print("  ✗ Please enter a number.")


# ── Mode configuration ────────────────────────────────────────────────────────

def select_mode() -> tuple[str, str]:
    """
    Prompt the user to pick a game mode.
    Returns (mode_key, description).
    """
    modes = {
        "1": ("human_minimax", "Human  vs  Minimax"),
        "2": ("human_mcts",    "Human  vs  MCTS"),
        "3": ("minimax_mcts",  "Minimax  vs  MCTS"),
    }
    print("\n  Select a game mode:\n")
    for key, (_, label) in modes.items():
        print(f"    [{key}]  {label}")
    print()

    while True:
        choice = input("  Enter 1 / 2 / 3: ").strip()
        if choice in modes:
            return modes[choice]
        print("  ✗ Invalid choice, please enter 1, 2, or 3.")


def select_depth(agent_name: str, default: int = 5) -> int:
    """Ask for Minimax search depth."""
    while True:
        try:
            raw = input(
                f"  Minimax depth for {agent_name} (3–7, default {default}): "
            ).strip()
            if raw == "":
                return default
            d = int(raw)
            if 1 <= d <= 10:
                return d
            print("  ✗ Please enter a number between 1 and 10.")
        except ValueError:
            print("  ✗ Please enter a number.")


def select_iterations(default: int = 1000) -> int:
    """Ask for MCTS simulation budget."""
    while True:
        try:
            raw = input(
                f"  MCTS iterations (100–5000, default {default}): "
            ).strip()
            if raw == "":
                return default
            n = int(raw)
            if 10 <= n <= 10_000:
                return n
            print("  ✗ Please enter a number between 10 and 10000.")
        except ValueError:
            print("  ✗ Please enter a number.")


def select_who_goes_first(name1: str, name2: str) -> bool:
    """Return True if the first-listed player goes first."""
    while True:
        choice = input(
            f"  Who goes first?  [1] {name1}   [2] {name2}  (default 1): "
        ).strip()
        if choice in ("", "1"):
            return True
        if choice == "2":
            return False
        print("  ✗ Enter 1 or 2.")


# ── Player abstraction ────────────────────────────────────────────────────────

class _Player:
    """Wraps a human or agent so the game loop stays uniform."""

    def __init__(self, name: str, is_human: bool, agent=None) -> None:
        self.name     = name
        self.is_human = is_human
        self._agent   = agent

    def choose_move(self, engine: GameEngine) -> int:
        if self.is_human:
            return get_human_move(engine)
        print(f"  {self.name} is thinking…", end="", flush=True)
        t0  = time.perf_counter()
        col = self._agent.choose_move(engine)
        elapsed = time.perf_counter() - t0
        print(f" plays column {col}  ({elapsed:.2f}s)")
        return col


# ── Game loop ─────────────────────────────────────────────────────────────────

def run_game(p1: _Player, p2: _Player) -> None:
    """Run a single game between two _Player objects."""
    engine  = GameEngine()
    players = {P1: p1, P2: p2}
    names   = {P1: p1.name, P2: p2.name}

    print(f"\n  {p1.name} (●) vs {p2.name} (○)\n")
    print_board(engine)

    while not engine.is_terminal():
        current = players[engine.get_current_player()]
        col     = current.choose_move(engine)
        engine.make_move(col)
        print_board(engine)

    print_result(engine, names)


# ── Setup factories ───────────────────────────────────────────────────────────

def build_players(mode: str) -> tuple[_Player, _Player]:
    """
    Construct the two _Player objects for the chosen mode.
    Prompts the user for any required parameters (depth, iterations, order).
    """
    print()

    if mode == "human_minimax":
        depth  = select_depth("Minimax")
        p1_first = select_who_goes_first("You (Human)", "Minimax")
        agent  = None  # assigned below after player order is known

        if p1_first:
            minimax = MinimaxAgent(player=P2, depth=depth)
            return (
                _Player("Human",   is_human=True),
                _Player("Minimax", is_human=False, agent=minimax),
            )
        else:
            minimax = MinimaxAgent(player=P1, depth=depth)
            return (
                _Player("Minimax", is_human=False, agent=minimax),
                _Player("Human",   is_human=True),
            )

    elif mode == "human_mcts":
        iters    = select_iterations()
        p1_first = select_who_goes_first("You (Human)", "MCTS")

        if p1_first:
            mcts = MCTSAgent(player=P2, iterations=iters)
            return (
                _Player("Human", is_human=True),
                _Player("MCTS",  is_human=False, agent=mcts),
            )
        else:
            mcts = MCTSAgent(player=P1, iterations=iters)
            return (
                _Player("MCTS",  is_human=False, agent=mcts),
                _Player("Human", is_human=True),
            )

    elif mode == "minimax_mcts":
        print("  Configure Minimax:")
        depth = select_depth("Minimax")
        print("  Configure MCTS:")
        iters    = select_iterations()
        p1_first = select_who_goes_first("Minimax", "MCTS")

        if p1_first:
            minimax = MinimaxAgent(player=P1, depth=depth)
            mcts    = MCTSAgent(player=P2, iterations=iters)
            return (
                _Player("Minimax", is_human=False, agent=minimax),
                _Player("MCTS",    is_human=False, agent=mcts),
            )
        else:
            minimax = MinimaxAgent(player=P2, depth=depth)
            mcts    = MCTSAgent(player=P1, iterations=iters)
            return (
                _Player("MCTS",    is_human=False, agent=mcts),
                _Player("Minimax", is_human=False, agent=minimax),
            )

    raise ValueError(f"Unknown mode: {mode!r}")


# ── Play-again loop ───────────────────────────────────────────────────────────

def ask_play_again() -> bool:
    choice = input("\n  Play again? [y/n] (default y): ").strip().lower()
    return choice in ("", "y", "yes")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    clear_screen()
    print_header()

    mode, description = select_mode()
    print(f"\n  Mode: {description}")

    while True:
        p1, p2 = build_players(mode)
        run_game(p1, p2)

        if not ask_play_again():
            print("\n  Thanks for playing. Goodbye!\n")
            break

        clear_screen()
        print_header()
        print(f"\n  Mode: {description}")


if __name__ == "__main__":
    main()