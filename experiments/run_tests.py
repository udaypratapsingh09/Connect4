"""
experiments/run_tests.py
========================
Headless experiment runner for Connect Four AI benchmarking.
No Pygame window required — pure Python, runs overnight if needed.

Experiment configurations (from project plan):
  Depth 3  vs  200  sims   → MCTS likely wins
  Depth 4  vs  500  sims   → Competitive — key comparison
  Depth 5  vs  1000 sims   → Near parity
  Depth 6  vs  2000 sims   → MCTS advantage emerges

Usage:
  # Run full experiment suite (all 4 configs, 50 games each)
  python experiments/run_tests.py

  # Quick smoke test (10 games per config)
  python experiments/run_tests.py --games 10

  # Single config
  python experiments/run_tests.py --depth 4 --sims 500 --games 50

  # Custom output path
  python experiments/run_tests.py --output experiments/my_results.csv

Results saved to: experiments/results.csv  (appended, not overwritten)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# Allow running from project root as  python experiments/run_tests.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game import GameEngine, P1, P2
from agents import MinimaxAgent, MCTSAgent


# ── Results CSV ───────────────────────────────────────────────────────────────

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "results.csv")

CSV_FIELDNAMES = [
    "config",
    "minimax_depth",
    "mcts_simulations",
    "n_games",
    "minimax_wins",
    "mcts_wins",
    "draws",
    "minimax_win_pct",
    "mcts_win_pct",
    "draw_pct",
    "avg_minimax_time",   # seconds per move
    "avg_mcts_time",      # seconds per move
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    config:            str
    minimax_depth:     int
    mcts_simulations:  int
    n_games:           int
    minimax_wins:      int = 0
    mcts_wins:         int = 0
    draws:             int = 0
    minimax_move_times: list[float] = field(default_factory=list)
    mcts_move_times:    list[float] = field(default_factory=list)

    # ── Derived stats ─────────────────────────────────────────────────────────

    @property
    def minimax_win_pct(self) -> float:
        return round(100 * self.minimax_wins / self.n_games, 1)

    @property
    def mcts_win_pct(self) -> float:
        return round(100 * self.mcts_wins / self.n_games, 1)

    @property
    def draw_pct(self) -> float:
        return round(100 * self.draws / self.n_games, 1)

    @property
    def avg_minimax_time(self) -> float:
        if not self.minimax_move_times:
            return 0.0
        return round(sum(self.minimax_move_times) / len(self.minimax_move_times), 4)

    @property
    def avg_mcts_time(self) -> float:
        if not self.mcts_move_times:
            return 0.0
        return round(sum(self.mcts_move_times) / len(self.mcts_move_times), 4)

    def to_csv_row(self) -> dict:
        return {
            "config":            self.config,
            "minimax_depth":     self.minimax_depth,
            "mcts_simulations":  self.mcts_simulations,
            "n_games":           self.n_games,
            "minimax_wins":      self.minimax_wins,
            "mcts_wins":         self.mcts_wins,
            "draws":             self.draws,
            "minimax_win_pct":   self.minimax_win_pct,
            "mcts_win_pct":      self.mcts_win_pct,
            "draw_pct":          self.draw_pct,
            "avg_minimax_time":  self.avg_minimax_time,
            "avg_mcts_time":     self.avg_mcts_time,
        }

    def print_summary(self) -> None:
        total = self.minimax_wins + self.mcts_wins + self.draws
        print(f"\n  {'─' * 46}")
        print(f"  Config : {self.config}")
        print(f"  Games  : {total}/{self.n_games}")
        print(f"  Minimax wins : {self.minimax_wins:>3}  ({self.minimax_win_pct}%)")
        print(f"  MCTS wins    : {self.mcts_wins:>3}  ({self.mcts_win_pct}%)")
        print(f"  Draws        : {self.draws:>3}  ({self.draw_pct}%)")
        print(f"  Avg Minimax move time : {self.avg_minimax_time:.4f}s")
        print(f"  Avg MCTS move time    : {self.avg_mcts_time:.4f}s")
        print(f"  {'─' * 46}")


# ── Single game ───────────────────────────────────────────────────────────────

def _play_one_game(
    minimax: MinimaxAgent,
    mcts:    MCTSAgent,
    minimax_is_p1: bool,
    result: MatchResult,
) -> Optional[int]:
    """
    Play one complete game between Minimax and MCTS.

    minimax_is_p1=True  → Minimax plays as P1 (goes first)
    minimax_is_p1=False → MCTS plays as P1 (goes first)

    Returns the winner token (P1 or P2) or None for a draw.
    Appends move times to result in-place.
    """
    engine = GameEngine()

    # Map player tokens to agents depending on who goes first
    if minimax_is_p1:
        agent_map = {P1: minimax, P2: mcts}
        time_map  = {P1: result.minimax_move_times, P2: result.mcts_move_times}
    else:
        agent_map = {P1: mcts,    P2: minimax}
        time_map  = {P1: result.mcts_move_times,    P2: result.minimax_move_times}

    while not engine.is_terminal():
        current = engine.get_current_player()
        agent   = agent_map[current]
        times   = time_map[current]

        t0  = time.perf_counter()
        col = agent.choose_move(engine)
        times.append(time.perf_counter() - t0)

        engine.make_move(col)

    return engine.get_winner()


# ── Match runner ──────────────────────────────────────────────────────────────

def run_match(
    minimax_depth:    int,
    mcts_simulations: int,
    n_games:          int = 50,
    verbose:          bool = True,
) -> MatchResult:
    """
    Run n_games between Minimax(depth) and MCTS(simulations).

    Alternates which agent goes first each game to eliminate first-move bias.
    Returns a fully populated MatchResult.
    """
    config = f"depth{minimax_depth}_vs_{mcts_simulations}sims"
    result = MatchResult(
        config           = config,
        minimax_depth    = minimax_depth,
        mcts_simulations = mcts_simulations,
        n_games          = n_games,
    )

    if verbose:
        print(f"\n  Running: {config}  ({n_games} games)")
        print(f"  {'Game':<6} {'First':>8} {'Winner':<12} {'MM time':>9} {'MCTS time':>10}")
        print(f"  {'─'*6} {'─'*8} {'─'*12} {'─'*9} {'─'*10}")

    for game_idx in range(n_games):
        # Alternate who goes first
        minimax_is_p1 = (game_idx % 2 == 0)

        # Re-instantiate agents each game so player tokens are correct
        if minimax_is_p1:
            minimax = MinimaxAgent(player=P1, depth=minimax_depth)
            mcts    = MCTSAgent(player=P2, iterations=mcts_simulations)
        else:
            minimax = MinimaxAgent(player=P2, depth=minimax_depth)
            mcts    = MCTSAgent(player=P1, iterations=mcts_simulations)

        winner_token = _play_one_game(minimax, mcts, minimax_is_p1, result)

        # Determine which agent won from the token
        if winner_token is None:
            result.draws += 1
            winner_label = "Draw"
        elif (winner_token == P1 and minimax_is_p1) or \
             (winner_token == P2 and not minimax_is_p1):
            result.minimax_wins += 1
            winner_label = "Minimax"
        else:
            result.mcts_wins += 1
            winner_label = "MCTS"

        if verbose:
            first_label = "Minimax" if minimax_is_p1 else "MCTS"
            mm_t   = result.minimax_move_times[-1] if result.minimax_move_times else 0
            mcts_t = result.mcts_move_times[-1]    if result.mcts_move_times    else 0
            print(
                f"  {game_idx+1:<6} {first_label:>8} {winner_label:<12} "
                f"{mm_t:>8.3f}s {mcts_t:>9.3f}s"
            )

    if verbose:
        result.print_summary()

    return result


# ── Experiment suite ──────────────────────────────────────────────────────────

# Default configs from the project plan
DEFAULT_CONFIGS = [
    (3, 200),    # MCTS likely wins
    (4, 500),    # Competitive — key comparison
    (5, 1000),   # Near parity
    (6, 2000),   # MCTS advantage emerges
]


def run_experiment(
    configs:  list[tuple[int, int]],
    n_games:  int     = 50,
    output:   str     = RESULTS_CSV,
    verbose:  bool    = True,
) -> list[MatchResult]:
    """
    Run the full experiment suite.

    configs : list of (minimax_depth, mcts_simulations) tuples
    n_games : games per configuration (minimum 50 recommended)
    output  : path to CSV file (rows appended, not overwritten)
    """
    results = []
    suite_start = time.perf_counter()

    print(f"\n{'═' * 50}")
    print(f"  CONNECT FOUR EXPERIMENT SUITE")
    print(f"  Configs : {len(configs)}   Games per config : {n_games}")
    print(f"  Output  : {output}")
    print(f"{'═' * 50}")

    for depth, sims in configs:
        result = run_match(
            minimax_depth    = depth,
            mcts_simulations = sims,
            n_games          = n_games,
            verbose          = verbose,
        )
        results.append(result)
        _append_csv(result, output)

    suite_elapsed = time.perf_counter() - suite_start
    _print_final_table(results)
    print(f"\n  Total time : {suite_elapsed/60:.1f} min")
    print(f"  Results saved to : {output}\n")

    return results


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _append_csv(result: MatchResult, path: str) -> None:
    """Append one result row to the CSV, writing the header if the file is new."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(result.to_csv_row())


def _print_final_table(results: list[MatchResult]) -> None:
    """Print a compact summary table of all experiment results."""
    print(f"\n{'═' * 70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═' * 70}")
    header = (
        f"  {'Config':<28} {'MM wins':>8} {'MCTS wins':>10} "
        f"{'Draws':>7} {'MM t/mv':>9} {'MC t/mv':>9}"
    )
    print(header)
    print(f"  {'─'*28} {'─'*8} {'─'*10} {'─'*7} {'─'*9} {'─'*9}")
    for r in results:
        print(
            f"  {r.config:<28} "
            f"{r.minimax_wins:>5} ({r.minimax_win_pct:>4}%)  "
            f"{r.mcts_wins:>5} ({r.mcts_win_pct:>4}%)  "
            f"{r.draws:>4} ({r.draw_pct:>4}%)  "
            f"{r.avg_minimax_time:>7.3f}s  "
            f"{r.avg_mcts_time:>7.3f}s"
        )
    print(f"{'═' * 70}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless Connect Four experiment runner (Minimax vs MCTS)"
    )
    parser.add_argument(
        "--games", type=int, default=50,
        help="Number of games per configuration (default: 50, minimum recommended: 50)"
    )
    parser.add_argument(
        "--output", type=str, default=RESULTS_CSV,
        help=f"CSV output path (default: {RESULTS_CSV})"
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help="Run a single config: Minimax depth (must pair with --sims)"
    )
    parser.add_argument(
        "--sims", type=int, default=None,
        help="Run a single config: MCTS simulations (must pair with --depth)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-game output (only print summaries)"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.games < 10:
        print("Warning: fewer than 10 games gives statistically meaningless results.")

    # Single config mode
    if args.depth is not None or args.sims is not None:
        if args.depth is None or args.sims is None:
            print("Error: --depth and --sims must be used together.")
            sys.exit(1)
        configs = [(args.depth, args.sims)]
    else:
        configs = DEFAULT_CONFIGS

    run_experiment(
        configs  = configs,
        n_games  = args.games,
        output   = args.output,
        verbose  = not args.quiet,
    )


if __name__ == "__main__":
    main()