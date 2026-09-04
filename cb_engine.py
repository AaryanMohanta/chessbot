"""Engine: owns per-game state and wires cb_time -> cb_search -> cb_eval.

``Engine.__init__`` is where one-time, expensive setup belongs (loading
weights, warming up a JIT, allocating tables) — it runs during the 60 s
init budget, before the game clock starts. Everything in ``get_move`` runs
under the per-move clock instead. v1 has no weights to load, so init is
just object construction; the timing log exists so days 5-7's numba JIT
warmup has a baseline to compare against.
"""
from __future__ import annotations

import sys
import time

import chess

import cb_search
from cb_time import INIT_BUDGET_MS, TimeManager
from cb_tt import TranspositionTable


def _position_key(board: chess.Board) -> tuple:
    """Repetition-relevant position identity: board + side to move +
    castling rights + en passant target. Deliberately excludes the
    halfmove clock and fullmove number (those don't affect whether a
    position "repeats" under the rules).

    NOTE: the wire protocol hands us a fresh FEN per call with no move
    history, and we're only asked to move on our own turns — so this can
    only see positions where it was our turn, not the full game history.
    It's therefore a partial repetition signal (good enough to notice "we
    keep landing back here"), not authoritative threefold detection. The
    referee claims threefold itself; this is bookkeeping for future search
    heuristics (e.g. contempt), not a legality check.
    """
    return (board.board_fen(), board.turn, board.castling_rights, board.ep_square)


class Engine:
    def __init__(self) -> None:
        start = time.perf_counter()

        self.time_manager = TimeManager()
        self.position_counts: dict[tuple, int] = {}

        # Owned for the whole game, not just one move: transpositions from
        # earlier in the game are still valid positions, and the
        # generation counter (bumped once per get_move call) lets the TT's
        # replacement policy prefer fresh entries over stale ones without
        # having to clear the whole table between moves.
        self.tt = TranspositionTable()
        self.generation = 0
        self.last_score: int | None = None  # centipawns, mover's perspective -- diagnostic only, see harness.match

        # TODO (v2/numba): load weights, warm up onnxruntime/numba JIT here,
        # inside this same budget.

        init_ms = (time.perf_counter() - start) * 1000
        print(f"[cb_engine] init took {init_ms:.1f} ms (budget {INIT_BUDGET_MS} ms)", file=sys.stderr)

    def get_move(self, fen: str, time_left_ms: int) -> str:
        """Per-move entry point. Raises on any internal failure — the
        caller (``agent.py``) is responsible for catching that, verifying
        legality, and falling back to a trivially-legal move."""
        board = chess.Board(fen)

        key = _position_key(board)
        self.position_counts[key] = self.position_counts.get(key, 0) + 1

        ply = board.ply()
        budget = self.time_manager.budget(time_left_ms, ply=ply)

        self.generation += 1
        move, score, _info = cb_search.search(board, budget.soft_ms, budget.hard_ms, self.tt, self.generation)
        self.last_score = score
        return move.uci()
