"""AI Chessathon submission entry point.

Ships at the ROOT of the submission zip alongside the ``cb_*.py`` modules
it imports and the ``cb_book.bin`` opening book. Exposes exactly one
public function:

    get_move(fen: str, time_left_ms: int) -> str

returning a UCI move string (e.g. "e2e4", "e7e8q").

Layered fallback, each layer independent of the ones "above" it and each
layer's answer independently re-verified legal before being trusted:

  1. cb_book: an instant, no-search Polyglot opening-book probe.
  2. cb_nb_engine.Engine: the numba bitboard search. Its own __init__
     already blocks compiling (up to a wall-clock deadline) then falls
     back to serving from v1 in the background if that deadline is hit
     -- see cb_nb_engine.py -- so this layer alone already degrades
     gracefully under a slow or failed compile.
  3. cb_engine.Engine (v1, python-chess): constructed independently here
     as well, not just reached via layer 2's internal fallback -- so a
     bug in cb_nb_engine's own wrapper code (not the numba compile
     itself, which layer 2 already isolates) still can't take v1 down
     with it.
  4. _fallback_move: first legal move python-chess enumerates. Depends
     on nothing but ``chess``, so it's as close to "obviously correct" as
     a fallback can be without reimplementing chess rules from scratch.

A crash mid-game is an unrecoverable loss; a slightly worse move is not
-- so every layer above is guarded, and any failure just falls through to
the next one instead of crashing the process.
"""
from __future__ import annotations

import time

_PROCESS_START = time.monotonic()  # first line, before any other import --
# cb_nb_engine's compile deadline is measured from here, not from when its
# own __init__ runs, so the (nontrivial) time spent importing numpy/numba/
# chess below still counts against that deadline.

import chess

try:
    import cb_book
except Exception:
    cb_book = None

_v1 = None
try:
    import cb_engine

    _v1 = cb_engine.Engine()
except Exception:
    _v1 = None

_engine = None
try:
    import cb_nb_engine

    _engine = cb_nb_engine.Engine(process_start=_PROCESS_START)
except Exception:
    _engine = None


def _fallback_move(fen: str) -> str:
    board = chess.Board(fen)
    for move in board.legal_moves:
        return move.uci()
    raise RuntimeError(f"no legal moves available in position: {fen}")


def _is_legal(fen: str, move_uci: str) -> bool:
    """Re-verify a move against the FEN independently of whatever
    internal logic produced it -- book, search, doesn't matter. A hash
    collision, a bug in a seam not yet wired up, or any other defect that
    produces a well-formed but illegal UCI string must be caught here —
    an unverified move is an illegal-move loss."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
    except Exception:
        return False
    return move in board.legal_moves


_last_score = None  # centipawns, mover's perspective -- diagnostic only, see get_last_score()


def get_last_score() -> float | None:
    """Optional hook, not part of the competition's get_move contract:
    the internal test harness (harness/stdio_runner.py) calls this after
    get_move, purely for score-based resign/draw adjudication in dev
    SPRT -- see harness/match.py. None whenever the answering move came
    from the book or the trivial fallback (nothing was searched), or
    whenever nothing has been played yet."""
    return _last_score


def get_move(fen: str, time_left_ms: int) -> str:
    global _last_score
    _last_score = None

    if cb_book is not None:
        try:
            move = cb_book.probe(fen)
            if move is not None and _is_legal(fen, move):
                return move
        except Exception:
            pass

    if _engine is not None:
        try:
            move = _engine.get_move(fen, time_left_ms)
            if move is not None and _is_legal(fen, move):
                _last_score = getattr(_engine, "last_score", None)
                return move
        except Exception:
            pass

    if _v1 is not None:
        try:
            move = _v1.get_move(fen, time_left_ms)
            if move is not None and _is_legal(fen, move):
                _last_score = getattr(_v1, "last_score", None)
                return move
        except Exception:
            pass

    return _fallback_move(fen)
