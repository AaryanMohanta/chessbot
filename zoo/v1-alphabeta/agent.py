"""AI Chessathon submission entry point.

Ships at the ROOT of the submission zip alongside the ``cb_*.py`` modules
it imports. Exposes exactly one public function:

    get_move(fen: str, time_left_ms: int) -> str

returning a UCI move string (e.g. "e2e4", "e7e8q").

Design: all real engine logic lives behind ``cb_engine.Engine``. Both the
import of that module and the construction of the engine are guarded here,
so any failure in our own code (bad weights, a bug in search, an
unavailable optional feature) degrades to a trivially-legal move instead of
crashing the process — a crash mid-game is an unrecoverable loss, a slightly
worse move is not.

The fallback deliberately depends on nothing but ``chess`` (python-chess is
guaranteed present in the competition environment): it parses the FEN and
returns the first legal move python-chess enumerates. That's the entire
dependency surface, so it is as close to "obviously correct" as a fallback
can be without reimplementing chess rules from scratch in the standard
library.
"""
from __future__ import annotations

import chess

_engine = None

try:
    import cb_engine

    _engine = cb_engine.Engine()
except Exception:
    _engine = None


def _fallback_move(fen: str) -> str:
    board = chess.Board(fen)
    for move in board.legal_moves:
        return move.uci()
    raise RuntimeError(f"no legal moves available in position: {fen}")


def _is_legal(fen: str, move_uci: str) -> bool:
    """Re-verify the engine's answer against the FEN independently of
    whatever internal logic produced it. A hash collision, a bug in a
    seam not yet wired up (TT, extensions), or any other engine defect
    that produces a well-formed but illegal UCI string must be caught
    here — an unverified engine move is an illegal-move loss."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
    except Exception:
        return False
    return move in board.legal_moves


def get_move(fen: str, time_left_ms: int) -> str:
    if _engine is not None:
        try:
            move = _engine.get_move(fen, time_left_ms)
            if move is not None and _is_legal(fen, move):
                return move
        except Exception:
            pass
    return _fallback_move(fen)
