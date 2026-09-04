"""SPRT/gauntlet-only agent: pure numba bitboard search, no book. v1 is
still reachable internally through cb_nb_engine.Engine as its own init-
warmup/failure fallback (see cb_nb_engine.py) -- that's intentional and
part of what's being measured here, since it's exactly what the real
agent.py does too. NOT part of the submission: not in
tools/build_zip.py's whitelist, not agent.py.
"""
from __future__ import annotations

import time

_PROCESS_START = time.monotonic()  # see agent.py -- cb_nb_engine's compile
# deadline is measured from here, not from when its own __init__ runs.

import chess

import cb_nb_engine

_engine = cb_nb_engine.Engine(process_start=_PROCESS_START)


def get_move(fen: str, time_left_ms: int) -> str:
    move = _engine.get_move(fen, time_left_ms)
    board = chess.Board(fen)
    if chess.Move.from_uci(move) in board.legal_moves:
        return move
    for legal_move in board.legal_moves:
        return legal_move.uci()
    raise RuntimeError(f"no legal moves available in position: {fen}")
