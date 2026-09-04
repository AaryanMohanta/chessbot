"""SPRT/gauntlet-only agent: pure v1 (python-chess) search, no book, no
numba -- exists so ratings/sprt.py can measure v1's strength in isolation
against cb_nb_agent.py's pure-numba counterpart. NOT part of the
submission: not in tools/build_zip.py's whitelist, not agent.py.
"""
from __future__ import annotations

import chess

import cb_engine

_engine = cb_engine.Engine()


def get_move(fen: str, time_left_ms: int) -> str:
    move = _engine.get_move(fen, time_left_ms)
    board = chess.Board(fen)
    if chess.Move.from_uci(move) in board.legal_moves:
        return move
    for legal_move in board.legal_moves:
        return legal_move.uci()
    raise RuntimeError(f"no legal moves available in position: {fen}")
