"""Baseline opponent: plays a uniformly random legal move.

Same entry-point contract as agent.py (get_move(fen, time_left_ms) -> str)
so the harness can run it through the same stdio wire protocol.
"""
from __future__ import annotations

import random

import chess


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    return random.choice(list(board.legal_moves)).uci()
