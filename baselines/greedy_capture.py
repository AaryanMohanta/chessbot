"""Baseline opponent: takes the highest-value capture available, else plays
a random legal move. No lookahead, no defense against being recaptured.

Same entry-point contract as agent.py (get_move(fen, time_left_ms) -> str)
so the harness can run it through the same stdio wire protocol.
"""
from __future__ import annotations

import random

import chess

_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _capture_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return _PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    return _PIECE_VALUES[captured.piece_type] if captured else 0


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    moves = list(board.legal_moves)

    best_move = None
    best_value = 0
    for move in moves:
        if not board.is_capture(move):
            continue
        value = _capture_value(board, move)
        if value > best_value:
            best_value, best_move = value, move

    return (best_move or random.choice(moves)).uci()
