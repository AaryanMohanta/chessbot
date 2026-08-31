"""Static position evaluation.

Stub: material count only, centipawns, from the side-to-move's perspective
(positive = side to move is better). Replace with a real evaluation
(PSTs, mobility, king safety, an NNUE, ...) without changing the signature.
"""
from __future__ import annotations

import chess

PIECE_VALUES_CP: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate(board: chess.Board) -> int:
    """Return a centipawn score from the perspective of ``board.turn``.

    Stub implementation: raw material difference only. No positional
    terms, no mobility, no king safety.
    """
    material = 0
    for piece_type, value in PIECE_VALUES_CP.items():
        material += value * (
            len(board.pieces(piece_type, chess.WHITE))
            - len(board.pieces(piece_type, chess.BLACK))
        )
    return material if board.turn == chess.WHITE else -material
