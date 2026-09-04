"""Static position evaluation.

v1: material + piece-square tables only, phase-tapered per design doc §6:

    score = (mg_score * phase + eg_score * (256 - phase)) // 256

Mate and stalemate are handled by the search, not here — evaluate() is
never called on a terminal position.
"""
from __future__ import annotations

import chess

from cb_tables import MAX_PHASE, PHASE_WEIGHTS, PIECE_VALUES_EG, PIECE_VALUES_MG, PST_EG, PST_MG

_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def evaluate(board: chess.Board) -> int:
    """Return a centipawn score from the perspective of ``board.turn``.

    Material, PST, and game-phase are all accumulated in one pass over
    each piece type — profiling showed a separate phase-computation pass
    was independently re-calling board.pieces() for the same piece types
    this loop already visits, roughly doubling the bitboard-scan cost for
    no reason.
    """
    mg_score = 0
    eg_score = 0
    phase = 0

    for piece_type in _PIECE_TYPES:
        mg_value = PIECE_VALUES_MG[piece_type]
        eg_value = PIECE_VALUES_EG[piece_type]
        mg_pst = PST_MG[piece_type]
        eg_pst = PST_EG[piece_type]
        phase_weight = PHASE_WEIGHTS[piece_type]

        white_squares = board.pieces(piece_type, chess.WHITE)
        black_squares = board.pieces(piece_type, chess.BLACK)

        for square in white_squares:
            mg_score += mg_value + mg_pst[square]
            eg_score += eg_value + eg_pst[square]

        for square in black_squares:
            mirrored = chess.square_mirror(square)
            mg_score -= mg_value + mg_pst[mirrored]
            eg_score -= eg_value + eg_pst[mirrored]

        if phase_weight:
            phase += phase_weight * (len(white_squares) + len(black_squares))

    phase = min(phase, MAX_PHASE)
    # phase is scaled 0..MAX_PHASE; §6's formula uses a 0..256 scale, so
    # rescale phase onto that range before interpolating.
    phase_256 = (phase * 256) // MAX_PHASE
    score = (mg_score * phase_256 + eg_score * (256 - phase_256)) // 256

    return score if board.turn == chess.WHITE else -score
