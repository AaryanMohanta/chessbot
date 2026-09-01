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


def _game_phase(board: chess.Board) -> int:
    """0..MAX_PHASE, MAX_PHASE = full material (opening), 0 = bare kings."""
    phase = 0
    for piece_type in _PIECE_TYPES:
        weight = PHASE_WEIGHTS[piece_type]
        if weight:
            count = len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
            phase += weight * count
    return min(phase, MAX_PHASE)


def evaluate(board: chess.Board) -> int:
    """Return a centipawn score from the perspective of ``board.turn``."""
    mg_score = 0
    eg_score = 0

    for piece_type in _PIECE_TYPES:
        mg_value = PIECE_VALUES_MG[piece_type]
        eg_value = PIECE_VALUES_EG[piece_type]
        mg_pst = PST_MG[piece_type]
        eg_pst = PST_EG[piece_type]

        for square in board.pieces(piece_type, chess.WHITE):
            mg_score += mg_value + mg_pst[square]
            eg_score += eg_value + eg_pst[square]

        for square in board.pieces(piece_type, chess.BLACK):
            mirrored = chess.square_mirror(square)
            mg_score -= mg_value + mg_pst[mirrored]
            eg_score -= eg_value + eg_pst[mirrored]

    phase = _game_phase(board)
    # phase is scaled 0..MAX_PHASE; §6's formula uses a 0..256 scale, so
    # rescale phase onto that range before interpolating.
    phase_256 = (phase * 256) // MAX_PHASE
    score = (mg_score * phase_256 + eg_score * (256 - phase_256)) // 256

    return score if board.turn == chess.WHITE else -score
