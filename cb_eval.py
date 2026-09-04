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

# Minimal king-exposure penalty (2026-09 stopgap): v1 is meant to answer
# only the first handful of moves, while cb_nb_engine's numba search is
# still JIT-compiling in the background -- but a real ladder loss (round
# 9) showed v1 answering as late as move 12 and choosing a needless king
# recapture (Kxe7 instead of Qxe7/Nxe7, all otherwise-equal) purely
# because v1's eval has zero opinion on king safety at all. Not a full
# port of cb_nb_fast.py's king-safety term (shield/file/attacker
# tracking would cost real nps in an engine that only exists to be fast
# and safe for a few opening moves) -- just the one piece of that logic
# cheap enough to matter here: penalize a king that has lost castling
# rights without castling while the opponent still has a queen or rook,
# using the same constants as cb_nb_fast.py's CB_NB_KING_SAFETY_V2 for
# consistency rather than inventing new ones.
_CASTLED_SQUARES = {chess.WHITE: (chess.G1, chess.C1), chess.BLACK: (chess.G8, chess.C8)}
UNCASTLED_EXPOSED_PENALTY_MG = 40


def _uncastled_exposure_penalty(board: chess.Board, color: bool) -> int:
    king_bb = chess.BB_H1 if color else chess.BB_H8
    queen_bb = chess.BB_A1 if color else chess.BB_A8
    has_rights = bool(board.castling_rights & (king_bb | queen_bb))
    if has_rights:
        return 0
    if board.king(color) in _CASTLED_SQUARES[color]:
        return 0
    enemy = not color
    if not (board.pieces(chess.QUEEN, enemy) or board.pieces(chess.ROOK, enemy)):
        return 0
    return UNCASTLED_EXPOSED_PENALTY_MG


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

    mg_score += _uncastled_exposure_penalty(board, chess.BLACK) - _uncastled_exposure_penalty(board, chess.WHITE)

    phase = min(phase, MAX_PHASE)
    # phase is scaled 0..MAX_PHASE; §6's formula uses a 0..256 scale, so
    # rescale phase onto that range before interpolating.
    phase_256 = (phase * 256) // MAX_PHASE
    score = (mg_score * phase_256 + eg_score * (256 - phase_256)) // 256

    return score if board.turn == chess.WHITE else -score
