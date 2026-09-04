"""Generated constants: piece-square tables, as plain nested lists.

These are v1 starting values, not a transcription of any specific
published engine's tables — they're derived from simple, documented
formulas (center-distance bonuses, back-rank-distance for king safety,
rank-based advancement for pawns) so their shape is easy to audit and
they're a clean base for the Texel tuner (design doc §6) to replace
later. No PST is claimed to be "correct" beyond "encodes the right
qualitative shape."

Indexing: each table is a flat 64-entry list in python-chess's own
square order (index 0 = a1, 63 = h8; index = rank*8 + file). Values are
from WHITE's perspective. For a black piece, look up
``table[chess.square_mirror(square)]`` instead — see cb_eval.py.

MG = middlegame, EG = endgame; cb_eval.py interpolates between them by
game phase.
"""
from __future__ import annotations

import chess

# fmt: off

_PAWN_MG_GRID = [
    [   0,    0,    5,   10,   10,    5,    0,    0],
    [   5,    5,   10,   15,   15,   10,    5,    5],
    [  10,   10,   15,   20,   20,   15,   10,   10],
    [  20,   20,   25,   30,   30,   25,   20,   20],
    [  35,   35,   40,   45,   45,   40,   35,   35],
    [  60,   60,   65,   70,   70,   65,   60,   60],
    [  90,   90,   95,  100,  100,   95,   90,   90],
    [   0,    0,    5,   10,   10,    5,    0,    0],
]

_PAWN_EG_GRID = [
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [  10,   10,   10,   10,   10,   10,   10,   10],
    [  20,   20,   20,   20,   20,   20,   20,   20],
    [  35,   35,   35,   35,   35,   35,   35,   35],
    [  60,   60,   60,   60,   60,   60,   60,   60],
    [ 100,  100,  100,  100,  100,  100,  100,  100],
    [   0,    0,    0,    0,    0,    0,    0,    0],
]

_KNIGHT_MG_GRID = [
    [ -40,  -30,  -20,  -10,  -10,  -20,  -30,  -40],
    [ -30,  -20,  -10,    0,    0,  -10,  -20,  -30],
    [ -20,  -10,    0,   10,   10,    0,  -10,  -20],
    [ -10,    0,   10,   20,   20,   10,    0,  -10],
    [ -10,    0,   10,   20,   20,   10,    0,  -10],
    [ -20,  -10,    0,   10,   10,    0,  -10,  -20],
    [ -30,  -20,  -10,    0,    0,  -10,  -20,  -30],
    [ -40,  -30,  -20,  -10,  -10,  -20,  -30,  -40],
]

_KNIGHT_EG_GRID = [
    [ -30,  -22,  -14,   -6,   -6,  -14,  -22,  -30],
    [ -22,  -14,   -6,    2,    2,   -6,  -14,  -22],
    [ -14,   -6,    2,   10,   10,    2,   -6,  -14],
    [  -6,    2,   10,   18,   18,   10,    2,   -6],
    [  -6,    2,   10,   18,   18,   10,    2,   -6],
    [ -14,   -6,    2,   10,   10,    2,   -6,  -14],
    [ -22,  -14,   -6,    2,    2,   -6,  -14,  -22],
    [ -30,  -22,  -14,   -6,   -6,  -14,  -22,  -30],
]

_BISHOP_MG_GRID = [
    [ -20,  -12,   -4,    4,    4,   -4,  -12,  -20],
    [ -12,   -4,    4,   12,   12,    4,   -4,  -12],
    [  -4,    4,   12,   20,   20,   12,    4,   -4],
    [   4,   12,   20,   28,   28,   20,   12,    4],
    [   4,   12,   20,   28,   28,   20,   12,    4],
    [  -4,    4,   12,   20,   20,   12,    4,   -4],
    [ -12,   -4,    4,   12,   12,    4,   -4,  -12],
    [ -20,  -12,   -4,    4,    4,   -4,  -12,  -20],
]

_BISHOP_EG_GRID = [
    [ -10,   -5,    0,    5,    5,    0,   -5,  -10],
    [  -5,    0,    5,   10,   10,    5,    0,   -5],
    [   0,    5,   10,   15,   15,   10,    5,    0],
    [   5,   10,   15,   20,   20,   15,   10,    5],
    [   5,   10,   15,   20,   20,   15,   10,    5],
    [   0,    5,   10,   15,   15,   10,    5,    0],
    [  -5,    0,    5,   10,   10,    5,    0,   -5],
    [ -10,   -5,    0,    5,    5,    0,   -5,  -10],
]

_ROOK_MG_GRID = [
    [   0,    0,    0,    4,    4,    0,    0,    0],
    [   0,    0,    0,    4,    4,    0,    0,    0],
    [   0,    0,    0,    4,    4,    0,    0,    0],
    [   0,    0,    0,    4,    4,    0,    0,    0],
    [   0,    0,    0,    4,    4,    0,    0,    0],
    [   0,    0,    0,    4,    4,    0,    0,    0],
    [  10,   10,   10,   14,   14,   10,   10,   10],
    [   0,    0,    0,    4,    4,    0,    0,    0],
]

_ROOK_EG_GRID = [
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [   0,    0,    0,    0,    0,    0,    0,    0],
    [  15,   15,   15,   15,   15,   15,   15,   15],
    [   0,    0,    0,    0,    0,    0,    0,    0],
]

_QUEEN_MG_GRID = [
    [  -5,   -3,   -1,    1,    1,   -1,   -3,   -5],
    [  -3,   -1,    1,    3,    3,    1,   -1,   -3],
    [  -1,    1,    3,    5,    5,    3,    1,   -1],
    [   1,    3,    5,    7,    7,    5,    3,    1],
    [   1,    3,    5,    7,    7,    5,    3,    1],
    [  -1,    1,    3,    5,    5,    3,    1,   -1],
    [  -3,   -1,    1,    3,    3,    1,   -1,   -3],
    [  -5,   -3,   -1,    1,    1,   -1,   -3,   -5],
]

_QUEEN_EG_GRID = [
    [  -5,   -2,    1,    4,    4,    1,   -2,   -5],
    [  -2,    1,    4,    7,    7,    4,    1,   -2],
    [   1,    4,    7,   10,   10,    7,    4,    1],
    [   4,    7,   10,   13,   13,   10,    7,    4],
    [   4,    7,   10,   13,   13,   10,    7,    4],
    [   1,    4,    7,   10,   10,    7,    4,    1],
    [  -2,    1,    4,    7,    7,    4,    1,   -2],
    [  -5,   -2,    1,    4,    4,    1,   -2,   -5],
]

_KING_MG_GRID = [
    [  20,   30,   10,  -10,  -10,   10,   30,   20],
    [   5,   15,   -5,  -25,  -25,   -5,   15,    5],
    [ -10,    0,  -20,  -40,  -40,  -20,    0,  -10],
    [ -25,  -15,  -35,  -55,  -55,  -35,  -15,  -25],
    [ -40,  -30,  -50,  -70,  -70,  -50,  -30,  -40],
    [ -55,  -45,  -65,  -85,  -85,  -65,  -45,  -55],
    [ -70,  -60,  -80, -100, -100,  -80,  -60,  -70],
    [ -85,  -75,  -95, -115, -115,  -95,  -75,  -85],
]

_KING_EG_GRID = [
    [ -30,  -22,  -14,   -6,   -6,  -14,  -22,  -30],
    [ -22,  -14,   -6,    2,    2,   -6,  -14,  -22],
    [ -14,   -6,    2,   10,   10,    2,   -6,  -14],
    [  -6,    2,   10,   18,   18,   10,    2,   -6],
    [  -6,    2,   10,   18,   18,   10,    2,   -6],
    [ -14,   -6,    2,   10,   10,    2,   -6,  -14],
    [ -22,  -14,   -6,    2,    2,   -6,  -14,  -22],
    [ -30,  -22,  -14,   -6,   -6,  -14,  -22,  -30],
]

# fmt: on


def _flatten(grid: list[list[int]]) -> list[int]:
    """grid[rank][file], rank 0 = rank1 -> flat[rank*8 + file] = chess.square order."""
    flat = [0] * 64
    for rank in range(8):
        for file in range(8):
            flat[rank * 8 + file] = grid[rank][file]
    return flat


PAWN_MG = _flatten(_PAWN_MG_GRID)
PAWN_EG = _flatten(_PAWN_EG_GRID)
KNIGHT_MG = _flatten(_KNIGHT_MG_GRID)
KNIGHT_EG = _flatten(_KNIGHT_EG_GRID)
BISHOP_MG = _flatten(_BISHOP_MG_GRID)
BISHOP_EG = _flatten(_BISHOP_EG_GRID)
ROOK_MG = _flatten(_ROOK_MG_GRID)
ROOK_EG = _flatten(_ROOK_EG_GRID)
QUEEN_MG = _flatten(_QUEEN_MG_GRID)
QUEEN_EG = _flatten(_QUEEN_EG_GRID)
KING_MG = _flatten(_KING_MG_GRID)
KING_EG = _flatten(_KING_EG_GRID)

PST_MG: dict[chess.PieceType, list[int]] = {
    chess.PAWN: PAWN_MG,
    chess.KNIGHT: KNIGHT_MG,
    chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG,
    chess.QUEEN: QUEEN_MG,
    chess.KING: KING_MG,
}

PST_EG: dict[chess.PieceType, list[int]] = {
    chess.PAWN: PAWN_EG,
    chess.KNIGHT: KNIGHT_EG,
    chess.BISHOP: BISHOP_EG,
    chess.ROOK: ROOK_EG,
    chess.QUEEN: QUEEN_EG,
    chess.KING: KING_EG,
}

# Material, centipawns.
PIECE_VALUES_MG: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
PIECE_VALUES_EG: dict[chess.PieceType, int] = dict(PIECE_VALUES_MG)

# Phase weights (design doc §6): phase computed from remaining non-pawn
# material. Standard "24 at full material, 0 at bare kings" scale.
PHASE_WEIGHTS: dict[chess.PieceType, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
    chess.KING: 0,
}
MAX_PHASE = 24  # 4 knights + 4 bishops + 4 rooks + 2 queens, one side's worth doubled
