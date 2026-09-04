"""Move ordering.

v1 (design doc §5): captures first, ordered by MVV-LVA, promotions scored
high among captures; quiet moves last, unordered among themselves. No TT
move, no killers, no history yet — those need a TT to key off of and are
days 3-4 work.
"""
from __future__ import annotations

import chess

# Indexed by chess.PieceType (1..6); index 0 unused.
_PIECE_VALUE = [0, 100, 320, 330, 500, 900, 20000]

# Promotions are scored above any ordinary capture so a queening move is
# always tried first regardless of what (if anything) it captures.
_PROMOTION_BONUS = 100_000


def _mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        victim_value = _PIECE_VALUE[chess.PAWN]
        attacker_value = _PIECE_VALUE[chess.PAWN]
    else:
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_value = _PIECE_VALUE[victim.piece_type] if victim else 0
        attacker_value = _PIECE_VALUE[attacker.piece_type] if attacker else 0
    # Most Valuable Victim, Least Valuable Attacker: sort by victim value
    # descending, then by attacker value ascending.
    score = victim_value * 16 - attacker_value
    if move.promotion:
        score += _PROMOTION_BONUS + _PIECE_VALUE[move.promotion]
    return score


def order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Return ``moves`` reordered: promotions and captures first (by
    MVV-LVA, promotions scored highest), then quiet moves."""

    def key(move: chess.Move) -> int:
        if move.promotion or board.is_capture(move):
            return _mvv_lva_score(board, move)
        return -1  # below every capture/promotion, ties broken by original order

    return sorted(moves, key=key, reverse=True)
