"""Move ordering (design doc §5).

Priority: TT move, then captures/promotions by MVV-LVA (promotions scored
highest), then killer moves (two per ply), then quiets by history score.
No SEE yet — MVV-LVA only, per §5 ("MVV-LVA, or SEE if you get there").
"""
from __future__ import annotations

import chess

# Indexed by chess.PieceType (1..6); index 0 unused.
_PIECE_VALUE = [0, 100, 320, 330, 500, 900, 20000]

# Promotions are scored above any ordinary capture so a queening move is
# always tried first regardless of what (if anything) it captures.
_PROMOTION_BONUS = 100_000

# Score bands so a single sort key totally orders every category without
# a capture ever landing below a killer, etc.
_TT_MOVE_SCORE = 1_000_000_000
_KILLER_BASE_SCORE = 500_000  # below every capture/promotion, above all other quiets


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


def order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    tt_move: chess.Move | None = None,
    killers: tuple[chess.Move | None, chess.Move | None] = (None, None),
    history: dict | None = None,
) -> list[chess.Move]:
    """Return ``moves`` reordered by the priority above.

    ``killers`` and ``history`` are ignored for captures/promotions —
    those are always ordered by MVV-LVA regardless (a killer is by
    definition a *quiet* move that caused a cutoff).
    """

    def key(move: chess.Move) -> tuple:
        if move == tt_move:
            return (_TT_MOVE_SCORE,)
        if move.promotion or board.is_capture(move):
            return (_mvv_lva_score(board, move),)
        if move == killers[0]:
            return (_KILLER_BASE_SCORE + 1,)
        if move == killers[1]:
            return (_KILLER_BASE_SCORE,)
        if history:
            piece = board.piece_at(move.from_square)
            piece_type = piece.piece_type if piece else 0
            return (history.get((piece_type, move.to_square), 0),)
        return (0,)

    return sorted(moves, key=key, reverse=True)
