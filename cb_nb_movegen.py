"""Move generation, attack detection, and make/unmake for the bitboard
board (numba movegen experiment, stage 2 — THE GATE).

Plain Python for now, deliberately. njit-decorating buggy logic just
makes the bug harder to see (numba's error messages for logic bugs vs.
type bugs are not always easy to tell apart, and pdb/print-debugging
inside jitted code is much more limited than in plain Python). The plan
is: get this exactly right and perft-clean here first, *then* convert to
njit as a mechanical, re-verified-by-perft-again transformation — not
debug bitboard chess rules and numba's type system at the same time.

Moves are plain tuples for the same reason (readability while this is
being gotten right): (from_square, to_square, promotion_piece_type or -1,
flag). Flags: 0=normal, 1=en passant capture, 2=castle kingside,
3=castle queenside, 4=double pawn push.
"""
from __future__ import annotations

from cb_nb_board import BBBoard, CASTLE_BK, CASTLE_BQ, CASTLE_WK, CASTLE_WQ
from cb_nb_tables import (
    BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE, BLACK,
    KING_ATTACKS, KNIGHT_ATTACKS, PAWN_ATTACKS,
    bishop_attacks, rook_attacks, queen_attacks,
)

FLAG_NORMAL, FLAG_EP, FLAG_CASTLE_K, FLAG_CASTLE_Q, FLAG_DOUBLE_PUSH = 0, 1, 2, 3, 4

_PROMO_RANK = {WHITE: 7, BLACK: 0}
_START_RANK = {WHITE: 1, BLACK: 6}
_PUSH_DIR = {WHITE: 8, BLACK: -8}

# Castling: king/rook home squares and the squares that must be empty /
# unattacked, a1=0 numbering.
_CASTLE_KING_FROM = {WHITE: 4, BLACK: 60}
_CASTLE_KING_TO = {(WHITE, FLAG_CASTLE_K): 6, (WHITE, FLAG_CASTLE_Q): 2, (BLACK, FLAG_CASTLE_K): 62, (BLACK, FLAG_CASTLE_Q): 58}
_CASTLE_ROOK_FROM = {(WHITE, FLAG_CASTLE_K): 7, (WHITE, FLAG_CASTLE_Q): 0, (BLACK, FLAG_CASTLE_K): 63, (BLACK, FLAG_CASTLE_Q): 56}
_CASTLE_ROOK_TO = {(WHITE, FLAG_CASTLE_K): 5, (WHITE, FLAG_CASTLE_Q): 3, (BLACK, FLAG_CASTLE_K): 61, (BLACK, FLAG_CASTLE_Q): 59}
_CASTLE_EMPTY_SQUARES = {
    (WHITE, FLAG_CASTLE_K): [5, 6],
    (WHITE, FLAG_CASTLE_Q): [1, 2, 3],
    (BLACK, FLAG_CASTLE_K): [61, 62],
    (BLACK, FLAG_CASTLE_Q): [57, 58, 59],
}
_CASTLE_KING_PATH = {  # squares the king passes through (incl. start/end), must not be attacked
    (WHITE, FLAG_CASTLE_K): [4, 5, 6],
    (WHITE, FLAG_CASTLE_Q): [4, 3, 2],
    (BLACK, FLAG_CASTLE_K): [60, 61, 62],
    (BLACK, FLAG_CASTLE_Q): [60, 59, 58],
}
_CASTLE_RIGHT_BIT = {(WHITE, FLAG_CASTLE_K): CASTLE_WK, (WHITE, FLAG_CASTLE_Q): CASTLE_WQ, (BLACK, FLAG_CASTLE_K): CASTLE_BK, (BLACK, FLAG_CASTLE_Q): CASTLE_BQ}


def is_square_attacked(board: BBBoard, square: int, by_color: int) -> bool:
    occ = board.occupied()
    opp = board.pieces[by_color]

    if int(KNIGHT_ATTACKS[square]) & int(opp[KNIGHT]):
        return True
    if int(KING_ATTACKS[square]) & int(opp[KING]):
        return True
    # Pawn attacks are asymmetric: "is `square` attacked by a by_color pawn"
    # means checking the squares a by_color pawn diagonally in front of
    # `square` would stand on -- i.e. the *other* color's pawn-attack table
    # from `square`.
    other = WHITE if by_color == BLACK else BLACK
    if int(PAWN_ATTACKS[other][square]) & int(opp[PAWN]):
        return True
    if bishop_attacks(square, occ) & (int(opp[BISHOP]) | int(opp[QUEEN])):
        return True
    if rook_attacks(square, occ) & (int(opp[ROOK]) | int(opp[QUEEN])):
        return True
    return False


def king_square(board: BBBoard, color: int) -> int:
    kings = int(board.pieces[color][KING])
    return kings.bit_length() - 1


def in_check(board: BBBoard, color: int) -> bool:
    opponent = WHITE if color == BLACK else BLACK
    return is_square_attacked(board, king_square(board, color), opponent)


def generate_pseudo_legal_moves(board: BBBoard) -> list[tuple]:
    moves = []
    color = board.turn
    opponent = WHITE if color == BLACK else BLACK
    own_occ = board.occupied_co(color)
    opp_occ = board.occupied_co(opponent)
    occ = own_occ | opp_occ

    # ---- Pawns ----
    pawns = int(board.pieces[color][PAWN])
    push_dir = _PUSH_DIR[color]
    promo_rank = _PROMO_RANK[color]
    start_rank = _START_RANK[color]
    sq = pawns
    while sq:
        from_sq = (sq & -sq).bit_length() - 1
        sq &= sq - 1

        one_step = from_sq + push_dir
        if 0 <= one_step < 64 and not (occ & (1 << one_step)):
            if one_step // 8 == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves.append((from_sq, one_step, promo, FLAG_NORMAL))
            else:
                moves.append((from_sq, one_step, -1, FLAG_NORMAL))
                if from_sq // 8 == start_rank:
                    two_step = from_sq + 2 * push_dir
                    if not (occ & (1 << two_step)):
                        moves.append((from_sq, two_step, -1, FLAG_DOUBLE_PUSH))

        attacks = int(PAWN_ATTACKS[color][from_sq])
        targets = attacks & opp_occ
        t = targets
        while t:
            to_sq = (t & -t).bit_length() - 1
            t &= t - 1
            if to_sq // 8 == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves.append((from_sq, to_sq, promo, FLAG_NORMAL))
            else:
                moves.append((from_sq, to_sq, -1, FLAG_NORMAL))

        if board.ep_square != -1 and (attacks & (1 << board.ep_square)):
            moves.append((from_sq, board.ep_square, -1, FLAG_EP))

    # ---- Knights ----
    knights = int(board.pieces[color][KNIGHT])
    sq = knights
    while sq:
        from_sq = (sq & -sq).bit_length() - 1
        sq &= sq - 1
        targets = int(KNIGHT_ATTACKS[from_sq]) & ~own_occ
        t = targets
        while t:
            to_sq = (t & -t).bit_length() - 1
            t &= t - 1
            moves.append((from_sq, to_sq, -1, FLAG_NORMAL))

    # ---- Bishops / Rooks / Queens ----
    for piece_type, attack_fn in ((BISHOP, bishop_attacks), (ROOK, rook_attacks), (QUEEN, queen_attacks)):
        pieces = int(board.pieces[color][piece_type])
        sq = pieces
        while sq:
            from_sq = (sq & -sq).bit_length() - 1
            sq &= sq - 1
            targets = attack_fn(from_sq, occ) & ~own_occ
            t = targets
            while t:
                to_sq = (t & -t).bit_length() - 1
                t &= t - 1
                moves.append((from_sq, to_sq, -1, FLAG_NORMAL))

    # ---- King (non-castling) ----
    king_from = king_square(board, color)
    targets = int(KING_ATTACKS[king_from]) & ~own_occ
    t = targets
    while t:
        to_sq = (t & -t).bit_length() - 1
        t &= t - 1
        moves.append((king_from, to_sq, -1, FLAG_NORMAL))

    # ---- Castling ----
    for flag in (FLAG_CASTLE_K, FLAG_CASTLE_Q):
        right_bit = _CASTLE_RIGHT_BIT[(color, flag)]
        if not (board.castling_rights & right_bit):
            continue
        if any(occ & (1 << s) for s in _CASTLE_EMPTY_SQUARES[(color, flag)]):
            continue
        if any(is_square_attacked(board, s, opponent) for s in _CASTLE_KING_PATH[(color, flag)]):
            continue
        moves.append((king_from, _CASTLE_KING_TO[(color, flag)], -1, flag))

    return moves


class UndoInfo:
    __slots__ = ("captured_piece_type", "captured_color", "captured_square", "castling_rights", "ep_square", "halfmove_clock")


def make_move(board: BBBoard, move: tuple) -> UndoInfo:
    from_sq, to_sq, promotion, flag = move
    color = board.turn
    opponent = WHITE if color == BLACK else BLACK

    undo = UndoInfo()
    undo.castling_rights = board.castling_rights
    undo.ep_square = board.ep_square
    undo.halfmove_clock = board.halfmove_clock
    undo.captured_piece_type = -1
    undo.captured_color = -1
    undo.captured_square = -1

    piece_type = board.piece_type_at(from_sq)

    # Capture (including en passant, whose captured square isn't to_sq)
    if flag == FLAG_EP:
        captured_square = to_sq - _PUSH_DIR[color]
        undo.captured_piece_type = PAWN
        undo.captured_color = opponent
        undo.captured_square = captured_square
        board.pieces[opponent][PAWN] &= ~(1 << captured_square) & ((1 << 64) - 1)
    else:
        captured_pt = board.piece_type_at(to_sq)
        if captured_pt != -1:
            undo.captured_piece_type = captured_pt
            undo.captured_color = opponent
            undo.captured_square = to_sq
            board.pieces[opponent][captured_pt] &= ~(1 << to_sq) & ((1 << 64) - 1)

    # Move the piece
    board.pieces[color][piece_type] &= ~(1 << from_sq) & ((1 << 64) - 1)
    final_piece_type = promotion if promotion != -1 else piece_type
    board.pieces[color][final_piece_type] |= 1 << to_sq

    # Castling: also move the rook
    if flag in (FLAG_CASTLE_K, FLAG_CASTLE_Q):
        rook_from = _CASTLE_ROOK_FROM[(color, flag)]
        rook_to = _CASTLE_ROOK_TO[(color, flag)]
        board.pieces[color][ROOK] &= ~(1 << rook_from) & ((1 << 64) - 1)
        board.pieces[color][ROOK] |= 1 << rook_to

    # En passant square for the *next* move
    if flag == FLAG_DOUBLE_PUSH:
        board.ep_square = from_sq + _PUSH_DIR[color]
    else:
        board.ep_square = -1

    # Castling rights: lost if king moves, or either rook moves/is captured
    rights = board.castling_rights
    if piece_type == KING:
        rights &= ~(CASTLE_WK | CASTLE_WQ) if color == WHITE else ~(CASTLE_BK | CASTLE_BQ)
    for sq_, bit in ((0, CASTLE_WQ), (7, CASTLE_WK), (56, CASTLE_BQ), (63, CASTLE_BK)):
        if from_sq == sq_ or to_sq == sq_:
            rights &= ~bit
    board.castling_rights = rights

    # Halfmove clock: reset on pawn move or capture
    if piece_type == PAWN or undo.captured_piece_type != -1:
        board.halfmove_clock = 0
    else:
        board.halfmove_clock += 1

    if color == BLACK:
        board.fullmove_number += 1
    board.turn = opponent

    return undo


def unmake_move(board: BBBoard, move: tuple, undo: UndoInfo) -> None:
    from_sq, to_sq, promotion, flag = move
    opponent = board.turn  # side that just moved is the *other* color from current board.turn
    color = WHITE if opponent == BLACK else BLACK

    board.turn = color
    if color == BLACK:
        board.fullmove_number -= 1
    board.castling_rights = undo.castling_rights
    board.ep_square = undo.ep_square
    board.halfmove_clock = undo.halfmove_clock

    final_piece_type = promotion if promotion != -1 else board.piece_type_at(to_sq)
    board.pieces[color][final_piece_type] &= ~(1 << to_sq) & ((1 << 64) - 1)
    original_piece_type = PAWN if promotion != -1 else final_piece_type
    board.pieces[color][original_piece_type] |= 1 << from_sq

    if flag in (FLAG_CASTLE_K, FLAG_CASTLE_Q):
        rook_from = _CASTLE_ROOK_FROM[(color, flag)]
        rook_to = _CASTLE_ROOK_TO[(color, flag)]
        board.pieces[color][ROOK] &= ~(1 << rook_to) & ((1 << 64) - 1)
        board.pieces[color][ROOK] |= 1 << rook_from

    if undo.captured_piece_type != -1:
        board.pieces[undo.captured_color][undo.captured_piece_type] |= 1 << undo.captured_square


def generate_legal_moves(board: BBBoard) -> list[tuple]:
    color = board.turn
    legal = []
    for move in generate_pseudo_legal_moves(board):
        undo = make_move(board, move)
        if not in_check(board, color):
            legal.append(move)
        unmake_move(board, move, undo)
    return legal
