"""Numba-jitted movegen + make/unmake, mechanically ported from the
validated plain-Python reference in cb_nb_board.py/cb_nb_movegen.py (FEN
round-trip: 0 errors/11,859 positions; perft: exact match on all six
standard positions through depth 3-4, startpos through depth 4). This
module re-derives the same logic in a numba-compatible flat-array form and
is re-verified against the *same* perft suite — converting to njit is
treated as a distinct, re-checked step, not assumed correct by extension.

State layout (no Python objects, no dicts, no lists — design doc §4):
    pieces:  uint64[12]   piece bitboards, index = color*6 + piece_type
                          (WHITE=0: 0=P,1=N,2=B,3=R,4=Q,5=K; BLACK=1: +6)
    mailbox: int64[64]    piece index (0-11) at each square, or -1
    meta:    int64[5]     [turn, castling_rights, ep_square(-1=none),
                           halfmove_clock, fullmove_number]

Moves are packed into a single int32:
    from_sq(6 bits) | to_sq(6 bits) << 6 | promo(3 bits) << 12 | flag(3 bits) << 15
promo: 0..5 = piece type, 6 = no promotion. flag: 0=normal, 1=en passant,
2=castle kingside, 3=castle queenside, 4=double pawn push.

Move buffers are preallocated (MAX_MOVES per position, generous — real
positions never exceed ~220) and passed in by the caller rather than
allocated per call, per the design doc's "never allocate per node".
"""
from __future__ import annotations

import numpy as np
from numba import njit

import cb_nb_tables as T

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
NO_PROMO = 6
FLAG_NORMAL, FLAG_EP, FLAG_CASTLE_K, FLAG_CASTLE_Q, FLAG_DOUBLE_PUSH = 0, 1, 2, 3, 4
MAX_MOVES = 256

U64_MASK = np.uint64((1 << 64) - 1)

# Flattened per-(color,flag) castling constants, indexed [color][0 or 1]
# (0 => kingside/FLAG_CASTLE_K, 1 => queenside/FLAG_CASTLE_Q).
CASTLE_KING_TO = np.array([[6, 2], [62, 58]], dtype=np.int64)
CASTLE_ROOK_FROM = np.array([[7, 0], [63, 56]], dtype=np.int64)
CASTLE_ROOK_TO = np.array([[5, 3], [61, 59]], dtype=np.int64)
CASTLE_RIGHT_BIT = np.array([[1, 2], [4, 8]], dtype=np.int64)  # WK,WQ / BK,BQ
# Empty/path squares, padded with -1 to a fixed width of 3 for a rectangular array.
CASTLE_EMPTY_SQUARES = np.array([[[5, 6, -1], [1, 2, 3]], [[61, 62, -1], [57, 58, 59]]], dtype=np.int64)
CASTLE_KING_PATH = np.array([[[4, 5, 6], [4, 3, 2]], [[60, 61, 62], [60, 59, 58]]], dtype=np.int64)

CASTLE_ALL_ROOK_SQUARES = np.array([0, 7, 56, 63], dtype=np.int64)
CASTLE_ALL_ROOK_BITS = np.array([2, 1, 8, 4], dtype=np.int64)  # WQ,WK,BQ,BK


@njit(cache=True)
def pack_move(from_sq, to_sq, promo, flag):
    return from_sq | (to_sq << 6) | (promo << 12) | (flag << 15)


@njit(cache=True)
def unpack_move(move):
    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    promo = (move >> 12) & 0x7
    flag = (move >> 15) & 0x7
    return from_sq, to_sq, promo, flag


_DEBRUIJN64 = np.uint64(0x03F79D71B4CB0A89)
_DEBRUIJN_TABLE = np.array([0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4, 62, 55, 59, 36, 53, 51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5, 63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11, 46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6], dtype=np.int64)


@njit(cache=True)
def _bit_scan(bb):
    """Index of the lowest set bit, via the standard De Bruijn multiply
    (numpy uint64 scalars don't support .bit_length() under numba).
    Caller guarantees bb != 0."""
    isolated = bb & (~bb + np.uint64(1))  # lowest set bit only, i.e. bb & -bb
    index = (isolated * _DEBRUIJN64) >> np.uint64(58)
    return _DEBRUIJN_TABLE[np.int64(index)]


@njit(cache=True)
def rook_attacks_fast(square, occupied, masks, magics, shifts, offsets, table):
    blockers = np.uint64(occupied) & masks[square]
    index = (blockers * magics[square]) >> np.uint64(shifts[square])
    return table[offsets[square] + np.int64(index)]


@njit(cache=True)
def bishop_attacks_fast(square, occupied, masks, magics, shifts, offsets, table):
    blockers = np.uint64(occupied) & masks[square]
    index = (blockers * magics[square]) >> np.uint64(shifts[square])
    return table[offsets[square] + np.int64(index)]


@njit(cache=True)
def occupied_co(pieces, color):
    occ = np.uint64(0)
    base = color * 6
    for i in range(6):
        occ |= pieces[base + i]
    return occ


@njit(cache=True)
def occupied_all(pieces):
    return occupied_co(pieces, 0) | occupied_co(pieces, 1)


@njit(cache=True)
def is_square_attacked(pieces, square, by_color,
                        rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                        bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                        knight_attacks, king_attacks, pawn_attacks):
    occ = occupied_all(pieces)
    base = by_color * 6

    if knight_attacks[square] & pieces[base + KNIGHT]:
        return True
    if king_attacks[square] & pieces[base + KING]:
        return True
    other = 1 - by_color
    if pawn_attacks[other * 64 + square] & pieces[base + PAWN]:
        return True
    bishop_like = pieces[base + BISHOP] | pieces[base + QUEEN]
    if bishop_attacks_fast(square, occ, bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table) & bishop_like:
        return True
    rook_like = pieces[base + ROOK] | pieces[base + QUEEN]
    if rook_attacks_fast(square, occ, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table) & rook_like:
        return True
    return False


@njit(cache=True)
def king_square(pieces, color):
    return _bit_scan(pieces[color * 6 + KING])


@njit(cache=True)
def in_check(pieces, color,
             rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
             bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
             knight_attacks, king_attacks, pawn_attacks):
    opponent = 1 - color
    ksq = king_square(pieces, color)
    return is_square_attacked(pieces, ksq, opponent,
                               rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                               bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                               knight_attacks, king_attacks, pawn_attacks)


@njit(cache=True)
def generate_pseudo_legal_moves(pieces, mailbox, meta, moves_buf,
                                 rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                 bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                 knight_attacks, king_attacks, pawn_attacks,
                                 castle_king_to, castle_rook_from, castle_right_bit,
                                 castle_empty_squares, castle_king_path):
    count = 0
    color = meta[0]
    opponent = 1 - color
    own_occ = occupied_co(pieces, color)
    opp_occ = occupied_co(pieces, opponent)
    occ = own_occ | opp_occ
    ep_square = meta[2]

    push_dir = 8 if color == 0 else -8
    promo_rank = 7 if color == 0 else 0
    start_rank = 1 if color == 0 else 6

    pawns = pieces[color * 6 + PAWN]
    bb = pawns
    while bb != np.uint64(0):
        from_sq = _bit_scan(bb)
        bb &= bb - np.uint64(1)

        one_step = from_sq + push_dir
        if 0 <= one_step < 64 and (occ & (np.uint64(1) << np.uint64(one_step))) == np.uint64(0):
            if one_step // 8 == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves_buf[count] = pack_move(from_sq, one_step, promo, FLAG_NORMAL)
                    count += 1
            else:
                moves_buf[count] = pack_move(from_sq, one_step, NO_PROMO, FLAG_NORMAL)
                count += 1
                if from_sq // 8 == start_rank:
                    two_step = from_sq + 2 * push_dir
                    if (occ & (np.uint64(1) << np.uint64(two_step))) == np.uint64(0):
                        moves_buf[count] = pack_move(from_sq, two_step, NO_PROMO, FLAG_DOUBLE_PUSH)
                        count += 1

        attacks = pawn_attacks[color * 64 + from_sq]
        targets = attacks & opp_occ
        t = targets
        while t != np.uint64(0):
            to_sq = _bit_scan(t)
            t &= t - np.uint64(1)
            if to_sq // 8 == promo_rank:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves_buf[count] = pack_move(from_sq, to_sq, promo, FLAG_NORMAL)
                    count += 1
            else:
                moves_buf[count] = pack_move(from_sq, to_sq, NO_PROMO, FLAG_NORMAL)
                count += 1

        if ep_square != -1 and (attacks & (np.uint64(1) << np.uint64(ep_square))) != np.uint64(0):
            moves_buf[count] = pack_move(from_sq, ep_square, NO_PROMO, FLAG_EP)
            count += 1

    knights = pieces[color * 6 + KNIGHT]
    bb = knights
    while bb != np.uint64(0):
        from_sq = _bit_scan(bb)
        bb &= bb - np.uint64(1)
        targets = knight_attacks[from_sq] & ~own_occ
        t = targets
        while t != np.uint64(0):
            to_sq = _bit_scan(t)
            t &= t - np.uint64(1)
            moves_buf[count] = pack_move(from_sq, to_sq, NO_PROMO, FLAG_NORMAL)
            count += 1

    for piece_type in (BISHOP, ROOK, QUEEN):
        bb = pieces[color * 6 + piece_type]
        while bb != np.uint64(0):
            from_sq = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            if piece_type == BISHOP:
                targets = bishop_attacks_fast(from_sq, occ, bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table) & ~own_occ
            elif piece_type == ROOK:
                targets = rook_attacks_fast(from_sq, occ, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table) & ~own_occ
            else:
                targets = (bishop_attacks_fast(from_sq, occ, bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table)
                           | rook_attacks_fast(from_sq, occ, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table)) & ~own_occ
            t = targets
            while t != np.uint64(0):
                to_sq = _bit_scan(t)
                t &= t - np.uint64(1)
                moves_buf[count] = pack_move(from_sq, to_sq, NO_PROMO, FLAG_NORMAL)
                count += 1

    king_from = king_square(pieces, color)
    targets = king_attacks[king_from] & ~own_occ
    t = targets
    while t != np.uint64(0):
        to_sq = _bit_scan(t)
        t &= t - np.uint64(1)
        moves_buf[count] = pack_move(king_from, to_sq, NO_PROMO, FLAG_NORMAL)
        count += 1

    castling_rights = meta[1]
    for side in (0, 1):  # 0=kingside, 1=queenside
        right_bit = castle_right_bit[color, side]
        if (castling_rights & right_bit) == 0:
            continue
        blocked = False
        for i in range(3):
            s = castle_empty_squares[color, side, i]
            if s == -1:
                continue
            if occ & (np.uint64(1) << np.uint64(s)):
                blocked = True
                break
        if blocked:
            continue
        attacked = False
        for i in range(3):
            s = castle_king_path[color, side, i]
            if is_square_attacked(pieces, s, opponent,
                                   rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                   bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                   knight_attacks, king_attacks, pawn_attacks):
                attacked = True
                break
        if attacked:
            continue
        flag = FLAG_CASTLE_K if side == 0 else FLAG_CASTLE_Q
        moves_buf[count] = pack_move(king_from, castle_king_to[color, side], NO_PROMO, flag)
        count += 1

    return count


@njit(cache=True)
def make_move(pieces, mailbox, meta, move, castle_rook_from, castle_rook_to, castle_all_rook_squares, castle_all_rook_bits):
    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    promo = (move >> 12) & 0x7
    flag = (move >> 15) & 0x7

    color = meta[0]
    opponent = 1 - color

    undo_castling_rights = meta[1]
    undo_ep_square = meta[2]
    undo_halfmove_clock = meta[3]
    undo_captured_piece = -1
    undo_captured_square = -1

    piece_idx = mailbox[from_sq]
    piece_type = piece_idx % 6
    push_dir = 8 if color == 0 else -8

    if flag == FLAG_EP:
        captured_square = to_sq - push_dir
        captured_idx = mailbox[captured_square]
        undo_captured_piece = captured_idx
        undo_captured_square = captured_square
        pieces[captured_idx] &= ~(np.uint64(1) << np.uint64(captured_square))
        mailbox[captured_square] = -1
    else:
        captured_idx = mailbox[to_sq]
        if captured_idx != -1:
            undo_captured_piece = captured_idx
            undo_captured_square = to_sq
            pieces[captured_idx] &= ~(np.uint64(1) << np.uint64(to_sq))

    pieces[piece_idx] &= ~(np.uint64(1) << np.uint64(from_sq))
    final_piece_type = promo if promo != NO_PROMO else piece_type
    final_piece_idx = color * 6 + final_piece_type
    pieces[final_piece_idx] |= np.uint64(1) << np.uint64(to_sq)
    mailbox[from_sq] = -1
    mailbox[to_sq] = final_piece_idx

    if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
        side = 0 if flag == FLAG_CASTLE_K else 1
        rook_from = castle_rook_from[color, side]
        rook_to = castle_rook_to[color, side]
        rook_idx = color * 6 + ROOK
        pieces[rook_idx] &= ~(np.uint64(1) << np.uint64(rook_from))
        pieces[rook_idx] |= np.uint64(1) << np.uint64(rook_to)
        mailbox[rook_from] = -1
        mailbox[rook_to] = rook_idx

    new_ep = -1
    if flag == FLAG_DOUBLE_PUSH:
        new_ep = from_sq + push_dir

    rights = meta[1]
    if piece_type == KING:
        if color == 0:
            rights &= ~3
        else:
            rights &= ~12
    for i in range(4):
        sq_ = castle_all_rook_squares[i]
        if from_sq == sq_ or to_sq == sq_:
            rights &= ~castle_all_rook_bits[i]

    if piece_type == PAWN or undo_captured_piece != -1:
        new_halfmove = 0
    else:
        new_halfmove = meta[3] + 1

    meta[0] = opponent
    meta[1] = rights
    meta[2] = new_ep
    meta[3] = new_halfmove
    meta[4] = meta[4] + (1 if color == 1 else 0)

    return undo_castling_rights, undo_ep_square, undo_halfmove_clock, undo_captured_piece, undo_captured_square


@njit(cache=True)
def unmake_move(pieces, mailbox, meta, move, undo_castling_rights, undo_ep_square, undo_halfmove_clock, undo_captured_piece, undo_captured_square, castle_rook_from, castle_rook_to):
    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    promo = (move >> 12) & 0x7
    flag = (move >> 15) & 0x7

    opponent = meta[0]
    color = 1 - opponent

    meta[0] = color
    meta[1] = undo_castling_rights
    meta[2] = undo_ep_square
    meta[3] = undo_halfmove_clock
    meta[4] = meta[4] - (1 if color == 1 else 0)

    final_piece_idx = mailbox[to_sq]
    final_piece_type = final_piece_idx % 6
    original_piece_type = PAWN if promo != NO_PROMO else final_piece_type
    original_piece_idx = color * 6 + original_piece_type

    pieces[final_piece_idx] &= ~(np.uint64(1) << np.uint64(to_sq))
    pieces[original_piece_idx] |= np.uint64(1) << np.uint64(from_sq)
    mailbox[to_sq] = -1
    mailbox[from_sq] = original_piece_idx

    if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
        side = 0 if flag == FLAG_CASTLE_K else 1
        rook_from = castle_rook_from[color, side]
        rook_to = castle_rook_to[color, side]
        rook_idx = color * 6 + ROOK
        pieces[rook_idx] &= ~(np.uint64(1) << np.uint64(rook_to))
        pieces[rook_idx] |= np.uint64(1) << np.uint64(rook_from)
        mailbox[rook_to] = -1
        mailbox[rook_from] = rook_idx

    if undo_captured_piece != -1:
        pieces[undo_captured_piece] |= np.uint64(1) << np.uint64(undo_captured_square)
        mailbox[undo_captured_square] = undo_captured_piece


@njit(cache=True)
def perft(pieces, mailbox, meta, depth, moves_buf_stack, stack_ptr,
          rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
          bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
          knight_attacks, king_attacks, pawn_attacks,
          castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
          castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits):
    if depth == 0:
        return 1

    color = meta[0]
    moves_buf = moves_buf_stack[stack_ptr]
    count = generate_pseudo_legal_moves(pieces, mailbox, meta, moves_buf,
                                         rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                         bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                         knight_attacks, king_attacks, pawn_attacks,
                                         castle_king_to, castle_rook_from, castle_right_bit,
                                         castle_empty_squares, castle_king_path)

    nodes = 0
    for i in range(count):
        move = moves_buf[i]
        undo = make_move(pieces, mailbox, meta, move, castle_rook_from, castle_rook_to, castle_all_rook_squares, castle_all_rook_bits)
        if not in_check(pieces, color,
                         rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                         bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                         knight_attacks, king_attacks, pawn_attacks):
            nodes += perft(pieces, mailbox, meta, depth - 1, moves_buf_stack, stack_ptr + 1,
                           rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                           bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                           knight_attacks, king_attacks, pawn_attacks,
                           castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                           castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits)
        unmake_move(pieces, mailbox, meta, move, undo[0], undo[1], undo[2], undo[3], undo[4], castle_rook_from, castle_rook_to)

    return nodes


# ---------------------------------------------------------------------
# Python-side convenience layer: FEN <-> flat state, and a perft driver
# that binds all the constant-table arguments so callers don't have to.
# ---------------------------------------------------------------------

PIECE_CHARS = "pnbrqk"


def fen_to_state(fen: str):
    pieces = np.zeros(12, dtype=np.uint64)
    mailbox = np.full(64, -1, dtype=np.int64)
    meta = np.zeros(5, dtype=np.int64)

    placement, active, castling, ep, halfmove, fullmove = fen.split()

    rank, file = 7, 0
    for ch in placement:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            color = 0 if ch.isupper() else 1
            piece_type = PIECE_CHARS.index(ch.lower())
            square = rank * 8 + file
            idx = color * 6 + piece_type
            pieces[idx] |= np.uint64(1 << square)
            mailbox[square] = idx
            file += 1

    meta[0] = 0 if active == "w" else 1
    rights = 0
    if "K" in castling:
        rights |= 1
    if "Q" in castling:
        rights |= 2
    if "k" in castling:
        rights |= 4
    if "q" in castling:
        rights |= 8
    meta[1] = rights
    meta[2] = -1 if ep == "-" else (int(ep[1]) - 1) * 8 + (ord(ep[0]) - ord("a"))
    meta[3] = int(halfmove)
    meta[4] = int(fullmove)
    return pieces, mailbox, meta


def state_to_fen(pieces, mailbox, meta) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            square = rank * 8 + file
            idx = mailbox[square]
            if idx == -1:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                color, piece_type = idx // 6, idx % 6
                ch = PIECE_CHARS[piece_type]
                row += ch.upper() if color == 0 else ch
        if empty:
            row += str(empty)
        rows.append(row)
    placement = "/".join(rows)

    active = "w" if meta[0] == 0 else "b"
    rights = meta[1]
    castling = ""
    if rights & 1:
        castling += "K"
    if rights & 2:
        castling += "Q"
    if rights & 4:
        castling += "k"
    if rights & 8:
        castling += "q"
    if not castling:
        castling = "-"

    ep_square = meta[2]
    if ep_square == -1:
        ep = "-"
    else:
        ep = chr(ord("a") + ep_square % 8) + str(ep_square // 8 + 1)

    return f"{placement} {active} {castling} {ep} {meta[3]} {meta[4]}"


class Tables:
    """Bundles all the constant arrays perft()/generate_pseudo_legal_moves()
    need, so call sites pass one object instead of 20 positional args."""

    def __init__(self):
        self.rook_masks = T.ROOK_MASKS
        self.rook_magics = T.ROOK_MAGICS_ARR
        self.rook_shifts = T.ROOK_SHIFTS_ARR
        self.rook_offsets = T.ROOK_OFFSETS
        self.rook_table = T.ROOK_ATTACK_TABLE
        self.bishop_masks = T.BISHOP_MASKS
        self.bishop_magics = T.BISHOP_MAGICS_ARR
        self.bishop_shifts = T.BISHOP_SHIFTS_ARR
        self.bishop_offsets = T.BISHOP_OFFSETS
        self.bishop_table = T.BISHOP_ATTACK_TABLE
        self.knight_attacks = T.KNIGHT_ATTACKS
        self.king_attacks = T.KING_ATTACKS
        self.pawn_attacks = T.PAWN_ATTACKS.reshape(-1)  # flattened to [color*64+square]
        self.castle_king_to = CASTLE_KING_TO
        self.castle_rook_from = CASTLE_ROOK_FROM
        self.castle_rook_to = CASTLE_ROOK_TO
        self.castle_right_bit = CASTLE_RIGHT_BIT
        self.castle_empty_squares = CASTLE_EMPTY_SQUARES
        self.castle_king_path = CASTLE_KING_PATH
        self.castle_all_rook_squares = CASTLE_ALL_ROOK_SQUARES
        self.castle_all_rook_bits = CASTLE_ALL_ROOK_BITS


_TABLES = Tables()


def run_perft(fen: str, depth: int) -> int:
    pieces, mailbox, meta = fen_to_state(fen)
    moves_buf_stack = np.zeros((depth + 1, MAX_MOVES), dtype=np.int64)
    t = _TABLES
    return perft(
        pieces, mailbox, meta, depth, moves_buf_stack, 0,
        t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks, t.king_attacks, t.pawn_attacks,
        t.castle_king_to, t.castle_rook_from, t.castle_rook_to, t.castle_right_bit,
        t.castle_empty_squares, t.castle_king_path, t.castle_all_rook_squares, t.castle_all_rook_bits,
    )
