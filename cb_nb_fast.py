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

import os

import numpy as np
import numba
from numba import njit

import cb_nb_tables as T
import cb_tables as V1_TABLES  # reuse the already-tuned v1 PST/material data
import cb_nb_pst_tuned as PST_TUNED

# CB_NB_TEXEL_TUNED_PST (default on): picks between cb_tables.py's
# hand-set PST_MG/PST_EG and the ones ratings/pst_tune.py fit on the
# full 725k-position dataset (material and the 37 scalar terms held
# fixed -- see cb_nb_pst_tuned.py's own docstring for the held-out
# validation number). Separate flag from CB_NB_TEXEL_TUNED since the two
# were fit independently (PST tuning came after, holding the scalar
# retune's output fixed as a base) and either should be independently
# A/B-testable.
_TEXEL_TUNED_PST = os.environ.get("CB_NB_TEXEL_TUNED_PST", "1") != "0"

# Same env-var-flag pattern as cb_nb_search.py's CB_NB_ENABLE_KILLERS_HISTORY
# etc.: lets ratings/sprt.py A/B the *same* compiled module with the six
# eval terms below (passed pawns, isolated/doubled pawns, rook files,
# bishop pair, mobility, king safety) on vs off, for the batch SPRT
# against the material+PST-only baseline.
ENABLE_EXTENDED_EVAL = os.environ.get("CB_NB_ENABLE_EXTENDED_EVAL", "1") != "0"

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

# ---------------------------------------------------------------------
# Stage 4: signed, mirrored material+PST tables, indexed [piece_idx][square]
# (piece_idx = color*6+piece_type, same as `pieces`/`mailbox`). Folding
# material, PST, sign (+white/-black), and black's vertical mirror into
# one lookup makes the incremental update at each add/remove site a
# single table read, mirroring how the Zobrist tables work in stage 3.
# Reuses cb_tables.py's v1 PST data directly -- same a1=0..h8=63 square
# numbering, so no reformulation needed, just re-signing for black.
# ---------------------------------------------------------------------
MAX_PHASE = V1_TABLES.MAX_PHASE


def _build_signed_pst():
    pst_mg = np.zeros((12, 64), dtype=np.int64)
    pst_eg = np.zeros((12, 64), dtype=np.int64)
    phase_weight = np.zeros(12, dtype=np.int64)
    piece_types = [T.PAWN, T.KNIGHT, T.BISHOP, T.ROOK, T.QUEEN, T.KING]
    for pt in piece_types:
        # cb_nb_tables uses 0-indexed piece types (PAWN=0..KING=5);
        # cb_tables.py (v1, reused here for its PST/material data) keys
        # its dicts by python-chess's own 1-indexed constants (PAWN=1..
        # KING=6) -- +1 bridges the two conventions.
        chess_pt = pt + 1
        if _TEXEL_TUNED_PST:
            # Material values here come from the SAME joint fit as the
            # PST cells below (ratings/joint_tune.py) -- mixing this PST
            # with the old hand-set material (or vice versa) was never
            # validated, so both switch on the one flag together.
            mg_val = PST_TUNED.PIECE_VALUES_MG[chess_pt]
            eg_val = PST_TUNED.PIECE_VALUES_EG[chess_pt]
            mg_pst = PST_TUNED.PST_MG[chess_pt]
            eg_pst = PST_TUNED.PST_EG[chess_pt]
        else:
            mg_val = V1_TABLES.PIECE_VALUES_MG[chess_pt]
            eg_val = V1_TABLES.PIECE_VALUES_EG[chess_pt]
            mg_pst = V1_TABLES.PST_MG[chess_pt]
            eg_pst = V1_TABLES.PST_EG[chess_pt]
        weight = V1_TABLES.PHASE_WEIGHTS[chess_pt]

        white_idx = 0 * 6 + pt
        black_idx = 1 * 6 + pt
        phase_weight[white_idx] = weight
        phase_weight[black_idx] = weight
        for square in range(64):
            pst_mg[white_idx, square] = mg_val + mg_pst[square]
            pst_eg[white_idx, square] = eg_val + eg_pst[square]
            mirrored = square ^ 56
            pst_mg[black_idx, square] = -(mg_val + mg_pst[mirrored])
            pst_eg[black_idx, square] = -(eg_val + eg_pst[mirrored])
    return pst_mg, pst_eg, phase_weight


PST_SIGNED_MG, PST_SIGNED_EG, PHASE_WEIGHT_BY_IDX = _build_signed_pst()

# CB_NB_TEXEL_TUNED (default on): picks between the original hand-set
# scalar weights below ("just be nonzero, Texel tuning finds the real
# values later") and the values an actual Texel tune produced.
#
# Retuned 2026-09 on the real Zurichess quiet-labeled.epd (725,000
# positions -- see ratings/data/quiet_labeled.csv, converted via
# tools/convert_quiet_labeled.py), replacing the first pass's 8,578
# self-play positions (ratings/texel_data.csv, kept only as a smaller
# fallback dataset). reg_lambda scaled down from 5e-7 to 5.9e-9
# (proportional to the ~85x data increase -- see texel_error's docstring
# on why the raw value doesn't self-scale with dataset size) so the much
# larger sample is actually allowed to move weights away from the prior;
# this is what fixed the previous tune's clamped-near-zero mobility
# weights (mob_knight_mg/mob_bishop_mg were 5 either way, but
# mob_knight_eg was clamped from -1 to 0 -- now a real, comfortably
# positive 8, not just "not negative").
#
# Three tuner outputs still came back negative and were clamped to 0
# before shipping, same policy as the first pass: mob_queen_mg (-2),
# king_shield_mg (-4), king_castled_mg (-1). The first is "more mobility
# is bad," already established as not something a dataset gets to
# overrule regardless of size; the latter two are worse -- a shielded or
# actually-castled king scoring as LESS safe than an unshielded,
# uncastled one is chess-nonsensical, and given how small all three
# magnitudes are (comfortably inside noise), reads as leftover
# collinearity between the king-safety sub-features rather than real
# signal. Left un-clamped: rook_open_eg (-13), a small enough magnitude
# on a weakly-informative endgame feature that a real "file control
# matters less once rook activity comes from king/pawn proximity
# instead" effect is plausible, unlike the three above.
_TEXEL_TUNED = os.environ.get("CB_NB_TEXEL_TUNED", "1") != "0"

# Passed-pawn bonus by relative rank (0 = own back rank, 7 = the square
# it'd promote from -- unreachable for a pawn still on the board, kept
# at 0 for safety).
if _TEXEL_TUNED:
    PASSED_PAWN_BONUS_MG = np.array([0, 8, 8, 2, 19, 40, 88, 0], dtype=np.int64)
    PASSED_PAWN_BONUS_EG = np.array([0, 58, 48, 67, 93, 153, 188, 0], dtype=np.int64)
else:
    PASSED_PAWN_BONUS_MG = np.array([0, 5, 10, 20, 35, 60, 100, 0], dtype=np.int64)
    PASSED_PAWN_BONUS_EG = np.array([0, 10, 20, 35, 60, 100, 150, 0], dtype=np.int64)

# Lazy eval: material+PST alone is decisive often enough (a position way
# outside the search window won't be changed by refining pawn structure/
# mobility/king safety) that computing those terms for every leaf is
# wasted work. If the cheap material+PST score already misses the
# [alpha, beta] window by more than this margin, it's returned as-is.
LAZY_EVAL_MARGIN = 250

# King-to-passed-pawn race bonus (2026-09, endgame pass): per unit of
# (enemy king's Chebyshev distance to the pawn's promotion square minus
# our own king's distance to it), EG-only. Was added after the first
# Texel pass ran, so it shipped un-tuned (a flat 5) for one round; now
# folded into CB_NB_TEXEL_TUNED like every other term now that
# ratings/texel_features.py extracts it too.
if _TEXEL_TUNED:
    KING_PAWN_PROXIMITY_WEIGHT_EG = 15
else:
    KING_PAWN_PROXIMITY_WEIGHT_EG = 5

if _TEXEL_TUNED:
    ISOLATED_PENALTY_MG, ISOLATED_PENALTY_EG = 26, 11
    DOUBLED_PENALTY_MG, DOUBLED_PENALTY_EG = 15, 30
else:
    ISOLATED_PENALTY_MG, ISOLATED_PENALTY_EG = 12, 20
    DOUBLED_PENALTY_MG, DOUBLED_PENALTY_EG = 10, 15

# Rook on an open (no pawns of either colour) or semi-open (no *own*
# pawns) file. Bigger in the middlegame, where an open file is a real
# attacking asset; smaller in the endgame, where rook activity tends to
# come from king/pawn proximity more than file control.
if _TEXEL_TUNED:
    ROOK_OPEN_FILE_BONUS_MG, ROOK_OPEN_FILE_BONUS_EG = 72, -13
    ROOK_SEMI_OPEN_FILE_BONUS_MG, ROOK_SEMI_OPEN_FILE_BONUS_EG = 31, 19
else:
    ROOK_OPEN_FILE_BONUS_MG, ROOK_OPEN_FILE_BONUS_EG = 20, 10
    ROOK_SEMI_OPEN_FILE_BONUS_MG, ROOK_SEMI_OPEN_FILE_BONUS_EG = 10, 5

# Bishop pair: two bishops covering both square colours are worth more
# than the sum of two same-colour minor pieces, especially as the board
# opens up in the endgame.
if _TEXEL_TUNED:
    BISHOP_PAIR_BONUS_MG, BISHOP_PAIR_BONUS_EG = 52, 51
else:
    BISHOP_PAIR_BONUS_MG, BISHOP_PAIR_BONUS_EG = 15, 30

# King safety, middlegame-only (no EG term): king safety matters far
# less once material is traded off, and kings often want to be active/
# central in the endgame rather than sheltered.
#
# Retest (2026-09): weights raised substantially from the original
# "just be nonzero, Texel tuning finds the real values later" starting
# point, after a real ladder loss (round 4) where the engine never
# castled, wrecked its own kingside pawns, and pushed queenside pawns
# while exposed -- the old weights (max realistic total ~15-70cp) were
# far too small to ever outweigh other eval terms, so the engine had
# effectively no opinion on king safety at all. Still Texel-tuning-
# pending like every other weight here, but "not actively harmful" is
# no longer the bar.
#
# CB_NB_KING_SAFETY_V2 (default on) picks between this new set and the
# original small-weights set -- same env-var-flag pattern as every other
# toggle in this file, here so the retest can A/B the whole batch (raised
# weights + castling incentive) in one SPRT against the original values,
# not because the shipped default is ever meant to be "0".
_KING_SAFETY_V2 = os.environ.get("CB_NB_KING_SAFETY_V2", "1") != "0"

if _KING_SAFETY_V2 and _TEXEL_TUNED:
    # king_shield_mg and king_castled_mg came back negative from the
    # retune (-4, -1) -- a shielded or actually-castled king scoring as
    # LESS safe than the alternative is chess-nonsensical, and the
    # magnitudes are small enough (well inside noise) to read as
    # collinearity between the king-safety sub-features rather than a
    # real effect. Rather than clamp to 0 (making the term fully inert),
    # these two specifically fall back to the previous tune's values --
    # not the hand-set original, the ones already validated by a real
    # ladder loss (round 4: no castling, wrecked kingside shield, exposed
    # queenside push) that this exact retune would otherwise silently
    # erase evidence for. See the CB_NB_TEXEL_TUNED comment above.
    KING_SHIELD_PENALTY_MG = 26
    KING_SHIELD_MATERIAL_BONUS_PER_UNIT_MG = 6
    KING_OPEN_FILE_PENALTY_MG = 60
    KING_SEMI_OPEN_FILE_PENALTY_MG = 11
    KING_ATTACKER_PENALTY_PER_UNIT_MG = 8
    CASTLED_BONUS_MG = 33
    UNCASTLED_EXPOSED_PENALTY_MG = 52
elif _KING_SAFETY_V2:
    KING_SHIELD_PENALTY_MG = 25  # per missing pawn in the 3-square shield in front of the king
    # Extra shield penalty per missing pawn, scaled by how much attacking
    # material the opponent still has (see KING_ATTACKER_WEIGHT-style
    # reasoning below) -- a wrecked shield in a queen+rooks middlegame is
    # dangerous; the same wrecked shield once the opponent's queen and
    # rooks are gone barely matters.
    KING_SHIELD_MATERIAL_BONUS_PER_UNIT_MG = 6
    KING_OPEN_FILE_PENALTY_MG = 35  # king's own file has no pawns of either colour
    KING_SEMI_OPEN_FILE_PENALTY_MG = 18  # king's own file has no *own* pawns (enemy pawns may be present)
    KING_ATTACKER_PENALTY_PER_UNIT_MG = 10
    # Castling incentive, deliberately separate from the structural terms
    # above: a flat bonus for having actually castled (king on g1/c1 or
    # g8/c8 with no remaining rights -- the only way to reach that square
    # with rights already gone, short of an extremely unlikely manual
    # walk), and a flat penalty for having *lost* castling rights without
    # castling while the opponent still has real attacking material (a
    # queen or a rook) -- exactly the round-4 pattern: king stuck near
    # the centre, queen and rooks still on the board on both sides. No
    # penalty once the opponent's queen and rooks are gone -- an
    # uncastled king in a quiet, major-piece-free position isn't a safety
    # problem, it's normal endgame/simplified play.
    CASTLED_BONUS_MG = 45
    UNCASTLED_EXPOSED_PENALTY_MG = 40
else:
    KING_SHIELD_PENALTY_MG = 10
    KING_SHIELD_MATERIAL_BONUS_PER_UNIT_MG = 0
    KING_OPEN_FILE_PENALTY_MG = 15
    KING_SEMI_OPEN_FILE_PENALTY_MG = 8
    KING_ATTACKER_PENALTY_PER_UNIT_MG = 4
    CASTLED_BONUS_MG = 0
    UNCASTLED_EXPOSED_PENALTY_MG = 0

# Indexed PAWN..KING; only KNIGHT/BISHOP/ROOK/QUEEN are nonzero -- how
# dangerous one enemy piece of this type attacking the king's immediate
# zone is considered. Also reused (queen/rook units only) as the
# "attacking material" scale for the shield-material-bonus and
# uncastled-exposure penalty above. Unaffected by CB_NB_KING_SAFETY_V2.
KING_ATTACKER_WEIGHT = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)
ATTACKING_MATERIAL_QUEEN_UNITS = 3
ATTACKING_MATERIAL_ROOK_UNITS = 1

# Mobility: per-square bonus for each pseudo-legally reachable square not
# occupied by a piece of the same colour, weighted by piece type
# (indexed PAWN..KING; only KNIGHT/BISHOP/ROOK/QUEEN are nonzero -- pawn
# and king "mobility" aren't meaningful the same way and aren't scored
# here). Higher weight on knights/bishops than rooks/queens since the
# latter naturally reach far more squares, so an equal per-square weight
# would overweight them relative to how much any one extra square
# actually matters.
if _TEXEL_TUNED:
    # mob_queen_mg came back slightly negative from the retune (-2) --
    # clamped to 0 rather than shipped negative, see the
    # CB_NB_TEXEL_TUNED comment above ("more mobility is bad" isn't
    # something a dataset gets to overrule regardless of size).
    # mob_rook_mg is genuinely 0, not clamped -- that's what the tune
    # returned directly.
    MOBILITY_UNIT_MG = np.array([0, 12, 8, 0, 0, 0], dtype=np.int64)
    MOBILITY_UNIT_EG = np.array([0, 8, 7, 15, 24, 0], dtype=np.int64)
else:
    MOBILITY_UNIT_MG = np.array([0, 4, 3, 2, 1, 0], dtype=np.int64)
    MOBILITY_UNIT_EG = np.array([0, 2, 3, 2, 2, 0], dtype=np.int64)

# Rebound to plain module-level names (not accessed as T.PASSED_PAWN_MASK
# etc.) so they resolve as njit globals the same well-established way
# _DEBRUIJN_TABLE below does -- cross-module attribute chains aren't used
# as njit globals anywhere else in this file, so this isn't a pattern
# worth introducing here either.
_PASSED_PAWN_MASK = T.PASSED_PAWN_MASK
_FILE_MASKS = T.FILE_MASKS
_ADJACENT_FILES_MASK = T.ADJACENT_FILES_MASK


@njit(numba.int64(numba.int64, numba.int64, numba.int64, numba.int64), cache=False)
def pack_move(from_sq, to_sq, promo, flag):
    return from_sq | (to_sq << 6) | (promo << 12) | (flag << 15)


_DEBRUIJN64 = np.uint64(0x03F79D71B4CB0A89)
_DEBRUIJN_TABLE = np.array([0, 1, 48, 2, 57, 49, 28, 3, 61, 58, 50, 42, 38, 29, 17, 4, 62, 55, 59, 36, 53, 51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5, 63, 47, 56, 27, 60, 41, 37, 16, 54, 35, 52, 21, 44, 32, 23, 11, 46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9, 13, 8, 7, 6], dtype=np.int64)


@njit(cache=False)
def _bit_scan(bb):
    """Index of the lowest set bit, via the standard De Bruijn multiply
    (numpy uint64 scalars don't support .bit_length() under numba).
    Caller guarantees bb != 0."""
    isolated = bb & (~bb + np.uint64(1))  # lowest set bit only, i.e. bb & -bb
    index = (isolated * _DEBRUIJN64) >> np.uint64(58)
    return _DEBRUIJN_TABLE[np.int64(index)]


@njit(cache=False)
def rook_attacks_fast(square, occupied, masks, magics, shifts, offsets, table):
    blockers = np.uint64(occupied) & masks[square]
    index = (blockers * magics[square]) >> np.uint64(shifts[square])
    return table[offsets[square] + np.int64(index)]


@njit(cache=False)
def bishop_attacks_fast(square, occupied, masks, magics, shifts, offsets, table):
    blockers = np.uint64(occupied) & masks[square]
    index = (blockers * magics[square]) >> np.uint64(shifts[square])
    return table[offsets[square] + np.int64(index)]


@njit(numba.uint64(numba.uint64[:], numba.int64), cache=False)
def occupied_co(pieces, color):
    occ = np.uint64(0)
    base = color * 6
    for i in range(6):
        occ |= pieces[base + i]
    return occ


@njit(cache=False)
def occupied_all(pieces):
    return occupied_co(pieces, 0) | occupied_co(pieces, 1)


@njit(cache=False)
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


@njit(cache=False)
def king_square(pieces, color):
    return _bit_scan(pieces[color * 6 + KING])


@njit(cache=False)
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


@njit(cache=False)
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


@njit(cache=False)
def make_move(pieces, mailbox, meta, move, castle_rook_from, castle_rook_to, castle_all_rook_squares, castle_all_rook_bits,
              zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
              eval_state, pst_mg, pst_eg, phase_weight):
    """``zobrist`` and ``eval_state`` ([mg_score, eg_score, phase], all
    int64) are mutable out-params (numba's array-of-length-1/3 idiom for
    "pass by reference"). Every site that adds or removes a piece bit
    updates both alongside the bitboard/mailbox change itself -- same
    principle as the Zobrist XORs in stage 3, just += / -= instead of ^=."""
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

    h = zobrist[0]
    mg, eg, phase = eval_state[0], eval_state[1], eval_state[2]

    h ^= zobrist_piece[piece_idx, from_sq]
    mg -= pst_mg[piece_idx, from_sq]
    eg -= pst_eg[piece_idx, from_sq]
    phase -= phase_weight[piece_idx]

    if flag == FLAG_EP:
        captured_square = to_sq - push_dir
        captured_idx = mailbox[captured_square]
        undo_captured_piece = captured_idx
        undo_captured_square = captured_square
        pieces[captured_idx] &= ~(np.uint64(1) << np.uint64(captured_square))
        mailbox[captured_square] = -1
        h ^= zobrist_piece[captured_idx, captured_square]
        mg -= pst_mg[captured_idx, captured_square]
        eg -= pst_eg[captured_idx, captured_square]
        phase -= phase_weight[captured_idx]
    else:
        captured_idx = mailbox[to_sq]
        if captured_idx != -1:
            undo_captured_piece = captured_idx
            undo_captured_square = to_sq
            pieces[captured_idx] &= ~(np.uint64(1) << np.uint64(to_sq))
            h ^= zobrist_piece[captured_idx, to_sq]
            mg -= pst_mg[captured_idx, to_sq]
            eg -= pst_eg[captured_idx, to_sq]
            phase -= phase_weight[captured_idx]

    pieces[piece_idx] &= ~(np.uint64(1) << np.uint64(from_sq))
    final_piece_type = promo if promo != NO_PROMO else piece_type
    final_piece_idx = color * 6 + final_piece_type
    pieces[final_piece_idx] |= np.uint64(1) << np.uint64(to_sq)
    mailbox[from_sq] = -1
    mailbox[to_sq] = final_piece_idx
    h ^= zobrist_piece[final_piece_idx, to_sq]
    mg += pst_mg[final_piece_idx, to_sq]
    eg += pst_eg[final_piece_idx, to_sq]
    phase += phase_weight[final_piece_idx]

    if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
        side = 0 if flag == FLAG_CASTLE_K else 1
        rook_from = castle_rook_from[color, side]
        rook_to = castle_rook_to[color, side]
        rook_idx = color * 6 + ROOK
        pieces[rook_idx] &= ~(np.uint64(1) << np.uint64(rook_from))
        pieces[rook_idx] |= np.uint64(1) << np.uint64(rook_to)
        mailbox[rook_from] = -1
        mailbox[rook_to] = rook_idx
        h ^= zobrist_piece[rook_idx, rook_from]
        h ^= zobrist_piece[rook_idx, rook_to]
        mg += pst_mg[rook_idx, rook_to] - pst_mg[rook_idx, rook_from]
        eg += pst_eg[rook_idx, rook_to] - pst_eg[rook_idx, rook_from]
        # phase unaffected: same rook, just relocated

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

    h ^= zobrist_castling[undo_castling_rights]
    h ^= zobrist_castling[rights]
    if undo_ep_square != -1:
        h ^= zobrist_ep_file[undo_ep_square % 8]
    if new_ep != -1:
        h ^= zobrist_ep_file[new_ep % 8]
    h ^= zobrist_side

    meta[0] = opponent
    meta[1] = rights
    meta[2] = new_ep
    meta[3] = new_halfmove
    meta[4] = meta[4] + (1 if color == 1 else 0)
    zobrist[0] = h
    eval_state[0], eval_state[1], eval_state[2] = mg, eg, phase

    return undo_castling_rights, undo_ep_square, undo_halfmove_clock, undo_captured_piece, undo_captured_square


@njit(cache=False)
def unmake_move(pieces, mailbox, meta, move, undo_castling_rights, undo_ep_square, undo_halfmove_clock, undo_captured_piece, undo_captured_square, castle_rook_from, castle_rook_to,
                zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                eval_state, pst_mg, pst_eg, phase_weight):
    """XOR is its own inverse, so the hash update redoes the same XOR
    terms make_move applied. eval_state isn't self-inverse the same way
    (+=/-= aren't involutions) -- each site below applies the exact
    opposite sign of what make_move did there."""
    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    promo = (move >> 12) & 0x7
    flag = (move >> 15) & 0x7

    opponent = meta[0]
    color = 1 - opponent
    current_rights = meta[1]
    current_ep = meta[2]

    meta[0] = color
    meta[1] = undo_castling_rights
    meta[2] = undo_ep_square
    meta[3] = undo_halfmove_clock
    meta[4] = meta[4] - (1 if color == 1 else 0)

    final_piece_idx = mailbox[to_sq]
    final_piece_type = final_piece_idx % 6
    original_piece_type = PAWN if promo != NO_PROMO else final_piece_type
    original_piece_idx = color * 6 + original_piece_type

    h = zobrist[0]
    mg, eg, phase = eval_state[0], eval_state[1], eval_state[2]

    h ^= zobrist_piece[final_piece_idx, to_sq]
    mg -= pst_mg[final_piece_idx, to_sq]
    eg -= pst_eg[final_piece_idx, to_sq]
    phase -= phase_weight[final_piece_idx]

    pieces[final_piece_idx] &= ~(np.uint64(1) << np.uint64(to_sq))
    pieces[original_piece_idx] |= np.uint64(1) << np.uint64(from_sq)
    mailbox[to_sq] = -1
    mailbox[from_sq] = original_piece_idx
    h ^= zobrist_piece[original_piece_idx, from_sq]
    mg += pst_mg[original_piece_idx, from_sq]
    eg += pst_eg[original_piece_idx, from_sq]
    phase += phase_weight[original_piece_idx]

    if flag == FLAG_CASTLE_K or flag == FLAG_CASTLE_Q:
        side = 0 if flag == FLAG_CASTLE_K else 1
        rook_from = castle_rook_from[color, side]
        rook_to = castle_rook_to[color, side]
        rook_idx = color * 6 + ROOK
        pieces[rook_idx] &= ~(np.uint64(1) << np.uint64(rook_to))
        pieces[rook_idx] |= np.uint64(1) << np.uint64(rook_from)
        mailbox[rook_to] = -1
        mailbox[rook_from] = rook_idx
        h ^= zobrist_piece[rook_idx, rook_to]
        h ^= zobrist_piece[rook_idx, rook_from]
        mg += pst_mg[rook_idx, rook_from] - pst_mg[rook_idx, rook_to]
        eg += pst_eg[rook_idx, rook_from] - pst_eg[rook_idx, rook_to]

    if undo_captured_piece != -1:
        pieces[undo_captured_piece] |= np.uint64(1) << np.uint64(undo_captured_square)
        mailbox[undo_captured_square] = undo_captured_piece
        h ^= zobrist_piece[undo_captured_piece, undo_captured_square]
        mg += pst_mg[undo_captured_piece, undo_captured_square]
        eg += pst_eg[undo_captured_piece, undo_captured_square]
        phase += phase_weight[undo_captured_piece]

    h ^= zobrist_castling[current_rights]
    h ^= zobrist_castling[undo_castling_rights]
    if current_ep != -1:
        h ^= zobrist_ep_file[current_ep % 8]
    if undo_ep_square != -1:
        h ^= zobrist_ep_file[undo_ep_square % 8]
    h ^= zobrist_side

    zobrist[0] = h
    eval_state[0], eval_state[1], eval_state[2] = mg, eg, phase


@njit(cache=False)
def perft(pieces, mailbox, meta, depth, moves_buf_stack, stack_ptr,
          rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
          bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
          knight_attacks, king_attacks, pawn_attacks,
          castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
          castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
          zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
          eval_state, pst_mg, pst_eg, phase_weight):
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
        undo = make_move(pieces, mailbox, meta, move, castle_rook_from, castle_rook_to, castle_all_rook_squares, castle_all_rook_bits,
                          zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                          eval_state, pst_mg, pst_eg, phase_weight)
        if not in_check(pieces, color,
                         rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                         bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                         knight_attacks, king_attacks, pawn_attacks):
            nodes += perft(pieces, mailbox, meta, depth - 1, moves_buf_stack, stack_ptr + 1,
                           rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                           bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                           knight_attacks, king_attacks, pawn_attacks,
                           castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                           castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                           zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                           eval_state, pst_mg, pst_eg, phase_weight)
        unmake_move(pieces, mailbox, meta, move, undo[0], undo[1], undo[2], undo[3], undo[4], castle_rook_from, castle_rook_to,
                    zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                    eval_state, pst_mg, pst_eg, phase_weight)

    return nodes


# ---------------------------------------------------------------------
# Python-side convenience layer: FEN <-> flat state, and a perft driver
# that binds all the constant-table arguments so callers don't have to.
# ---------------------------------------------------------------------

PIECE_CHARS = "pnbrqk"
_PROMO_LETTER = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}


def move_to_uci(move: int) -> str:
    """Packed int32 move -> UCI string (e.g. "e2e4", "e7e8q")."""
    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    promo = (move >> 12) & 0x7
    files = "abcdefgh"
    uci = f"{files[from_sq % 8]}{from_sq // 8 + 1}{files[to_sq % 8]}{to_sq // 8 + 1}"
    if promo != NO_PROMO:
        uci += _PROMO_LETTER[promo]
    return uci


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
        self.zobrist_piece = T.ZOBRIST_PIECE.reshape(12, 64)  # [color*6+piece_type, square]
        self.zobrist_castling = T.ZOBRIST_CASTLING
        self.zobrist_ep_file = T.ZOBRIST_EP_FILE
        self.zobrist_side = T.ZOBRIST_SIDE
        self.pst_mg = PST_SIGNED_MG
        self.pst_eg = PST_SIGNED_EG
        self.phase_weight = PHASE_WEIGHT_BY_IDX


_TABLES = Tables()


@njit(cache=False)
def zobrist_hash_from_scratch(pieces, meta, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side):
    h = np.uint64(0)
    for idx in range(12):
        bb = pieces[idx]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            h ^= zobrist_piece[idx, square]
    h ^= zobrist_castling[meta[1]]
    if meta[2] != -1:
        h ^= zobrist_ep_file[meta[2] % 8]
    if meta[0] == 1:
        h ^= zobrist_side
    return h


def compute_hash(pieces, meta) -> int:
    t = _TABLES
    return int(zobrist_hash_from_scratch(pieces, meta, t.zobrist_piece, t.zobrist_castling, t.zobrist_ep_file, t.zobrist_side))


@njit(cache=False)
def eval_state_from_scratch(pieces, pst_mg, pst_eg, phase_weight):
    mg = np.int64(0)
    eg = np.int64(0)
    phase = np.int64(0)
    for idx in range(12):
        bb = pieces[idx]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            mg += pst_mg[idx, square]
            eg += pst_eg[idx, square]
            phase += phase_weight[idx]
    return mg, eg, phase


def compute_eval_state(pieces) -> tuple:
    t = _TABLES
    mg, eg, phase = eval_state_from_scratch(pieces, t.pst_mg, t.pst_eg, t.phase_weight)
    return int(mg), int(eg), int(phase)


@njit(cache=False)
def _popcount(bb):
    """Number of set bits. numba doesn't support uint64.bit_count()
    (same gap that made _bit_scan below need a manual De Bruijn multiply
    instead of .bit_length()) -- pawns-per-file is at most a handful, so
    a clear-the-lowest-bit loop is plenty fast without needing an O(1)
    SWAR trick."""
    count = 0
    while bb != np.uint64(0):
        count += 1
        bb &= bb - np.uint64(1)
    return count


@njit(cache=False)
def _pawn_structure_score(pieces):
    """White-minus-black (mg, eg) isolated/doubled pawn penalties.
    Doubled: a penalty per pawn beyond the first on a file. Isolated: no
    friendly pawn on either adjacent file, checked once per occupied
    file (every pawn on that file shares the same verdict). Like passed
    pawns, computed fresh from the piece bitboards at eval time rather
    than maintained incrementally."""
    mg = 0
    eg = 0
    for color in (0, 1):
        pawns = pieces[color * 6 + PAWN]
        for file in range(8):
            count = _popcount(pawns & _FILE_MASKS[file])
            if count == 0:
                continue
            penalty_mg = 0
            penalty_eg = 0
            if count > 1:
                penalty_mg += DOUBLED_PENALTY_MG * (count - 1)
                penalty_eg += DOUBLED_PENALTY_EG * (count - 1)
            if (pawns & _ADJACENT_FILES_MASK[file]) == np.uint64(0):
                penalty_mg += ISOLATED_PENALTY_MG
                penalty_eg += ISOLATED_PENALTY_EG
            if color == 0:
                mg -= penalty_mg
                eg -= penalty_eg
            else:
                mg += penalty_mg
                eg += penalty_eg
    return mg, eg


@njit(cache=False)
def _mobility_score(pieces, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                     bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                     knight_attacks):
    """White-minus-black (mg, eg) mobility bonus for knights/bishops/
    rooks/queens: a small per-piece-type-weighted bonus per pseudo-
    legally reachable square not occupied by a piece of the same colour.
    Pseudo-legal on purpose (doesn't check whether a move would leave
    the king in check) -- this is a positional heuristic evaluated at
    every leaf, not a legality check, and full legality filtering here
    would cost far more than the term is worth. King mobility isn't
    scored here (folded into king safety instead, when that term
    lands)."""
    mg = 0
    eg = 0
    occ = occupied_all(pieces)
    for color in (0, 1):
        not_own = ~occupied_co(pieces, color)
        term_mg = 0
        term_eg = 0

        bb = pieces[color * 6 + KNIGHT]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            n = _popcount(knight_attacks[square] & not_own)
            term_mg += n * MOBILITY_UNIT_MG[KNIGHT]
            term_eg += n * MOBILITY_UNIT_EG[KNIGHT]

        bb = pieces[color * 6 + BISHOP]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            attacks = bishop_attacks_fast(square, occ, bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table)
            n = _popcount(attacks & not_own)
            term_mg += n * MOBILITY_UNIT_MG[BISHOP]
            term_eg += n * MOBILITY_UNIT_EG[BISHOP]

        bb = pieces[color * 6 + ROOK]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            attacks = rook_attacks_fast(square, occ, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table)
            n = _popcount(attacks & not_own)
            term_mg += n * MOBILITY_UNIT_MG[ROOK]
            term_eg += n * MOBILITY_UNIT_EG[ROOK]

        bb = pieces[color * 6 + QUEEN]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            r_attacks = rook_attacks_fast(square, occ, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table)
            b_attacks = bishop_attacks_fast(square, occ, bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table)
            n = _popcount((r_attacks | b_attacks) & not_own)
            term_mg += n * MOBILITY_UNIT_MG[QUEEN]
            term_eg += n * MOBILITY_UNIT_EG[QUEEN]

        if color == 0:
            mg += term_mg
            eg += term_eg
        else:
            mg -= term_mg
            eg -= term_eg
    return mg, eg


@njit(cache=False)
def _king_safety_score(pieces, castling_rights, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                        bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                        knight_attacks, king_attacks):
    """White-minus-black middlegame-only king safety penalty:

      - pawn shield: the 3 squares one rank in front of the king (its own
        file and the two adjacent ones) -- a penalty per missing pawn,
        scaled up by how much attacking material (queen/rooks) the
        opponent still has.
      - king's own file open (no pawns of either colour) or semi-open
        (no *own* pawns) -- undefended by a friendly pawn wall.
      - enemy knights/bishops/rooks/queens that pseudo-legally attack any
        square in the king's immediate zone (its square + 8 neighbours,
        via king_attacks -- reused as a "zone" mask, not a legal-king-
        move check), weighted by piece type.
      - castling incentive: bonus for having castled, penalty for having
        lost the right to without castling while the opponent still has
        real attacking material (see CASTLED_BONUS_MG's comment).

    Reuses mobility's exact same attack-table parameters (already
    threaded through evaluate_from_state); castling_rights is meta[1],
    threaded in by every caller alongside the position itself."""
    mg = 0
    occ = occupied_all(pieces)
    for color in (0, 1):
        enemy = 1 - color
        king_sq = _bit_scan(pieces[color * 6 + KING])
        king_rank, king_file = divmod(king_sq, 8)
        own_pawns = pieces[color * 6 + PAWN]
        enemy_pawns = pieces[enemy * 6 + PAWN]
        penalty = 0

        attacking_units = (
            _popcount(pieces[enemy * 6 + QUEEN]) * ATTACKING_MATERIAL_QUEEN_UNITS
            + _popcount(pieces[enemy * 6 + ROOK]) * ATTACKING_MATERIAL_ROOK_UNITS
        )

        shield_rank = king_rank + 1 if color == 0 else king_rank - 1
        if 0 <= shield_rank < 8:
            for f in (king_file - 1, king_file, king_file + 1):
                if 0 <= f < 8:
                    shield_sq = shield_rank * 8 + f
                    if (own_pawns & (np.uint64(1) << np.uint64(shield_sq))) == np.uint64(0):
                        penalty += KING_SHIELD_PENALTY_MG + KING_SHIELD_MATERIAL_BONUS_PER_UNIT_MG * attacking_units

        king_file_mask = _FILE_MASKS[king_file]
        if (own_pawns & king_file_mask) == np.uint64(0):
            if (enemy_pawns & king_file_mask) == np.uint64(0):
                penalty += KING_OPEN_FILE_PENALTY_MG
            else:
                penalty += KING_SEMI_OPEN_FILE_PENALTY_MG

        zone = king_attacks[king_sq]
        attacker_weight = 0

        bb = pieces[enemy * 6 + KNIGHT]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            if (knight_attacks[square] & zone) != np.uint64(0):
                attacker_weight += KING_ATTACKER_WEIGHT[KNIGHT]

        bb = pieces[enemy * 6 + BISHOP]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            attacks = bishop_attacks_fast(square, occ, bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table)
            if (attacks & zone) != np.uint64(0):
                attacker_weight += KING_ATTACKER_WEIGHT[BISHOP]

        bb = pieces[enemy * 6 + ROOK]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            attacks = rook_attacks_fast(square, occ, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table)
            if (attacks & zone) != np.uint64(0):
                attacker_weight += KING_ATTACKER_WEIGHT[ROOK]

        bb = pieces[enemy * 6 + QUEEN]
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            r_attacks = rook_attacks_fast(square, occ, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table)
            b_attacks = bishop_attacks_fast(square, occ, bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table)
            if ((r_attacks | b_attacks) & zone) != np.uint64(0):
                attacker_weight += KING_ATTACKER_WEIGHT[QUEEN]

        penalty += attacker_weight * KING_ATTACKER_PENALTY_PER_UNIT_MG

        king_side_bit = np.int64(1) if color == 0 else np.int64(4)
        queen_side_bit = np.int64(2) if color == 0 else np.int64(8)
        has_rights = (castling_rights & (king_side_bit | queen_side_bit)) != 0
        castled_kingside_sq = 6 if color == 0 else 62
        castled_queenside_sq = 2 if color == 0 else 58
        is_on_castled_square = king_sq == castled_kingside_sq or king_sq == castled_queenside_sq

        if not has_rights:
            if is_on_castled_square:
                penalty -= CASTLED_BONUS_MG
            elif attacking_units > 0:
                penalty += UNCASTLED_EXPOSED_PENALTY_MG

        if color == 0:
            mg -= penalty
        else:
            mg += penalty
    return mg


@njit(cache=False)
def _bishop_pair_score(pieces):
    """White-minus-black (mg, eg) bonus for having both bishops. Just a
    popcount check on the bishop bitboard per side -- no need for the
    per-piece bit-scan loop the other terms use."""
    mg = 0
    eg = 0
    white_bishops = _popcount(pieces[0 * 6 + BISHOP])
    black_bishops = _popcount(pieces[1 * 6 + BISHOP])
    if white_bishops >= 2:
        mg += BISHOP_PAIR_BONUS_MG
        eg += BISHOP_PAIR_BONUS_EG
    if black_bishops >= 2:
        mg -= BISHOP_PAIR_BONUS_MG
        eg -= BISHOP_PAIR_BONUS_EG
    return mg, eg


@njit(cache=False)
def _rook_file_score(pieces):
    """White-minus-black (mg, eg) bonus for rooks on open/semi-open
    files. Computed fresh at eval time like the other pawn-structure-
    dependent terms above -- "are there pawns of either colour on this
    file" isn't something a rook's own make_move/unmake_move touches."""
    mg = 0
    eg = 0
    for color in (0, 1):
        enemy = 1 - color
        own_pawns = pieces[color * 6 + PAWN]
        enemy_pawns = pieces[enemy * 6 + PAWN]
        rooks = pieces[color * 6 + ROOK]
        bb = rooks
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            file_mask = _FILE_MASKS[square % 8]
            has_own = (own_pawns & file_mask) != np.uint64(0)
            has_enemy = (enemy_pawns & file_mask) != np.uint64(0)
            if has_own:
                continue  # own pawn on the file -- neither open nor semi-open
            bonus_mg = ROOK_OPEN_FILE_BONUS_MG if not has_enemy else ROOK_SEMI_OPEN_FILE_BONUS_MG
            bonus_eg = ROOK_OPEN_FILE_BONUS_EG if not has_enemy else ROOK_SEMI_OPEN_FILE_BONUS_EG
            if color == 0:
                mg += bonus_mg
                eg += bonus_eg
            else:
                mg -= bonus_mg
                eg -= bonus_eg
    return mg, eg


@njit(cache=False)
def _chebyshev_distance(sq_a, sq_b):
    file_diff = abs((sq_a % 8) - (sq_b % 8))
    rank_diff = abs((sq_a // 8) - (sq_b // 8))
    return file_diff if file_diff > rank_diff else rank_diff


@njit(cache=False)
def _passed_pawn_score(pieces):
    """White-minus-black (mg, eg) passed-pawn bonus, plus (2026-09,
    endgame-strength pass) a king-proximity term, EG-only: for each
    passed pawn, the side whose king is closer to that pawn's promotion
    square than the opponent's king gets a bonus scaled by the distance
    difference. This is one of the most standard, highest-value
    endgame heuristics in any engine ("the side with the active king
    wins king-and-pawn races") and was entirely absent before -- the
    king's EG PST already rewards general centralization, but has no
    idea a specific passed pawn needs escorting or stopping.

    Unlike material/PST, none of this is maintained incrementally
    through make_move/unmake_move -- "is this pawn's file-triple clear
    of enemy pawns" and "how far is each king from here" don't decompose
    into simple per-move deltas the way a piece landing on/leaving a
    square does, so it's recomputed from the piece bitboards at every
    leaf eval instead (cheap: a handful of pawns, one mask-and-compare
    and two distance checks each)."""
    mg = 0
    eg = 0
    for color in (0, 1):
        enemy = 1 - color
        own_pawns = pieces[color * 6 + PAWN]
        enemy_pawns = pieces[enemy * 6 + PAWN]
        own_king_sq = _bit_scan(pieces[color * 6 + KING])
        enemy_king_sq = _bit_scan(pieces[enemy * 6 + KING])
        bb = own_pawns
        while bb != np.uint64(0):
            square = _bit_scan(bb)
            bb &= bb - np.uint64(1)
            mask = _PASSED_PAWN_MASK[color, square]
            if (enemy_pawns & mask) == np.uint64(0):
                rank = square // 8
                rel_rank = rank if color == 0 else 7 - rank
                file = square % 8
                promo_sq = file + (56 if color == 0 else 0)
                own_dist = _chebyshev_distance(own_king_sq, promo_sq)
                enemy_dist = _chebyshev_distance(enemy_king_sq, promo_sq)
                king_bonus_eg = KING_PAWN_PROXIMITY_WEIGHT_EG * (enemy_dist - own_dist)
                if color == 0:
                    mg += PASSED_PAWN_BONUS_MG[rel_rank]
                    eg += PASSED_PAWN_BONUS_EG[rel_rank] + king_bonus_eg
                else:
                    mg -= PASSED_PAWN_BONUS_MG[rel_rank]
                    eg -= PASSED_PAWN_BONUS_EG[rel_rank] + king_bonus_eg
    return mg, eg


@njit(cache=False)
def evaluate_from_state(eval_state, turn, pieces, castling_rights, alpha, beta,
                         rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                         bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                         knight_attacks, king_attacks):
    """Tapered score from ``turn``'s perspective, design doc §6 formula --
    same tapering as v1's cb_eval.py, just reading incrementally
    maintained material+PST totals instead of rescanning the board, plus
    the (non-incremental) pawn-structure/mobility/king-safety terms
    computed fresh here. Takes the magic-bitboard attack tables (needed
    for mobility's and king safety's bishop/rook/queen attacks) because
    numba njit functions in this codebase take flat array parameters
    rather than a bundled object -- same reason root_search/quiescence/
    etc. all have long parameter lists instead of taking one Tables
    instance.

    Lazy eval: material+PST alone (the cheap, incrementally-maintained
    part) is computed first. If it already misses the caller's
    [alpha, beta] search window by more than LAZY_EVAL_MARGIN, the
    position is decided either way and it's returned as-is -- refining
    it with pawn structure/mobility/etc. wouldn't change what the search
    does with it, so that work (magic-bitboard mobility lookups
    especially) is skipped for the (majority of) leaves where it can't
    matter. alpha/beta are in ``turn``'s perspective, matching every
    other alpha/beta in this search (negamax convention).

    ENABLE_EXTENDED_EVAL (CB_NB_ENABLE_EXTENDED_EVAL) turns all six terms
    off at once, always returning the material+PST score -- for the
    batch SPRT of "all six terms" against the material+PST-only
    baseline, not per-term (a per-term SPRT was rejected: six separate
    attribution runs when only the batched accept/reject decision will
    actually be acted on)."""
    mg, eg, phase = eval_state[0], eval_state[1], eval_state[2]
    phase = min(phase, MAX_PHASE)
    phase_256 = (phase * 256) // MAX_PHASE

    lazy_score = (mg * phase_256 + eg * (256 - phase_256)) // 256
    lazy_score = lazy_score if turn == 0 else -lazy_score
    if not ENABLE_EXTENDED_EVAL:
        return lazy_score
    if lazy_score < alpha - LAZY_EVAL_MARGIN or lazy_score > beta + LAZY_EVAL_MARGIN:
        return lazy_score

    pp_mg, pp_eg = _passed_pawn_score(pieces)
    ps_mg, ps_eg = _pawn_structure_score(pieces)
    rf_mg, rf_eg = _rook_file_score(pieces)
    bp_mg, bp_eg = _bishop_pair_score(pieces)
    mob_mg, mob_eg = _mobility_score(pieces, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                      bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                      knight_attacks)
    ks_mg = _king_safety_score(pieces, castling_rights, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                knight_attacks, king_attacks)
    mg += pp_mg + ps_mg + rf_mg + bp_mg + mob_mg + ks_mg
    eg += pp_eg + ps_eg + rf_eg + bp_eg + mob_eg
    score = (mg * phase_256 + eg * (256 - phase_256)) // 256
    return score if turn == 0 else -score


def new_eval_state(pieces):
    mg, eg, phase = compute_eval_state(pieces)
    return np.array([mg, eg, phase], dtype=np.int64)


def run_perft(fen: str, depth: int) -> int:
    pieces, mailbox, meta = fen_to_state(fen)
    moves_buf_stack = np.zeros((depth + 1, MAX_MOVES), dtype=np.int64)
    zobrist = np.zeros(1, dtype=np.uint64)  # perft doesn't use the hash/eval; just needs to satisfy make_move's signature
    eval_state = new_eval_state(pieces)
    t = _TABLES
    return perft(
        pieces, mailbox, meta, depth, moves_buf_stack, 0,
        t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks, t.king_attacks, t.pawn_attacks,
        t.castle_king_to, t.castle_rook_from, t.castle_rook_to, t.castle_right_bit,
        t.castle_empty_squares, t.castle_king_path, t.castle_all_rook_squares, t.castle_all_rook_bits,
        zobrist, t.zobrist_piece, t.zobrist_castling, t.zobrist_ep_file, t.zobrist_side,
        eval_state, t.pst_mg, t.pst_eg, t.phase_weight,
    )


def make_move_simple(pieces, mailbox, meta, move, zobrist, eval_state):
    """Convenience wrapper binding the constant tables, for debug-mode
    verification and future search code that doesn't want to thread all
    the table arguments through by hand."""
    t = _TABLES
    return make_move(pieces, mailbox, meta, move, t.castle_rook_from, t.castle_rook_to, t.castle_all_rook_squares, t.castle_all_rook_bits,
                      zobrist, t.zobrist_piece, t.zobrist_castling, t.zobrist_ep_file, t.zobrist_side,
                      eval_state, t.pst_mg, t.pst_eg, t.phase_weight)


def unmake_move_simple(pieces, mailbox, meta, move, undo, zobrist, eval_state):
    t = _TABLES
    unmake_move(pieces, mailbox, meta, move, undo[0], undo[1], undo[2], undo[3], undo[4], t.castle_rook_from, t.castle_rook_to,
                zobrist, t.zobrist_piece, t.zobrist_castling, t.zobrist_ep_file, t.zobrist_side,
                eval_state, t.pst_mg, t.pst_eg, t.phase_weight)


def generate_legal_moves_simple(pieces, mailbox, meta):
    """Full legal-move generation (pseudo-legal + king-safety filter),
    tables bound. For debug/verification harnesses, not the search hot
    path (which should stay inlined in njit -- see stage 5)."""
    t = _TABLES
    moves_buf = np.zeros(MAX_MOVES, dtype=np.int64)
    count = generate_pseudo_legal_moves(
        pieces, mailbox, meta, moves_buf,
        t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks, t.king_attacks, t.pawn_attacks,
        t.castle_king_to, t.castle_rook_from, t.castle_right_bit,
        t.castle_empty_squares, t.castle_king_path,
    )
    color = meta[0]
    legal = []
    zobrist_scratch = np.zeros(1, dtype=np.uint64)
    eval_scratch = new_eval_state(pieces)
    for i in range(count):
        move = int(moves_buf[i])
        undo = make_move_simple(pieces, mailbox, meta, move, zobrist_scratch, eval_scratch)
        if not in_check(pieces, color,
                         t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
                         t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
                         t.knight_attacks, t.king_attacks, t.pawn_attacks):
            legal.append(move)
        unmake_move_simple(pieces, mailbox, meta, move, undo, zobrist_scratch, eval_scratch)
    return legal
