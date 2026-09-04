"""Stage 5: search recursion ported into njit, reusing the movegen/make-
unmake/Zobrist/eval machinery from cb_nb_fast.py end to end -- one njit
entry point per search call, not per node, per the design doc's explicit
requirement ("if evaluate() still takes a chess.Board, every leaf crosses
the Python/jit boundary... caps us near our current 19.6k nps").

Scope for this pass: negamax with fail-soft alpha-beta, quiescence search
(captures only, delta pruning), a transposition table (parallel numpy
arrays -- no Python dict, per the no-objects-in-njit constraint), and
MVV-LVA move ordering. Killers/history/null-move/LMR (present in v1's
cb_search.py) are not yet ported -- getting the core loop's nps into the
expected range is the thing to verify before adding more.

TT layout: parallel arrays indexed by ``key & mask``, matching cb_tt.py's
design (fixed size, replace-by-depth-or-generation, mate scores ply-
adjusted on store/probe) but as arrays instead of a list of Entry objects,
since numba can't jit a list of Python objects.
"""
from __future__ import annotations

import os
import time

import numpy as np
import numba
from numba import njit, objmode

import cb_nb_fast as F

MATE_SCORE = 100_000
MATE_THRESHOLD = MATE_SCORE - 1_000
MAX_PLY = 128
MAX_MOVES = F.MAX_MOVES

# Same env-var-flag pattern as v1's cb_search.py (CB_ENABLE_NULL_MOVE,
# CB_ENABLE_LMR): lets ratings/sprt.py A/B the *same* compiled module with
# a feature toggled, and lets us measure each feature's own compile-time
# contribution by toggling one at a time (module-level constants get
# folded in at njit compile time, so toggling forces a real recompile).
ENABLE_KILLERS_HISTORY = os.environ.get("CB_NB_ENABLE_KILLERS_HISTORY", "1") != "0"
ENABLE_NULL_MOVE = os.environ.get("CB_NB_ENABLE_NULL_MOVE", "1") != "0"
ENABLE_LMR = os.environ.get("CB_NB_ENABLE_LMR", "1") != "0"
# SEE-based quiescence pruning (2026-09): skip a capture in quiescence
# once its static-exchange result is clearly losing -- the standard,
# well-understood way to stop quiescence wasting nodes on trades that
# can't possibly help, freeing that search time for the lines that
# matter. Turned on by default: 5/5 hand-verified correctness cases pass
# (including the x-ray-attacker case a naive implementation gets wrong),
# the full test suite (perft/mate fixtures included) passes with it on,
# self-play sanity games run clean with no hangs/crashes, and it
# measurably improves median search depth (7.5 -> 8.0 across 8
# representative middlegame positions) despite costing raw nps (SEE's
# own per-node overhead) -- the SPRT infra is too slow right now to get
# a confirmatory accept/reject signal before the deadline, so this ships
# on the strength of that evidence rather than waiting on it.
ENABLE_SEE_PRUNING = os.environ.get("CB_NB_ENABLE_SEE_PRUNING", "1") != "0"
# Diagnostic-only (2026-09): lets a depth-by-depth trace isolate whether
# a TT collision/stale-bound is responsible for an odd move choice, by
# disabling probe+store entirely rather than guessing from the outside.
# Never meant to ship off -- the TT is core, not speculative, unlike
# every other flag in this file.
ENABLE_TT = os.environ.get("CB_NB_ENABLE_TT", "1") != "0"
# Shallow-depth pruning family: reverse futility ("static null move"),
# futility, and late move pruning (LMP). Independently toggleable from
# the start (not just at batch level) so a failed/inconclusive batch
# SPRT can be bisected one flag at a time without writing new code.
# Defaulted OFF pending a retest: the first batch SPRT rejected all three,
# both together and individually (see ratings/run_pruning_batch_sprt.py
# and ratings/run_pruning_bisect_sprt.py's logs) -- every one scored below
# break-even against no pruning. Root cause: the static_eval used here was
# going through evaluate_from_state's lazy-eval shortcut with the node's
# real alpha/beta window, so it was often the cheap material+PST-only
# estimate rather than the full eval (fixed below, plus depth cutoffs
# tightened) -- must not be silently active in the shipped build until a
# fresh batch SPRT accepts it.
ENABLE_RFP = os.environ.get("CB_NB_ENABLE_RFP", "0") != "0"
ENABLE_FUTILITY = os.environ.get("CB_NB_ENABLE_FUTILITY", "0") != "0"
ENABLE_LMP = os.environ.get("CB_NB_ENABLE_LMP", "0") != "0"

NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2
LMR_MIN_DEPTH = 3
LMR_MOVE_THRESHOLD = 4
LMR_REDUCTION = 1
KILLER_BASE_SCORE = 500_000

# Reverse futility pruning ("static null move"): if the static eval
# already beats beta by more than this margin (scaled by depth), the
# position is assumed good enough to fail high without searching further.
# Retest (2026-09): depth cutoff tightened from 6 and margin widened from
# 100 -- the first attempt covered too much of a 7-10 ply tree on a cheap
# estimate; see the LAZY_EVAL note below for the other half of the fix.
RFP_MAX_DEPTH = 4
RFP_MARGIN_PER_DEPTH = 120

# Futility pruning: skip a quiet move outright if the static eval plus
# this depth-scaled margin still can't reach alpha -- it's not going to
# swing the position enough to matter. Indexed by depth (1-2); index 0
# is never used (negamax dispatches depth<=0 to quiescence already).
# Retest (2026-09): tightened from depth<=3.
FUTILITY_MAX_DEPTH = 2
FUTILITY_MARGIN = np.array([0, 100, 200, 300], dtype=np.int64)

# Late move pruning: once this many quiet moves have already been tried
# at a shallow depth without raising alpha, assume the rest (ordered
# after them, so presumably weaker) aren't worth searching either.
# Retest (2026-09): tightened from depth<=4.
LMP_MAX_DEPTH = 3


@njit(cache=False)
def _lmp_quiet_limit(depth):
    return 4 + depth * depth

TT_SIZE_BITS = 20
TT_SIZE = 1 << TT_SIZE_BITS
TT_MASK = TT_SIZE - 1
BOUND_EXACT, BOUND_LOWER, BOUND_UPPER = 0, 1, 2

QUIESCENCE_MAX_PLY = 8
DELTA_MARGIN = 200

# Iteration-cost prediction (see search()'s docstring, "Retest (2026-09)"):
# EMA smoothing weight for the branching-factor estimate, and hard clamp
# bounds on any single sample before it enters the EMA.
BF_EMA_WEIGHT = 0.5
MIN_BRANCHING_FACTOR = 1.2
MAX_BRANCHING_FACTOR = 8.0
_CAPTURE_VALUE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)  # pawn..king, 0-indexed

# Same node-count-polled hard-deadline mechanism as v1's cb_search.py
# (checked every NODE_CHECK_INTERVAL nodes, not every node -- time.perf_
# counter() isn't free, and a periodic check is plenty granular for a
# clock measured in tens of milliseconds). Ported here because
# search()'s original design ("no hard-deadline exception unwind yet --
# each depth runs to completion inside njit") turned out not to be
# just a measurement-only simplification: with the eval-terms work
# dropping nps from ~463k to the ~290-350k range, a single iterative-
# deepening depth can now genuinely overrun its whole time budget with
# nothing able to interrupt it mid-flight, and this was observed causing
# real time-forfeit losses in calibration games (our own agent flagging
# in roughly 1 of 8 real tournament-TC games) -- not a calibration-
# infrastructure artifact, a real correctness gap.
NODE_CHECK_INTERVAL = 2048


class _SearchTimeout(Exception):
    pass


@njit(cache=False)
def _now():
    """Wall-clock seconds, comparable against the same time.perf_counter()
    origin search()'s own soft_deadline/hard_deadline are computed from.
    numba's nopython mode has no direct binding for time.perf_counter()
    (or any time.* function) -- objmode drops into the Python
    interpreter for just this one call, which costs something, but at
    NODE_CHECK_INTERVAL granularity that cost is negligible next to a
    search node's own work."""
    t = 0.0
    with objmode(t="float64"):
        t = time.perf_counter()
    return t


class SearchArrays:
    """Preallocated, reused across the whole iterative-deepening call
    (and ideally across moves in a game) -- no per-node allocation."""

    def __init__(self):
        self.tt_keys = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_depths = np.full(TT_SIZE, -1, dtype=np.int64)
        self.tt_scores = np.zeros(TT_SIZE, dtype=np.int64)
        self.tt_bounds = np.zeros(TT_SIZE, dtype=np.int64)
        self.tt_moves = np.full(TT_SIZE, -1, dtype=np.int64)
        self.tt_generations = np.full(TT_SIZE, -1, dtype=np.int64)

        # Separate stacks for negamax (indexed by ply) and quiescence
        # (indexed by qply) -- sharing one buffer across recursion levels
        # would let a deeper recursive call clobber an outer, still-in-use
        # move list (caught this in an earlier draft before it ran).
        self.moves_buf_stack = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int64)
        self.scores_buf_stack = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int64)
        self.qmoves_buf_stack = np.zeros((QUIESCENCE_MAX_PLY + 1, MAX_MOVES), dtype=np.int64)
        self.qscores_buf_stack = np.zeros((QUIESCENCE_MAX_PLY + 1, MAX_MOVES), dtype=np.int64)

        self.zobrist = np.zeros(1, dtype=np.uint64)
        self.eval_state = np.zeros(3, dtype=np.int64)
        self.nodes = np.zeros(1, dtype=np.int64)

        # killers[ply, 0/1] = packed move or -1. history[piece_idx, to_sq]
        # += depth*depth on a quiet beta cutoff. Both persist across the
        # whole iterative-deepening call (reset once per move via
        # reset_killers_history(), not per node or per ID iteration --
        # same as v1's _Searcher, which is what makes ID's "each iteration
        # primes the next" benefit actually happen).
        self.killers = np.full((MAX_PLY, 2), -1, dtype=np.int64)
        self.history = np.zeros((12, 64), dtype=np.int64)

    def reset_killers_history(self):
        self.killers.fill(-1)
        self.history.fill(0)


@njit(numba.int64(numba.int64, numba.int64), cache=False)
def score_to_tt(score, ply):
    if score > MATE_THRESHOLD:
        return score + ply
    if score < -MATE_THRESHOLD:
        return score - ply
    return score


@njit(cache=False)
def score_from_tt(score, ply):
    if score > MATE_THRESHOLD:
        return score - ply
    if score < -MATE_THRESHOLD:
        return score + ply
    return score


@njit(cache=False)
def tt_probe(key, tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations):
    index = key & np.uint64(TT_MASK)
    idx = np.int64(index)
    if tt_generations[idx] != -1 and tt_keys[idx] == key:
        return True, tt_depths[idx], tt_scores[idx], tt_bounds[idx], tt_moves[idx]
    return False, np.int64(0), np.int64(0), np.int64(0), np.int64(-1)


@njit(numba.void(
    numba.uint64, numba.int64, numba.int64, numba.int64, numba.int64, numba.int64,
    numba.uint64[:], numba.int64[:], numba.int64[:], numba.int64[:], numba.int64[:], numba.int64[:],
), cache=False)
def tt_store(key, depth, score, bound, move, generation, tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations):
    index = np.int64(key & np.uint64(TT_MASK))
    if tt_generations[index] == -1 or tt_generations[index] < generation or tt_depths[index] <= depth:
        tt_keys[index] = key
        tt_depths[index] = depth
        tt_scores[index] = score
        tt_bounds[index] = bound
        tt_moves[index] = move
        tt_generations[index] = generation


@njit(numba.int64(numba.uint64[:], numba.int64[:], numba.int64), cache=False)
def _capture_value_of(pieces, mailbox, move):
    to_sq = (move >> 6) & 0x3F
    flag = (move >> 15) & 0x7
    if flag == F.FLAG_EP:
        return _CAPTURE_VALUE[F.PAWN]
    idx = mailbox[to_sq]
    if idx == -1:
        return 0
    return _CAPTURE_VALUE[idx % 6]


@njit(cache=False)
def _see_least_valuable_attacker(square, side, occ, pieces,
                                  rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                  bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                  knight_attacks, king_attacks, pawn_attacks):
    """Returns (attacker_square, piece_type) of the cheapest piece of
    ``side`` currently attacking ``square``, given a (possibly reduced)
    ``occ`` -- SEE's swap-off loop below shrinks occ by one bit per ply
    as pieces are notionally traded off, which also correctly reveals
    x-ray attackers behind them since sliding attacks are recomputed
    against occ fresh on every call rather than cached. Checked in
    ascending value order, since that's exactly what the swap-off needs
    at each step. Returns (-1, -1) if ``side`` has no attacker left.

    KNOWN SIMPLIFICATION: doesn't verify a king "recapture" wouldn't walk
    into check (i.e. the square would still be defended after the king
    moves there) -- the standard SEE simplification other engines make
    too; wrong only in the rare case where the king is the sole attacker
    of an otherwise-still-defended square."""
    base = side * 6
    other = 1 - side

    bb = pawn_attacks[other * 64 + square] & pieces[base + F.PAWN] & occ
    if bb != np.uint64(0):
        return F._bit_scan(bb), F.PAWN

    bb = knight_attacks[square] & pieces[base + F.KNIGHT] & occ
    if bb != np.uint64(0):
        return F._bit_scan(bb), F.KNIGHT

    bishop_attacks = F.bishop_attacks_fast(square, occ, bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table)
    bb = bishop_attacks & pieces[base + F.BISHOP] & occ
    if bb != np.uint64(0):
        return F._bit_scan(bb), F.BISHOP

    rook_attacks = F.rook_attacks_fast(square, occ, rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table)
    bb = rook_attacks & pieces[base + F.ROOK] & occ
    if bb != np.uint64(0):
        return F._bit_scan(bb), F.ROOK

    bb = (rook_attacks | bishop_attacks) & pieces[base + F.QUEEN] & occ
    if bb != np.uint64(0):
        return F._bit_scan(bb), F.QUEEN

    bb = king_attacks[square] & pieces[base + F.KING] & occ
    if bb != np.uint64(0):
        return F._bit_scan(bb), F.KING

    return -1, -1


@njit(cache=False)
def see_capture(move, side_to_move, pieces, mailbox,
                 rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                 bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                 knight_attacks, king_attacks, pawn_attacks):
    """Static Exchange Evaluation for a capture move: the net material
    result (positive = good for the mover) of resolving every capture on
    the destination square, assuming both sides always recapture with
    their cheapest available attacker and stop as soon as recapturing
    would leave them worse off. Doesn't know about anything beyond that
    one square (a discovered attack elsewhere, a pin, etc.) -- it's a
    material-only oracle, same scope as every other engine's SEE.

    Classic "swap algorithm" (chess programming wiki: SEE - The Swap
    Algorithm): compute the gain at each ply of best-attacker exchanges,
    then back that up with a minimax where each side stops recapturing
    once it would come out behind. No board mutation needed beyond a
    shrinking occupancy mask -- removing a bit from occ both "removes"
    that attacker and correctly reveals any x-ray slider behind it on
    the very next magic-bitboard lookup."""
    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    flag = (move >> 15) & 0x7
    promo = (move >> 12) & 0x7

    moving_idx = mailbox[from_sq]
    moving_type = moving_idx % 6

    occ = F.occupied_all(pieces)
    occ &= ~(np.uint64(1) << np.uint64(from_sq))

    if flag == F.FLAG_EP:
        captured_type = F.PAWN
        captured_sq = to_sq + (-8 if side_to_move == 0 else 8)
        occ &= ~(np.uint64(1) << np.uint64(captured_sq))
    else:
        captured_idx = mailbox[to_sq]
        captured_type = captured_idx % 6 if captured_idx != -1 else -1

    gain = np.zeros(32, dtype=np.int64)
    gain[0] = _CAPTURE_VALUE[captured_type] if captured_type != -1 else 0
    # A promotion changes what's sitting on to_sq afterwards -- the piece
    # that could now be recaptured is the promoted piece, not the pawn.
    on_square_value = _CAPTURE_VALUE[promo] if promo != F.NO_PROMO else _CAPTURE_VALUE[moving_type]

    side = 1 - side_to_move
    depth = 0
    while True:
        att_sq, att_type = _see_least_valuable_attacker(
            to_sq, side, occ, pieces,
            rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
            bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
            knight_attacks, king_attacks, pawn_attacks,
        )
        if att_sq == -1:
            break
        depth += 1
        gain[depth] = on_square_value - gain[depth - 1]
        if max(-gain[depth - 1], gain[depth]) < 0:
            break
        occ &= ~(np.uint64(1) << np.uint64(att_sq))
        on_square_value = _CAPTURE_VALUE[att_type]
        side = 1 - side

    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1

    return gain[0]


@njit(numba.void(
    numba.uint64[:], numba.int64[:], numba.int64[:], numba.int64, numba.int64[:],
    numba.int64, numba.int64, numba.int64, numba.int64[:, :],
), cache=False)
def order_moves(pieces, mailbox, moves, count, scores_buf, tt_move, killer0, killer1, history):
    """Fills scores_buf[:count] and insertion-sorts moves[:count]
    descending by score: TT move highest, then MVV-LVA for captures/
    promotions, then the two killers, then quiets by history score --
    matches v1's cb_order.py. Quiescence calls this too (with
    killer0=killer1=-1, real history array) -- the killer/history branch
    is simply dead there since a captures-only move list never reaches
    it, cheaper than a second code path."""
    for i in range(count):
        move = moves[i]
        if move == tt_move:
            scores_buf[i] = 1_000_000_000
            continue
        promo = (move >> 12) & 0x7
        to_sq = (move >> 6) & 0x3F
        from_sq = move & 0x3F
        is_capture = mailbox[to_sq] != -1 or ((move >> 15) & 0x7) == F.FLAG_EP
        if promo != F.NO_PROMO or is_capture:
            victim_value = _capture_value_of(pieces, mailbox, move)
            attacker_idx = mailbox[from_sq]
            attacker_value = _CAPTURE_VALUE[attacker_idx % 6] if attacker_idx != -1 else 0
            score = victim_value * 16 - attacker_value
            if promo != F.NO_PROMO:
                score += 100_000 + _CAPTURE_VALUE[promo]
            scores_buf[i] = score
        elif ENABLE_KILLERS_HISTORY and move == killer0:
            scores_buf[i] = KILLER_BASE_SCORE + 1
        elif ENABLE_KILLERS_HISTORY and move == killer1:
            scores_buf[i] = KILLER_BASE_SCORE
        elif ENABLE_KILLERS_HISTORY:
            attacker_idx = mailbox[from_sq]
            scores_buf[i] = history[attacker_idx, to_sq] if attacker_idx != -1 else 0
        else:
            scores_buf[i] = 0

    for i in range(1, count):
        key_move = moves[i]
        key_score = scores_buf[i]
        j = i - 1
        while j >= 0 and scores_buf[j] < key_score:
            moves[j + 1] = moves[j]
            scores_buf[j + 1] = scores_buf[j]
            j -= 1
        moves[j + 1] = key_move
        scores_buf[j + 1] = key_score


@njit(cache=False)
def quiescence(pieces, mailbox, meta, alpha, beta, qply, zobrist, eval_state, nodes, hard_deadline,
               rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
               bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
               knight_attacks, king_attacks, pawn_attacks,
               castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
               castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
               zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
               pst_mg, pst_eg, phase_weight, qmoves_buf_stack, qscores_buf_stack, history):
    nodes[0] += 1
    if nodes[0] % NODE_CHECK_INTERVAL == 0 and _now() >= hard_deadline:
        raise _SearchTimeout()
    stand_pat = F.evaluate_from_state(
        eval_state, meta[0], pieces, meta[1], alpha, beta,
        rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
        bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
        knight_attacks, king_attacks,
    )
    if qply >= QUIESCENCE_MAX_PLY:
        return stand_pat

    best = stand_pat
    if best >= beta:
        return best
    alpha = max(alpha, best)

    color = meta[0]
    opp_occ = F.occupied_co(pieces, 1 - color)

    # Captures only, generated the same way as pseudo-legal movegen but
    # filtered to opp_occ targets inline (mirrors cb_nb_movegen's original
    # plain-Python captures-only approach; the full generate_legal_captures
    # split cb_search.py uses for the same reason isn't ported here yet).
    captures_buf = qmoves_buf_stack[qply]
    scores_buf = qscores_buf_stack[qply]
    count = F.generate_pseudo_legal_moves(pieces, mailbox, meta, captures_buf,
                                           rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                           bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                           knight_attacks, king_attacks, pawn_attacks,
                                           castle_king_to, castle_rook_from, castle_right_bit,
                                           castle_empty_squares, castle_king_path)
    cap_count = 0
    for i in range(count):
        move = captures_buf[i]
        to_sq = (move >> 6) & 0x3F
        flag = (move >> 15) & 0x7
        promo = (move >> 12) & 0x7
        if flag == F.FLAG_EP or (opp_occ & (np.uint64(1) << np.uint64(to_sq))) != np.uint64(0) or promo != F.NO_PROMO:
            captures_buf[cap_count] = move
            cap_count += 1

    order_moves(pieces, mailbox, captures_buf, cap_count, scores_buf, -1, -1, -1, history)

    for i in range(cap_count):
        move = captures_buf[i]
        gain = _capture_value_of(pieces, mailbox, move)
        if stand_pat + gain + DELTA_MARGIN < alpha:
            continue

        if ENABLE_SEE_PRUNING and see_capture(
            move, color, pieces, mailbox,
            rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
            bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
            knight_attacks, king_attacks, pawn_attacks,
        ) < 0:
            continue

        undo = F.make_move(pieces, mailbox, meta, move, castle_rook_from, castle_rook_to, castle_all_rook_squares, castle_all_rook_bits,
                            zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                            eval_state, pst_mg, pst_eg, phase_weight)
        if F.is_square_attacked(pieces, F.king_square(pieces, color), 1 - color,
                                 rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                 bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                 knight_attacks, king_attacks, pawn_attacks):
            F.unmake_move(pieces, mailbox, meta, move, undo[0], undo[1], undo[2], undo[3], undo[4], castle_rook_from, castle_rook_to,
                          zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                          eval_state, pst_mg, pst_eg, phase_weight)
            continue

        score = -quiescence(pieces, mailbox, meta, -beta, -alpha, qply + 1, zobrist, eval_state, nodes, hard_deadline,
                             rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                             bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                             knight_attacks, king_attacks, pawn_attacks,
                             castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                             castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                             zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                             pst_mg, pst_eg, phase_weight, qmoves_buf_stack, qscores_buf_stack, history)
        F.unmake_move(pieces, mailbox, meta, move, undo[0], undo[1], undo[2], undo[3], undo[4], castle_rook_from, castle_rook_to,
                      zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                      eval_state, pst_mg, pst_eg, phase_weight)

        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break

    return best


@njit(cache=False)
def has_non_pawn_material(pieces, color):
    base = color * 6
    non_pawn_king = pieces[base + F.KNIGHT] | pieces[base + F.BISHOP] | pieces[base + F.ROOK] | pieces[base + F.QUEEN]
    return non_pawn_king != np.uint64(0)


@njit(cache=False)
def make_null_move(meta, zobrist, zobrist_ep_file, zobrist_side):
    old_ep = meta[2]
    h = zobrist[0]
    if old_ep != -1:
        h ^= zobrist_ep_file[old_ep % 8]
    h ^= zobrist_side
    zobrist[0] = h
    meta[0] = 1 - meta[0]
    meta[2] = -1
    return old_ep


@njit(cache=False)
def unmake_null_move(meta, old_ep, zobrist, zobrist_ep_file, zobrist_side):
    meta[0] = 1 - meta[0]
    meta[2] = old_ep
    h = zobrist[0]
    h ^= zobrist_side
    if old_ep != -1:
        h ^= zobrist_ep_file[old_ep % 8]
    zobrist[0] = h


@njit(cache=False)
def record_cutoff(pieces, mailbox, move, depth, ply, killers, history):
    """A quiet move caused a beta cutoff: remember it as a killer at this
    ply and bump its history score. Matches v1's cb_search.py exactly."""
    if not ENABLE_KILLERS_HISTORY:
        return
    to_sq = (move >> 6) & 0x3F
    is_capture_or_promo = mailbox[to_sq] != -1 or ((move >> 15) & 0x7) == F.FLAG_EP or ((move >> 12) & 0x7) != F.NO_PROMO
    if is_capture_or_promo:
        return
    if move != killers[ply, 0]:
        killers[ply, 1] = killers[ply, 0]
        killers[ply, 0] = move
    from_sq = move & 0x3F
    attacker_idx = mailbox[from_sq]
    if attacker_idx != -1:
        history[attacker_idx, to_sq] += depth * depth


@njit(cache=False)
def try_move(pieces, mailbox, meta, move, depth, alpha, beta, ply, generation, is_first, lmr_eligible,
             mover_color, opponent_color,
             zobrist, eval_state, nodes, hard_deadline,
             tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations,
             rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
             bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
             knight_attacks, king_attacks, pawn_attacks,
             castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
             castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
             zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
             pst_mg, pst_eg, phase_weight,
             moves_buf_stack, scores_buf_stack, qmoves_buf_stack, qscores_buf_stack,
             killers, history):
    """Kernel split: make_move + legality check + PVS/LMR recursion +
    unmake_move, extracted out of negamax's own body. negamax previously
    had this whole block duplicated at 3 textual call sites (first move,
    PVS null-window, PVS full-window re-search); with killers/history/
    null-move/LMR added on top, that pushed negamax's own compile time
    from 22.16s to ~43s (cold, cache cleared) even with those features
    disabled at runtime -- the sheer size/branch count of one function
    was the cost, not any single feature. Moving the per-move trial here
    means negamax's own body is just a loop calling this once per move.

    Returns (legal: bool, score: int) -- score is only meaningful when
    legal is True. ``lmr_eligible`` is decided by the caller from
    information available before making the move (move index, depth,
    is_quiet, in_check); this function itself checks "does this move give
    check" (only knowable after making it) before actually applying the
    reduction, matching v1's cb_search.py's LMR condition exactly.
    """
    undo = F.make_move(pieces, mailbox, meta, move, castle_rook_from, castle_rook_to, castle_all_rook_squares, castle_all_rook_bits,
                        zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                        eval_state, pst_mg, pst_eg, phase_weight)
    if F.is_square_attacked(pieces, F.king_square(pieces, mover_color), opponent_color,
                             rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                             bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                             knight_attacks, king_attacks, pawn_attacks):
        F.unmake_move(pieces, mailbox, meta, move, undo[0], undo[1], undo[2], undo[3], undo[4], castle_rook_from, castle_rook_to,
                      zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                      eval_state, pst_mg, pst_eg, phase_weight)
        return False, 0

    reduction = 0
    if ENABLE_LMR and lmr_eligible:
        gives_check = F.is_square_attacked(pieces, F.king_square(pieces, opponent_color), mover_color,
                                            rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                            bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                            knight_attacks, king_attacks, pawn_attacks)
        if not gives_check:
            reduction = LMR_REDUCTION

    if is_first:
        score = -negamax(pieces, mailbox, meta, depth - 1, -beta, -alpha, ply + 1, generation, True,
                          zobrist, eval_state, nodes, hard_deadline,
                          tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations,
                          rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                          bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                          knight_attacks, king_attacks, pawn_attacks,
                          castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                          castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                          zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                          pst_mg, pst_eg, phase_weight,
                          moves_buf_stack, scores_buf_stack, qmoves_buf_stack, qscores_buf_stack,
                          killers, history)
    else:
        score = -negamax(pieces, mailbox, meta, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, generation, True,
                          zobrist, eval_state, nodes, hard_deadline,
                          tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations,
                          rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                          bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                          knight_attacks, king_attacks, pawn_attacks,
                          castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                          castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                          zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                          pst_mg, pst_eg, phase_weight,
                          moves_buf_stack, scores_buf_stack, qmoves_buf_stack, qscores_buf_stack,
                          killers, history)
        if score > alpha:
            score = -negamax(pieces, mailbox, meta, depth - 1, -beta, -alpha, ply + 1, generation, True,
                              zobrist, eval_state, nodes, hard_deadline,
                              tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations,
                              rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                              bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                              knight_attacks, king_attacks, pawn_attacks,
                              castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                              castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                              zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                              pst_mg, pst_eg, phase_weight,
                              moves_buf_stack, scores_buf_stack, qmoves_buf_stack, qscores_buf_stack,
                              killers, history)

    F.unmake_move(pieces, mailbox, meta, move, undo[0], undo[1], undo[2], undo[3], undo[4], castle_rook_from, castle_rook_to,
                  zobrist, zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                  eval_state, pst_mg, pst_eg, phase_weight)
    return True, score


@njit(cache=False)
def negamax(pieces, mailbox, meta, depth, alpha, beta, ply, generation, null_allowed,
            zobrist, eval_state, nodes, hard_deadline,
            tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations,
            rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
            bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
            knight_attacks, king_attacks, pawn_attacks,
            castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
            castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
            zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
            pst_mg, pst_eg, phase_weight,
            moves_buf_stack, scores_buf_stack, qmoves_buf_stack, qscores_buf_stack,
            killers, history):
    nodes[0] += 1
    if nodes[0] % NODE_CHECK_INTERVAL == 0 and _now() >= hard_deadline:
        raise _SearchTimeout()
    alpha_orig = alpha

    key = zobrist[0]
    found, tt_depth, tt_score_raw, tt_bound, tt_move = tt_probe(key, tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations)
    if not ENABLE_TT:
        found = False
        tt_move = -1
    if found and tt_depth >= depth:
        tt_score = score_from_tt(tt_score_raw, ply)
        if tt_bound == BOUND_EXACT:
            return tt_score
        if tt_bound == BOUND_LOWER:
            alpha = max(alpha, tt_score)
        elif tt_bound == BOUND_UPPER:
            beta = min(beta, tt_score)
        if alpha >= beta:
            return tt_score

    color = meta[0]
    opponent = 1 - color
    in_chk = F.is_square_attacked(pieces, F.king_square(pieces, color), opponent,
                                   rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                   bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                   knight_attacks, king_attacks, pawn_attacks)
    if in_chk:
        depth += 1  # check extension, unconditional -- same as v1's cb_search.py

    if depth <= 0:
        return quiescence(pieces, mailbox, meta, alpha, beta, 0, zobrist, eval_state, nodes, hard_deadline,
                           rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                           bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                           knight_attacks, king_attacks, pawn_attacks,
                           castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                           castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                           zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                           pst_mg, pst_eg, phase_weight, qmoves_buf_stack, qscores_buf_stack, history)

    is_pv = (beta - alpha) > 1

    # Shallow-depth pruning family: reverse futility ("static null move"),
    # futility, and (in the move loop below) late move pruning. All three
    # are guarded against PV nodes, being in check, and near-mate bounds
    # (abs(), not just one sign -- a static eval comparison against
    # either a near-checkmating or near-getting-mated bound isn't
    # meaningful, the position needs to be searched properly either way).
    # static_eval is computed at most once per node and shared between
    # RFP and futility rather than evaluated twice.
    #
    # Bug fix (2026-09): this call used to pass the node's real alpha/beta,
    # which fed straight into evaluate_from_state's own lazy-eval shortcut
    # (return the cheap material+PST estimate once it's LAZY_EVAL_MARGIN
    # past that same window). RFP/futility then used whatever came back as
    # if it were the full eval, comparing it against margins meant for a
    # real score -- but a value that only promises "outside the window by
    # at least LAZY_EVAL_MARGIN" isn't safe to compare against a fixed
    # margin the way an exact eval is: the missing pawn-structure/mobility/
    # king-safety terms are exactly what would have kept some of these
    # positions from being prunable. Passing a window wide enough that the
    # lazy shortcut can never fire forces the real full evaluation here,
    # at the cost of losing the lazy fast-path for these two callers only
    # (quiescence's stand_pat, above, still uses its own real window on
    # purpose -- there the lazy bound itself is the thing being compared,
    # not fed into a margin).
    static_eval = 0
    can_rfp = ENABLE_RFP and not is_pv and not in_chk and depth <= RFP_MAX_DEPTH and abs(beta) < MATE_THRESHOLD
    can_futility = ENABLE_FUTILITY and not is_pv and not in_chk and depth <= FUTILITY_MAX_DEPTH and abs(alpha) < MATE_THRESHOLD
    if can_rfp or can_futility:
        static_eval = F.evaluate_from_state(eval_state, color, pieces, meta[1], -MATE_SCORE, MATE_SCORE,
                                             rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                             bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                             knight_attacks, king_attacks)
        if can_rfp and static_eval - RFP_MARGIN_PER_DEPTH * depth >= beta:
            return static_eval

    if (ENABLE_NULL_MOVE and null_allowed and not in_chk and depth >= NULL_MOVE_MIN_DEPTH
            and beta < MATE_THRESHOLD and has_non_pawn_material(pieces, color)):
        old_ep = make_null_move(meta, zobrist, zobrist_ep_file, zobrist_side)
        null_score = -negamax(pieces, mailbox, meta, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, ply + 1, generation, False,
                               zobrist, eval_state, nodes, hard_deadline,
                               tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations,
                               rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                               bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                               knight_attacks, king_attacks, pawn_attacks,
                               castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                               castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                               zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                               pst_mg, pst_eg, phase_weight,
                               moves_buf_stack, scores_buf_stack, qmoves_buf_stack, qscores_buf_stack,
                               killers, history)
        unmake_null_move(meta, old_ep, zobrist, zobrist_ep_file, zobrist_side)
        if null_score >= beta:
            return null_score

    moves = moves_buf_stack[ply]
    scores_buf = scores_buf_stack[ply]
    count = F.generate_pseudo_legal_moves(pieces, mailbox, meta, moves,
                                           rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                           bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                           knight_attacks, king_attacks, pawn_attacks,
                                           castle_king_to, castle_rook_from, castle_right_bit,
                                           castle_empty_squares, castle_king_path)
    killer0 = killers[ply, 0] if ply < MAX_PLY else -1
    killer1 = killers[ply, 1] if ply < MAX_PLY else -1
    order_moves(pieces, mailbox, moves, count, scores_buf, tt_move, killer0, killer1, history)

    best = -MATE_SCORE - 1
    best_move = -1
    legal_seen = 0
    quiets_tried = 0
    # -1 sentinel: LMP not active at this node (depth*depth+4 is always
    # >= 4, so -1 never collides with a real limit).
    lmp_limit = _lmp_quiet_limit(depth) if (ENABLE_LMP and not is_pv and not in_chk and depth <= LMP_MAX_DEPTH) else -1

    for i in range(count):
        move = moves[i]
        is_quiet = mailbox[(move >> 6) & 0x3F] == -1 and ((move >> 15) & 0x7) != F.FLAG_EP and ((move >> 12) & 0x7) == F.NO_PROMO
        is_first = legal_seen == 0

        # Late move pruning and futility pruning both only ever skip a
        # quiet, non-first move -- "not is_first" (equivalently,
        # legal_seen > 0) guarantees at least one legal move has already
        # been searched before either can trigger, so best_move can never
        # come back as "no move tried" even if every remaining move gets
        # skipped without a legality check.
        if is_quiet and not is_first:
            if lmp_limit >= 0 and quiets_tried >= lmp_limit:
                continue
            if can_futility and static_eval + FUTILITY_MARGIN[depth] < alpha:
                continue

        # legal_seen here is a pre-increment count (legal moves found
        # *before* this one) -- >= is the correct comparison to match
        # v1's post-increment "legal_seen > LMR_MOVE_THRESHOLD" (off-by-
        # one if left as a plain >, caught before this ever ran).
        lmr_eligible = legal_seen >= LMR_MOVE_THRESHOLD and depth >= LMR_MIN_DEPTH and is_quiet and not in_chk

        legal, score = try_move(pieces, mailbox, meta, move, depth, alpha, beta, ply, generation, is_first, lmr_eligible,
                                 color, opponent,
                                 zobrist, eval_state, nodes, hard_deadline,
                                 tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations,
                                 rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                 bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                 knight_attacks, king_attacks, pawn_attacks,
                                 castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                                 castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                                 zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                                 pst_mg, pst_eg, phase_weight,
                                 moves_buf_stack, scores_buf_stack, qmoves_buf_stack, qscores_buf_stack,
                                 killers, history)
        if not legal:
            continue
        legal_seen += 1
        if is_quiet:
            quiets_tried += 1

        if score > best:
            best = score
            best_move = move
        if best > alpha:
            alpha = best
        if alpha >= beta:
            record_cutoff(pieces, mailbox, move, depth, ply, killers, history)
            break

    if legal_seen == 0:
        return -MATE_SCORE + ply if in_chk else 0

    bound = BOUND_EXACT
    if best <= alpha_orig:
        bound = BOUND_UPPER
    elif best >= beta:
        bound = BOUND_LOWER
    if ENABLE_TT:
        tt_store(key, depth, score_to_tt(best, ply), bound, best_move, generation, tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations)

    return best


@njit(cache=False)
def root_search(pieces, mailbox, meta, depth, generation,
                 zobrist, eval_state, nodes, hard_deadline,
                 tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations,
                 rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                 bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                 knight_attacks, king_attacks, pawn_attacks,
                 castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                 castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                 zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                 pst_mg, pst_eg, phase_weight,
                 moves_buf_stack, scores_buf_stack, qmoves_buf_stack, qscores_buf_stack,
                 killers, history):
    # Root never null-moves (needs a real move to return) -- LMR isn't
    # applied here either, matching v1: root's window is already
    # (near-)full on move 1 so PVS's own null-window narrowing does most
    # of the work, and reducing the very moves we might return adds risk
    # for little benefit.
    key = zobrist[0]
    found, _d, _s, _b, tt_move = tt_probe(key, tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations)
    if not ENABLE_TT:
        found = False
        tt_move = -1
    if not found:
        tt_move = -1

    moves = moves_buf_stack[0]
    scores_buf = scores_buf_stack[0]
    count = F.generate_pseudo_legal_moves(pieces, mailbox, meta, moves,
                                           rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                           bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                           knight_attacks, king_attacks, pawn_attacks,
                                           castle_king_to, castle_rook_from, castle_right_bit,
                                           castle_empty_squares, castle_king_path)
    killer0 = killers[0, 0]
    killer1 = killers[0, 1]
    order_moves(pieces, mailbox, moves, count, scores_buf, tt_move, killer0, killer1, history)

    color = meta[0]
    opponent = 1 - color
    alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
    best_score = -MATE_SCORE - 1
    best_move = -1
    legal_seen = 0

    for i in range(count):
        move = moves[i]
        is_first = legal_seen == 0

        legal, score = try_move(pieces, mailbox, meta, move, depth, alpha, beta, 0, generation, is_first, False,
                                 color, opponent,
                                 zobrist, eval_state, nodes, hard_deadline,
                                 tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations,
                                 rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                 bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                 knight_attacks, king_attacks, pawn_attacks,
                                 castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                                 castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                                 zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                                 pst_mg, pst_eg, phase_weight,
                                 moves_buf_stack, scores_buf_stack, qmoves_buf_stack, qscores_buf_stack,
                                 killers, history)
        if not legal:
            continue
        legal_seen += 1

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    if ENABLE_TT:
        tt_store(key, depth, score_to_tt(best_score, 0), BOUND_EXACT, best_move, generation, tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations)
    return best_score, best_move


# ---------------------------------------------------------------------
# Python-side: iterative deepening driver
# ---------------------------------------------------------------------

def search(fen: str, soft_ms: float, hard_ms: float, arrays: "SearchArrays", generation: int = 1, max_depth: int = 64):
    """Iterative deepening from depth 1. Same node-count-polled hard-
    deadline mechanism as v1's cb_search.py now (see NODE_CHECK_INTERVAL/
    _SearchTimeout above): negamax/quiescence raise _SearchTimeout when
    the clock is blown mid-iteration, caught here, falling back to the
    last *completed* iteration's move/score -- a single very deep
    iteration can no longer run unbounded past the hard deadline.

    That's the safety net; it isn't budget management. Before *starting*
    each new iteration, its cost is predicted from the previous
    iteration's wall-clock time scaled by an EMA of the empirical
    branching factor (node-count ratio between consecutive completed
    iterations), and the iteration is skipped if that projection would
    finish past soft_deadline. Getting rescued by the hard deadline
    mid-iteration means whatever time that partial iteration burned was
    wasted -- for the SAME total wall-clock spent, stopping one depth
    shallower but complete is strictly better than starting a deeper one
    and having it cut off part-way. No prediction is attempted for the
    first two iterations (nothing to compare yet); they're cheap enough
    that this doesn't matter in practice.

    Retest (2026-09): the single-sample ratio used to just feed straight
    into the prediction, but real per-depth node growth is noisy enough
    (measured swinging between ~1.8x and ~7.4x between consecutive depths
    on ordinary middlegame positions) that one unlucky sample could
    predict a wildly inflated next-iteration cost and bail out of
    iterative deepening early with soft budget still unspent -- observed
    in practice as moves finishing in well under a second against a
    multi-second budget. Smoothing the ratio with an EMA across all prior
    iterations (not just the last one) damps that single-sample noise
    while still tracking a real trend across depths; clamping it to
    [MIN_BRANCHING_FACTOR, MAX_BRANCHING_FACTOR] is a backstop against a
    single pathological outlier (e.g. a check-evasion-driven node-count
    explosion) dominating the EMA in one step.
    """
    import time

    pieces, mailbox, meta = F.fen_to_state(fen)
    t = F._TABLES

    arrays.zobrist[0] = np.uint64(F.compute_hash(pieces, meta))
    mg, eg, phase = F.compute_eval_state(pieces)
    arrays.eval_state[0], arrays.eval_state[1], arrays.eval_state[2] = mg, eg, phase
    arrays.nodes[0] = 0
    arrays.reset_killers_history()  # per-move, like v1's fresh _Searcher each call

    start = time.perf_counter()
    soft_deadline = start + max(0.0, soft_ms) / 1000
    hard_deadline = start + max(0.0, hard_ms) / 1000

    best_move, best_score, depth_reached = -1, 0, 0
    prev_nodes = None
    smoothed_bf = None

    depth = 1
    while depth <= max_depth:
        nodes_before = int(arrays.nodes[0])
        iter_start = time.perf_counter()
        try:
            score, move = root_search(
                pieces, mailbox, meta, depth, generation,
                arrays.zobrist, arrays.eval_state, arrays.nodes, hard_deadline,
                arrays.tt_keys, arrays.tt_depths, arrays.tt_scores, arrays.tt_bounds, arrays.tt_moves, arrays.tt_generations,
                t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
                t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
                t.knight_attacks, t.king_attacks, t.pawn_attacks,
                t.castle_king_to, t.castle_rook_from, t.castle_rook_to, t.castle_right_bit,
                t.castle_empty_squares, t.castle_king_path, t.castle_all_rook_squares, t.castle_all_rook_bits,
                t.zobrist_piece, t.zobrist_castling, t.zobrist_ep_file, t.zobrist_side,
                t.pst_mg, t.pst_eg, t.phase_weight,
                arrays.moves_buf_stack, arrays.scores_buf_stack, arrays.qmoves_buf_stack, arrays.qscores_buf_stack,
                arrays.killers, arrays.history,
            )
        except _SearchTimeout:
            # This iteration never finished -- its partial TT/killer
            # writes are still valid (each is written as it's found, not
            # rolled back), but there's no (score, move) to trust from an
            # interrupted root_search, so keep whatever the last
            # *completed* iteration returned instead.
            break

        now = time.perf_counter()
        this_duration = now - iter_start
        this_nodes = int(arrays.nodes[0]) - nodes_before

        best_move, best_score, depth_reached = move, score, depth

        if now >= soft_deadline or now >= hard_deadline:
            break
        if abs(best_score) >= MATE_THRESHOLD:
            break

        if prev_nodes is not None and prev_nodes > 0 and this_nodes > 0:
            raw_bf = this_nodes / prev_nodes
            raw_bf = min(max(raw_bf, MIN_BRANCHING_FACTOR), MAX_BRANCHING_FACTOR)
            smoothed_bf = raw_bf if smoothed_bf is None else (
                BF_EMA_WEIGHT * raw_bf + (1.0 - BF_EMA_WEIGHT) * smoothed_bf
            )
            predicted_next_duration = this_duration * smoothed_bf
            if now + predicted_next_duration > soft_deadline:
                break

        prev_nodes = this_nodes
        depth += 1

    return best_move, best_score, {"depth": depth_reached, "nodes": int(arrays.nodes[0])}
