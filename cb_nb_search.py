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

import chess
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
# Continuation history (ContHist, 2026-09): item 2 of the 3-change plan,
# right after CorrHist. Killers/history alone only ever ask "has this
# quiet move worked well historically, anywhere" -- ContHist asks "has
# this quiet move worked well specifically as a REPLY to what the
# opponent just played (1 ply back) and to what we ourselves played two
# plies ago (2 plies back, the move that set up the tactical theme this
# node continues)". Move ordering quality in quiet positions is exactly
# what decides whether LMR/pruning (both downstream of ordering) end up
# skipping the moves that actually hold a small advantage or defend a
# small disadvantage -- see the real-loss diagnosis that motivated
# CorrHist above, which this targets from the other side (search
# selectivity, not eval bias).
#
# Indexing is the actual risk here, not the update math (same
# +=depth*depth formula as the existing history table) -- a widely-
# circulated real implementation has a copy-paste bug reading (ss-1)
# for BOTH the 1-ply and 2-ply lookup. See _cont_hist_context and
# tests/test_conthist.py, which exists specifically to catch that class
# of bug rather than trust the indexing because it compiles.
ENABLE_CONTHIST = os.environ.get("CB_NB_ENABLE_CONTHIST", "0") != "0"
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
#
# Defaulted ON as of 2026-09's retest (ratings/run_pruning_batch_sprt.py,
# 108 valid pairs / 216 games at 5+0.05, LLR=+3.190 crossing the H1 bound
# of +2.944 -- candidate confirmed >= 15 Elo stronger). The first batch
# SPRT had rejected all three, both together and individually (see
# ratings/run_pruning_bisect_sprt.py's logs): the static_eval used here
# was going through evaluate_from_state's lazy-eval shortcut with the
# node's real alpha/beta window, so it was often the cheap
# material+PST-only estimate rather than the full eval. That's fixed
# below (plus tightened depth cutoffs), and this retest is what finally
# confirmed the fix worked instead of leaving it an open question.
ENABLE_RFP = os.environ.get("CB_NB_ENABLE_RFP", "1") != "0"
ENABLE_FUTILITY = os.environ.get("CB_NB_ENABLE_FUTILITY", "1") != "0"
ENABLE_LMP = os.environ.get("CB_NB_ENABLE_LMP", "1") != "0"

NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2
LMR_MIN_DEPTH = 3
LMR_MOVE_THRESHOLD = 4
LMR_REDUCTION = 1
# LMR retest (2026-09): the base reduction above is flat regardless of
# how deep the remaining search still has to go -- standard practice in
# most engines scales the reduction up at higher depth too (there's more
# search budget left to recover from an over-aggressive reduction via
# re-search), not just by how late the move is in the ordering. Gated
# separately from ENABLE_LMR so it's SPRT-testable against the flat
# scheme alone rather than folded in blind.
ENABLE_LMR_DEPTH_SCALING = os.environ.get("CB_NB_ENABLE_LMR_DEPTH_SCALING", "0") != "0"
LMR_DEEP_DEPTH = 6
LMR_DEEP_EXTRA_REDUCTION = 1

# Contempt (2026-09, the 600-ply-draw rule change): a repetition draw
# used to be strictly worse than almost any alternative when losing
# (the position would otherwise run to the 300-ply material-adjudication
# cutoff and lose outright) -- now a draw reached by ply 600 is a real
# half point regardless of material, so a draw score of exactly 0
# undersells it when we're clearly worse and oversells it when we're
# clearly better. Signed offset scaled by the mover's own current static
# eval at the point a repetition is found: negative eval (mover worse)
# pushes the draw score up (more attractive, encourages steering into or
# accepting the repeat); positive eval (mover better) pushes it down
# (less attractive, encourages playing on rather than settling).
# CONTEMPT_EVAL_CAP bounds how much a single very lopsided position can
# swing this -- past a few pawns down, "more losing" shouldn't make a
# draw progressively more attractive than it already is at the cap.
ENABLE_CONTEMPT = os.environ.get("CB_NB_ENABLE_CONTEMPT", "1") != "0"
CONTEMPT_SCALE = 0.15
CONTEMPT_EVAL_CAP = 300

# Correction history (CorrHist, 2026-09): real ladder losses this session
# showed no tactical blunder in any of them -- a sustained, ~10-30cp/move
# static-eval bias instead, for 10-15 moves at a stretch, in three of
# four cases starting from an equal-or-better position and eroding move
# by move. That shape is exactly what CorrHist targets: it records the
# gap between static eval and the score search actually found, keyed by
# a pawn-structure feature (a pawn-only Zobrist hash -- see
# pawn_zobrist_hash_from_scratch in cb_nb_fast.py and this file's zobrist
# array, now 2 elements: [0] main key, [1] pawn-only key), and corrects
# future static evals sharing that same pawn structure. Introduced in
# Caissa (Oct 2023), now widely adopted; expected to matter more at our
# own 120+0.5 time control than at fast test-suite TCs, since a real
# correction needs enough completed searches at meaningful depth to
# accumulate a signal.
ENABLE_CORRHIST = os.environ.get("CB_NB_ENABLE_CORRHIST", "0") != "0"
CORRHIST_SIZE = 16384
CORRHIST_MASK = CORRHIST_SIZE - 1
CORRHIST_GRAIN = 256
CORRHIST_SCALE = 256
CORRHIST_MAX = 32 * CORRHIST_GRAIN


@njit(cache=False)
def corrhist_update(corrhist, side, pawn_key, depth, search_score, static_eval):
    """Exponential moving average, weighted more heavily by a deeper
    search (up to a cap): the whole point is that a deep search's score
    is trusted more than a shallow one's as evidence of the *true* value
    of this pawn structure relative to what static eval says it's worth.
    Callers are responsible for only calling this when the update
    preconditions hold (not in check; best move quiet or none; the score
    isn't a bound on the wrong side of static_eval) -- see negamax's own
    call site for exactly which conditions and why."""
    idx = int(pawn_key & np.uint64(CORRHIST_MASK))
    scaled_diff = (search_score - static_eval) * CORRHIST_GRAIN
    new_weight = min(depth * depth + 2 * depth + 1, 128)
    entry = corrhist[side, idx]
    entry = (entry * (CORRHIST_SCALE - new_weight) + scaled_diff * new_weight) // CORRHIST_SCALE
    entry = max(-CORRHIST_MAX, min(CORRHIST_MAX, entry))
    corrhist[side, idx] = entry


@njit(cache=False)
def corrhist_correct(corrhist, side, pawn_key, static_eval):
    """static_eval, nudged by whatever this pawn structure's accumulated
    correction says -- clamped so it can never reach or cross a mate
    score (a correction is a static-eval adjustment, never a claim about
    forced mate, which only a real search result should ever report).
    Flag-agnostic on purpose -- see this module's ENABLE_CORRHIST for why
    the check belongs at the call site, not here."""
    idx = int(pawn_key & np.uint64(CORRHIST_MASK))
    corrected = static_eval + corrhist[side, idx] // CORRHIST_GRAIN
    if corrected >= MATE_THRESHOLD:
        corrected = MATE_THRESHOLD - 1
    elif corrected <= -MATE_THRESHOLD:
        corrected = -(MATE_THRESHOLD - 1)
    return corrected


ENABLE_ASPIRATION = os.environ.get("CB_NB_ENABLE_ASPIRATION", "1") != "0"

# Aspiration windows (2026-09, design doc sec.5): from ASPIRATION_MIN_DEPTH
# on, search()'s ID loop starts each iteration with a narrow window around
# the previous iteration's score instead of full width -- most positions'
# evaluation doesn't swing much between consecutive depths, so a narrow
# window lets alpha-beta prune far more of the tree for the same result.
# When the previous score is close to it, though, the window doesn't
# help (see ASPIRATION_MATE_MARGIN below) and re-search costs make it a
# net loss to even try. On a fail-low/fail-high, root_search's returned
# score is only a bound (see its own BOUND_UPPER/BOUND_LOWER comment) --
# widen and re-search the SAME depth before trusting it or moving on.
ASPIRATION_MIN_DEPTH = 5
ASPIRATION_WINDOW = 30
ASPIRATION_MATE_MARGIN = MATE_THRESHOLD - 2_000  # stay clear of MATE_THRESHOLD even after widening once
KILLER_BASE_SCORE = 500_000
# Ordering bases for the two capture classes SEE distinguishes below
# (2026-09, replacing raw MVV-LVA for ordering -- SEE was already
# computed for quiescence pruning, this reuses it): a winning-or-equal
# capture is a forcing tactical move and belongs ahead of the killer
# heuristic, which only exists to approximate that value for QUIET
# moves; a losing capture is worse than an average quiet move and design
# doc §5's own priority list puts it dead last, after history. Bug fixed
# in the same change: the previous MVV-LVA scores (victim*16 - attacker,
# max ~14.4k for QxQ) never actually exceeded KILLER_BASE_SCORE
# (500,000) despite the docstring's claimed "captures before killers"
# ordering -- captures were silently being searched AFTER killers this
# entire time. WINNING_CAPTURE_BASE/LOSING_CAPTURE_BASE are offset far
# enough from 0 that a realistic SEE value (bounded by piece values,
# at most a few thousand even for a multi-piece exchange) can't cross
# into the neighboring band.
WINNING_CAPTURE_BASE = 600_000
LOSING_CAPTURE_BASE = -1_000_000

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

# TT_SIZE_BITS retest (2026-09): each entry is 6 separate int64/uint64
# arrays (keys, depths, scores, bounds, moves, generations) -- 48 bytes/
# entry, not a packed struct -- so 2^20 is ~50MB and 2^23 is ~403MB,
# comfortably inside the 2GB budget either way. At ~1M nps and a
# multi-second move budget, a single move alone can generate several
# million node visits against a 2^20 (~1.05M entry) table, thrashing
# replacement within one move, before even counting that the table
# persists and accumulates across the whole game. Env-var override
# (matching every other tunable flag in this file) so the two sizes are
# SPRT-testable against each other rather than just asserted better.
TT_SIZE_BITS = int(os.environ.get("CB_NB_TT_SIZE_BITS", "20"))
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
# Material values for SEE (move ordering + quiescence pruning), pawn..
# king, 0-indexed -- pulled from whichever material table cb_nb_fast.py's
# CB_NB_TEXEL_TUNED_PST flag actually has active, not a separate
# hardcoded copy, so SEE's idea of a piece's worth can't silently drift
# out of sync with what the eval itself uses (2026-09, added when
# material values became tunable via ratings/joint_tune.py). SEE is a
# single flat scale, not mg/eg-tapered, so this uses the mg values --
# same simplification every other engine's SEE makes.
if F._TEXEL_TUNED_PST:
    _MATERIAL_MG = F.PST_TUNED.PIECE_VALUES_MG
else:
    _MATERIAL_MG = F.V1_TABLES.PIECE_VALUES_MG
_CAPTURE_VALUE = np.array([
    _MATERIAL_MG[chess.PAWN], _MATERIAL_MG[chess.KNIGHT], _MATERIAL_MG[chess.BISHOP],
    _MATERIAL_MG[chess.ROOK], _MATERIAL_MG[chess.QUEEN], _MATERIAL_MG[chess.KING],
], dtype=np.int64)

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

        self.zobrist = np.zeros(2, dtype=np.uint64)  # [0]=main key, [1]=pawn-only key (CorrHist)
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

        # Correction history (CorrHist, 2026-09) -- [side_to_move,
        # pawn_key & CORRHIST_MASK]. Persists for the whole game/process
        # lifetime like the TT, NOT reset per move (see
        # reset_killers_history) -- the accumulated correction across the
        # whole game is the point, not a per-move scratch value.
        self.corrhist = np.zeros((2, CORRHIST_SIZE), dtype=np.int64)

        # Continuation history (ContHist, 2026-09) -- cont_piece[p]/
        # cont_to[p] record the (piece, to-square) of the move that was
        # played to REACH ply p (written by try_move for the child it's
        # about to recurse into); cont_hist_1ply/2ply are indexed
        # [prev_piece, prev_to, piece, to]. Same per-move reset lifecycle
        # as killers/history below (see reset_killers_history) -- unlike
        # CorrHist/the TT, this is deliberately NOT carried across moves.
        self.cont_piece = np.full(MAX_PLY + 2, -1, dtype=np.int64)
        self.cont_to = np.full(MAX_PLY + 2, -1, dtype=np.int64)
        self.cont_hist_1ply = np.zeros((12, 64, 12, 64), dtype=np.int64)
        self.cont_hist_2ply = np.zeros((12, 64, 12, 64), dtype=np.int64)

        # Repetition detection (2026-09, after a real ladder loss --
        # round 15 -- where the engine gave up a completely won game,
        # up a full queen, by walking a checking sequence into a
        # threefold-repetition draw entirely on its own initiative). Two
        # parts:
        #   path_keys[ply]: the Zobrist key at each ply of the CURRENT
        #     line being explored in THIS search call, index 0 = the
        #     root position itself. Reused across iterative-deepening
        #     iterations without resetting -- only path_keys[0:ply] is
        #     ever read at a given node, so stale deeper entries from a
        #     previous (deeper) iteration are simply never looked at.
        #   game_history_keys[0:game_history_count]: Zobrist keys of
        #     every REAL position seen so far at the start of one of our
        #     own turns (see cb_nb_engine.py) -- persists for the whole
        #     game, unlike path_keys. This is a partial history (the
        #     wire protocol hands us a fresh FEN per call with no move
        #     list, so we only ever observe positions where it's our own
        #     turn, never the opponent's intermediate replies), same
        #     limitation v1's cb_engine.py already documents for its own
        #     (currently unused) position_counts dict -- good enough to
        #     catch "I've already been in exactly this position with the
        #     move now available to me," which is exactly what a search-
        #     tree-only check can't see across separate get_move calls.
        self.path_keys = np.zeros(MAX_PLY, dtype=np.uint64)
        self.game_history_keys = np.zeros(512, dtype=np.uint64)
        self.game_history_count = 0

    def record_game_position(self, key) -> None:
        """Call once per get_move, with the position's Zobrist key
        BEFORE searching -- see cb_nb_engine.py."""
        if self.game_history_count < len(self.game_history_keys):
            self.game_history_keys[self.game_history_count] = key
            self.game_history_count += 1

    def reset_killers_history(self):
        self.killers.fill(-1)
        self.history.fill(0)
        self.cont_piece.fill(-1)
        self.cont_to.fill(-1)
        self.cont_hist_1ply.fill(0)
        self.cont_hist_2ply.fill(0)


@njit(cache=False)
def _is_repetition(key, ply, path_keys, game_history_keys, game_history_count):
    """True if ``key`` matches an earlier position either in the current
    search line (path_keys[0:ply], where index 0 is the root position
    itself) or in the real game so far (game_history_keys[0:
    game_history_count]) -- treating a single earlier match as enough to
    call a node a draw, not waiting to actually reach a third occurrence.
    Standard practice: two occurrences already means the position is
    reachable again, which is exactly the signal a search needs to
    avoid it when ahead (or seek it when behind) -- waiting for a literal
    third repetition before scoring it as a draw would only find that
    out one ply too late to route around it."""
    for i in range(ply):
        if path_keys[i] == key:
            return True
    for i in range(game_history_count):
        if game_history_keys[i] == key:
            return True
    return False


@njit(cache=False)
def _contempt_draw_score(pieces, mailbox, meta, eval_state,
                          rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                          bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                          knight_attacks, king_attacks):
    """The draw score for a detected repetition, adjusted by contempt --
    see ENABLE_CONTEMPT's comment above. Returns 0 outright when contempt
    is disabled, matching the pre-2026-09 behavior exactly. mailbox is
    unused but kept for signature symmetry with the other node-local
    helpers that take a full board-state parameter list."""
    if not ENABLE_CONTEMPT:
        return 0
    color = meta[0]
    static_eval = F.evaluate_from_state(
        eval_state, color, pieces, meta[1], -1_000_000, 1_000_000,
        rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
        bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
        knight_attacks, king_attacks,
    )
    capped = min(max(static_eval, -CONTEMPT_EVAL_CAP), CONTEMPT_EVAL_CAP)
    return np.int64(-CONTEMPT_SCALE * capped)


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


@njit(cache=False)
def order_moves(pieces, mailbox, moves, count, scores_buf, tt_move, killer0, killer1, history,
                 cont_hist_1ply, cont_hist_2ply, prev1_piece, prev1_to, prev2_piece, prev2_to,
                 rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                 bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                 knight_attacks, king_attacks, pawn_attacks):
    """Fills scores_buf[:count] and insertion-sorts moves[:count]
    descending by score: TT move highest, then winning-or-equal captures
    and promotions by SEE value, then the two killers, then quiets by
    history score (2026-09: history + ContHist's 1-ply context + half its
    2-ply context, see ENABLE_CONTHIST), then losing captures by SEE
    value -- see WINNING_CAPTURE_BASE/LOSING_CAPTURE_BASE above for why
    captures split around the killer band instead of all sitting above
    it. Quiescence calls this too (with killer0=killer1=-1, prev1/prev2
    piece/to = -1, real table references) -- the killer/history/ContHist
    branch is simply dead there since a captures-only move list never
    reaches it, cheaper than a second code path."""
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
            attacker_idx = mailbox[from_sq]
            side_to_move = attacker_idx // 6
            see = see_capture(
                move, side_to_move, pieces, mailbox,
                rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                knight_attacks, king_attacks, pawn_attacks,
            ) if is_capture else 0
            if promo != F.NO_PROMO:
                see += _CAPTURE_VALUE[promo]
            scores_buf[i] = WINNING_CAPTURE_BASE + see if see >= 0 else LOSING_CAPTURE_BASE + see
        elif ENABLE_KILLERS_HISTORY and move == killer0:
            scores_buf[i] = KILLER_BASE_SCORE + 1
        elif ENABLE_KILLERS_HISTORY and move == killer1:
            scores_buf[i] = KILLER_BASE_SCORE
        elif ENABLE_KILLERS_HISTORY:
            attacker_idx = mailbox[from_sq]
            if attacker_idx == -1:
                scores_buf[i] = 0
            else:
                quiet_score = history[attacker_idx, to_sq]
                if ENABLE_CONTHIST:
                    if prev1_piece != -1:
                        quiet_score += cont_hist_1ply[prev1_piece, prev1_to, attacker_idx, to_sq]
                    if prev2_piece != -1:
                        quiet_score += cont_hist_2ply[prev2_piece, prev2_to, attacker_idx, to_sq] // 2
                scores_buf[i] = quiet_score
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
               pst_mg, pst_eg, phase_weight, qmoves_buf_stack, qscores_buf_stack, history, corrhist,
               cont_hist_1ply, cont_hist_2ply):
    nodes[0] += 1
    if nodes[0] % NODE_CHECK_INTERVAL == 0 and _now() >= hard_deadline:
        raise _SearchTimeout()
    stand_pat = F.evaluate_from_state(
        eval_state, meta[0], pieces, meta[1], alpha, beta,
        rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
        bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
        knight_attacks, king_attacks,
    )
    if ENABLE_CORRHIST:
        stand_pat = corrhist_correct(corrhist, meta[0], zobrist[1], stand_pat)
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

    order_moves(pieces, mailbox, captures_buf, cap_count, scores_buf, -1, -1, -1, history,
                cont_hist_1ply, cont_hist_2ply, -1, -1, -1, -1,
                rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                knight_attacks, king_attacks, pawn_attacks)

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
                             pst_mg, pst_eg, phase_weight, qmoves_buf_stack, qscores_buf_stack, history, corrhist, cont_hist_1ply, cont_hist_2ply)
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
def _cont_hist_context(cont_piece, cont_to, ply):
    """(prev1_piece, prev1_to, prev2_piece, prev2_to) for ContHist
    lookups at the node currently at ``ply``. cont_piece[p]/cont_to[p]
    holds the (piece, to-square) of the move that was played TO REACH
    ply p -- written by the parent node right before recursing (see
    try_move). So at this node: 1-ply-back ("what did the opponent just
    play") is cont_piece[ply] itself; 2-ply-back ("what did WE play last
    time it was our move") is cont_piece[ply - 1], one slot further back
    up the same stack -- NOT cont_piece[ply] again (that's the specific
    copy-paste bug this function's own tests exist to catch). -1
    sentinels when that much history doesn't exist yet (near the root)."""
    if ply >= 1:
        prev1_piece, prev1_to = cont_piece[ply], cont_to[ply]
    else:
        prev1_piece, prev1_to = -1, -1
    if ply >= 2:
        prev2_piece, prev2_to = cont_piece[ply - 1], cont_to[ply - 1]
    else:
        prev2_piece, prev2_to = -1, -1
    return prev1_piece, prev1_to, prev2_piece, prev2_to


@njit(cache=False)
def record_cutoff_conthist(cont_hist_1ply, cont_hist_2ply, piece, to_sq,
                            prev1_piece, prev1_to, prev2_piece, prev2_to, depth):
    """The ContHist half of record_cutoff's job, isolated so its indexing
    can be tested directly against a hand-built context (see
    tests/test_conthist.py) rather than only through a full search.
    Each offset updates independently and is skipped outright when its
    own context doesn't exist yet -- a missing 2-ply context must never
    fall back to reusing the 1-ply one."""
    if prev1_piece != -1:
        cont_hist_1ply[prev1_piece, prev1_to, piece, to_sq] += depth * depth
    if prev2_piece != -1:
        cont_hist_2ply[prev2_piece, prev2_to, piece, to_sq] += depth * depth


@njit(cache=False)
def record_cutoff(pieces, mailbox, move, depth, ply, killers, history,
                   cont_hist_1ply, cont_hist_2ply, prev1_piece, prev1_to, prev2_piece, prev2_to):
    """A quiet move caused a beta cutoff: remember it as a killer at this
    ply, bump its history score, and (2026-09) its continuation-history
    score against the 1-ply and 2-ply context this node was reached
    through. Matches v1's cb_search.py exactly for the killers/history
    part."""
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
        if ENABLE_CONTHIST:
            record_cutoff_conthist(cont_hist_1ply, cont_hist_2ply, attacker_idx, to_sq,
                                    prev1_piece, prev1_to, prev2_piece, prev2_to, depth)


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
             killers, history, path_keys, game_history_keys, game_history_count, corrhist, cont_piece, cont_to, cont_hist_1ply, cont_hist_2ply):
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

    if ENABLE_CONTHIST:
        # Recorded for the CHILD node (ply + 1) we're about to enter --
        # "the move that led here", read back via _cont_hist_context at
        # that node's own ply. mailbox[to_sq] is read AFTER make_move on
        # purpose: it's the piece as it now stands (the promoted piece,
        # for a promotion), not the pawn that used to be at from_sq --
        # moot for what actually reads this (only quiet moves ever
        # consult ContHist, and a quiet move can't be a promotion), but
        # this is the natural value to reach for at this point in the
        # function regardless.
        to_sq_for_context = (move >> 6) & 0x3F
        cont_piece[ply + 1] = mailbox[to_sq_for_context]
        cont_to[ply + 1] = to_sq_for_context

    reduction = 0
    if ENABLE_LMR and lmr_eligible:
        gives_check = F.is_square_attacked(pieces, F.king_square(pieces, opponent_color), mover_color,
                                            rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                            bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                            knight_attacks, king_attacks, pawn_attacks)
        if not gives_check:
            reduction = LMR_REDUCTION
            if ENABLE_LMR_DEPTH_SCALING and depth >= LMR_DEEP_DEPTH:
                reduction += LMR_DEEP_EXTRA_REDUCTION

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
                          killers, history, path_keys, game_history_keys, game_history_count, corrhist, cont_piece, cont_to, cont_hist_1ply, cont_hist_2ply)
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
                          killers, history, path_keys, game_history_keys, game_history_count, corrhist, cont_piece, cont_to, cont_hist_1ply, cont_hist_2ply)
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
                              killers, history, path_keys, game_history_keys, game_history_count, corrhist, cont_piece, cont_to, cont_hist_1ply, cont_hist_2ply)

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
            killers, history, path_keys, game_history_keys, game_history_count, corrhist, cont_piece, cont_to, cont_hist_1ply, cont_hist_2ply):
    nodes[0] += 1
    if nodes[0] % NODE_CHECK_INTERVAL == 0 and _now() >= hard_deadline:
        raise _SearchTimeout()
    alpha_orig = alpha

    key = zobrist[0]
    # Repetition check BEFORE the TT probe, and skipping the TT entirely
    # for a drawn-by-repetition node: whether this exact position is a
    # repeat depends on the PATH taken to reach it, not just the
    # position itself, so a repetition-draw score cached under this
    # key would be wrong to reuse from a different line where the same
    # position isn't a repeat at all.
    if _is_repetition(key, ply, path_keys, game_history_keys, game_history_count):
        return _contempt_draw_score(
            pieces, mailbox, meta, eval_state,
            rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
            bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
            knight_attacks, king_attacks,
        )
    path_keys[ply] = key
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

    prev1_piece, prev1_to, prev2_piece, prev2_to = _cont_hist_context(cont_piece, cont_to, ply)

    if depth <= 0:
        return quiescence(pieces, mailbox, meta, alpha, beta, 0, zobrist, eval_state, nodes, hard_deadline,
                           rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                           bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                           knight_attacks, king_attacks, pawn_attacks,
                           castle_king_to, castle_rook_from, castle_rook_to, castle_right_bit,
                           castle_empty_squares, castle_king_path, castle_all_rook_squares, castle_all_rook_bits,
                           zobrist_piece, zobrist_castling, zobrist_ep_file, zobrist_side,
                           pst_mg, pst_eg, phase_weight, qmoves_buf_stack, qscores_buf_stack, history, corrhist, cont_hist_1ply, cont_hist_2ply)

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
    # static_eval feeds RFP/futility (via corrected_eval, below) exactly
    # as before CorrHist; CorrHist itself (2026-09) additionally needs it
    # computed at every not-in-check node regardless of RFP/futility's own
    # depth cutoffs, both to correct that node's own decisions and to
    # have something to record an update against at the node's exit --
    # so the eval call now also fires whenever ENABLE_CORRHIST is on,
    # not just when can_rfp/can_futility already needed it. When the flag
    # is off this is unchanged from before: still lazy, still skipped at
    # PV nodes and deeper non-PV nodes past RFP_MAX_DEPTH/FUTILITY_MAX_DEPTH.
    static_eval = 0
    corrected_eval = 0
    can_rfp = ENABLE_RFP and not is_pv and not in_chk and depth <= RFP_MAX_DEPTH and abs(beta) < MATE_THRESHOLD
    can_futility = ENABLE_FUTILITY and not is_pv and not in_chk and depth <= FUTILITY_MAX_DEPTH and abs(alpha) < MATE_THRESHOLD
    if can_rfp or can_futility or (ENABLE_CORRHIST and not in_chk):
        static_eval = F.evaluate_from_state(eval_state, color, pieces, meta[1], -MATE_SCORE, MATE_SCORE,
                                             rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                                             bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                                             knight_attacks, king_attacks)
        corrected_eval = corrhist_correct(corrhist, color, zobrist[1], static_eval) if ENABLE_CORRHIST else static_eval
        if can_rfp and corrected_eval - RFP_MARGIN_PER_DEPTH * depth >= beta:
            return corrected_eval

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
                               killers, history, path_keys, game_history_keys, game_history_count, corrhist, cont_piece, cont_to, cont_hist_1ply, cont_hist_2ply)
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
    order_moves(pieces, mailbox, moves, count, scores_buf, tt_move, killer0, killer1, history,
                cont_hist_1ply, cont_hist_2ply, prev1_piece, prev1_to, prev2_piece, prev2_to,
                rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                knight_attacks, king_attacks, pawn_attacks)

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
            if can_futility and corrected_eval + FUTILITY_MARGIN[depth] < alpha:
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
                                 killers, history, path_keys, game_history_keys, game_history_count, corrhist, cont_piece, cont_to, cont_hist_1ply, cont_hist_2ply)
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
            record_cutoff(pieces, mailbox, move, depth, ply, killers, history,
                           cont_hist_1ply, cont_hist_2ply, prev1_piece, prev1_to, prev2_piece, prev2_to)
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

    # CorrHist update (2026-09): only when in check never applies (a
    # static eval taken while in check isn't meaningful -- the position
    # is about to change by force); only when the best move found is
    # quiet or there wasn't one to prefer over "no capture/promo changed
    # the position" (best_move == -1 can't actually happen here -- an
    # early return above already covers legal_seen == 0 -- kept as an
    # explicit check anyway per the design rather than relying on that);
    # and never against a bound that's already known to disagree with
    # static_eval in the direction that would matter (a lower bound below
    # static_eval, or an upper bound above it, tells us nothing about
    # whether static_eval itself was wrong in this position).
    if ENABLE_CORRHIST and not in_chk:
        to_sq = (best_move >> 6) & 0x3F
        best_is_quiet = best_move == -1 or (
            mailbox[to_sq] == -1 and ((best_move >> 15) & 0x7) != F.FLAG_EP and ((best_move >> 12) & 0x7) == F.NO_PROMO
        )
        if best_is_quiet:
            bad_lower = bound == BOUND_LOWER and best < static_eval
            bad_upper = bound == BOUND_UPPER and best > static_eval
            if not bad_lower and not bad_upper:
                corrhist_update(corrhist, color, zobrist[1], depth, best, static_eval)

    return best


@njit(cache=False)
def root_search(pieces, mailbox, meta, depth, generation, alpha, beta,
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
                 killers, history, path_keys, game_history_keys, game_history_count, corrhist, cont_piece, cont_to, cont_hist_1ply, cont_hist_2ply):
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
    # Root has no ContHist context of its own (ply 0 -- no move has been
    # made yet in this search line), so it always passes the -1 "no
    # context" sentinels; only cont_hist_1ply/2ply are real, threaded
    # through to try_move below so ply 1's own context gets recorded.
    order_moves(pieces, mailbox, moves, count, scores_buf, tt_move, killer0, killer1, history,
                cont_hist_1ply, cont_hist_2ply, -1, -1, -1, -1,
                rook_masks, rook_magics, rook_shifts, rook_offsets, rook_table,
                bishop_masks, bishop_magics, bishop_shifts, bishop_offsets, bishop_table,
                knight_attacks, king_attacks, pawn_attacks)

    color = meta[0]
    opponent = 1 - color
    alpha_orig = alpha
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
                                 killers, history, path_keys, game_history_keys, game_history_count, corrhist, cont_piece, cont_to, cont_hist_1ply, cont_hist_2ply)
        if not legal:
            continue
        legal_seen += 1

        if score > best_score:
            best_score = score
            best_move = move
        if best_score > alpha:
            alpha = best_score

    # Aspiration windows (2026-09) mean this call's alpha/beta may be a
    # narrow window around the previous iteration's score, not the
    # historically-always-full [-MATE-1, MATE+1] -- so best_score can now
    # genuinely be only a bound rather than the exact score, same
    # fail-soft classification negamax already does at every other node.
    # search()'s ID loop is the one that notices this (best_score <=
    # alpha or >= beta) and re-searches this depth with a wider window;
    # this only has to get the TT bound type right, not decide to widen.
    bound = BOUND_EXACT
    if best_score <= alpha_orig:
        bound = BOUND_UPPER
    elif best_score >= beta:
        bound = BOUND_LOWER
    if ENABLE_TT:
        tt_store(key, depth, score_to_tt(best_score, 0), bound, best_move, generation, tt_keys, tt_depths, tt_scores, tt_bounds, tt_moves, tt_generations)
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
    arrays.zobrist[1] = np.uint64(F.compute_pawn_hash(pieces))
    mg, eg, phase = F.compute_eval_state(pieces)
    arrays.eval_state[0], arrays.eval_state[1], arrays.eval_state[2] = mg, eg, phase
    arrays.nodes[0] = 0
    arrays.reset_killers_history()  # per-move, like v1's fresh _Searcher each call
    arrays.path_keys[0] = arrays.zobrist[0]

    start = time.perf_counter()
    soft_deadline = start + max(0.0, soft_ms) / 1000
    hard_deadline = start + max(0.0, hard_ms) / 1000

    best_move, best_score, depth_reached = -1, 0, 0
    prev_nodes = None
    smoothed_bf = None
    prev_score = None

    depth = 1
    while depth <= max_depth:
        if (not ENABLE_ASPIRATION or depth < ASPIRATION_MIN_DEPTH or prev_score is None
                or abs(prev_score) >= ASPIRATION_MATE_MARGIN):
            alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
        else:
            alpha = max(prev_score - ASPIRATION_WINDOW, -MATE_SCORE - 1)
            beta = min(prev_score + ASPIRATION_WINDOW, MATE_SCORE + 1)

        nodes_before = int(arrays.nodes[0])
        iter_start = time.perf_counter()
        try:
            while True:
                score, move = root_search(
                    pieces, mailbox, meta, depth, generation, alpha, beta,
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
                    arrays.path_keys, arrays.game_history_keys, arrays.game_history_count, arrays.corrhist,
                    arrays.cont_piece, arrays.cont_to, arrays.cont_hist_1ply, arrays.cont_hist_2ply,
                )
                if score <= alpha and alpha > -MATE_SCORE - 1:
                    alpha = -MATE_SCORE - 1  # fail-low: widen down, same depth
                    continue
                if score >= beta and beta < MATE_SCORE + 1:
                    beta = MATE_SCORE + 1  # fail-high: widen up, same depth
                    continue
                break
        except _SearchTimeout:
            # This iteration never finished -- its partial TT/killer
            # writes are still valid (each is written as it's found, not
            # rolled back), but there's no (score, move) to trust from an
            # interrupted root_search, so keep whatever the last
            # *completed* iteration returned instead.
            break
        prev_score = score

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
