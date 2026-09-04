"""Numba bitboard engine wrapper: deadline-based blocking compile with a
background-compile fallback.

NOT part of the shipped submission yet: not in tools/build_zip.py's
whitelist, not imported by agent.py. v1 (cb_engine.py) remains the
fallback and the perft oracle throughout -- this module is where the
eventual agent.py switchover happens once the numba path clears the
remaining validation steps (head-to-head SPRT vs v1, tournament-TC
calibration; see README's "further out" section).

Why blocking-then-background, not background-only: the competition gives
a 60s init budget before the game clock starts, with no contention for
the single core during that window. A background-only design (the first
cut of this module) throws that budget away and instead pays the full
JIT compile cost *during play*, where numba's LLVM codegen competes with
the live search for the one available core -- so the opening moves served
from v1 during warmup end up slower and weaker than v1 normally is, not
just "v1 instead of numba".

Instead, __init__ compiles normally (blocking) against a wall-clock
deadline measured from *process start* (see the process_start param --
agent.py will eventually capture this before any imports run, since
interpreter/import overhead already eats into the 60s budget before this
class is even constructed). Compilation runs in priority order --
movegen/make-unmake first, then eval/Zobrist, then the search kernel
last -- so if the deadline is hit partway through, whatever's already
warm is the highest-value subset available: the search kernel is both by
far the largest compile unit and the one v1 substitutes for most cleanly,
so it's the one left to finish in the background if time runs out.

Three outcomes, all covered by tests/test_nb_insurance_policy.py:
  1. All three stages finish inside the deadline (expected on reference
     hardware) -- init returns with numba fully warm, at zero cost to
     live play.
  2. The deadline is hit after some stages -- init returns immediately
     with whatever's warm, the remaining stages finish on a background
     thread, and get_move serves from v1 until they do. This is the
     original (now fallback-only) lever #7 path.
  3. A stage raises -- treated as a permanent numba failure; get_move
     falls back to v1 for the rest of the game, no retries.
"""
from __future__ import annotations

import sys
import threading
import time

import chess

import cb_engine
import cb_nb_fast as F
import cb_nb_search as S

_WARMUP_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Wall-clock budget (seconds, measured from process start -- see the
# process_start constructor param, not from when compilation begins) for
# blocking compilation during init. Tune against real numbers from the
# Linux reference container (see README): it trades "risk of overrunning
# the 60s hard init budget" against "risk of falling back to the slower,
# core-contended background-compile path for the opening moves".
COMPILE_DEADLINE_S = 40.0


class Engine:
    def __init__(self, process_start: float | None = None) -> None:
        start = time.monotonic()
        self._deadline = (process_start if process_start is not None else start) + COMPILE_DEADLINE_S

        # v1 is both the immediate-availability fallback during warmup and
        # the permanent fallback if numba never becomes ready. It owns its
        # own TimeManager, which this class reuses rather than duplicating
        # the §7 budget formula.
        self._v1 = cb_engine.Engine()
        self._arrays = S.SearchArrays()
        self._generation = 0

        self._numba_ready = threading.Event()
        self._numba_failed = False
        self.last_score: float | None = None  # centipawns, mover's perspective -- diagnostic only, see harness.match

        stages_done = self._compile_blocking()
        if self._numba_failed:
            self._numba_ready.set()
            status = "compile failed, permanently using v1"
        elif stages_done >= len(self._STAGES):
            self._numba_ready.set()
            status = "fully warm"
        else:
            threading.Thread(target=self._warm_up_remaining, args=(stages_done,), daemon=True).start()
            status = f"deadline hit after stage {stages_done}/{len(self._STAGES)}, finishing in background"

        init_ms = (time.monotonic() - start) * 1000
        print(f"[cb_nb_engine] init returned after {init_ms:.1f} ms ({status})", file=sys.stderr)

    def _stage_movegen(self) -> None:
        """Stage 1 (highest priority): movegen + make/unmake, and the
        attack-table/bit-scan/pack_move helpers they depend on. This also
        compiles the incremental Zobrist/eval maintenance inlined inside
        make_move/unmake_move's own bodies -- same functions, a different
        call site from stage 2's from-scratch versions."""
        F.run_perft(_WARMUP_FEN, 2)

    def _stage_eval_zobrist(self) -> None:
        """Stage 2: from-scratch Zobrist hashing and material+PST eval --
        separate compiled units from make_move/unmake_move's incremental
        maintenance. Needed once per search() call to seed the starting
        position (compute_hash/compute_eval_state) and at every search
        leaf (evaluate_from_state)."""
        pieces, mailbox, meta = F.fen_to_state(_WARMUP_FEN)
        F.compute_hash(pieces, meta)
        eval_state = F.new_eval_state(pieces)
        t = F._TABLES
        # alpha/beta wide enough that the lazy short-circuit never fires --
        # this warmup call needs to actually execute the full-eval path
        # (not just have it present in the compiled code) so every
        # sub-function it calls (mobility, etc.) gets compiled too, not
        # just the material+PST fast path.
        F.evaluate_from_state(
            eval_state, meta[0], pieces, meta[1], -1_000_000, 1_000_000,
            t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
            t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
            t.knight_attacks, t.king_attacks,
        )

    def _stage_search(self) -> None:
        """Stage 3 (lowest priority, by far the largest compile unit):
        the negamax/PVS/quiescence/TT search kernel. Left for last
        deliberately -- it's the one v1 substitutes for most cleanly, so
        it's the most affordable to leave for the background thread if
        the deadline is tight."""
        warmup_arrays = S.SearchArrays()
        S.search(_WARMUP_FEN, soft_ms=1, hard_ms=50, arrays=warmup_arrays)

    _STAGES = (_stage_movegen, _stage_eval_zobrist, _stage_search)

    def _run_stage(self, stage_fn) -> bool:
        """Runs one compile stage. Returns False (and marks the numba
        path permanently failed) if it raises -- callers must stop
        attempting further stages, blocking or background, once this
        returns False."""
        try:
            stage_fn(self)
            return True
        except Exception:
            self._numba_failed = True
            return False

    def _compile_blocking(self) -> int:
        """Runs stages in priority order until either all finish or the
        deadline is reached. The deadline is only checked *between*
        stages -- a single stage's njit compile call can't be preempted
        partway through -- so it's checked before starting each stage,
        deliberately leaving the (much larger) search stage for last:
        that's what bounds the overrun risk to whichever stage happens
        to be in flight when time runs out, rather than the whole
        compile."""
        stages_done = 0
        for stage_fn in self._STAGES:
            if self._numba_failed or time.monotonic() >= self._deadline:
                break
            if not self._run_stage(stage_fn):
                break
            stages_done += 1
        return stages_done

    def _warm_up_remaining(self, stages_done: int) -> None:
        """Background continuation of _compile_blocking after a deadline
        hit -- finishes whatever stages didn't make it in before init
        returned. get_move keeps serving from v1 until this sets
        _numba_ready."""
        try:
            for stage_fn in self._STAGES[stages_done:]:
                if self._numba_failed:
                    return
                if not self._run_stage(stage_fn):
                    return
        finally:
            self._numba_ready.set()

    def get_move(self, fen: str, time_left_ms: int) -> str:
        """Raises on any internal failure, same contract as cb_engine.Engine
        -- agent.py is responsible for catching that, verifying legality,
        and falling back further to a trivially-legal move."""
        if self._numba_ready.is_set() and not self._numba_failed:
            move = self._get_move_numba(fen, time_left_ms)
        else:
            move = self._v1.get_move(fen, time_left_ms)
            self.last_score = self._v1.last_score
        return move

    def _get_move_numba(self, fen: str, time_left_ms: int) -> str:
        ply = chess.Board(fen).ply()
        budget = self._v1.time_manager.budget(time_left_ms, ply=ply)

        self._generation += 1
        move, score, _info = S.search(
            fen, budget.soft_ms, budget.hard_ms, self._arrays, self._generation,
        )
        self.last_score = score
        return F.move_to_uci(move)
