"""Numba bitboard engine wrapper: deadline-based blocking compile with a
background-compile fallback.

Shipped: in tools/build_zip.py's whitelist and imported by agent.py as
the primary engine. v1 (cb_engine.py) remains the fallback whenever
compilation hasn't finished in time or the numba path fails outright,
and stays the perft oracle. (This docstring is a holdover from before
the agent.py switchover -- corrected 2026-09, see git history for when
the numba path actually went live.)

Why blocking-then-background, not background-only: the competition gives
a 90s init budget (raised from 60s, 2026-09) before the game clock
starts, with no contention for the single core during that window. A
background-only design (the first cut of this module) throws that
budget away and instead pays the full JIT compile cost *during play*,
where numba's LLVM codegen competes with the live search for the one
available core -- so the opening moves served from v1 during warmup end
up slower and weaker than v1 normally is, not just "v1 instead of numba".

Instead, __init__ compiles normally (blocking) against a wall-clock
deadline measured from *process start* (see the process_start param --
agent.py will eventually capture this before any imports run, since
interpreter/import overhead already eats into the 90s budget before this
class is even constructed). Compilation runs in priority order --
movegen/make-unmake first, then eval/Zobrist, then the search kernel,
then a purely-optional CPU/cache warmup pass last -- so if the deadline
is hit partway through, whatever's already warm is the highest-value
subset available: the search kernel is both by far the largest compile
unit and the one v1 substitutes for most cleanly, so it's the one left
to finish in the background if time runs out; the warmup pass (2026-09,
see _stage_cpu_warmup) is lowest priority of all since it's not needed
for correctness at all, just spending otherwise-idle init time (real
compile finishes in ~35-42s, well under the 90s budget) on CPU frequency
scaling and page-cache state before the real clock starts.

Four outcomes, the first three covered by tests/test_nb_insurance_policy.py:
  1. All stages finish inside the deadline (expected on reference
     hardware) -- init returns with numba fully warm, at zero cost to
     live play.
  2. The deadline is hit after some stages -- init returns immediately
     with whatever's warm, the remaining stages finish on a background
     thread, and get_move serves from v1 until they do. This is the
     original (now fallback-only) lever #7 path.
  3. A stage raises -- treated as a permanent numba failure; get_move
     falls back to v1 for the rest of the game, no retries.
  4. Everything through the search kernel finishes but the deadline is
     hit during the CPU-warmup stage -- functionally identical to
     outcome 1 (numba was already fully compiled and correct after stage
     3), just without the warmup's minor latency benefit for the
     opening moves.
"""
from __future__ import annotations

import sys
import threading
import time

import chess
import numpy as np

import cb_engine
import cb_nb_fast as F
import cb_nb_search as S
from cb_time import INIT_BUDGET_MS

_WARMUP_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Wall-clock budget (seconds, measured from process start -- see the
# process_start constructor param, not from when compilation begins) for
# blocking compilation during init. Raised 40.0 -> 70.0 (2026-09) to match
# the rules update's 60s -> 90s init budget, keeping the same ~20s safety
# margin before the hard cutoff: real ladder hardware has consistently
# compiled in ~28-33s (see round 13-15 match logs), so 70s gives large
# headroom for v1 to never be the one actually playing, while still
# trading "risk of overrunning the hard init budget" against "risk of
# falling back to the slower, core-contended background-compile path for
# the opening moves" the same way the old 40s/60s pair did.
COMPILE_DEADLINE_S = 70.0

# Safety margin (seconds) _stage_cpu_warmup keeps against the real 90s
# init budget when sizing itself to fill whatever's left -- separate from
# COMPILE_DEADLINE_S's own margin since this stage runs strictly after
# compilation already succeeded, so it can safely use more of the gap
# between 70s and 90s than the compile stages themselves are trusted with.
_CPU_WARMUP_SAFETY_S = 10.0


class Engine:
    def __init__(self, process_start: float | None = None) -> None:
        start = time.monotonic()
        origin = process_start if process_start is not None else start
        self._deadline = origin + COMPILE_DEADLINE_S
        # The real hard cutoff (2026-09): used only by _stage_cpu_warmup to
        # size itself against whatever's actually left of the real 90s
        # budget, not the more conservative 70s compile deadline above --
        # this stage isn't compiling anything, so it can safely use the
        # margin between the two.
        self._init_budget_deadline = origin + INIT_BUDGET_MS / 1000

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
        self.last_depth: int | None = None  # diagnostic only, see agent.py's per-move stderr log
        self.last_nodes: int | None = None
        self.last_layer: str | None = None  # "numba" or "v1", whichever path actually produced the last move
        self._stage_times_ms: list[tuple[str, float]] = []

        stages_done = self._compile_blocking()
        ready_before_move1 = False
        if self._numba_failed:
            self._numba_ready.set()
            status = "compile failed, permanently using v1"
        elif stages_done >= len(self._STAGES):
            self._numba_ready.set()
            status = "fully warm"
            ready_before_move1 = True
        else:
            threading.Thread(target=self._warm_up_remaining, args=(stages_done,), daemon=True).start()
            status = f"deadline hit after stage {stages_done}/{len(self._STAGES)}, finishing in background"

        init_ms = (time.monotonic() - start) * 1000
        # Terse, parseable per-stage breakdown (2026-09 telemetry) --
        # short names so this stays legible inside the 8KB-per-game log's
        # first-4KB slice alongside everything else that happens at
        # init. ready_before_move1 is the direct answer to "did numba
        # ever get a chance to play the opening" without having to infer
        # it from the status string.
        stage_summary = " ".join(f"{name.lstrip('_')}={ms:.0f}ms" for name, ms in self._stage_times_ms)
        print(f"[cb_nb_engine] init returned after {init_ms:.1f} ms ({status}) "
              f"ready_before_move1={ready_before_move1} stages: {stage_summary}", file=sys.stderr)

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

    def _stage_cpu_warmup(self) -> None:
        """Stage 4 (2026-09), lowest priority of all: a longer, more
        realistic-depth search on the warmup position, spending some of
        the otherwise-idle init budget (real compile finishes in ~35-42s
        on reference hardware, well under the 90s budget) actually
        exercising the CPU at sustained load before the real game clock
        starts.

        NOT for compilation coverage -- confirmed empirically that numba
        compiles a whole function ahead of time on its first call rather
        than lazily per branch, so stage 3's tiny 50ms warmup already
        fully compiles every function in the real search call graph
        (root_search/try_move/negamax/quiescence/order_moves/see_capture
        all get invoked, and therefore fully compiled, even at depth 1).
        The actual target here is CPU frequency scaling and OS page-cache
        state: a brief burst of real load is what gets a CPU up to its
        sustained boost clock and its instruction/data caches populated
        with the hot search code, and the first few real moves of the
        game are exactly when that would otherwise still be cold.

        Deliberately last in _STAGES: if the deadline is already tight
        after the compile stages, this is what gets deferred to the
        background thread instead, at zero extra risk to the stages that
        actually matter.

        Budget (2026-09 retest): fills whatever's actually left of the
        real 90s init budget, not a fixed few seconds -- compile alone
        typically leaves ~30-50s completely idle, and none of it costs
        real game-clock time either way, so there's no reason to leave
        most of it on the table. _CPU_WARMUP_SAFETY_S keeps a hard
        buffer against the 90s cutoff (this stage's own timing checks are
        wall-clock-based like every other search call, not perfectly
        exact) rather than trusting the arithmetic down to the second."""
        remaining_s = self._init_budget_deadline - time.monotonic() - _CPU_WARMUP_SAFETY_S
        if remaining_s <= 0:
            return
        warmup_arrays = S.SearchArrays()
        S.search(_WARMUP_FEN, soft_ms=remaining_s * 1000, hard_ms=remaining_s * 1000, arrays=warmup_arrays)

    _STAGES = (_stage_movegen, _stage_eval_zobrist, _stage_search, _stage_cpu_warmup)

    def _run_stage(self, stage_fn) -> bool:
        """Runs one compile stage. Returns False (and marks the numba
        path permanently failed) if it raises -- callers must stop
        attempting further stages, blocking or background, once this
        returns False. Records its own wall-clock time in
        self._stage_times_ms (2026-09, per-stage init telemetry) whether
        it succeeds or fails, since a slow FAILING stage is exactly the
        case worth seeing in the log."""
        t0 = time.monotonic()
        try:
            stage_fn(self)
            return True
        except Exception:
            self._numba_failed = True
            return False
        finally:
            self._stage_times_ms.append((stage_fn.__name__, (time.monotonic() - t0) * 1000))

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
        t0 = time.monotonic()
        try:
            for stage_fn in self._STAGES[stages_done:]:
                if self._numba_failed:
                    return
                if not self._run_stage(stage_fn):
                    return
        finally:
            self._numba_ready.set()
            # 2026-09 telemetry: v1 serves however many real moves land
            # between init returning and this firing -- worth its own
            # line since the main init line's ready_before_move1=False
            # only says numba WASN'T ready at move 1, not when (or
            # whether) it ever became ready at all.
            print(f"[cb_nb_engine] background warmup finished after {(time.monotonic() - t0) * 1000:.1f} ms "
                  f"(numba_failed={self._numba_failed})", file=sys.stderr)

    def get_move(self, fen: str, time_left_ms: int) -> str:
        """Raises on any internal failure, same contract as cb_engine.Engine
        -- agent.py is responsible for catching that, verifying legality,
        and falling back further to a trivially-legal move."""
        if self._numba_ready.is_set() and not self._numba_failed:
            move = self._get_move_numba(fen, time_left_ms)
        else:
            move = self._v1.get_move(fen, time_left_ms)
            self.last_score = self._v1.last_score
            self.last_depth = self._v1.last_depth
            self.last_nodes = self._v1.last_nodes
            self.last_layer = "v1"  # internal fallback -- agent.py's own layer name stays "numba" either way
        return move

    def _get_move_numba(self, fen: str, time_left_ms: int) -> str:
        ply = chess.Board(fen).ply()
        budget = self._v1.time_manager.budget(time_left_ms, ply=ply)

        pieces, _mailbox, meta = F.fen_to_state(fen)
        key = np.uint64(F.compute_hash(pieces, meta))
        self._arrays.record_game_position(key)

        self._generation += 1
        move, score, info = S.search(
            fen, budget.soft_ms, budget.hard_ms, self._arrays, self._generation,
        )
        self.last_score = score
        self.last_depth = info["depth"]
        self.last_nodes = info["nodes"]
        self.last_layer = "numba"
        return F.move_to_uci(move)
