"""Time management: converts a clock reading into a per-move thinking budget.

This is plumbing, not strategy — it knows nothing about chess positions,
only about clocks. ``cb_engine`` asks it "how long should I think" and gets
back a soft target (aim to stop around here) and a hard ceiling (never
exceed this, on pain of flagging).

Competition clock: 120_000 ms + 500 ms increment per side, with a separate
60_000 ms one-time init budget spent *before* the clock starts (see
``INIT_BUDGET_MS`` — this module doesn't enforce it, ``cb_engine.__init__``
does, but the constant lives here so both sides agree on the number).
"""
from __future__ import annotations

import dataclasses

# One-time budget for process start + engine init (weight loading, warmup),
# spent before the per-move clock begins. Not consumed by budget().
INIT_BUDGET_MS = 60_000

DEFAULT_INCREMENT_MS = 500

# Below this remaining time, stop planning normally and just survive.
PANIC_THRESHOLD_MS = 3_000
PANIC_SOFT_MS = 50
PANIC_HARD_FRACTION = 0.5  # never use more than half of what's left, even panicking

# Never plan to leave less than this much time on the clock after a move
# (covers IPC/OS scheduling jitter around the engine's own budget).
MIN_RESERVE_MS = 200

# Moves-to-go estimate: assume the game runs about this many full moves in
# total, tapering down as it progresses, floored so the late game still
# gets a sane per-move slice instead of assuming "40 more" forever.
ASSUMED_GAME_FULLMOVES = 40
MIN_MOVES_TO_GO = 10

# Hard ceiling is allowed to stretch past the soft target (search can keep
# a deepening iteration going) but not by an unbounded amount, and never
# past this fraction of remaining time.
HARD_MULTIPLE_OF_SOFT = 4
HARD_FRACTION_OF_REMAINING = 0.5


@dataclasses.dataclass(frozen=True)
class TimeBudget:
    """A single move's allotment. Both values are milliseconds of *thinking*
    time, not wall-clock deadlines."""

    soft_ms: int
    hard_ms: int


class TimeManager:
    """Stateless in practice today (moves_to_go can be estimated purely from
    ply), but kept as a class so per-game state — e.g. a running average of
    actual time spent per move — can be added later without changing the
    call site in ``cb_engine``."""

    def estimate_moves_to_go(self, ply: int) -> int:
        fullmove_number = ply // 2 + 1
        return max(MIN_MOVES_TO_GO, ASSUMED_GAME_FULLMOVES - fullmove_number)

    def budget(
        self,
        time_left_ms: int,
        increment_ms: int = DEFAULT_INCREMENT_MS,
        ply: int = 0,
        moves_to_go: int | None = None,
    ) -> TimeBudget:
        """Compute this move's thinking budget.

        Args:
            time_left_ms: our remaining clock time, as reported by the
                runner for this move (already net of everything spent so far).
            increment_ms: increment added back after each move.
            ply: half-moves played so far in the game (0 at game start),
                used to estimate how many moves remain when ``moves_to_go``
                isn't supplied.
            moves_to_go: override the moves-remaining estimate (e.g. for a
                real moves-to-next-time-control rule); falls back to a
                ply-based heuristic when omitted.
        """
        if time_left_ms <= 0:
            return TimeBudget(soft_ms=0, hard_ms=0)

        if time_left_ms <= PANIC_THRESHOLD_MS:
            soft = min(PANIC_SOFT_MS, time_left_ms)
            hard = max(soft, int(time_left_ms * PANIC_HARD_FRACTION))
            hard = min(hard, max(0, time_left_ms - 1))
            return TimeBudget(soft_ms=soft, hard_ms=hard)

        remaining_moves = moves_to_go or self.estimate_moves_to_go(ply)
        base_ms = time_left_ms / remaining_moves + increment_ms * 0.8

        soft = int(base_ms)
        hard = int(
            min(
                time_left_ms - MIN_RESERVE_MS,
                base_ms * HARD_MULTIPLE_OF_SOFT,
                time_left_ms * HARD_FRACTION_OF_REMAINING,
            )
        )

        soft = max(1, min(soft, time_left_ms - MIN_RESERVE_MS))
        hard = max(soft, min(hard, time_left_ms - 1))

        return TimeBudget(soft_ms=soft, hard_ms=hard)
