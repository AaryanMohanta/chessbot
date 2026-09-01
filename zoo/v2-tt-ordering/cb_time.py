"""Time management: converts a clock reading into a per-move thinking budget.

Implements the design doc's §7 formula exactly:

    reserve      = 2000 ms
    moves_to_go  = max(20, 55 - ply // 2)
    usable       = max(0, time_left - reserve)
    soft         = increment * 0.9 + usable / moves_to_go
    hard         = min(soft * 4, usable * 0.3)

``ply`` is total game half-moves played so far (``board.ply()``), not a
per-side move count — moves_to_go decreasing as the game progresses means
the *quotient* usable/moves_to_go grows, so later moves get a bigger slice
of what's left. That's deliberate (§7: "bank time early, spend it in the
middlegame").

Usage: soft is checked between iterative-deepening iterations (don't start
another depth once exceeded); hard is checked inside the search, every
2048 nodes, and enforced by raising a private timeout there.

KNOWN PROPERTY (not a bug): when usable time is small relative to the
increment, this formula can produce soft > hard (e.g. time_left=2500ms,
increment=500ms -> usable=500ms -> soft=459ms, hard=150ms). That's
harmless: hard's wall-clock deadline still lands earlier than soft's in
that regime, so the in-search hard cutoff always fires first regardless of
which number is nominally larger. See tests/test_time_management.py.
"""
from __future__ import annotations

import dataclasses

# One-time budget for process start + engine init (weight loading, warmup),
# spent before the per-move clock begins. Not consumed by budget().
INIT_BUDGET_MS = 60_000

DEFAULT_INCREMENT_MS = 500

RESERVE_MS = 2_000
MIN_MOVES_TO_GO = 20
MOVES_TO_GO_BASE = 55
HARD_SOFT_MULTIPLE = 4
HARD_FRACTION_OF_USABLE = 0.3


@dataclasses.dataclass(frozen=True)
class TimeBudget:
    """A single move's allotment. Both values are milliseconds of *thinking*
    time, not wall-clock deadlines."""

    soft_ms: int
    hard_ms: int


class TimeManager:
    def estimate_moves_to_go(self, ply: int) -> int:
        return max(MIN_MOVES_TO_GO, MOVES_TO_GO_BASE - ply // 2)

    def budget(
        self,
        time_left_ms: int,
        increment_ms: int = DEFAULT_INCREMENT_MS,
        ply: int = 0,
    ) -> TimeBudget:
        """Compute this move's thinking budget per design doc §7.

        Args:
            time_left_ms: our remaining clock time, as reported for this move.
            increment_ms: increment added back after each move.
            ply: total game half-moves played so far (``board.ply()``).
        """
        moves_to_go = self.estimate_moves_to_go(ply)
        usable_ms = max(0, time_left_ms - RESERVE_MS)

        soft_ms = increment_ms * 0.9 + usable_ms / moves_to_go
        hard_ms = min(soft_ms * HARD_SOFT_MULTIPLE, usable_ms * HARD_FRACTION_OF_USABLE)

        return TimeBudget(soft_ms=int(soft_ms), hard_ms=int(hard_ms))
