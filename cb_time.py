"""Time management: converts a clock reading into a per-move thinking budget.

Based on the design doc's §7 formula, with MIN_MOVES_TO_GO/MOVES_TO_GO_BASE
retuned (2026-09) from real game-length data instead of the original guess:

    reserve      = 2000 ms
    moves_to_go  = max(15, 50 - ply // 2)
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
# spent before the per-move clock begins. Not consumed by budget(). Raised
# 60_000 -> 90_000 (2026-09) per the competition rules update; real ladder
# hardware compiles in ~28-33s either way, so this mainly widens the
# margin before a legitimate compile is ever mistaken for a hang.
INIT_BUDGET_MS = 90_000

DEFAULT_INCREMENT_MS = 500

RESERVE_MS = 2_000
# Retest (2026-09): re-derived from real game lengths rather than the
# original guess -- the 20-game match vs. a comparable-strength opponent
# at this exact TC had a median length of 47 full moves (mean 45, range
# 13.5-69.5), not ~55. MIN lowered from 20 and BASE from 55 so later
# moves get a meaningfully bigger slice of what's left, without being so
# aggressive it risks flagging on the long tail of real games (simulated
# steady-state spend: ~80% of the clock by move 40, ~87% by move 47).
MIN_MOVES_TO_GO = 15
MOVES_TO_GO_BASE = 50
HARD_SOFT_MULTIPLE = 4
HARD_FRACTION_OF_USABLE = 0.3

# 600-ply-draw retest (2026-09): moves_to_go's floor of 15 was tuned
# against 47-move real games, not the new rule's 300-move-per-side cap.
# Simulating an always-spend-soft 300-of-our-own-moves game shows this
# does NOT actually flag -- moves_to_go staying fixed at 15 forever past
# ply 70 makes usable_ms geometrically decay toward a stable equilibrium
# (~1.5x increment) rather than bleeding to zero, since the increment
# keeps replenishing it every move. But hard_ms converges to well under
# the increment itself (~0.3x the equilibrium usable, i.e. ~45% of one
# increment) for the entire rest of a long endgame -- exactly the phase
# where actually converting or defending a drawn-out position matters
# most, and now that a 600-ply draw is a real half-point instead of a
# material-based loss, that's worth protecting. Flooring hard_ms at the
# increment itself (see budget() below) is always sustainable regardless
# of how long the game runs: RESERVE_MS is already carved out of
# usable_ms before any of this math happens, so spending up to one full
# increment per move can never be the thing that actually empties the
# real clock.


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
        # Floor (2026-09, see the comment above MIN_MOVES_TO_GO) -- only
        # once moves_to_go has hit its floor, i.e. only in the long-game
        # regime this is actually targeting: an early-game move that's
        # genuinely short on time (moves_to_go still large) must NOT get
        # this floor, or it starts spending a much bigger share of a
        # tiny usable_ms than the formula intends (see
        # test_soft_can_exceed_hard_when_time_is_low, which depends on
        # exactly that regime staying untouched). Capped by usable_ms
        # itself so this can never claim more than what's actually
        # banked regardless.
        if moves_to_go == MIN_MOVES_TO_GO:
            hard_ms = max(hard_ms, min(increment_ms, usable_ms))

        return TimeBudget(soft_ms=int(soft_ms), hard_ms=int(hard_ms))
