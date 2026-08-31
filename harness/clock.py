"""Per-side chess clock enforcement for the harness (mirrors the real
120_000 ms + 500 ms increment competition clock, but with configurable
values so tests can use faster clocks)."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SideClock:
    remaining_ms: float
    increment_ms: float = 500

    def consume(self, elapsed_ms: float) -> bool:
        """Deduct time spent thinking; add the increment if it didn't flag.

        Returns True if this consumption flagged the clock (remaining time
        ran out before the move arrived).
        """
        self.remaining_ms -= elapsed_ms
        if self.remaining_ms <= 0:
            return True
        self.remaining_ms += self.increment_ms
        return False
