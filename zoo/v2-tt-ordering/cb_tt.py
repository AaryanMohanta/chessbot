"""Transposition table (design doc §5).

Zobrist keying uses python-chess's own ``chess.polyglot.zobrist_hash`` — a
standard, well-tested 64-bit scheme that already folds in side-to-move,
castling rights, and en passant file, rather than hand-rolling a fresh
Zobrist table (a hand-rolled one is a classic source of subtle collision
bugs, e.g. forgetting to include castling rights in the key). It's fixed
at import time (built into the library), satisfying the "generated at
import from a fixed seed" intent without re-deriving something python-chess
already gets right.

Layout: a flat Python list of size ``2**size_bits``, indexed by
``key & mask``. Each slot holds a full ``Entry`` (including the *full*
64-bit key, not just the index) so an index collision can be detected and
treated as a miss rather than silently returning another position's data.

Replacement: replace on an empty slot, on a stale slot (older search
generation than the current one), or on a same-or-newer-generation slot
whose stored depth is <= the incoming depth ("replace by depth, preferring
the current generation" per §5).

Mate scores are stored and probed ply-adjusted (§5's explicit warning):
a raw score near +-MATE_SCORE encodes "mate in K plies from the root of
THIS particular search call", which is meaningless once reattached to a
different node (a different ply-from-root) via a transposition. Storing
converts to "mate in K plies from this position" (ply-independent);
probing converts back using the *current* node's ply, which may differ
from the ply the entry was stored at.
"""
from __future__ import annotations

import dataclasses
import enum

MATE_SCORE = 100_000
MATE_THRESHOLD = MATE_SCORE - 1_000  # anything at least this extreme is "a mate score"

DEFAULT_SIZE_BITS = 20  # 2**20 = ~1M entries


class Bound(enum.IntEnum):
    EXACT = 0
    LOWER = 1  # fail-high: true score >= stored score
    UPPER = 2  # fail-low: true score <= stored score


@dataclasses.dataclass
class Entry:
    key: int
    depth: int
    score: int
    bound: Bound
    best_move: "chess.Move | None"
    generation: int


def score_to_tt(score: int, ply: int) -> int:
    if score > MATE_THRESHOLD:
        return score + ply
    if score < -MATE_THRESHOLD:
        return score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    if score > MATE_THRESHOLD:
        return score - ply
    if score < -MATE_THRESHOLD:
        return score + ply
    return score


class TranspositionTable:
    def __init__(self, size_bits: int = DEFAULT_SIZE_BITS):
        self.size = 1 << size_bits
        self.mask = self.size - 1
        self._table: list[Entry | None] = [None] * self.size

    def probe(self, key: int) -> Entry | None:
        entry = self._table[key & self.mask]
        if entry is not None and entry.key == key:
            return entry
        return None

    def store(self, key: int, depth: int, score: int, bound: Bound, best_move, generation: int) -> None:
        index = key & self.mask
        existing = self._table[index]
        if (
            existing is None
            or existing.generation < generation
            or existing.depth <= depth
        ):
            self._table[index] = Entry(key, depth, score, bound, best_move, generation)

    def clear(self) -> None:
        self._table = [None] * self.size
