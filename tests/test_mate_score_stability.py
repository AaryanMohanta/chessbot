"""search()'s iterative-deepening loop must not trust a mate-range score
for its early-exit until it's been found at real depth (2026-09, after a
real ladder loss -- round 53 -- where a K+Q vs K endgame, completely won,
drew by threefold repetition: the match log showed depth collapsing to 1
(some iterations as low as 55-30 total nodes) for essentially every move
from the point a mate-range score first appeared, for the rest of the
game. root_search's own TT probe can return a cached BOUND_EXACT score
for the exact current position from an earlier (and possibly, in a
recurring king-and-queen dance, no-longer-quite-applicable) deeper
search -- a single depth-1 iteration hitting that cache is not the same
as THIS move's search having verified the mate is still there, and
`abs(best_score) >= MATE_THRESHOLD: break` trusted it anyway, permanently
starving every later move of the extra plies that might have caught the
mistake or found genuine progress.

Uses a fake root_search (mirrors tests/test_nb_insurance_policy.py's own
mocking style) that reports a mate-range score starting at depth 1, so
the ID loop's behavior can be tested directly without needing a position
that actually reproduces stale TT pollution.
"""
import numpy as np

import cb_nb_search as s


def _startpos_state():
    import cb_nb_fast as f
    pieces, mailbox, meta = f.fen_to_state(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    return pieces, mailbox, meta


class _FakeRootSearchAlwaysMate:
    """Reports a mate-range score at EVERY depth starting from 1 -- the
    exact shape of the real bug (not "eventually finds mate at a
    reasonable depth", but "claims mate from the very first, shallowest
    iteration"). Records every depth it was called at."""

    def __init__(self):
        self.depths_called: list[int] = []

    def __call__(self, pieces, mailbox, meta, depth, generation, alpha, beta, *rest):
        self.depths_called.append(depth)
        return s.MATE_SCORE - 5, 12345  # mate-range score, a fixed dummy move


def test_does_not_stop_at_depth_1_just_because_the_score_is_mate_range(monkeypatch):
    fake = _FakeRootSearchAlwaysMate()
    monkeypatch.setattr(s, "root_search", fake)

    pieces, mailbox, meta = _startpos_state()
    import cb_nb_fast as f
    arrays = s.SearchArrays()
    arrays.zobrist[0] = np.uint64(f.compute_hash(pieces, meta))
    arrays.zobrist[1] = np.uint64(f.compute_pawn_hash(pieces))
    mg, eg, phase = f.compute_eval_state(pieces)
    arrays.eval_state[0], arrays.eval_state[1], arrays.eval_state[2] = mg, eg, phase

    move, score, info = s.search(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        soft_ms=5000, hard_ms=10000, arrays=arrays, generation=1,
    )

    assert max(fake.depths_called) >= s.MATE_FOUND_MIN_DEPTH, (
        f"stopped after depths {fake.depths_called} -- a mate-range score at shallow "
        f"depth must not short-circuit the ID loop before {s.MATE_FOUND_MIN_DEPTH}"
    )


def test_still_stops_reasonably_promptly_once_min_depth_is_reached(monkeypatch):
    """The fix must not turn into "never trust mate, always burn the full
    budget" -- once MATE_FOUND_MIN_DEPTH is reached, it should still stop
    rather than continuing to MAX_DEPTH for no reason."""
    fake = _FakeRootSearchAlwaysMate()
    monkeypatch.setattr(s, "root_search", fake)

    pieces, mailbox, meta = _startpos_state()
    import cb_nb_fast as f
    arrays = s.SearchArrays()
    arrays.zobrist[0] = np.uint64(f.compute_hash(pieces, meta))
    arrays.zobrist[1] = np.uint64(f.compute_pawn_hash(pieces))
    mg, eg, phase = f.compute_eval_state(pieces)
    arrays.eval_state[0], arrays.eval_state[1], arrays.eval_state[2] = mg, eg, phase

    s.search(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        soft_ms=5000, hard_ms=10000, arrays=arrays, generation=1,
    )

    assert max(fake.depths_called) < s.MATE_FOUND_MIN_DEPTH + 3, (
        f"kept iterating to depths {fake.depths_called} well past the confirmation "
        f"floor -- once a mate is confirmed at real depth, still stop promptly"
    )
