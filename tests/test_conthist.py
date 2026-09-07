"""Continuation history (ContHist, 2026-09): the indexing is the actual
risk here, not the update arithmetic (which reuses the exact same
+=depth*depth formula the existing killers/history heuristic already
uses, at cb_nb_search.py's record_cutoff) -- a widely-circulated real
implementation has a copy-paste bug where the 2-ply lookup reads (ss-1)
instead of (ss-2), silently degrading to "the same context twice" rather
than crashing. These tests build a known move sequence and assert the
1-ply and 2-ply contexts resolve to genuinely distinct (piece, to)
values, then that the two update sites land in distinct table cells --
never trusting it just because it compiles and runs.
"""
import numpy as np

import cb_nb_search as s


def test_context_is_empty_at_the_root():
    cont_piece = np.full(10, -1, dtype=np.int64)
    cont_to = np.full(10, -1, dtype=np.int64)
    p1p, p1t, p2p, p2t = s._cont_hist_context(cont_piece, cont_to, 0)
    assert (p1p, p1t, p2p, p2t) == (-1, -1, -1, -1)


def test_context_has_only_one_ply_of_history_at_ply_one():
    cont_piece = np.full(10, -1, dtype=np.int64)
    cont_to = np.full(10, -1, dtype=np.int64)
    cont_piece[1], cont_to[1] = 3, 20  # the move that led to ply 1
    p1p, p1t, p2p, p2t = s._cont_hist_context(cont_piece, cont_to, 1)
    assert (p1p, p1t) == (3, 20)
    assert (p2p, p2t) == (-1, -1)  # nothing 2 plies back yet


def test_context_distinguishes_1ply_from_2ply_at_ply_two():
    """The core regression case: a known 3-node sequence (root -> ply1 ->
    ply2) where the move into ply1 and the move into ply2 use CLEARLY
    DIFFERENT (piece, to) -- if 1-ply and 2-ply ever read the same slot
    (the ss-1-instead-of-ss-2 bug), this fails by both pairs coming back
    identical instead of matching their own distinct recorded move."""
    cont_piece = np.full(10, -1, dtype=np.int64)
    cont_to = np.full(10, -1, dtype=np.int64)
    cont_piece[1], cont_to[1] = 1, 10   # move made to reach ply 1 (2 plies back from ply 2)
    cont_piece[2], cont_to[2] = 7, 55   # move made to reach ply 2 (1 ply back from ply 2)

    p1p, p1t, p2p, p2t = s._cont_hist_context(cont_piece, cont_to, 2)
    assert (p1p, p1t) == (7, 55)
    assert (p2p, p2t) == (1, 10)
    assert (p1p, p1t) != (p2p, p2t)


def test_context_at_ply_three_reads_the_immediately_preceding_two_plies():
    cont_piece = np.full(10, -1, dtype=np.int64)
    cont_to = np.full(10, -1, dtype=np.int64)
    cont_piece[1], cont_to[1] = 1, 10
    cont_piece[2], cont_to[2] = 7, 55
    cont_piece[3], cont_to[3] = 4, 33

    p1p, p1t, p2p, p2t = s._cont_hist_context(cont_piece, cont_to, 3)
    assert (p1p, p1t) == (4, 33)
    assert (p2p, p2t) == (7, 55)


def _fresh_cont_hist():
    return np.zeros((12, 64, 12, 64), dtype=np.int64)


def test_update_writes_1ply_and_2ply_at_distinct_cells():
    """record_cutoff's own two update sites, exercised directly: with
    genuinely distinct 1-ply and 2-ply contexts, the two tables' updated
    cells must be independently addressable -- updating one must never
    silently also update the other via a shared/miscomputed index."""
    cont_hist_1ply = _fresh_cont_hist()
    cont_hist_2ply = _fresh_cont_hist()
    piece, to_sq = 0, 28  # white pawn moving to e4

    s.record_cutoff_conthist(cont_hist_1ply, cont_hist_2ply, piece, to_sq,
                              prev1_piece=7, prev1_to=55, prev2_piece=1, prev2_to=10, depth=5)

    updated_1ply = cont_hist_1ply[7, 55, piece, to_sq]
    updated_2ply = cont_hist_2ply[1, 10, piece, to_sq]
    assert updated_1ply > 0
    assert updated_2ply > 0
    # The 2-ply table's cell at the 1-ply context's coordinates (the bug
    # this whole test file exists to catch) must be untouched.
    assert cont_hist_2ply[7, 55, piece, to_sq] == 0
    assert cont_hist_1ply[1, 10, piece, to_sq] == 0


def test_update_skips_1ply_when_context_absent():
    cont_hist_1ply = _fresh_cont_hist()
    cont_hist_2ply = _fresh_cont_hist()
    s.record_cutoff_conthist(cont_hist_1ply, cont_hist_2ply, 0, 28,
                              prev1_piece=-1, prev1_to=-1, prev2_piece=-1, prev2_to=-1, depth=5)
    assert np.count_nonzero(cont_hist_1ply) == 0
    assert np.count_nonzero(cont_hist_2ply) == 0
