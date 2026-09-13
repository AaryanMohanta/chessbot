"""Correction history (CorrHist, 2026-09): hand-verified tests for the
EMA update and correction-application arithmetic in isolation, before any
of it is wired into negamax/quiescence. Real losses this session showed
no tactical blunder -- a slow, sustained static-eval bias instead -- and
CorrHist targets exactly that by keying a correction on a pawn-structure
feature (see the pawn-hash tests in tests/test_nb_incremental.py for the
key itself).

Both functions here are deliberately flag-agnostic (no ENABLE_CORRHIST
check inside): the flag gates whether callers reach for this table at
all, matching the existing ENABLE_RFP/ENABLE_FUTILITY pattern in
cb_nb_search.py rather than branching inside the arithmetic itself.
"""
import numpy as np

import cb_nb_search as s

CORRHIST_SIZE = s.CORRHIST_SIZE


def _fresh_table():
    return np.zeros((2, CORRHIST_SIZE), dtype=np.int64)


def test_correct_returns_static_eval_unchanged_before_any_update():
    table = _fresh_table()
    assert s.corrhist_correct(table, 0, np.uint64(123), 57) == 57


def test_single_update_then_correct_shifts_toward_the_observed_bias():
    """search_score consistently higher than static_eval at this pawn
    structure -- corrected eval should move up towards that bias, not
    stay pinned at the raw static eval."""
    table = _fresh_table()
    s.corrhist_update(table, 0, np.uint64(5), depth=4, search_score=150, static_eval=50)
    corrected = s.corrhist_correct(table, 0, np.uint64(5), 50)
    assert corrected > 50


def test_repeated_updates_converge_towards_the_bias():
    """A +20cp bias stays well inside CORRHIST_MAX's +/-32cp cap (see
    test_update_is_bounded_by_corrhist_max for the clamp itself), so
    repeated updates should visibly converge toward it rather than get
    stopped short by clamping."""
    table = _fresh_table()
    for _ in range(50):
        s.corrhist_update(table, 0, np.uint64(9), depth=6, search_score=20, static_eval=0)
    corrected = s.corrhist_correct(table, 0, np.uint64(9), 0)
    assert corrected >= 18


def test_update_is_bounded_by_corrhist_max():
    table = _fresh_table()
    for _ in range(200):
        s.corrhist_update(table, 0, np.uint64(1), depth=20, search_score=100_000, static_eval=-100_000)
    corrected = s.corrhist_correct(table, 0, np.uint64(1), 0)
    assert corrected <= s.CORRHIST_MAX // s.CORRHIST_GRAIN
    # And the symmetric negative case.
    for _ in range(200):
        s.corrhist_update(table, 0, np.uint64(2), depth=20, search_score=-100_000, static_eval=100_000)
    corrected_neg = s.corrhist_correct(table, 0, np.uint64(2), 0)
    assert corrected_neg >= -(s.CORRHIST_MAX // s.CORRHIST_GRAIN)


def test_different_pawn_keys_do_not_interfere():
    table = _fresh_table()
    s.corrhist_update(table, 0, np.uint64(5), depth=8, search_score=300, static_eval=0)
    untouched = s.corrhist_correct(table, 0, np.uint64(6), 0)
    assert untouched == 0


def test_different_sides_do_not_interfere():
    table = _fresh_table()
    s.corrhist_update(table, 0, np.uint64(5), depth=8, search_score=300, static_eval=0)
    untouched = s.corrhist_correct(table, 1, np.uint64(5), 0)
    assert untouched == 0


def test_higher_depth_applies_a_larger_first_update_weight():
    """From a fresh (zero) entry, a single update's resulting entry is
    exactly scaled_diff*weight/SCALE -- a strictly larger depth (below the
    weight cap) must produce a strictly larger correction."""
    table_shallow = _fresh_table()
    table_deep = _fresh_table()
    s.corrhist_update(table_shallow, 0, np.uint64(1), depth=1, search_score=100, static_eval=0)
    s.corrhist_update(table_deep, 0, np.uint64(1), depth=8, search_score=100, static_eval=0)
    shallow_corrected = s.corrhist_correct(table_shallow, 0, np.uint64(1), 0)
    deep_corrected = s.corrhist_correct(table_deep, 0, np.uint64(1), 0)
    assert deep_corrected > shallow_corrected


def test_weight_is_capped_at_128_past_depth_eleven():
    """(depth+1)^2 crosses 128 between depth 10 (121) and depth 11 (144,
    capped) -- depth 11 and depth 20 must produce identical results since
    both hit the same capped weight."""
    table_11 = _fresh_table()
    table_20 = _fresh_table()
    s.corrhist_update(table_11, 0, np.uint64(1), depth=11, search_score=100, static_eval=0)
    s.corrhist_update(table_20, 0, np.uint64(1), depth=20, search_score=100, static_eval=0)
    assert s.corrhist_correct(table_11, 0, np.uint64(1), 0) == s.corrhist_correct(table_20, 0, np.uint64(1), 0)


def test_correction_never_reaches_mate_threshold():
    table = _fresh_table()
    for _ in range(500):
        s.corrhist_update(table, 0, np.uint64(1), depth=30, search_score=100_000, static_eval=-100_000)
    corrected = s.corrhist_correct(table, 0, np.uint64(1), s.MATE_THRESHOLD - 1)
    assert -s.MATE_THRESHOLD < corrected < s.MATE_THRESHOLD


def test_correction_never_crosses_negative_mate_threshold():
    table = _fresh_table()
    for _ in range(500):
        s.corrhist_update(table, 0, np.uint64(1), depth=30, search_score=-100_000, static_eval=100_000)
    corrected = s.corrhist_correct(table, 0, np.uint64(1), -(s.MATE_THRESHOLD - 1))
    assert -s.MATE_THRESHOLD < corrected < s.MATE_THRESHOLD
