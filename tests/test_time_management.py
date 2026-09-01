"""Tests for cb_time's exact §7 formula. This module is real plumbing, so
it gets real tests, not stubs.
"""
import pytest

from cb_time import RESERVE_MS, TimeManager


@pytest.fixture
def tm():
    return TimeManager()


def test_soft_can_exceed_hard_when_time_is_low(tm):
    """Documents a real property of the exact §7 formula, not a bug: when
    usable time is small relative to the increment, soft (increment*0.9 +
    usable/moves_to_go) can come out above hard (min(soft*4, usable*0.3)).
    E.g. time_left=2500ms, increment=500ms -> usable=500ms -> soft=459ms,
    hard=150ms. This is harmless because hard is enforced independently
    inside the search (every 2048 nodes) and its deadline still lands
    earlier in wall-clock time than soft's, so the hard cutoff always
    fires first regardless of which number is larger; soft's "aim to stop
    around here between ID iterations" simply becomes moot in this regime.
    """
    budget = tm.budget(2_500, ply=0)
    assert budget.soft_ms > budget.hard_ms
    assert budget.hard_ms > 0


def test_hard_never_exceeds_usable_time(tm):
    for time_left in (2_500, 5_000, 10_000, 60_000, 120_000):
        budget = tm.budget(time_left, ply=0)
        usable = max(0, time_left - RESERVE_MS)
        assert budget.hard_ms <= usable


def test_near_reserve_yields_zero_hard_budget(tm):
    # time_left below the reserve leaves nothing usable at all.
    budget = tm.budget(1_000, ply=0)
    assert budget.hard_ms == 0


def test_moves_to_go_floors_at_twenty(tm):
    assert tm.estimate_moves_to_go(0) == 55
    assert tm.estimate_moves_to_go(200) == 20
    assert tm.estimate_moves_to_go(1_000) == 20


def test_budget_grows_as_moves_to_go_shrinks_with_fixed_time(tm):
    # Same remaining time, deeper into the game -> smaller moves_to_go ->
    # a bigger slice of what's left per move (§7: bank early, spend later).
    early = tm.budget(60_000, ply=0)
    late = tm.budget(60_000, ply=120)
    assert late.soft_ms >= early.soft_ms


def test_two_hundred_moves_never_exceed_hard_budget_or_go_negative(tm):
    """Simulate 200 of our own moves under the real clock (120s + 0.5s
    increment), always spending exactly the hard budget (worst case), and
    assert the clock never goes negative and the hard budget requested
    never exceeds what's actually left on the clock.
    """
    time_left = 120_000
    increment = 500
    ply = 0

    for _ in range(200):
        budget = tm.budget(time_left, increment_ms=increment, ply=ply)
        assert budget.hard_ms <= time_left

        time_left -= budget.hard_ms
        assert time_left >= 0

        time_left += increment
        ply += 2  # our move + the opponent's reply
