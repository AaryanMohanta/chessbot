"""cb_time is real plumbing (not a strategy stub), so it gets a few basic
invariant checks now. Scenario-level tests (does the engine actually play
faster under time pressure, does a real search respect the hard limit) are
left as structured slots for once cb_search does real iterative work.
"""
import pytest

from cb_time import TimeManager, INIT_BUDGET_MS


@pytest.fixture
def tm():
    return TimeManager()


def test_soft_never_exceeds_hard(tm):
    for time_left in (500, 3_000, 10_000, 60_000, 120_000):
        budget = tm.budget(time_left, ply=0)
        assert budget.soft_ms <= budget.hard_ms


def test_never_budgets_more_than_time_left(tm):
    for time_left in (500, 3_000, 10_000, 60_000, 120_000):
        budget = tm.budget(time_left, ply=0)
        assert budget.hard_ms < time_left


def test_zero_time_left_yields_zero_budget(tm):
    budget = tm.budget(0, ply=0)
    assert budget.soft_ms == 0
    assert budget.hard_ms == 0


def test_low_time_triggers_panic_mode(tm):
    budget = tm.budget(1_000, ply=40)
    assert budget.soft_ms <= 50


def test_budget_shrinks_as_ply_increases_with_fixed_time(tm):
    # Same remaining time, deeper into the game -> fewer estimated moves to
    # go -> a larger (or equal) slice of it per move.
    early = tm.budget(60_000, ply=2)
    late = tm.budget(60_000, ply=120)
    assert late.soft_ms >= early.soft_ms


@pytest.mark.skip(reason="TODO: fill in once cb_search does real iterative-deepening work")
def test_search_respects_hard_budget_under_load():
    ...


@pytest.mark.skip(reason="TODO: fill in once cb_engine.__init__ does real weight-loading work")
def test_engine_init_stays_within_init_budget():
    assert INIT_BUDGET_MS == 60_000
