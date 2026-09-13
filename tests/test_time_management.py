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


def test_moves_to_go_floors_at_fifteen(tm):
    assert tm.estimate_moves_to_go(0) == 50
    assert tm.estimate_moves_to_go(200) == 15
    assert tm.estimate_moves_to_go(1_000) == 15


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


def test_six_hundred_ply_game_never_flags_even_at_worst_case_spend(tm):
    """The 600-ply draw rule (2026-09, replacing the old 300-ply material
    adjudication) makes a 300-of-our-own-moves game a real possibility,
    not just a pathological edge case -- moves_to_go's floor of 15 was
    tuned against ~47-move real games. Same worst-case-spend methodology
    as the 200-move test above, extended to the new rule's actual limit,
    both for the realistic case (always spend soft_ms) and the adversarial
    one (always spend the full hard_ms)."""
    increment = 500

    for spend_field in ("soft_ms", "hard_ms"):
        time_left = 120_000
        ply = 0
        for move_num in range(300):
            budget = tm.budget(time_left, increment_ms=increment, ply=ply)
            spend = getattr(budget, spend_field)
            assert spend <= time_left, f"move {move_num} ({spend_field}): budget exceeds actual time left"

            time_left -= spend
            assert time_left >= 0, f"FLAGGED at move {move_num} spending {spend_field}"

            time_left += increment
            ply += 2


def test_hard_floor_keeps_deep_endgame_budget_near_a_full_increment(tm):
    """The concrete problem the floor fixes, not just "doesn't flag":
    before this floor existed, hard_ms in the steady-state deep-endgame
    regime (moves_to_go stuck at its floor) converged to well under half
    an increment (~229ms out of a 500ms increment, ~46%) -- this asserts
    it now converges to a comfortably larger fraction instead, since that
    phase is exactly where actually converting or defending a long,
    drawn-out position matters most under the new 600-ply rule."""
    increment = 500
    time_left = 120_000
    ply = 0
    for _ in range(200):  # run well past the point of reaching steady state
        budget = tm.budget(time_left, increment_ms=increment, ply=ply)
        time_left = time_left - budget.soft_ms + increment
        ply += 2

    final_budget = tm.budget(time_left, increment_ms=increment, ply=ply)
    assert final_budget.hard_ms >= increment * 0.9
