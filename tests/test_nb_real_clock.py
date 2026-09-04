"""Regression gate for the real-clock time-management bug: cb_nb_search's
iterative-deepening driver used to let a single depth iteration run
unbounded inside njit, checked only *between* depths. Once the eval-terms
work dropped nps enough for a single deep iteration to plausibly outlast
the whole clock, that showed up as real time-forfeit losses -- roughly
12% of our own agent's games in a tournament-TC calibration run, before
the node-count-polled hard deadline (see cb_nb_search.py's search()
docstring) and the soft-limit iteration-cost prediction were added.

Every other test in this suite passed while that bug was live, because
none of them played a full game under a real clock: mate fixtures and
perft don't touch the time budget at all, and the deadline-stress tests
in test_nb_search.py call cb_nb_search.search() directly with synthetic
soft_ms/hard_ms rather than going through a real per-move clock across a
whole game. This test plays the real, shipped agent.py through a
complete game at the actual tournament time control and asserts neither
side's reason is a time flag -- the only kind of check that would have
actually caught the bug that shipped.
"""
import pytest

from harness.match import DEFAULT_INCREMENT_MS, DEFAULT_TIME_MS, play_game

AGENT = "agent.py"
BASELINE = "baselines/random_mover.py"


@pytest.mark.slow
def test_agent_as_white_completes_a_full_game_without_flagging():
    result = play_game(AGENT, BASELINE, time_ms=DEFAULT_TIME_MS, increment_ms=DEFAULT_INCREMENT_MS)
    assert "flag" not in result.reason, f"flagged on time: {result.reason} (agent played white)"


@pytest.mark.slow
def test_agent_as_black_completes_a_full_game_without_flagging():
    result = play_game(BASELINE, AGENT, time_ms=DEFAULT_TIME_MS, increment_ms=DEFAULT_INCREMENT_MS)
    assert "flag" not in result.reason, f"flagged on time: {result.reason} (agent played black)"
