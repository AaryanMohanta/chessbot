"""Tests for the joint rating fit (ratings/elo.py) on synthetic game
records — no engine/subprocess involved, purely the math.
"""
import math

from ratings.db import GameRecord
from ratings.elo import fit_ratings, fit_rating_vs_known_opponents


def _game(white, black, result, i):
    return GameRecord(
        timestamp=f"2026-01-01T00:00:{i:02d}",
        white_version=white,
        black_version=black,
        opening_id="test",
        result=result,
        termination="checkmate",
        plies=40,
    )


def test_stronger_version_gets_higher_rating():
    # A beats B decisively, B beats C decisively; anchor = C.
    games = []
    for i in range(20):
        games.append(_game("A", "B", "1-0", i))
    for i in range(20, 40):
        games.append(_game("B", "C", "1-0", i))

    fit = fit_ratings(games, anchor="C")
    ratings = {e.version: e.rating for e in fit.entries}

    assert ratings["A"] > ratings["B"] > ratings["C"] == 0
    assert all(math.isfinite(v) for v in ratings.values())


def test_anchor_pinned_at_zero_even_with_games():
    games = [_game("A", "ANCHOR", "1-0", i) for i in range(10)]
    fit = fit_ratings(games, anchor="ANCHOR")
    entry = fit.by_version()["ANCHOR"]
    assert entry.rating == 0.0
    assert "anchor" in entry.flags


def test_anchor_with_no_games_is_flagged():
    games = [_game("A", "B", "1-0", i) for i in range(10)]
    fit = fit_ratings(games, anchor="NEVER_PLAYED")
    entry = fit.by_version()["NEVER_PLAYED"]
    assert entry.rating == 0.0
    assert "no_games" in entry.flags


def test_undefeated_version_is_flagged_but_finite():
    # A never loses a single game against the anchor -> unregularized MLE
    # would be +infinity; the prior must keep it finite.
    games = [_game("A", "ANCHOR", "1-0", i) for i in range(15)]
    fit = fit_ratings(games, anchor="ANCHOR")
    entry = fit.by_version()["A"]
    assert "undefeated" in entry.flags
    assert math.isfinite(entry.rating)
    assert entry.rating > 0


def test_disconnected_component_does_not_crash_and_stays_ordered():
    # {v0, v1} connect to the anchor; {v2, v3} only play each other and
    # never touch the anchor's component at all.
    games = []
    for i in range(10):
        games.append(_game("v1", "v0", "1-0", i))
    for i in range(10, 20):
        games.append(_game("v2", "v3", "1-0", i))

    fit = fit_ratings(games, anchor="v0")
    ratings = {e.version: e.rating for e in fit.entries}

    assert all(math.isfinite(v) for v in ratings.values())
    assert ratings["v1"] > ratings["v0"] == 0
    assert ratings["v2"] > ratings["v3"]  # internally consistent even though disconnected


def test_all_draws_leaves_ratings_near_anchor():
    games = [_game("A", "ANCHOR", "1/2-1/2", i) for i in range(10)]
    fit = fit_ratings(games, anchor="ANCHOR")
    entry = fit.by_version()["A"]
    assert abs(entry.rating) < 1.0  # should sit right at parity
    assert "undefeated" not in entry.flags  # never lost, but also never won -> not flagged undefeated (0 losses AND draws counted separately)


def test_fit_vs_known_opponents_converges_when_far_from_zero():
    """Regression test: Newton's method starting from r=0 while opponents
    sit at real Elo levels (1300-1900) put the initial sigmoid evaluation
    deep in its flat tail (near-zero Hessian), which caused a divergent
    oscillation the first time this ran — a 3-1-0 record against a
    1500-rated opponent came back as a rating of -1410, not the ~1650-1700
    it should be. Fixed by initializing at the mean opponent Elo instead
    of 0, plus a step clamp as a second line of defense.
    """
    observations = [(1500.0, 1.0), (1500.0, 1.0), (1500.0, 1.0), (1500.0, 0.5)]
    rating, ci95 = fit_rating_vs_known_opponents(observations)
    assert math.isfinite(rating)
    assert 1500 < rating < 2200  # comfortably above the opponent, not a wild value
    assert math.isfinite(ci95)


def test_fit_vs_known_opponents_lands_near_crossover():
    # Strong at 1320, roughly even at 1500, weak at 1700 and 1900 ->
    # true strength should land close to the 1500 draw level.
    observations = (
        [(1320.0, 1.0)] * 4
        + [(1500.0, 0.5)] * 4
        + [(1700.0, 0.0)] * 4
        + [(1900.0, 0.0)] * 4
    )
    rating, _ci95 = fit_rating_vs_known_opponents(observations)
    assert 1350 < rating < 1650
