"""ratings/ladder_report.py: per-build_tag aggregation and the
known-opponent-Elo performance-rating gate (MIN_GAMES_FOR_RATING) --
the numbers this session's "did LMR depth-scaling actually help on the
real ladder" comparison will be read off of.
"""
from __future__ import annotations

from ratings.ladder_db import LadderGameRecord
from ratings.ladder_report import MIN_GAMES_FOR_RATING, print_report, summarize_tag


def _record(**overrides) -> LadderGameRecord:
    defaults = dict(
        timestamp="2026-09-06T12:00:00+00:00",
        build_tag="tag-a",
        our_color="white",
        opponent="house-bot",
        opponent_elo="2400",
        score_for_us=1.0,
        termination="checkmate",
        plies=40,
        median_depth=9.0,
        total_clock_spend_s=80.0,
        v1_ever_played=False,
        depth_collapse_count=0,
        moves_uci="",
        source_log="",
        source_pgn="",
    )
    defaults.update(overrides)
    return LadderGameRecord(**defaults)


def test_summarize_tag_counts_wins_draws_losses():
    games = [
        _record(score_for_us=1.0),
        _record(score_for_us=0.5),
        _record(score_for_us=0.0),
        _record(score_for_us=0.0),
    ]
    s = summarize_tag("tag-a", games)
    assert s["games"] == 4
    assert s["wins"] == 1
    assert s["draws"] == 1
    assert s["losses"] == 2
    assert s["score_rate"] == 1.5 / 4


def test_summarize_tag_no_rating_below_minimum_games():
    games = [_record(opponent_elo="2400") for _ in range(MIN_GAMES_FOR_RATING - 1)]
    s = summarize_tag("tag-a", games)
    assert s["rating"] is None
    assert s["rated_games"] == MIN_GAMES_FOR_RATING - 1


def test_summarize_tag_computes_rating_at_minimum_games():
    games = [_record(opponent_elo="2400", score_for_us=1.0) for _ in range(MIN_GAMES_FOR_RATING)]
    s = summarize_tag("tag-a", games)
    assert s["rating"] is not None
    # All wins against a fixed 2400 opponent -> our estimated rating should
    # land comfortably above 2400, not at or below it.
    assert s["rating"] > 2400


def test_summarize_tag_ignores_games_with_no_opponent_elo():
    games = [_record(opponent_elo="") for _ in range(MIN_GAMES_FOR_RATING + 5)]
    s = summarize_tag("tag-a", games)
    assert s["rated_games"] == 0
    assert s["rating"] is None


def test_summarize_tag_avg_median_depth_ignores_missing_values():
    games = [_record(median_depth=8.0), _record(median_depth=10.0), _record(median_depth=None)]
    s = summarize_tag("tag-a", games)
    assert s["avg_median_depth"] == 9.0


def test_summarize_tag_handles_empty_group():
    s = summarize_tag("tag-a", [])
    assert s["games"] == 0
    assert s["score_rate"] == 0.0
    assert s["avg_median_depth"] is None


def test_print_report_handles_missing_db_without_crashing(tmp_path, capsys):
    print_report(db_path=tmp_path / "does_not_exist.csv")
    out = capsys.readouterr().out
    assert "No ladder games recorded" in out


def test_print_report_smoke_with_two_tags(tmp_path, capsys):
    from ratings.ladder_db import append_game

    db_path = tmp_path / "ladder_games.csv"
    for i in range(MIN_GAMES_FOR_RATING):
        append_game(_record(build_tag="lmr-on", opponent_elo="2400", score_for_us=1.0,
                             source_log=f"a{i}.log"), db_path=db_path)
        append_game(_record(build_tag="lmr-off", opponent_elo="2400", score_for_us=0.5,
                             source_log=f"b{i}.log"), db_path=db_path)

    print_report(db_path=db_path)
    out = capsys.readouterr().out
    assert "lmr-on" in out
    assert "lmr-off" in out
    assert "performance_rating=" in out
