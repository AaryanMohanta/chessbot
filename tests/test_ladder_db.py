"""ratings/ladder_db.py: CSV round-trip and dedup-key helpers, same
"append-only, load it all back" contract as ratings/db.py's GameRecord,
but exercised against ladder_db's own schema (None median_depth, bool
v1_ever_played, and the (source_log, source_pgn) dedup key specifically).
"""
from __future__ import annotations

from ratings.ladder_db import LadderGameRecord, already_ingested_sources, append_game, load_games


def _record(**overrides) -> LadderGameRecord:
    defaults = dict(
        timestamp="2026-09-06T12:00:00+00:00",
        build_tag="abc1234",
        our_color="white",
        opponent="house-bot-2400",
        opponent_elo="2400",
        score_for_us=1.0,
        termination="checkmate",
        plies=47,
        median_depth=9.5,
        total_clock_spend_s=88.2,
        v1_ever_played=False,
        depth_collapse_count=0,
        moves_uci="e2e4 e7e5",
        source_log="round19.log",
        source_pgn="round19.pgn",
    )
    defaults.update(overrides)
    return LadderGameRecord(**defaults)


def test_append_and_load_round_trips_all_fields(tmp_path):
    db_path = tmp_path / "ladder_games.csv"
    record = _record()
    append_game(record, db_path=db_path)
    loaded = load_games(db_path=db_path)
    assert loaded == [record]


def test_median_depth_none_round_trips_as_none(tmp_path):
    db_path = tmp_path / "ladder_games.csv"
    record = _record(median_depth=None)
    append_game(record, db_path=db_path)
    loaded = load_games(db_path=db_path)
    assert loaded[0].median_depth is None


def test_v1_ever_played_bool_round_trips(tmp_path):
    db_path = tmp_path / "ladder_games.csv"
    append_game(_record(v1_ever_played=True), db_path=db_path)
    append_game(_record(v1_ever_played=False, source_log="round20.log", source_pgn="round20.pgn"), db_path=db_path)
    loaded = load_games(db_path=db_path)
    assert loaded[0].v1_ever_played is True
    assert loaded[1].v1_ever_played is False


def test_load_games_returns_empty_list_for_missing_db(tmp_path):
    assert load_games(db_path=tmp_path / "does_not_exist.csv") == []


def test_already_ingested_sources_tracks_log_pgn_pairs(tmp_path):
    db_path = tmp_path / "ladder_games.csv"
    append_game(_record(source_log="round19.log", source_pgn="round19.pgn"), db_path=db_path)
    append_game(_record(source_log="round20.log", source_pgn=""), db_path=db_path)
    seen = already_ingested_sources(db_path=db_path)
    assert ("round19.log", "round19.pgn") in seen
    assert ("round20.log", "") in seen
    assert ("round21.log", "round21.pgn") not in seen


def test_multiple_appends_accumulate(tmp_path):
    db_path = tmp_path / "ladder_games.csv"
    for i in range(5):
        append_game(_record(source_log=f"round{i}.log", source_pgn=f"round{i}.pgn"), db_path=db_path)
    assert len(load_games(db_path=db_path)) == 5
