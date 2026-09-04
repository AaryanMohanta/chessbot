"""Tests for the append-only CSV game database."""
from ratings.db import GameRecord, append_game, load_games, played_pairings, result_string


def test_append_and_load_round_trip(tmp_path):
    db_path = tmp_path / "games.csv"
    record = GameRecord(
        timestamp="2026-01-01T00:00:00",
        white_version="v1",
        black_version="v2",
        opening_id="italian",
        result="1-0",
        termination="checkmate",
        plies=41,
        moves_uci="e2e4 e7e5",
    )
    append_game(record, db_path=db_path)

    loaded = load_games(db_path)
    assert len(loaded) == 1
    assert loaded[0] == record


def test_never_overwrites_across_multiple_appends(tmp_path):
    db_path = tmp_path / "games.csv"
    for i in range(5):
        append_game(
            GameRecord(
                timestamp=f"2026-01-01T00:00:0{i}",
                white_version="v1",
                black_version="v2",
                opening_id="italian",
                result="1-0",
                termination="checkmate",
                plies=40 + i,
            ),
            db_path=db_path,
        )
    assert len(load_games(db_path)) == 5


def test_white_score_property():
    win = GameRecord("t", "a", "b", "o", "1-0", "checkmate", 10)
    draw = GameRecord("t", "a", "b", "o", "1/2-1/2", "stalemate", 10)
    loss = GameRecord("t", "a", "b", "o", "0-1", "checkmate", 10)
    assert win.white_score == 1.0
    assert draw.white_score == 0.5
    assert loss.white_score == 0.0


def test_result_string():
    assert result_string("white") == "1-0"
    assert result_string("black") == "0-1"
    assert result_string(None) == "1/2-1/2"


def test_played_pairings_used_for_resumability(tmp_path):
    db_path = tmp_path / "games.csv"
    append_game(
        GameRecord("t", "v1", "v2", "italian", "1-0", "checkmate", 30),
        db_path=db_path,
    )
    pairings = played_pairings(db_path)
    assert ("v1", "v2", "italian") in pairings
    assert ("v2", "v1", "italian") not in pairings
