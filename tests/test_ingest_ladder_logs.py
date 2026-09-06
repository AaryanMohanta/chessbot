"""ratings/ingest_ladder_logs.py: pairing log/PGN files by stem, colour
determination (log ply-parity first, --our-name PGN-header fallback
second), score-for-us computation, and dedup-on-rerun -- all of which
ratings/ladder_report.py's numbers depend on being right, since a wrong
colour flips every win/loss in that group.
"""
from __future__ import annotations

import sys

from ratings.ingest_ladder_logs import (
    _collect_files,
    _group_by_stem,
    _our_color_from_log,
    _our_color_from_pgn_headers,
    _score_for_us,
    ingest_one,
    main,
)
from ratings.ladder_db import load_games

_LOG_WE_ARE_WHITE = """\
[agent] build_tag=abc1234
mv ply=0 layer=book d=- sc=- n=- t=0.00s left=120.0s
mv ply=2 layer=numba d=9 sc=+20 n=294146 t=1.02s left=118.5s
mv ply=4 layer=numba d=8 sc=+35 n=100000 t=0.80s left=117.0s
"""

_LOG_WE_ARE_BLACK = """\
[agent] build_tag=xyz5678
mv ply=1 layer=numba d=10 sc=-15 n=200000 t=1.10s left=119.0s
"""

_PGN_WE_WIN_AS_WHITE = """\
[Event "Ladder Round 19"]
[White "chessbot"]
[Black "house-bot-2400"]
[Result "1-0"]
[WhiteElo "?"]
[BlackElo "2400"]
[Termination "checkmate"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 1-0
"""

_PGN_WE_LOSE_AS_BLACK = """\
[Event "Ladder Round 20"]
[White "house-bot-2600"]
[Black "chessbot"]
[Result "1-0"]
[WhiteElo "2600"]
[BlackElo "?"]

1. e4 e5 2. Nf3 1-0
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_our_color_from_log_white_when_first_ply_even():
    assert _our_color_from_log(_LOG_WE_ARE_WHITE) == "white"


def test_our_color_from_log_black_when_first_ply_odd():
    assert _our_color_from_log(_LOG_WE_ARE_BLACK) == "black"


def test_our_color_from_log_none_when_no_moves():
    assert _our_color_from_log("nothing useful here") is None


def test_our_color_from_pgn_headers_matches_case_insensitively():
    headers = {"White": "ChessBot-v37", "Black": "house-bot-2400"}
    assert _our_color_from_pgn_headers(headers, "chessbot") == "white"


def test_our_color_from_pgn_headers_none_without_our_name():
    headers = {"White": "chessbot", "Black": "house-bot-2400"}
    assert _our_color_from_pgn_headers(headers, None) is None


def test_score_for_us():
    assert _score_for_us("1-0", "white") == 1.0
    assert _score_for_us("1-0", "black") == 0.0
    assert _score_for_us("0-1", "black") == 1.0
    assert _score_for_us("1/2-1/2", "white") == 0.5
    assert _score_for_us("*", "white") is None


def test_collect_files_expands_directories(tmp_path):
    _write(tmp_path, "a.log", "x")
    _write(tmp_path, "a.pgn", "x")
    _write(tmp_path, "b.txt", "x")
    files = _collect_files([str(tmp_path)])
    assert {f.name for f in files} == {"a.log", "a.pgn", "b.txt"}


def test_group_by_stem_pairs_matching_names(tmp_path):
    log = _write(tmp_path, "round19.log", "x")
    pgn = _write(tmp_path, "round19.pgn", "x")
    groups = _group_by_stem([log, pgn])
    assert groups == {"round19": {"log": log, "pgn": pgn}}


def test_ingest_one_full_pair_white_win(tmp_path):
    log = _write(tmp_path, "round19.log", _LOG_WE_ARE_WHITE)
    pgn = _write(tmp_path, "round19.pgn", _PGN_WE_WIN_AS_WHITE)
    record = ingest_one("round19", {"log": log, "pgn": pgn}, our_name=None)
    assert record is not None
    assert record.build_tag == "abc1234"
    assert record.our_color == "white"
    assert record.score_for_us == 1.0
    assert record.opponent == "house-bot-2400"
    assert record.opponent_elo == "2400"
    assert record.termination == "checkmate"
    assert record.median_depth == 8.5  # searched depths 9, 8 -> median 8.5
    assert record.v1_ever_played is False
    assert record.source_log == "round19.log"
    assert record.source_pgn == "round19.pgn"


def test_ingest_one_full_pair_black_loss(tmp_path):
    log = _write(tmp_path, "round20.log", _LOG_WE_ARE_BLACK)
    pgn = _write(tmp_path, "round20.pgn", _PGN_WE_LOSE_AS_BLACK)
    record = ingest_one("round20", {"log": log, "pgn": pgn}, our_name=None)
    assert record is not None
    assert record.our_color == "black"
    assert record.score_for_us == 0.0
    assert record.opponent == "house-bot-2600"
    assert record.opponent_elo == "2600"


def test_ingest_one_pgn_only_uses_our_name_fallback(tmp_path):
    pgn = _write(tmp_path, "round21.pgn", _PGN_WE_WIN_AS_WHITE)
    record = ingest_one("round21", {"pgn": pgn}, our_name="chessbot")
    assert record is not None
    assert record.our_color == "white"
    assert record.score_for_us == 1.0
    assert record.build_tag == "unknown"  # no log -> no build_tag line


def test_ingest_one_returns_none_when_color_unresolvable(tmp_path):
    pgn = _write(tmp_path, "round22.pgn", _PGN_WE_WIN_AS_WHITE)
    record = ingest_one("round22", {"pgn": pgn}, our_name=None)
    assert record is None


def test_main_ingests_and_skips_on_rerun(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "round19.log", _LOG_WE_ARE_WHITE)
    _write(tmp_path, "round19.pgn", _PGN_WE_WIN_AS_WHITE)
    db_path = tmp_path / "ladder_games.csv"

    import ratings.ingest_ladder_logs as mod
    monkeypatch.setattr(mod, "DB_PATH", db_path)

    monkeypatch.setattr(sys, "argv", ["ingest_ladder_logs.py", str(tmp_path)])
    main()
    games = load_games(db_path=db_path)
    assert len(games) == 1

    # Re-running over the same directory must not double-ingest.
    main()
    games_after_rerun = load_games(db_path=db_path)
    assert len(games_after_rerun) == 1
