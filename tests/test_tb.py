"""cb_tb.py gate: probes the real 35-file 3-4 man WDL set shipped under
syzygy/ (small enough -- ~1.3 MB -- that using the real files directly is
simpler and more meaningful than faking a synthetic set), and checks
piece-count gating, legality re-verification, move-selection quality
(prefers an actual win over a drawn/losing alternative), and graceful
degradation when the tablebase directory is missing/empty -- all of
which agent.py depends on to never let a tablebase problem break a real
game.
"""
from __future__ import annotations

import chess
import pytest

import cb_tb


@pytest.fixture(autouse=True)
def _reset_tablebase_cache(monkeypatch):
    """Every test gets a fresh, unmemoized _get_tablebase() -- probe()
    lazily caches the opened Tablebase at module scope on first use, which
    would otherwise leak state (and an open file-descriptor pool) between
    tests that intentionally point _CANDIDATE_DIRS elsewhere."""
    monkeypatch.setattr(cb_tb, "_tablebase", None)
    monkeypatch.setattr(cb_tb, "_tablebase_unavailable", False)
    yield
    if cb_tb._tablebase is not None:
        cb_tb._tablebase.close()
        monkeypatch.setattr(cb_tb, "_tablebase", None)


def test_probe_returns_a_legal_winning_move_for_kqvk():
    move = cb_tb.probe("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1")
    assert move is not None
    board = chess.Board("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1")
    assert chess.Move.from_uci(move) in board.legal_moves


def test_probe_avoids_a_drawing_trap_when_a_winning_move_exists():
    """White to move, up a queen: g6-h6 is a known KQvK trap that throws
    the win away (verified directly against this same tablebase set --
    every other legal king/queen move keeps WDL=2, only g6h6 drops to 0).
    A WDL-only best-child search should never pick it while a winning
    alternative is on the board."""
    move = cb_tb.probe("7k/8/6K1/8/8/8/8/6Q1 w - - 0 1")
    assert move is not None
    assert move != "g6h6"


def test_probe_returns_none_when_more_than_four_pieces():
    assert cb_tb.probe(chess.STARTING_FEN) is None


def test_probe_returns_a_legal_move_for_a_drawn_material_combo():
    # KBvK: a lone minor can't force mate -- every legal move is a draw,
    # but probe() should still return a real (legal) move, not bail out
    # just because nothing is decisive.
    move = cb_tb.probe("4k3/8/8/3B4/8/8/8/4K3 w - - 0 1")
    assert move is not None
    board = chess.Board("4k3/8/8/3B4/8/8/8/4K3 w - - 0 1")
    assert chess.Move.from_uci(move) in board.legal_moves


def test_probe_never_returns_an_illegal_move_across_several_positions():
    fens = [
        "4k3/8/8/3Q4/8/8/8/4K3 b - - 0 1",
        "4k3/8/8/3R4/8/8/8/4K3 w - - 0 1",
        "8/8/3k4/8/3PK3/8/8/8 w - - 0 1",  # KPvK
        "8/8/8/4k3/8/4N3/4K3/8 w - - 0 1",  # KNvK, always drawn
    ]
    for fen in fens:
        move = cb_tb.probe(fen)
        if move is None:
            continue
        board = chess.Board(fen)
        assert chess.Move.from_uci(move) in board.legal_moves, f"{move} illegal in {fen}"


def test_probe_returns_none_when_tablebase_directory_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cb_tb, "_CANDIDATE_DIRS", [str(tmp_path / "does_not_exist")])
    assert cb_tb.probe("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1") is None


def test_probe_returns_none_when_tablebase_directory_is_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(cb_tb, "_CANDIDATE_DIRS", [str(tmp_path)])
    assert cb_tb.probe("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1") is None


def test_probe_falls_back_to_the_second_candidate_directory(monkeypatch, tmp_path):
    """Dev repo layout is syzygy/<file>, but the flat submission zip ships
    every file (including the tablebase set) at the zip root -- probe()
    must find the real files via the second candidate when the first
    (syzygy/-style) directory is empty."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    real_dir = cb_tb._MODULE_DIR + "/syzygy"
    monkeypatch.setattr(cb_tb, "_CANDIDATE_DIRS", [str(empty_dir), real_dir])
    move = cb_tb.probe("4k3/8/8/3Q4/8/8/8/4K3 w - - 0 1")
    assert move is not None


def test_probe_returns_none_for_a_position_with_castling_rights():
    # Impossible at <=4 pieces in a real game, but defend the invariant
    # directly rather than relying on it never coming up: Syzygy tables
    # never contain positions with castling rights.
    board = chess.Board("4k2r/8/8/8/8/8/8/4K3 w k - 0 1")  # 3 pieces, rights on
    assert cb_tb.probe(board.fen()) is None


def test_probe_returns_none_for_an_invalid_fen():
    assert cb_tb.probe("not a fen") is None
