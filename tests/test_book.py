"""cb_book.py gate: builds a small synthetic Polyglot book (independent
of the real ~2-8MB cb_book.bin, which is generated offline from a
third-party download and isn't required for tests to run), and checks
weighted-random selection, legality re-verification, and graceful
degradation when the book is missing/corrupt or the position isn't in
it -- all of which get_move depends on to never let a book problem break
a real game.
"""
from __future__ import annotations

import random
import struct

import chess
import chess.polyglot
import pytest

import cb_book

ENTRY_STRUCT = struct.Struct(">QHHI")
STARTPOS = chess.STARTING_FEN


def _encode_move(move: chess.Move) -> int:
    promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]
    return (promo << 12) | (move.from_square << 6) | move.to_square


def _write_book(path, entries: list[tuple[chess.Board, chess.Move, int]]) -> None:
    """entries: (board-before-the-move, move, weight)."""
    packed = [
        (chess.polyglot.zobrist_hash(board), _encode_move(move), weight)
        for board, move, weight in entries
    ]
    packed.sort(key=lambda e: e[0])
    with open(path, "wb") as f:
        for key, raw_move, weight in packed:
            f.write(ENTRY_STRUCT.pack(key, raw_move, weight, 0))


@pytest.fixture
def two_move_book(tmp_path, monkeypatch):
    """A book with exactly two moves from the startpos: e2e4 (weight 90)
    and d2d4 (weight 10) -- skewed enough that weighted selection over
    many draws should overwhelmingly favor e2e4."""
    board = chess.Board(STARTPOS)
    path = tmp_path / "test_book.bin"
    _write_book(path, [
        (board, chess.Move.from_uci("e2e4"), 90),
        (board, chess.Move.from_uci("d2d4"), 10),
    ])
    monkeypatch.setattr(cb_book, "_BOOK_PATH", str(path))
    monkeypatch.setattr(cb_book, "_reader", None)
    monkeypatch.setattr(cb_book, "_reader_unavailable", False)
    yield path
    if cb_book._reader is not None:
        cb_book._reader.close()
        monkeypatch.setattr(cb_book, "_reader", None)


def test_probe_returns_a_legal_book_move(two_move_book):
    move = cb_book.probe(STARTPOS, rng=random.Random(0))
    assert move in ("e2e4", "d2d4")


def test_probe_never_returns_an_illegal_move(two_move_book):
    for seed in range(50):
        move = cb_book.probe(STARTPOS, rng=random.Random(seed))
        if move is not None:
            board = chess.Board(STARTPOS)
            assert chess.Move.from_uci(move) in board.legal_moves


def test_weighted_selection_favors_the_heavier_entry(two_move_book):
    rng = random.Random(42)
    counts = {"e2e4": 0, "d2d4": 0}
    for _ in range(2000):
        move = cb_book.probe(STARTPOS, rng=rng)
        counts[move] += 1
    # 90:10 weighting -- allow generous slack, this only needs to prove
    # "skewed", not pin down an exact ratio.
    assert counts["e2e4"] > counts["d2d4"] * 3


def test_selection_is_not_fully_deterministic(two_move_book):
    seen = {cb_book.probe(STARTPOS, rng=random.Random(seed)) for seed in range(30)}
    assert len(seen) > 1, "weighted-random selection should surface both book moves across enough draws"


def test_returns_none_for_a_position_not_in_the_book(two_move_book):
    board = chess.Board(STARTPOS)
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert cb_book.probe(board.fen()) is None


def test_returns_none_when_book_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(cb_book, "_BOOK_PATH", str(tmp_path / "does_not_exist.bin"))
    monkeypatch.setattr(cb_book, "_reader", None)
    monkeypatch.setattr(cb_book, "_reader_unavailable", False)
    assert cb_book.probe(STARTPOS) is None


def test_returns_none_when_book_file_is_corrupt(monkeypatch, tmp_path):
    bad_path = tmp_path / "corrupt.bin"
    bad_path.write_bytes(b"not a real polyglot book, just garbage bytes" * 3)
    monkeypatch.setattr(cb_book, "_BOOK_PATH", str(bad_path))
    monkeypatch.setattr(cb_book, "_reader", None)
    monkeypatch.setattr(cb_book, "_reader_unavailable", False)
    # Garbage-but-right-size-ish data shouldn't crash probe() even if it
    # produces nonsense entries -- corrupt data degrades to "no book move"
    # or a still-independently-legality-checked move, never an exception.
    result = cb_book.probe(STARTPOS)
    if result is not None:
        assert chess.Move.from_uci(result) in chess.Board(STARTPOS).legal_moves
