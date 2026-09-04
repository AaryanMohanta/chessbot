"""Design doc §3's non-negotiable: get_move must never raise and never
return an illegal move, even if the engine underneath is actively broken.
Verified by deliberately breaking it, not by reading the code and hoping.

Every test here disables the book (monkeypatch agent.cb_book to None):
agent.get_move tries the book before the engine, and the starting
position is very much in the real book -- without disabling it, these
tests would silently stop exercising the engine-guard path they exist to
test (the book would answer before the deliberately-broken stub engine
is ever reached) rather than failing loudly, which is worse.
"""
import chess
import pytest

import agent


@pytest.fixture(autouse=True)
def no_book(monkeypatch):
    monkeypatch.setattr(agent, "cb_book", None)


def test_falls_back_when_engine_raises(monkeypatch):
    class ExplodingEngine:
        def get_move(self, fen, time_left_ms):
            raise RuntimeError("deliberate failure for the fallback guard test")

    monkeypatch.setattr(agent, "_engine", ExplodingEngine())

    fen = chess.STARTING_FEN
    move = agent.get_move(fen, 5_000)

    board = chess.Board(fen)
    assert chess.Move.from_uci(move) in board.legal_moves


def test_falls_back_when_engine_returns_illegal_move(monkeypatch):
    class LyingEngine:
        def get_move(self, fen, time_left_ms):
            return "e2e5"  # well-formed UCI, illegal from the start position

    monkeypatch.setattr(agent, "_engine", LyingEngine())

    fen = chess.STARTING_FEN
    move = agent.get_move(fen, 5_000)

    board = chess.Board(fen)
    assert chess.Move.from_uci(move) in board.legal_moves


def test_falls_back_when_engine_returns_garbage(monkeypatch):
    class GarbageEngine:
        def get_move(self, fen, time_left_ms):
            return "not-a-move"

    monkeypatch.setattr(agent, "_engine", GarbageEngine())

    fen = chess.STARTING_FEN
    move = agent.get_move(fen, 5_000)

    board = chess.Board(fen)
    assert chess.Move.from_uci(move) in board.legal_moves


def test_returns_engines_move_when_it_is_legal(monkeypatch):
    class HonestEngine:
        def get_move(self, fen, time_left_ms):
            board = chess.Board(fen)
            return next(iter(board.legal_moves)).uci()

    monkeypatch.setattr(agent, "_engine", HonestEngine())

    fen = chess.STARTING_FEN
    board = chess.Board(fen)
    expected = next(iter(board.legal_moves)).uci()

    assert agent.get_move(fen, 5_000) == expected
