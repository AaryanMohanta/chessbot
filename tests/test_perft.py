"""Perft harness: verifies legal move generation node counts against known
values for the six standard test positions.

Depths <= 3 run by default (fast, seconds total). Depths >= 4 are real
verification (millions of nodes for some positions, several minutes total
in pure python-chess) and are marked slow: run them explicitly with
``pytest -m slow tests/test_perft.py``.
"""
import chess
import pytest

from tests.positions import PERFT_POSITIONS, SLOW_DEPTH_THRESHOLD


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


_ALL_CASES = [
    (pos["name"], pos["fen"], depth, expected)
    for pos in PERFT_POSITIONS
    for depth, expected in pos["nodes"].items()
]
_FAST_CASES = [c for c in _ALL_CASES if c[2] < SLOW_DEPTH_THRESHOLD]
_SLOW_CASES = [c for c in _ALL_CASES if c[2] >= SLOW_DEPTH_THRESHOLD]


@pytest.mark.parametrize(
    "name,fen,depth,expected",
    _FAST_CASES,
    ids=[f"{name}-depth{depth}" for name, _, depth, _ in _FAST_CASES],
)
def test_perft_fast(name, fen, depth, expected):
    board = chess.Board(fen)
    assert perft(board, depth) == expected


@pytest.mark.slow
@pytest.mark.parametrize(
    "name,fen,depth,expected",
    _SLOW_CASES,
    ids=[f"{name}-depth{depth}" for name, _, depth, _ in _SLOW_CASES],
)
def test_perft_slow(name, fen, depth, expected):
    board = chess.Board(fen)
    assert perft(board, depth) == expected
