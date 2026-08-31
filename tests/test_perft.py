"""Perft harness: verifies legal move generation node counts against known
values for standard test positions."""
import chess
import pytest

from tests.positions import PERFT_POSITIONS


def perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += perft(board, depth - 1)
        board.pop()
    return nodes


_CASES = [
    (pos["name"], pos["fen"], depth, expected)
    for pos in PERFT_POSITIONS
    for depth, expected in pos["nodes"].items()
]


@pytest.mark.parametrize(
    "name,fen,depth,expected",
    _CASES,
    ids=[f"{name}-depth{depth}" for name, _, depth, _ in _CASES],
)
def test_perft(name, fen, depth, expected):
    board = chess.Board(fen)
    assert perft(board, depth) == expected
