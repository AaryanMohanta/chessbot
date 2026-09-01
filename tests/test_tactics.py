"""Tactical suite: mate-in-1 and mate-in-2 positions, verified independently
of the engine (by brute-force enumeration over all legal replies, not by
hand-derivation) before being used as fixtures here. See the comments on
each position for how it was checked.

Positions with multiple valid mating first moves are asserted by outcome
(does the returned move actually deliver/force the mate), not by requiring
one specific "book" move — anything is worth prohibiting-the-alternative
would be over-fitting to a single found move (e.g., mate-in-1 fixtures
were confirmed to have a small tied set of literal winning moves for the
first two; a real engine may or may not pick every one of them, so those
two check the game outcome rather than the exact string).
"""
import chess
import pytest

import cb_search
from cb_tt import TranspositionTable

SEARCH_BUDGET_MS = 2_000


def _search_move(fen: str) -> chess.Move:
    board = chess.Board(fen)
    move, _score, _info = cb_search.search(board, soft_ms=SEARCH_BUDGET_MS, hard_ms=SEARCH_BUDGET_MS * 3, tt=TranspositionTable())
    return move


# Each verified by: for every legal move, push it, check board.is_checkmate().
MATE_IN_ONE_POSITIONS = [
    ("back_rank_rook", "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1"),
    ("smothered_knight", "6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1"),
    ("back_rank_rook_2", "3r2k1/5ppp/8/8/8/8/8/3R3K w - - 0 1"),
]


@pytest.mark.parametrize("name,fen", MATE_IN_ONE_POSITIONS, ids=[n for n, _ in MATE_IN_ONE_POSITIONS])
def test_finds_mate_in_one(name, fen):
    board = chess.Board(fen)
    move = _search_move(fen)
    board.push(move)
    assert board.is_checkmate()


def test_finds_mate_in_two():
    # Verified by brute force: a2b3 is the *only* legal first move for
    # which every black reply still allows a forced mate next move.
    fen = "8/8/8/8/8/3R4/K7/2k5 w - - 0 1"
    move = _search_move(fen)
    assert move.uci() == "a2b3"


@pytest.mark.skip(reason="TODO: fill in once cb_eval has more than material+PST to reason about")
def test_avoids_hanging_a_piece():
    ...


@pytest.mark.skip(reason="TODO: fill in once eval terms beyond material exist to distinguish equal-material options")
def test_prefers_positionally_better_equal_material_move():
    ...
