"""Tactical test slots — empty until cb_search / cb_eval are implemented.

Intended shape once filled in: give a FEN with a known best move (mate-in-N,
a forced win of material, a must-not-blunder position) and assert
cb_engine.Engine().search(...) finds it within a small time budget.
"""
import pytest


@pytest.mark.skip(reason="TODO: fill in once cb_search finds moves better than random")
def test_finds_mate_in_one():
    ...


@pytest.mark.skip(reason="TODO: fill in once cb_search finds moves better than random")
def test_finds_mate_in_two():
    ...


@pytest.mark.skip(reason="TODO: fill in once cb_search finds moves better than random")
def test_avoids_hanging_a_piece():
    ...


@pytest.mark.skip(reason="TODO: fill in once cb_eval scores beyond material")
def test_prefers_material_winning_move():
    ...
