"""Perft gate for the numba bitboard movegen experiment (cb_nb_*.py).

This is THE gate the design doc calls out explicitly: if this doesn't
match exactly, the numba path does not proceed further regardless of
speed. Marked slow (millions of nodes, tens of seconds to minutes each)
— run explicitly with `pytest -m slow tests/test_nb_perft.py`.

cb_nb_*.py is a parallel experiment, not wired into agent.py. v1
(cb_search.py on python-chess) stays the shipped path and the perft
oracle throughout — these modules are not in tools/build_zip.py's
whitelist and never will be until this gate and stages 3-5 all pass.
"""
import pytest

import cb_nb_fast as nb

# (name, fen, depth, expected_nodes) -- depth 5 for all standard positions
# except startpos, which the design doc calls out for depth 6.
POSITIONS = [
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 6, 119060324),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5, 193690690),
    ("position_3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624),
    ("position_4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 5, 15833292),
    ("position_5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 5, 89941194),
    ("position_6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 5, 164075551),
]


@pytest.mark.slow
@pytest.mark.parametrize("name,fen,depth,expected", POSITIONS, ids=[p[0] for p in POSITIONS])
def test_nb_perft_gate(name, fen, depth, expected):
    assert nb.run_perft(fen, depth) == expected


def test_nb_perft_fast_sanity():
    """Cheap, always-run sanity check (depth <= 3) so a build regression
    shows up in the default suite, not only when someone remembers to run
    the slow gate."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert nb.run_perft(fen, 1) == 20
    assert nb.run_perft(fen, 2) == 400
    assert nb.run_perft(fen, 3) == 8902
