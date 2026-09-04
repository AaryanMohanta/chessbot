"""Known perft positions and node counts: the six standard
chess-programming-wiki test positions (see
https://www.chessprogramming.org/Perft_Results), keyed by depth.

Depths <= 3 run in the default (fast) test suite. Depths 4-5 are real but
expensive in pure python-chess (millions of nodes for some of these), so
they're marked slow (see test_perft.py) and run on request with
``pytest -m slow`` rather than on every default run.
"""

PERFT_POSITIONS = [
    {
        "name": "position_1_startpos",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "nodes": {1: 20, 2: 400, 3: 8_902, 4: 197_281, 5: 4_865_609},
    },
    {
        "name": "position_2_kiwipete",
        "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "nodes": {1: 48, 2: 2_039, 3: 97_862, 4: 4_085_603},
    },
    {
        "name": "position_3",
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "nodes": {1: 14, 2: 191, 3: 2_812, 4: 43_238},
    },
    {
        "name": "position_4",
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "nodes": {1: 6, 2: 264, 3: 9_467, 4: 422_333},
    },
    {
        "name": "position_5",
        "fen": "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "nodes": {1: 44, 2: 1_486, 3: 62_379, 4: 2_103_487},
    },
    {
        "name": "position_6",
        "fen": "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "nodes": {1: 46, 2: 2_079, 3: 89_890, 4: 3_894_594},
    },
]

# Depths at or above this run only under -m slow (see test_perft.py).
SLOW_DEPTH_THRESHOLD = 4
