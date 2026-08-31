"""Known perft positions and node counts (standard chess-programming-wiki
test set), keyed by depth. Kept shallow (depth <= 3) so the suite stays
fast; python-chess's own move generator is what's actually under test here
since this repo doesn't implement custom movegen — it's a sanity check on
the environment/library, and a ready-made scaffold if that ever changes.
"""

PERFT_POSITIONS = [
    {
        "name": "startpos",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "nodes": {1: 20, 2: 400, 3: 8902},
    },
    {
        "name": "kiwipete",
        "fen": "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "nodes": {1: 48, 2: 2039, 3: 97862},
    },
    {
        "name": "position_3",
        "fen": "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "nodes": {1: 14, 2: 191, 3: 2812},
    },
    {
        "name": "position_4",
        "fen": "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "nodes": {1: 6, 2: 264, 3: 9467},
    },
]
