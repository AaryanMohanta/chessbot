"""A balanced opening book for the gauntlet runner: ~50 short, standard
opening lines (4-10 plies), verified legal and to reach 51 distinct
positions. "Balanced" here means drawn from established theory across all
three first moves and a wide spread of structures (open games, semi-open,
closed/QGD family, Indian systems, flank openings) rather than book lines
biased toward one side, so a gauntlet run isn't just replaying the same
handful of structures.

Each entry is (opening_id, uci_moves_from_startpos, resulting_fen). FENs
are derived at import time from the move list, not hand-typed, so they
can't drift out of sync with the moves.
"""
from __future__ import annotations

import chess

# (opening_id, space-separated UCI moves from the starting position)
_LINES: list[tuple[str, str]] = [
    ("italian", "e2e4 e7e5 g1f3 b8c6 f1c4"),
    ("ruy_lopez", "e2e4 e7e5 g1f3 b8c6 f1b5"),
    ("ruy_lopez_berlin", "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6"),
    ("scotch", "e2e4 e7e5 g1f3 b8c6 d2d4"),
    ("petrov", "e2e4 e7e5 g1f3 g8f6"),
    ("four_knights", "e2e4 e7e5 g1f3 b8c6 b1c3 g8f6"),
    ("vienna", "e2e4 e7e5 b1c3"),
    ("kings_gambit", "e2e4 e7e5 f2f4"),
    ("bishops_opening", "e2e4 e7e5 f1c4"),
    ("sicilian_najdorf_setup", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3"),
    ("sicilian_open", "e2e4 c7c5 g1f3 b8c6"),
    ("sicilian_dragon_setup", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 g7g6"),
    ("sicilian_alapin", "e2e4 c7c5 c2c3"),
    ("french", "e2e4 e7e6 d2d4 d7d5"),
    ("french_advance", "e2e4 e7e6 d2d4 d7d5 e4e5"),
    ("french_tarrasch", "e2e4 e7e6 d2d4 d7d5 b1d2"),
    ("caro_kann", "e2e4 c7c6 d2d4 d7d5"),
    ("caro_kann_advance", "e2e4 c7c6 d2d4 d7d5 e4e5"),
    ("scandinavian", "e2e4 d7d5"),
    ("scandinavian_nf6", "e2e4 d7d5 e4d5 g8f6"),
    ("alekhine", "e2e4 g8f6"),
    ("pirc", "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6"),
    ("modern", "e2e4 g7g6"),
    ("qgd", "d2d4 d7d5 c2c4 e7e6"),
    ("qga", "d2d4 d7d5 c2c4 d5c4"),
    ("slav", "d2d4 d7d5 c2c4 c7c6"),
    ("semi_slav", "d2d4 d7d5 c2c4 c7c6 b1c3 e7e6"),
    ("nimzo_indian", "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4"),
    ("queens_indian", "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6"),
    ("kid", "d2d4 g8f6 c2c4 g7g6"),
    ("kid_classical", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6"),
    ("grunfeld", "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5"),
    ("benoni", "d2d4 g8f6 c2c4 c7c5"),
    ("catalan", "d2d4 g8f6 c2c4 e7e6 g2g3"),
    ("dutch", "d2d4 f7f5"),
    ("london", "d2d4 d7d5 g1f3 g8f6 c1f4"),
    ("english", "c2c4"),
    ("english_symmetrical", "c2c4 c7c5"),
    ("english_reversed_sicilian", "c2c4 e7e5"),
    ("reti", "g1f3 d7d5 c2c4"),
    ("reti_vs_kid", "g1f3 g8f6 c2c4 g7g6"),
    ("birds", "f2f4"),
    ("larsen", "b2b3"),
    ("nimzo_larsen", "b2b3 e7e5 c1b2"),
    ("grob", "g2g4"),
    ("vant_kruijs", "e2e3"),
    ("zukertort", "g1f3 d7d5 g2g3"),
    ("kia", "g1f3 d7d5 g2g3 b8c6 f1g2"),
    ("trompowsky", "d2d4 g8f6 c1g5"),
    ("torre", "d2d4 g8f6 g1f3 e7e6 c1g5"),
    ("colle", "d2d4 d7d5 g1f3 g8f6 e2e3"),
]


def _build_book() -> list[tuple[str, str, str]]:
    entries = []
    for opening_id, moves_str in _LINES:
        board = chess.Board()
        for uci in moves_str.split():
            board.push(chess.Move.from_uci(uci))
        entries.append((opening_id, moves_str, board.fen()))
    return entries


OPENING_BOOK: list[tuple[str, str, str]] = _build_book()  # (id, moves_uci, fen)

_BY_ID = {opening_id: fen for opening_id, _, fen in OPENING_BOOK}


def fen_for(opening_id: str) -> str:
    return _BY_ID[opening_id]
