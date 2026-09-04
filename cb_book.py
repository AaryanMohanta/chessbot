"""Polyglot opening-book probe. Ships at the zip root alongside agent.py
and cb_book.bin (built offline by tools/build_opening_book.py from the
Lichess Elite Database -- see that script's docstring for the source and
binary format).

Book moves are instant and need no search -- if the first ~10-12 moves
come from the book, cb_nb_engine's deadline-based numba compile finishes
in the background well before the engine is ever asked to think for
itself, which is what actually neutralizes the compile-time risk in
practice (see cb_nb_engine.py).

Loading is lazy and memory-mapped (chess.polyglot.open_reader mmaps the
file rather than reading it into memory) so importing this module, or
constructing the engine, never touches the book file or spends any of
the init budget on it -- the first real cost happens on the first probe()
call, once we're already inside a per-move clock.
"""
from __future__ import annotations

import os
import random

import chess
import chess.polyglot

_BOOK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cb_book.bin")

_reader = None
_reader_unavailable = False


def _get_reader():
    global _reader, _reader_unavailable
    if _reader is None and not _reader_unavailable:
        try:
            _reader = chess.polyglot.open_reader(_BOOK_PATH)
        except Exception:
            _reader_unavailable = True
    return _reader


def probe(fen: str, rng: random.Random | None = None) -> str | None:
    """Weighted-random UCI move from the book for this position (not
    always-best, so play isn't fully deterministic), or None if the
    position isn't in the book, the book file is missing/corrupt, or
    anything else goes wrong.

    chess.polyglot's own find_all/weighted_choice already filter entries
    to moves that are legal in the given position -- but a book move is
    exactly the kind of thing that came from an external, hand-built
    binary blob rather than our own verified search, so it gets the same
    "never trust it blindly, reverify independently" treatment agent.py
    already applies to the search engine's own output before playing it.
    """
    reader = _get_reader()
    if reader is None:
        return None

    try:
        board = chess.Board(fen)
        entry = reader.weighted_choice(board, random=rng or random)
        move = entry.move
    except (IndexError, ValueError):
        return None  # IndexError: no book entries for this position
    except Exception:
        return None  # corrupt book, mmap failure, etc. -- never let a book problem break get_move

    if move not in board.legal_moves:
        return None
    return move.uci()
