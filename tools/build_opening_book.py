"""Builds cb_book.bin: a Polyglot opening book shipped in the submission
zip, so the agent can play its first ~10-12 moves instantly with no
search at all -- which, per the numba compile-time investigation, is
what actually neutralizes the compile-time risk in practice: if the book
covers the opening, the (up to ~40s) deadline-based compile in
cb_nb_engine.py finishes in the background well before the engine is
ever asked to think for itself.

Source: the Lichess Elite Database (https://database.nikonoel.fr/), a
third-party monthly filtering of Lichess's public game dump to only
2500+-rated-vs-2300+-rated players' games (no further rating filter
needed here -- that's already the entire point of this source over the
raw Lichess monthly dumps, which are mostly low-rated bullet games).
Downloaded on demand into tools/books/ (gitignored, like the Stockfish
binary and ratings/pgn_book.py's calibration book -- a third-party asset,
not authored here).

python-chess can only *read* Polyglot books (chess.polyglot.open_reader),
not write them, so the binary format is written by hand here. It's a
simple, well-documented format: entries sorted ascending by
chess.polyglot.zobrist_hash(position-before-the-move), each 16 bytes
big-endian (key: uint64, move: uint16, weight: uint16, learn: uint32).
The move encoding is exactly (from_square << 6) | to_square, plus a
promotion nibble -- verified against chess.polyglot's own reader source
(MemoryMappedReader.__getitem__) rather than assumed from a spec:
castling can be encoded as a plain two-square king move (e1g1) because
Board._from_chess960 passes non-king-takes-rook moves through unchanged,
so there's no special-case encoding needed for it here.

Usage: python tools/build_opening_book.py [--min-count N] [--max-ply N]
"""
from __future__ import annotations

import argparse
import struct
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot

BOOK_MONTH = "2025-11"
DOWNLOAD_URL = f"https://database.nikonoel.fr/lichess_elite_{BOOK_MONTH}.zip"
BOOKS_DIR = Path(__file__).resolve().parent / "books"
ZIP_PATH = BOOKS_DIR / f"lichess_elite_{BOOK_MONTH}.zip"
PGN_PATH = BOOKS_DIR / f"lichess_elite_{BOOK_MONTH}.pgn"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "cb_book.bin"

ENTRY_STRUCT = struct.Struct(">QHHI")  # key, move, weight, learn -- matches chess.polyglot.ENTRY_STRUCT

_PROMOTION_TO_POLYGLOT = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}


def ensure_downloaded() -> Path:
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    if not PGN_PATH.exists():
        if not ZIP_PATH.exists():
            print(f"Downloading {DOWNLOAD_URL} ...", file=sys.stderr)
            urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_PATH)
        print(f"Extracting {ZIP_PATH} ...", file=sys.stderr)
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(BOOKS_DIR)
    if not PGN_PATH.exists():
        raise FileNotFoundError(f"expected {PGN_PATH} after extracting {ZIP_PATH}")
    return PGN_PATH


def encode_move(move: chess.Move) -> int:
    promo = _PROMOTION_TO_POLYGLOT[move.promotion]
    return (promo << 12) | (move.from_square << 6) | move.to_square


def collect_move_counts(pgn_path: Path, max_ply: int) -> Counter[tuple[int, int]]:
    """(zobrist_key, raw_move) -> number of games in which this move was
    played from this position, restricted to the first max_ply plies of
    each game."""
    counts: Counter[tuple[int, int]] = Counter()
    games_seen = 0
    with open(pgn_path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_seen += 1
            if games_seen % 20_000 == 0:
                print(f"  ...{games_seen} games processed, {len(counts)} distinct (position, move) pairs so far", file=sys.stderr)

            if game.headers.get("Variant", "Standard") != "Standard":
                continue

            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= max_ply:
                    break
                key = chess.polyglot.zobrist_hash(board)
                counts[(key, encode_move(move))] += 1
                board.push(move)

    print(f"Done: {games_seen} games, {len(counts)} distinct (position, move) pairs", file=sys.stderr)
    return counts


def write_book(counts: Counter[tuple[int, int]], min_count: int, output_path: Path) -> int:
    entries = [
        (key, raw_move, min(count, 0xFFFF))
        for (key, raw_move), count in counts.items()
        if count >= min_count
    ]
    entries.sort(key=lambda e: e[0])  # ascending by key -- required for the reader's binary search

    with open(output_path, "wb") as f:
        for key, raw_move, weight in entries:
            f.write(ENTRY_STRUCT.pack(key, raw_move, weight, 0))

    return len(entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-count", type=int, default=3,
                         help="drop (position, move) pairs played fewer than this many times (default: 3)")
    parser.add_argument("--max-ply", type=int, default=24,
                         help="only index moves within the first N plies of each game (default: 24 = 12 full moves)")
    args = parser.parse_args()

    pgn_path = ensure_downloaded()
    counts = collect_move_counts(pgn_path, args.max_ply)
    n_entries = write_book(counts, args.min_count, OUTPUT_PATH)

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUTPUT_PATH}: {n_entries} entries, {size_mb:.2f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
