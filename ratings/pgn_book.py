"""Loads opening positions from a real PGN opening book (default:
official-stockfish/books' 8moves_v3.pgn — 34.7k distinct 8-move lines) for
calibration/SPRT gauntlets, instead of the small 51-line hand-picked book
in ratings/book.py. More distinct openings means a calibration result is
less likely to be an artifact of a handful of specific lines our engine
happens to play well or badly.

Downloaded on demand into tools/books/ (gitignored, like the Stockfish
binary — it's a third-party asset, not authored here) rather than checked
into the repo.
"""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

import chess.pgn

DEFAULT_URL = "https://raw.githubusercontent.com/official-stockfish/books/master/8moves_v3.pgn.zip"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "tools" / "books" / "8moves_v3.pgn"


def ensure_book(path: Path = DEFAULT_PATH, url: str = DEFAULT_URL) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = path.with_suffix(".pgn.zip")
    print(f"Downloading opening book from {url} ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path.parent)
    if not path.exists():
        raise FileNotFoundError(f"expected {path} after extracting {zip_path}")
    return path


def load_openings(path: Path = DEFAULT_PATH, limit: int = 1000) -> list[tuple[str, str]]:
    """Up to ``limit`` (opening_id, fen) pairs, fen being the position
    after each game's recorded moves (8 full moves for 8moves_v3.pgn)."""
    ensure_book(path)
    openings = []
    with open(path, encoding="utf-8", errors="replace") as f:
        while len(openings) < limit:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.end().board()
            eco = game.headers.get("Eco", "?")
            opening_id = f"pgn_{len(openings):05d}_{eco}"
            openings.append((opening_id, board.fen()))
    return openings
