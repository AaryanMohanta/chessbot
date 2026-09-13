"""Syzygy 3-4 man WDL tablebase probe (2026-09, the 600-ply-draw rule
adaptation's item 3c -- see the ratings/parse_match_log.py and cb_time.py
2026-09 comments for the other two parts of that same change).

Ships at the zip root alongside agent.py and the 35 .rtbw files under
syzygy/ (~1.3 MB total, well under the 3 MB budgeted in the design doc --
see tools/build_zip.py's SHIPPED_TABLEBASE_FILES). WDL only, no DTZ: a
5-6-7 man set would blow the 50 MB cap by itself, and even full 3-4-5 man
WDL+DTZ is unnecessary when the only thing being fixed is "don't misplay
a trivial 3-4 man endgame" -- something a WDL-only classification already
solves exactly, just without knowledge of the fastest mate.

Motivation: before this, a KRvK or similar endgame was scored by the
hand-tuned eval + a shallow search, both of which can misjudge wrong-
color-bishop/stalemate-trick corners even when materially "obviously"
winning. Under the old 300-ply material-adjudication rule a bad technique
mistake in a winning endgame usually still won on material before the
cutoff; under the new 600-ply flat-draw rule (see cb_time.py) it can
instead run out the clock as a real draw -- so guaranteeing correct
technique in the 3-4 man cases this table set actually covers is worth
more now than it was.

Loading is lazy (first probe() call, not import or Engine construction)
and each table is only mapped in on its first actual use, same reasoning
as cb_book.py's lazy Polyglot reader: never spend any of the init budget
on something that might not come up for 50+ moves, if at all.
"""
from __future__ import annotations

import os

import chess
import chess.syzygy

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# Dev repo layout keeps the 35 files organized under syzygy/ rather than
# loose at the repo root; the flat submission zip (tools/build_zip.py
# writes every shipped file with arcname=path.name, no subfolders) ships
# them at the zip root instead, alongside every .py module and cb_book.bin
# -- so at runtime they're siblings of this file, not in a subdirectory.
# Try the dev layout first, then fall back to this file's own directory.
_CANDIDATE_DIRS = [os.path.join(_MODULE_DIR, "syzygy"), _MODULE_DIR]

# 3-4 man WDL only: a position with more pieces than this has no table in
# our set (see _CANDIDATE_DIRS above), so probe() bails out immediately
# rather than paying for a guaranteed MissingTableError.
_TB_MAX_PIECES = 4

_tablebase = None
_tablebase_unavailable = False


def _get_tablebase():
    global _tablebase, _tablebase_unavailable
    if _tablebase is None and not _tablebase_unavailable:
        for directory in _CANDIDATE_DIRS:
            try:
                tb = chess.syzygy.open_tablebase(directory, load_dtz=False)
            except Exception:
                continue
            if len(tb.wdl) > 0:
                _tablebase = tb
                break
        if _tablebase is None:
            _tablebase_unavailable = True
    return _tablebase


def probe(fen: str) -> str | None:
    """Best move by WDL classification (win beats cursed-win beats draw
    beats cursed-loss beats loss) when at most _TB_MAX_PIECES pieces
    remain on the board, or None if the tablebase can't answer -- too
    many pieces, tables missing/unavailable, castling rights still live
    (Syzygy tables never contain such positions -- impossible anyway at
    <=4 pieces, checked defensively), no legal moves, or anything else
    going wrong. Callers treat None exactly like a book miss and fall
    through to the real search.

    Deliberately probes each *resulting* position via probe_wdl rather
    than the root position: WDL alone doesn't rank same-classification
    moves by speed of conversion (that needs DTZ, which we don't ship),
    but it's exactly enough to never pick a move that throws away a win
    or lets a draw slip into a loss, which is the actual failure mode
    being fixed here.
    """
    tb = _get_tablebase()
    if tb is None:
        return None

    try:
        board = chess.Board(fen)
    except Exception:
        return None

    if chess.popcount(board.occupied) > _TB_MAX_PIECES:
        return None
    if board.castling_rights:
        return None

    best_move = None
    best_score = None
    for move in board.legal_moves:
        board.push(move)
        try:
            score = -tb.probe_wdl(board)
        except Exception:
            score = None
        finally:
            board.pop()

        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_move = move

    if best_move is None:
        return None
    return best_move.uci()
