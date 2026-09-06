"""AI Chessathon submission entry point.

Ships at the ROOT of the submission zip alongside the ``cb_*.py`` modules
it imports and the ``cb_book.bin`` opening book. Exposes exactly one
public function:

    get_move(fen: str, time_left_ms: int) -> str

returning a UCI move string (e.g. "e2e4", "e7e8q").

Layered fallback, each layer independent of the ones "above" it and each
layer's answer independently re-verified legal before being trusted:

  1. cb_book: an instant, no-search Polyglot opening-book probe.
  2. cb_tb: an instant, no-search Syzygy 3-4 man WDL tablebase probe
     (2026-09) -- authoritative whenever it applies (<=4 pieces left), so
     it's tried before the search layers rather than after them.
  3. cb_nb_engine.Engine: the numba bitboard search. Its own __init__
     already blocks compiling (up to a wall-clock deadline) then falls
     back to serving from v1 in the background if that deadline is hit
     -- see cb_nb_engine.py -- so this layer alone already degrades
     gracefully under a slow or failed compile.
  4. cb_engine.Engine (v1, python-chess): constructed independently here
     as well, not just reached via layer 2's internal fallback -- so a
     bug in cb_nb_engine's own wrapper code (not the numba compile
     itself, which layer 2 already isolates) still can't take v1 down
     with it.
  5. _fallback_move: first legal move python-chess enumerates. Depends
     on nothing but ``chess``, so it's as close to "obviously correct" as
     a fallback can be without reimplementing chess rules from scratch.

A crash mid-game is an unrecoverable loss; a slightly worse move is not
-- so every layer above is guarded, and any failure just falls through to
the next one instead of crashing the process.
"""
from __future__ import annotations

import sys
import time
import traceback

_PROCESS_START = time.monotonic()  # first line, before any other import --
# cb_nb_engine's compile deadline is measured from here, not from when its
# own __init__ runs, so the (nontrivial) time spent importing numpy/numba/
# chess below still counts against that deadline.

import chess


def _log_layer_failure(layer_name: str) -> None:
    """Every fallback layer below silently swallows its own exception on
    purpose (a crash mid-game is an unrecoverable loss, a degraded layer
    is not) -- but "silent" used to mean genuinely invisible, with
    nothing printed anywhere. A real ladder loss (round 11) showed the
    engine playing the trivial first-legal-move fallback for an entire
    game with "nothing written to stderr" in the match log, and no way
    to tell whether that was v1, numba, or both failing, or why. This
    always writes to stderr (which every real match log captures, per
    the competition's own tooling) so a repeat is actually diagnosable
    instead of just visible-in-hindsight from behavior alone."""
    exc_type, exc_value, _ = sys.exc_info()
    print(f"[agent] {layer_name} failed to initialize: {exc_type.__name__}: {exc_value}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()


try:
    import cb_book
except Exception:
    _log_layer_failure("cb_book")
    cb_book = None

try:
    import cb_tb
except Exception:
    _log_layer_failure("cb_tb")
    cb_tb = None

_v1 = None
try:
    import cb_engine

    _v1 = cb_engine.Engine()
except Exception:
    _log_layer_failure("cb_engine (v1)")
    _v1 = None

_engine = None
try:
    import cb_nb_engine

    _engine = cb_nb_engine.Engine(process_start=_PROCESS_START)
except Exception:
    _log_layer_failure("cb_nb_engine (numba)")
    _engine = None


def _fallback_move(fen: str) -> str:
    board = chess.Board(fen)
    for move in board.legal_moves:
        return move.uci()
    raise RuntimeError(f"no legal moves available in position: {fen}")


def _is_legal(fen: str, move_uci: str) -> bool:
    """Re-verify a move against the FEN independently of whatever
    internal logic produced it -- book, search, doesn't matter. A hash
    collision, a bug in a seam not yet wired up, or any other defect that
    produces a well-formed but illegal UCI string must be caught here —
    an unverified move is an illegal-move loss."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
    except Exception:
        return False
    return move in board.legal_moves


_last_score = None  # centipawns, mover's perspective -- diagnostic only, see get_last_score()


def get_last_score() -> float | None:
    """Optional hook, not part of the competition's get_move contract:
    the internal test harness (harness/stdio_runner.py) calls this after
    get_move, purely for score-based resign/draw adjudication in dev
    SPRT -- see harness/match.py. None whenever the answering move came
    from the book or the trivial fallback (nothing was searched), or
    whenever nothing has been played yet."""
    return _last_score


_logged_move_failures: set[str] = set()  # log each layer's first per-move failure only, not every move


def _log_move_failure(layer_name: str) -> None:
    """Same reasoning as _log_layer_failure, for a layer that initialized
    fine but then raised on a specific get_move call. Logged once per
    layer per process (not once per move) -- a persistently-failing
    layer would otherwise flood stderr with the same traceback every
    single move for the rest of the game."""
    if layer_name in _logged_move_failures:
        return
    _logged_move_failures.add(layer_name)
    exc_type, exc_value, _ = sys.exc_info()
    print(f"[agent] {layer_name}.get_move failed: {exc_type.__name__}: {exc_value}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()


def _log_move(ply: int, layer: str, depth, score, nodes, elapsed_ms: float, time_left_ms: int) -> None:
    """One compact stderr line per move (2026-09, per the updated rules'
    8KB-per-game log: first 4KB + last 4KB, PGN shown alongside on the
    dashboard). Deliberately terse and single-line -- with truncation
    keeping only the opening and the endgame, a verbose or multi-line
    format would fit far fewer real moves into that budget. Fixed,
    short field names (not a natural-language sentence) so a parser can
    pull these into the ratings database alongside the PGN: ply, which
    layer actually answered (book/tb/numba/v1/trivial -- numba's own
    last_layer distinguishes an internal v1 fallback from a real numba
    answer, see cb_nb_engine.Engine.get_move), search depth reached,
    score (mover's POV, centipawns), node count, wall-clock time spent
    on this decision, and the time_left_ms this decision was actually
    made against (not the value after -- the wire protocol doesn't tell
    us the increment, so this is the honest number to log)."""
    d = "-" if depth is None else str(depth)
    sc = "-" if score is None else f"{score:+.0f}"
    n = "-" if nodes is None else str(nodes)
    print(f"mv ply={ply} layer={layer} d={d} sc={sc} n={n} t={elapsed_ms/1000:.2f}s left={time_left_ms/1000:.1f}s",
          file=sys.stderr, flush=True)


def get_move(fen: str, time_left_ms: int) -> str:
    global _last_score
    _last_score = None
    ply = chess.Board(fen).ply()
    move_start = time.perf_counter()

    if cb_book is not None:
        try:
            move = cb_book.probe(fen)
            if move is not None and _is_legal(fen, move):
                _log_move(ply, "book", None, None, None, (time.perf_counter() - move_start) * 1000, time_left_ms)
                return move
        except Exception:
            _log_move_failure("cb_book")

    if cb_tb is not None:
        try:
            move = cb_tb.probe(fen)
            if move is not None and _is_legal(fen, move):
                _log_move(ply, "tb", None, None, None, (time.perf_counter() - move_start) * 1000, time_left_ms)
                return move
        except Exception:
            _log_move_failure("cb_tb")

    if _engine is not None:
        try:
            move = _engine.get_move(fen, time_left_ms)
            if move is not None and _is_legal(fen, move):
                _last_score = getattr(_engine, "last_score", None)
                layer = getattr(_engine, "last_layer", None) or "numba"
                _log_move(ply, layer, getattr(_engine, "last_depth", None), _last_score,
                           getattr(_engine, "last_nodes", None), (time.perf_counter() - move_start) * 1000, time_left_ms)
                return move
        except Exception:
            _log_move_failure("cb_nb_engine (numba)")

    if _v1 is not None:
        try:
            move = _v1.get_move(fen, time_left_ms)
            if move is not None and _is_legal(fen, move):
                _last_score = getattr(_v1, "last_score", None)
                _log_move(ply, "v1", getattr(_v1, "last_depth", None), _last_score,
                           getattr(_v1, "last_nodes", None), (time.perf_counter() - move_start) * 1000, time_left_ms)
                return move
        except Exception:
            _log_move_failure("cb_engine (v1)")

    move = _fallback_move(fen)
    _log_move(ply, "trivial", None, None, None, (time.perf_counter() - move_start) * 1000, time_left_ms)
    return move
