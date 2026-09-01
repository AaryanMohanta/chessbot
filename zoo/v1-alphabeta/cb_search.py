"""Move search (design doc §5, v1 scope: days 1-2).

Iterative deepening -> negamax with fail-soft alpha-beta -> quiescence
search at the horizon. No transposition table, no null move, no LMR, no
killers/history yet (those are days 3-4, each gated behind its own SPRT
test per the design doc) — the seams for them are left as comments where
they'd plug in.
"""
from __future__ import annotations

import time

import chess

import cb_order
from cb_eval import evaluate

SearchInfo = dict

MATE_SCORE = 100_000
MAX_DEPTH = 64
NODE_CHECK_INTERVAL = 2048

# Quiescence search: capture-only, MVV-LVA ordered, delta-pruned, capped so
# a pathological capture chain can't blow the stack or the clock.
QUIESCENCE_MAX_PLY = 8
DELTA_MARGIN = 200  # cp; safety margin so delta pruning doesn't cut a move
# that's bad materially but good positionally. The design doc's sketch
# scores this off SEE; v1 has no SEE (ordering is MVV-LVA only per §5), so
# this uses the captured piece's raw value as the stand-in.
_CAPTURE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


class _SearchTimeout(Exception):
    """Raised internally when the hard time budget is exceeded; caught by
    the iterative-deepening loop in search(), never propagates further."""


def _capture_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return _CAPTURE_VALUE[chess.PAWN]
    victim = board.piece_at(move.to_square)
    return _CAPTURE_VALUE[victim.piece_type] if victim else 0


class _Searcher:
    """Holds the mutable state of a single search() call: node count and
    the hard deadline. A fresh instance per call keeps search() reentrant
    and thread-safe (relevant later for pondering)."""

    def __init__(self, hard_deadline: float):
        self.nodes = 0
        self.hard_deadline = hard_deadline

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes % NODE_CHECK_INTERVAL == 0 and time.perf_counter() >= self.hard_deadline:
            raise _SearchTimeout()

    def quiescence(self, board: chess.Board, alpha: int, beta: int, qply: int) -> int:
        self._tick()

        stand_pat = evaluate(board)
        if qply >= QUIESCENCE_MAX_PLY:
            return stand_pat

        best = stand_pat  # fail-soft: our own score is a valid lower bound
        if best >= beta:
            return best
        alpha = max(alpha, best)

        captures = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
        for move in cb_order.order_moves(board, captures):
            gain_estimate = _capture_value(board, move)
            if stand_pat + gain_estimate + DELTA_MARGIN < alpha:
                continue  # delta pruning: even winning the piece can't help

            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, qply + 1)
            board.pop()

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        return best

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._tick()

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension (§5): unconditional, no cap yet

        if depth <= 0:
            return self.quiescence(board, alpha, beta, 0)

        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0

        # SEAM: TT probe would go here (depth-sufficient hit -> early return
        # or bound tightening; TT move surfaced to cb_order as the top
        # ordering priority). Days 3-4.

        best = -MATE_SCORE - 1  # fail-soft: track the true best, not just alpha
        for move in cb_order.order_moves(board, moves):
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break  # fail-soft cutoff: `best` may exceed beta, and that's reported as-is

        # SEAM: TT store would go here (mate scores ply-adjusted per §5's
        # warning), plus killer/history updates on a beta cutoff. Days 3-4.

        return best

    def root_search(self, board: chess.Board, depth: int) -> tuple[int, chess.Move]:
        moves = cb_order.order_moves(board, list(board.legal_moves))
        alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
        best_move = moves[0]
        best_score = -MATE_SCORE - 1

        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score

        return best_score, best_move


def search(
    board: chess.Board,
    soft_ms: float,
    hard_ms: float,
    stop_check=lambda: False,
) -> tuple[chess.Move, int, SearchInfo]:
    """Iterative deepening from depth 1, returning the best move from the
    last *completed* iteration (design doc §5, §7).

    Args:
        board: position to search from (side to move = ``board.turn``).
        soft_ms: checked between iterations; once elapsed, don't start
            another depth.
        hard_ms: checked inside the search every ``NODE_CHECK_INTERVAL``
            nodes; on hit, the in-progress iteration is abandoned.
        stop_check: reserved for an external abort signal (e.g. pondering
            being told to stand down); not used by v1's ID loop directly,
            but plumbed through so callers can extend it later.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("search() called on a position with no legal moves")

    start = time.perf_counter()
    soft_deadline = start + max(0.0, soft_ms) / 1000
    hard_deadline = start + max(0.0, hard_ms) / 1000

    best_move = legal_moves[0]
    best_score = 0
    info: SearchInfo = {"depth": 0, "nodes": 0}

    depth = 1
    while depth <= MAX_DEPTH:
        if stop_check():
            break
        searcher = _Searcher(hard_deadline)
        try:
            score, move = searcher.root_search(board, depth)
        except _SearchTimeout:
            break

        best_score, best_move = score, move
        info = {"depth": depth, "nodes": searcher.nodes}

        if time.perf_counter() >= soft_deadline:
            break
        if abs(best_score) >= MATE_SCORE - MAX_DEPTH:
            break  # found a forced mate; deepening further can't improve on it
        depth += 1

    return best_move, best_score, info
