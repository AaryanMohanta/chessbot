"""Move search (design doc §5).

Iterative deepening -> negamax with fail-soft alpha-beta and PVS ->
quiescence search at the horizon, backed by a transposition table with
killer-move and history-heuristic ordering on top of MVV-LVA, plus null-
move pruning and late move reductions.

The transposition table, killer table, and history table are all created
*once* per ``search()`` call and threaded through every iteration of
iterative deepening (including one that gets aborted by the hard time
limit) — earlier profiling found that recreating them per-iteration was
silently discarding the "each iteration primes the next" benefit ID is
supposed to provide (design doc §5), and was also why the previously
reported node counts only reflected the last *completed* iteration rather
than the true total work done.

Null-move pruning and LMR are each behind an environment-variable flag
(read once at import, same pattern as baselines/stockfish_agent.py's
strength settings) specifically so ratings/sprt.py can A/B two subprocess
instances of the *same* agent.py with one flag toggled off, via
harness.match.play_game's existing white_env/black_env — no new plumbing
needed to SPRT-test a search feature in isolation.
"""
from __future__ import annotations

import os
import time

import chess
import chess.polyglot

import cb_order
from cb_eval import evaluate
from cb_tt import MATE_SCORE, MATE_THRESHOLD, Bound, TranspositionTable, score_from_tt, score_to_tt

SearchInfo = dict

MAX_DEPTH = 64
NODE_CHECK_INTERVAL = 2048

QUIESCENCE_MAX_PLY = 8
DELTA_MARGIN = 200  # cp; see cb_search's earlier note on why this isn't SEE-based
_CAPTURE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

ENABLE_NULL_MOVE = os.environ.get("CB_ENABLE_NULL_MOVE", "1") != "0"
ENABLE_LMR = os.environ.get("CB_ENABLE_LMR", "1") != "0"

NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2  # R

LMR_MIN_DEPTH = 3
LMR_MOVE_THRESHOLD = 4  # start reducing after this many moves at a node
LMR_REDUCTION = 1


class _SearchTimeout(Exception):
    """Raised internally when the hard time budget is exceeded; caught by
    the iterative-deepening loop in search(), never propagates further."""


def _capture_value(board: chess.Board, move: chess.Move) -> int:
    if board.is_en_passant(move):
        return _CAPTURE_VALUE[chess.PAWN]
    victim = board.piece_at(move.to_square)
    return _CAPTURE_VALUE[victim.piece_type] if victim else 0


def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
    """Null-move pruning is unsound in king-and-pawn-only endgames (the
    "free move" assumption breaks down exactly where zugzwang is common)
    — skip it when the side to move has nothing but pawns and a king."""
    occupied = board.occupied_co[color]
    non_pawn_king = occupied & ~board.pawns & ~board.kings
    return non_pawn_king != 0


_PROMOTION_RANK = {chess.WHITE: chess.BB_RANK_7, chess.BLACK: chess.BB_RANK_2}


def _quiescence_moves(board: chess.Board) -> list[chess.Move]:
    """Captures (incl. en passant) plus promotions, generated directly
    rather than via ``list(board.legal_moves)`` + a Python-level filter —
    see the README for the profiling that motivated this."""
    moves = list(board.generate_legal_captures())
    promoting_pawns = board.pawns & board.occupied_co[board.turn] & _PROMOTION_RANK[board.turn]
    if promoting_pawns:
        capture_targets = {m.to_square for m in moves}
        for move in board.generate_legal_moves(from_mask=promoting_pawns):
            if move.promotion and move.to_square not in capture_targets:
                moves.append(move)
    return moves


class _Searcher:
    """Mutable state for one search() call: node count, hard deadline, the
    (persistent, owned by cb_engine) transposition table, and killer/
    history tables that live for the whole call — across every iterative-
    deepening iteration, not reset each time."""

    def __init__(self, hard_deadline: float, tt: TranspositionTable, generation: int):
        self.nodes = 0
        self.hard_deadline = hard_deadline
        self.tt = tt
        self.generation = generation
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_DEPTH + 16)]
        self.history: dict[tuple[int, int], int] = {}

    def _tick(self) -> None:
        self.nodes += 1
        if self.nodes % NODE_CHECK_INTERVAL == 0 and time.perf_counter() >= self.hard_deadline:
            raise _SearchTimeout()

    def _record_cutoff(self, board: chess.Board, move: chess.Move, depth: int, ply: int) -> None:
        """A quiet move caused a beta cutoff: remember it as a killer at
        this ply, and bump its history score."""
        if board.is_capture(move) or move.promotion:
            return
        killers = self.killers[ply]
        if move != killers[0]:
            killers[1] = killers[0]
            killers[0] = move
        piece = board.piece_at(move.from_square)
        if piece is not None:
            key = (piece.piece_type, move.to_square)
            self.history[key] = self.history.get(key, 0) + depth * depth

    def quiescence(self, board: chess.Board, alpha: int, beta: int, qply: int) -> int:
        self._tick()

        stand_pat = evaluate(board)
        if qply >= QUIESCENCE_MAX_PLY:
            return stand_pat

        best = stand_pat  # fail-soft: our own score is a valid lower bound
        if best >= beta:
            return best
        alpha = max(alpha, best)

        captures = _quiescence_moves(board)
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

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, null_allowed: bool = True) -> int:
        self._tick()
        alpha_orig = alpha

        key = chess.polyglot.zobrist_hash(board)
        tt_move: chess.Move | None = None
        tt_entry = self.tt.probe(key)
        if tt_entry is not None:
            tt_move = tt_entry.best_move
            if tt_entry.depth >= depth:
                tt_score = score_from_tt(tt_entry.score, ply)
                if tt_entry.bound == Bound.EXACT:
                    return tt_score
                if tt_entry.bound == Bound.LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_entry.bound == Bound.UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension (§5): unconditional, no cap yet

        if depth <= 0:
            return self.quiescence(board, alpha, beta, 0)

        if (
            ENABLE_NULL_MOVE
            and null_allowed
            and not in_check
            and depth >= NULL_MOVE_MIN_DEPTH
            and beta < MATE_THRESHOLD  # don't prune when we're already chasing a mate score
            and _has_non_pawn_material(board, board.turn)
        ):
            board.push(chess.Move.null())
            null_score = -self.negamax(board, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, ply + 1, null_allowed=False)
            board.pop()
            if null_score >= beta:
                return null_score  # fail-soft: a verified lower bound, not just "prune to beta"

        moves = list(board.legal_moves)
        if not moves:
            return -MATE_SCORE + ply if in_check else 0

        killers = self.killers[ply] if ply < len(self.killers) else (None, None)
        ordered = cb_order.order_moves(board, moves, tt_move=tt_move, killers=tuple(killers), history=self.history)

        best = -MATE_SCORE - 1  # fail-soft: track the true best, not just alpha
        best_move = ordered[0]
        for i, move in enumerate(ordered):
            is_quiet = not (board.is_capture(move) or move.promotion)
            board.push(move)
            gives_check = board.is_check()

            reduction = 0
            if (
                ENABLE_LMR
                and i >= LMR_MOVE_THRESHOLD
                and depth >= LMR_MIN_DEPTH
                and is_quiet
                and not in_check
                and not gives_check
            ):
                reduction = LMR_REDUCTION

            if i == 0:
                score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # PVS + LMR combined: a reduced and/or null-window search
                # first; a full-depth, full-window re-search only if that
                # suggested this move might actually beat alpha (i.e. the
                # reduction was wrong, or it genuinely beats the null
                # window) — one re-search condition covers both.
                score = -self.negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                if score > alpha:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()

            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                self._record_cutoff(board, move, depth, ply)
                break

        bound = Bound.EXACT
        if best <= alpha_orig:
            bound = Bound.UPPER
        elif best >= beta:
            bound = Bound.LOWER
        self.tt.store(key, depth, score_to_tt(best, ply), bound, best_move, self.generation)

        return best

    def root_search(self, board: chess.Board, depth: int) -> tuple[int, chess.Move]:
        key = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.probe(key)
        tt_move = tt_entry.best_move if tt_entry is not None else None

        moves = cb_order.order_moves(board, list(board.legal_moves), tt_move=tt_move, killers=tuple(self.killers[0]), history=self.history)
        alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
        best_move = moves[0]
        best_score = -MATE_SCORE - 1

        for i, move in enumerate(moves):
            board.push(move)
            if i == 0:
                score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            else:
                score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                if alpha < score < beta:
                    score = -self.negamax(board, depth - 1, -beta, -alpha, 1)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score

        self.tt.store(key, depth, score_to_tt(best_score, 0), Bound.EXACT, best_move, self.generation)
        return best_score, best_move


def search(
    board: chess.Board,
    soft_ms: float,
    hard_ms: float,
    tt: TranspositionTable,
    generation: int = 0,
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
        tt: the transposition table to use — owned by the caller
            (``cb_engine.Engine``) so it persists across moves within a
            game, not just within this one search() call.
        generation: search generation for the TT's replacement policy;
            the caller bumps this once per move.
        stop_check: reserved for an external abort signal (e.g. pondering
            being told to stand down); not used by the ID loop directly.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("search() called on a position with no legal moves")

    start = time.perf_counter()
    soft_deadline = start + max(0.0, soft_ms) / 1000
    hard_deadline = start + max(0.0, hard_ms) / 1000

    searcher = _Searcher(hard_deadline, tt, generation)
    best_move = legal_moves[0]
    best_score = 0
    info: SearchInfo = {"depth": 0, "nodes": 0}

    depth = 1
    while depth <= MAX_DEPTH:
        if stop_check():
            break
        try:
            score, move = searcher.root_search(board, depth)
        except _SearchTimeout:
            break

        best_score, best_move = score, move
        info = {"depth": depth, "nodes": searcher.nodes}

        if time.perf_counter() >= soft_deadline:
            break
        if abs(best_score) >= MATE_THRESHOLD:
            break  # found a forced mate; deepening further can't improve on it
        depth += 1

    info["nodes"] = searcher.nodes  # true total, incl. any aborted final iteration
    return best_move, best_score, info
