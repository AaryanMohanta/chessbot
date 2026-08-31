"""Move search.

Stub: picks a uniformly random legal move and reports a zero score. No
algorithm — replace with iterative deepening / negamax / PVS / MCTS /
whatever, respecting ``budget_ms`` and polling ``stop_check`` so the caller
(``cb_engine``) can enforce the time budget from ``cb_time``.
"""
from __future__ import annotations

import random

import chess

# What cb_search.search() returns: the chosen move, a centipawn score from
# the side-to-move's perspective, and a free-form diagnostics dict.
SearchInfo = dict


def search(
    board: chess.Board,
    budget_ms: int,
    stop_check=lambda: False,
) -> tuple[chess.Move, int, SearchInfo]:
    """Search ``board`` for up to ``budget_ms`` milliseconds.

    Args:
        board: position to search from (side to move = ``board.turn``).
        budget_ms: soft time budget in milliseconds; a real search should
            treat this as "try to stop around here" and check ``stop_check``
            / a hard deadline to avoid ever exceeding the caller's hard limit.
        stop_check: callable returning True when the search must return
            immediately (e.g. hard time limit reached). The stub ignores it
            since it does no iterative work.

    Returns:
        (move, score_cp, info) where ``score_cp`` is from the side-to-move's
        perspective and ``info`` carries arbitrary diagnostics (depth,
        node count, principal variation, ...).
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("search() called on a position with no legal moves")

    move = random.choice(legal_moves)
    info: SearchInfo = {"depth": 0, "nodes": len(legal_moves), "budget_ms": budget_ms}
    return move, 0, info
