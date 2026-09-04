"""Stage 5 gate: search recursion in njit (negamax/quiescence/TT/MVV-LVA),
one njit call tree per search() call, no per-node Python/jit boundary
crossings.

Correctness is checked against known mate-in-1/mate-in-2 fixtures (same
ones v1's test_tactics.py verified independently) -- eval-magnitude-
independent since mate scores dominate. A fixed-depth score/move
cross-check against v1's own search used to live here too; see the
comment at the bottom of this file for why it was retired once eval
started diverging from v1 by design.
"""
import pytest

import cb_nb_fast as F
import cb_nb_search as S


def _search_move(fen: str, soft_ms=2000, hard_ms=6000):
    arrays = S.SearchArrays()
    move, score, info = S.search(fen, soft_ms=soft_ms, hard_ms=hard_ms, arrays=arrays)
    return move, score, info


def _uci(move: int) -> str:
    from_sq, to_sq = move & 0x3F, (move >> 6) & 0x3F
    files = "abcdefgh"
    return f"{files[from_sq % 8]}{from_sq // 8 + 1}{files[to_sq % 8]}{to_sq // 8 + 1}"


# Same fixtures as tests/test_tactics.py (v1), verified there by brute-force
# enumeration over all legal replies before being trusted as fixtures.
MATE_IN_ONE_POSITIONS = [
    ("back_rank_rook", "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1", "e1e8"),
    ("back_rank_rook_2", "3r2k1/5ppp/8/8/8/8/8/3R3K w - - 0 1", "d1d8"),
]


@pytest.mark.parametrize("name,fen,expected_uci", MATE_IN_ONE_POSITIONS, ids=[n for n, _, _ in MATE_IN_ONE_POSITIONS])
def test_finds_mate_in_one(name, fen, expected_uci):
    move, score, info = _search_move(fen)
    assert _uci(move) == expected_uci
    assert score >= S.MATE_THRESHOLD


def test_finds_mate_in_two():
    # Verified by brute force in v1's test suite: a2b3 is the *only*
    # first move for which every black reply still allows a forced mate.
    move, score, info = _search_move("8/8/8/8/8/3R4/K7/2k5 w - - 0 1")
    assert _uci(move) == "a2b3"


_MIDDLEGAME_FEN = "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 7"


@pytest.mark.parametrize("hard_ms", [5, 20, 50, 200])
def test_hard_deadline_bounds_a_single_iteration_and_stays_legal(hard_ms):
    """negamax/quiescence's node-count-polled _SearchTimeout must stop a
    single iteration from running unbounded -- a real bug this exact
    test would have caught: before it was fixed, one iterative-deepening
    depth ran to completion inside njit no matter how long it took, only
    checked *between* depths, which was directly responsible for real
    time-forfeit losses once the eval-terms work dropped nps enough for
    a single deep iteration to blow through the whole clock (see
    cb_nb_search.py's search() docstring). soft_ms is set absurdly high
    so only the hard deadline is in play.

    info['nodes'] reflects the last *completed* iteration (the one that
    got interrupted is discarded, not reported), so it has no reason to
    land on a NODE_CHECK_INTERVAL boundary itself -- what's actually
    guaranteed is that the interrupted iteration was stopped within one
    check-interval's worth of nodes past the deadline, which bounds
    elapsed wall-clock time, not the reported node count."""
    arrays = S.SearchArrays()
    import time

    t0 = time.perf_counter()
    move, score, info = S.search(_MIDDLEGAME_FEN, soft_ms=100_000, hard_ms=hard_ms, arrays=arrays, max_depth=64)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert move != -1
    import chess

    board = chess.Board(_MIDDLEGAME_FEN)
    assert chess.Move.from_uci(_uci(move)) in board.legal_moves
    # Generous bound: a few node-check-intervals' worth of overrun is
    # expected (the interrupted iteration runs a bit past the deadline
    # before its own check fires), unbounded growth is not.
    assert elapsed_ms < hard_ms + 500


@pytest.mark.parametrize("soft_ms", [50, 100, 300, 1000])
def test_soft_limit_prediction_bounds_overshoot(soft_ms):
    """Before starting a new iteration, search() predicts its cost from
    the previous iteration's node-count growth and skips it if that
    would finish past soft_deadline -- 'under-committing beats being
    rescued': the hard deadline being wide open (hard_ms huge) must not
    let a single iteration blow far past soft_ms just because nothing
    would flag on time for it. Checked as a bounded-overshoot property,
    not an exact figure -- the prediction is a heuristic, not a
    guarantee, so some overshoot from the one iteration already
    in flight when the prediction was made is expected and fine."""
    arrays = S.SearchArrays()
    import time

    t0 = time.perf_counter()
    move, score, info = S.search(_MIDDLEGAME_FEN, soft_ms=soft_ms, hard_ms=100_000, arrays=arrays, max_depth=64)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert move != -1
    assert elapsed_ms < soft_ms * 5, (
        f"elapsed {elapsed_ms:.0f}ms vs soft_ms={soft_ms}: the predictive skip should keep this "
        f"bounded, not let hard_ms's wide-open 100s budget run the search unchecked"
    )


# A test that once cross-checked numba's search against v1's at fixed
# depths (null-move/LMR disabled in v1 for a fair comparison of the
# shared negamax/PVS/TT/MVV-LVA algorithm) lived here. Retired once eval
# started diverging by design: numba's evaluate_from_state now carries
# terms (passed pawns, isolated/doubled pawns, and more to come per the
# eval-terms roadmap) that v1's cb_eval.py doesn't. A search 4-5 plies
# deep touches pawn-structure-changing lines almost regardless of which
# root FEN is picked, so "matches v1 exactly" stopped being something a
# cleaner starting position could rescue -- it was the eval staying
# identical that made the comparison meaningful, not just the search
# code. The search algorithm itself is still validated by the mate
# fixtures above (eval-magnitude-independent, since mate scores
# dominate) and by perft (movegen, unrelated to eval); playing-strength
# comparisons between v1 and numba now belong to
# ratings/run_numba_vs_v1_sprt.py, which measures real outcomes rather
# than requiring identical internal scores.
