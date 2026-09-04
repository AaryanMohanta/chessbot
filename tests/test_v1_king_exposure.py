"""Tests for cb_eval.py's uncastled-king-exposure stopgap (2026-09,
added after a real ladder loss -- round 9 -- where v1 was still
answering moves while cb_nb_engine's numba search was JIT-compiling in
the background, and chose a needless king recapture (Kxe7 over
Qxe7/Nxe7, all otherwise equal) purely because v1's eval had no opinion
on king safety at all.
"""
import chess

import cb_eval


def test_round_9_position_now_favours_queen_recapture_over_king():
    """The exact position from the real loss: Black to move, king on e8
    with full rights, queen/knight/king can all recapture on e7. Losing
    castling rights via Kxe7 should now score worse than keeping them
    via Qxe7, from Black's own perspective."""
    board = chess.Board("r2qk2r/1p2Nppp/p1npb3/4p3/4P3/4N3/PPP2PPP/R2QKB1R b KQkq - 0 12")

    after_kxe7 = board.copy()
    after_kxe7.push_san("Kxe7")
    score_kxe7 = cb_eval.evaluate(after_kxe7)

    after_qxe7 = board.copy()
    after_qxe7.push_san("Qxe7")
    score_qxe7 = cb_eval.evaluate(after_qxe7)

    # Both evaluate() calls return a score from the mover-to-follow's
    # perspective (White, in both resulting positions), so lower is
    # better for Black (who just moved) -- Kxe7 should look worse to
    # Black than Qxe7 by at least the exposure penalty.
    assert score_kxe7 > score_qxe7


def test_no_penalty_once_enemy_has_no_major_pieces():
    """A king with no rights, off its castled square, is NOT penalised
    once the opponent has no queen or rook left -- exactly the
    "uncastled in a quiet endgame is normal" reasoning the numba version
    of this term documents."""
    board = chess.Board("4k3/8/8/8/8/8/4K3/8 w - - 0 1")
    assert cb_eval._uncastled_exposure_penalty(board, chess.WHITE) == 0
    assert cb_eval._uncastled_exposure_penalty(board, chess.BLACK) == 0


def test_no_penalty_for_a_completed_castle():
    """A king that reached g1 with no rights left (a real castle, not a
    manual walk) isn't penalised."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/5RK1 w kq - 0 1")
    assert cb_eval._uncastled_exposure_penalty(board, chess.WHITE) == 0


def test_no_penalty_while_rights_are_still_held():
    """A king still on its home square with rights intact isn't
    penalised -- it hasn't committed to anything yet."""
    board = chess.Board(chess.STARTING_FEN)
    assert cb_eval._uncastled_exposure_penalty(board, chess.WHITE) == 0
    assert cb_eval._uncastled_exposure_penalty(board, chess.BLACK) == 0
