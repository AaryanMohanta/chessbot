"""Hand-verified correctness tests for see_capture (static exchange
evaluation) -- every expected value below is computed independently by
hand from the position's material, not derived from the implementation
under test. See cb_nb_search.py's see_capture docstring for the
algorithm (the standard "swap" algorithm).

Expected values are expressed in terms of s._CAPTURE_VALUE (the live
material scale SEE actually uses -- see that array's own comment: it
tracks whichever material table CB_NB_TEXEL_TUNED_PST has active, not a
fixed hand-set constant), not hardcoded literals -- the exchange
ARITHMETIC being verified here (who wins what) doesn't depend on the
specific tuning, and pinning literal numbers just means every future
retune breaks these tests for no reason.
"""
import cb_nb_fast as f
import cb_nb_search as s

_T = f._TABLES
_PAWN, _KNIGHT, _BISHOP, _ROOK, _QUEEN, _KING = s._CAPTURE_VALUE


def _sq(name):
    file = ord(name[0]) - ord("a")
    rank = int(name[1]) - 1
    return rank * 8 + file


def _see(fen, from_sq_name, to_sq_name, side, promo=f.NO_PROMO, flag=0):
    pieces, mailbox, meta = f.fen_to_state(fen)
    move = _sq(from_sq_name) | (_sq(to_sq_name) << 6) | (promo << 12) | (flag << 15)
    return s.see_capture(
        move, side, pieces, mailbox,
        _T.rook_masks, _T.rook_magics, _T.rook_shifts, _T.rook_offsets, _T.rook_table,
        _T.bishop_masks, _T.bishop_magics, _T.bishop_shifts, _T.bishop_offsets, _T.bishop_table,
        _T.knight_attacks, _T.king_attacks, _T.pawn_attacks,
    )


def test_undefended_capture_wins_full_value():
    """Rxe5 against a knight with no defenders -- nothing recaptures,
    so the result is simply the knight's value."""
    v = _see("4k3/8/8/4n3/8/8/8/4RK2 w - - 0 1", "e1", "e5", 0)
    assert v == _KNIGHT


def test_defended_capture_by_cheaper_piece_loses():
    """Nxe5 where the pawn on e5 is defended by another pawn on d6 --
    trading a knight for a pawn nets pawn - knight (a loss)."""
    v = _see("4k3/8/3p4/4p3/8/5N2/8/4K3 w - - 0 1", "f3", "e5", 0)
    assert v == _PAWN - _KNIGHT


def test_equal_pawn_trade_is_zero():
    """exd5 where d5 is defended by a c6 pawn -- pawn for pawn, net 0."""
    v = _see("4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4", "d5", 0)
    assert v == 0


def test_overdefended_capture_loses_after_full_exchange():
    """Rxe5 against a knight defended by a queen behind it -- rook wins
    the knight but the queen recaptures the rook: knight - rook (a loss)."""
    v = _see("4k3/8/8/4n3/8/8/4q3/4RK2 w - - 0 1", "e1", "e5", 0)
    assert v == _KNIGHT - _ROOK


def test_xray_attacker_revealed_after_first_capture():
    """Doubled white rooks on the a-file (a1 behind a2) vs. a black
    knight on a5 defended by a bishop on c7. Rxa5, Bxa5, Rxa5 (the a1
    rook, only reachable once a2's rook has moved) -- net
    knight - rook + bishop. This is the classic case a naive
    "attackers computed once, not recomputed as occupancy shrinks"
    implementation gets wrong."""
    v = _see("4k3/2b5/8/n7/8/8/R7/R3K3 w - - 0 1", "a2", "a5", 0)
    assert v == _KNIGHT - _ROOK + _BISHOP
