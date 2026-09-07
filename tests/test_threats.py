"""Threats eval term (2026-09, item 3 of the 3-change plan): Stockfish's
largest non-material eval group, previously entirely absent. Two cheap,
unambiguous cases, each hand-verified against a constructed position
rather than derived from the implementation under test:
  1. A bonus for each enemy minor/major piece one of our pawns attacks.
  2. A penalty for each of our own minor/major pieces that is both
     undefended and attacked by a strictly cheaper enemy piece.
"""
import cb_nb_fast as f

_T = f._TABLES


def _threats(fen):
    pieces, mailbox, meta = f.fen_to_state(fen)
    return f._threats_score(
        pieces, _T.rook_masks, _T.rook_magics, _T.rook_shifts, _T.rook_offsets, _T.rook_table,
        _T.bishop_masks, _T.bishop_magics, _T.bishop_shifts, _T.bishop_offsets, _T.bishop_table,
        _T.knight_attacks, _T.king_attacks, _T.pawn_attacks,
    )


def test_zero_on_an_empty_quiet_position():
    # Kings only, far apart -- no pawns, no pieces, nothing to threaten.
    assert _threats("4k3/8/8/8/8/8/8/4K3 w - - 0 1") == 0


def test_pawn_attacking_a_knight_scores_positive_for_white():
    # White pawn on d4 attacks a black knight on e5.
    score = _threats("4k3/8/8/4n3/3P4/8/8/4K3 w - - 0 1")
    assert score > 0


def test_pawn_attacking_a_knight_scores_negative_when_black_attacks():
    # Mirror: black pawn on e5 attacks a white knight on d4.
    score = _threats("4k3/8/8/4p3/3N4/8/8/4K3 w - - 0 1")
    assert score < 0


def test_undefended_rook_attacked_by_bishop_is_penalized():
    # White rook on d4, black bishop on a1 attacks it along the diagonal,
    # nothing of white's defends d4.
    score = _threats("4k3/8/8/8/3R4/8/8/b3K3 w - - 0 1")
    assert score < 0


def test_defended_rook_attacked_by_bishop_is_not_penalized():
    # Same as above, but a white pawn on e3 now defends d4 -- e3, not
    # c3, so it doesn't also block the bishop's own a1-d4 diagonal.
    undefended_score = _threats("4k3/8/8/8/3R4/8/8/b3K3 w - - 0 1")
    defended_score = _threats("4k3/8/8/8/3R4/4P3/8/b3K3 w - - 0 1")
    assert defended_score > undefended_score


def test_knight_attacked_by_bishop_is_not_a_lesser_piece_threat():
    """A bishop is not strictly cheaper than a knight -- this must not
    trigger the hanging-to-a-cheaper-piece penalty the way a rook or
    queen in the same spot would."""
    score = _threats("4k3/8/8/8/3N4/8/8/b3K3 w - - 0 1")
    assert score == 0


def test_undefended_queen_attacked_by_rook_is_penalized():
    # Black rook on d1 attacks the white queen on d4 along the d-file.
    score = _threats("4k3/8/8/8/3Q4/8/8/3r3K w - - 0 1")
    assert score < 0


def test_hanging_to_pawn_and_hanging_to_minor_both_register_as_threats():
    """The pawn-attack penalty and the cheaper-minor penalty are
    different constants -- just confirm both directions register as a
    real threat (this test would also catch a copy-paste where one
    penalty accidentally aliases the other's sign)."""
    pawn_threat = _threats("4k3/8/8/4p3/3R4/8/8/4K3 w - - 0 1")  # black pawn e5 attacks Rd4
    minor_threat = _threats("4k3/8/8/8/3R4/8/8/b3K3 w - - 0 1")  # black bishop a1 attacks Rd4
    assert pawn_threat < 0
    assert minor_threat < 0
