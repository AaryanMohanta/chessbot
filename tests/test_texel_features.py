"""Validates ratings/texel_features.py against the real njit eval terms
in cb_nb_fast.py -- if these ever disagree, the Texel tuner would be
silently tuning weights for the wrong formula. Every position here must
match to the exact integer.
"""
import chess

import cb_nb_fast as f
from ratings.texel_features import INITIAL_WEIGHTS, eval_extra_terms, extract_features

_T = f._TABLES

_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqk2r/ppp2ppp/2n2n2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 7",
    "r1bq1rk1/2p1bppp/p1n2n2/1p1pp3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 0 9",
    "1k6/ppp5/8/8/8/8/5PPP/6K1 w KQkq - 0 1",
    "1k6/ppp5/8/8/8/8/6PP/6K1 w KQkq - 0 1",
    "1k6/ppp5/6r1/8/8/8/6PP/6K1 w KQkq - 0 1",
    "4k3/3ppp2/8/8/8/8/5PPP/6K1 w k - 0 1",
    "4k2r/3ppp2/8/8/8/3PPP2/4K3/8 w kq - 0 1",
    "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 8",
    "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 w - - 0 1",
]


def _real_extra_terms(fen: str) -> tuple[int, int]:
    pieces, mailbox, meta = f.fen_to_state(fen)
    pp_mg, pp_eg = f._passed_pawn_score(pieces)
    ps_mg, ps_eg = f._pawn_structure_score(pieces)
    rf_mg, rf_eg = f._rook_file_score(pieces)
    bp_mg, bp_eg = f._bishop_pair_score(pieces)
    mob_mg, mob_eg = f._mobility_score(
        pieces, _T.rook_masks, _T.rook_magics, _T.rook_shifts, _T.rook_offsets, _T.rook_table,
        _T.bishop_masks, _T.bishop_magics, _T.bishop_shifts, _T.bishop_offsets, _T.bishop_table,
        _T.knight_attacks,
    )
    ks_mg = f._king_safety_score(
        pieces, meta[1], _T.rook_masks, _T.rook_magics, _T.rook_shifts, _T.rook_offsets, _T.rook_table,
        _T.bishop_masks, _T.bishop_magics, _T.bishop_shifts, _T.bishop_offsets, _T.bishop_table,
        _T.knight_attacks, _T.king_attacks,
    )
    mg = pp_mg + ps_mg + rf_mg + bp_mg + mob_mg + ks_mg
    eg = pp_eg + ps_eg + rf_eg + bp_eg + mob_eg
    return mg, eg


def test_feature_extraction_matches_real_eval_exactly():
    mismatches = []
    for fen in _POSITIONS:
        board = chess.Board(fen)
        real_mg, real_eg = _real_extra_terms(fen)
        features = extract_features(board)
        got_mg, got_eg = eval_extra_terms(features, INITIAL_WEIGHTS)
        if (got_mg, got_eg) != (real_mg, real_eg):
            mismatches.append((fen, (real_mg, real_eg), (got_mg, got_eg)))
    assert not mismatches, f"feature/real eval mismatches: {mismatches}"
