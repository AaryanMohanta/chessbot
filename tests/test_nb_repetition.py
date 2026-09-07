"""Hand-verified tests for repetition detection in cb_nb_search.py, added
after a real ladder loss (round 15) where the engine threw away a fully
winning game -- up a queen -- by walking a checking sequence into a
threefold-repetition draw with no repetition awareness at all.

Two layers are tested independently:
  - `_is_repetition`, the njit helper, against hand-picked path/game
    history arrays where the expected answer is obvious by inspection.
  - The effect on search via `try_move`: a queen move in a trivially won
    K+Q vs K position must score as a plain "we're winning" value when
    its resulting position is new, and score EXACTLY 0 (the hard-coded
    draw score) when that same resulting position's Zobrist key is
    already present in game_history_keys -- proving the search itself,
    not just the helper in isolation, treats a repeat as a draw.
"""
import numpy as np

import cb_nb_fast as f
import cb_nb_search as s

_T = f._TABLES


def _sq(name):
    file = ord(name[0]) - ord("a")
    rank = int(name[1]) - 1
    return rank * 8 + file


def _move(from_sq_name, to_sq_name, promo=f.NO_PROMO, flag=0):
    return _sq(from_sq_name) | (_sq(to_sq_name) << 6) | (promo << 12) | (flag << 15)


def _key_after(fen, from_sq_name, to_sq_name):
    """Zobrist key of the position reached from fen by one quiet move."""
    pieces, mailbox, meta = f.fen_to_state(fen)
    zobrist = np.zeros(2, dtype=np.uint64)
    zobrist[0] = np.uint64(f.compute_hash(pieces, meta))
    eval_state = np.zeros(3, dtype=np.int64)
    mg, eg, phase = f.compute_eval_state(pieces)
    eval_state[0], eval_state[1], eval_state[2] = mg, eg, phase
    move = _move(from_sq_name, to_sq_name)
    f.make_move(pieces, mailbox, meta, move,
                _T.castle_rook_from, _T.castle_rook_to, _T.castle_all_rook_squares, _T.castle_all_rook_bits,
                zobrist, _T.zobrist_piece, _T.zobrist_castling, _T.zobrist_ep_file, _T.zobrist_side,
                eval_state, _T.pst_mg, _T.pst_eg, _T.phase_weight)
    return zobrist[0]


def _try_move(fen, from_sq_name, to_sq_name, arrays):
    pieces, mailbox, meta = f.fen_to_state(fen)
    zobrist = np.zeros(2, dtype=np.uint64)
    zobrist[0] = np.uint64(f.compute_hash(pieces, meta))
    mg, eg, phase = f.compute_eval_state(pieces)
    arrays.eval_state[0], arrays.eval_state[1], arrays.eval_state[2] = mg, eg, phase
    arrays.nodes[0] = 0
    arrays.path_keys[0] = zobrist[0]

    color = meta[0]
    opponent = 1 - color
    move = _move(from_sq_name, to_sq_name)
    hard_deadline = s._now() + 5.0

    return s.try_move(
        pieces, mailbox, meta, move, 1, -s.MATE_SCORE - 1, s.MATE_SCORE + 1, 0, 1, True, False,
        color, opponent,
        zobrist, arrays.eval_state, arrays.nodes, hard_deadline,
        arrays.tt_keys, arrays.tt_depths, arrays.tt_scores, arrays.tt_bounds, arrays.tt_moves, arrays.tt_generations,
        _T.rook_masks, _T.rook_magics, _T.rook_shifts, _T.rook_offsets, _T.rook_table,
        _T.bishop_masks, _T.bishop_magics, _T.bishop_shifts, _T.bishop_offsets, _T.bishop_table,
        _T.knight_attacks, _T.king_attacks, _T.pawn_attacks,
        _T.castle_king_to, _T.castle_rook_from, _T.castle_rook_to, _T.castle_right_bit,
        _T.castle_empty_squares, _T.castle_king_path, _T.castle_all_rook_squares, _T.castle_all_rook_bits,
        _T.zobrist_piece, _T.zobrist_castling, _T.zobrist_ep_file, _T.zobrist_side,
        _T.pst_mg, _T.pst_eg, _T.phase_weight,
        arrays.moves_buf_stack, arrays.scores_buf_stack, arrays.qmoves_buf_stack, arrays.qscores_buf_stack,
        arrays.killers, arrays.history, arrays.path_keys, arrays.game_history_keys, arrays.game_history_count,
        arrays.corrhist, arrays.cont_piece, arrays.cont_to, arrays.cont_hist_1ply, arrays.cont_hist_2ply,
    )


# -- _is_repetition: direct, hand-picked cases -------------------------

def test_is_repetition_true_when_key_matches_current_search_path():
    path_keys = np.array([np.uint64(111), np.uint64(222), np.uint64(0)], dtype=np.uint64)
    game_history_keys = np.zeros(4, dtype=np.uint64)
    assert s._is_repetition(np.uint64(111), 2, path_keys, game_history_keys, 0) is True


def test_is_repetition_true_when_key_matches_real_game_history():
    path_keys = np.zeros(4, dtype=np.uint64)
    game_history_keys = np.array([np.uint64(555), np.uint64(0), np.uint64(0), np.uint64(0)], dtype=np.uint64)
    assert s._is_repetition(np.uint64(555), 0, path_keys, game_history_keys, 1) is True


def test_is_repetition_false_when_no_match_anywhere():
    path_keys = np.array([np.uint64(111), np.uint64(222)], dtype=np.uint64)
    game_history_keys = np.array([np.uint64(333), np.uint64(444)], dtype=np.uint64)
    assert s._is_repetition(np.uint64(999), 2, path_keys, game_history_keys, 2) is False


def test_is_repetition_ignores_game_history_beyond_count():
    # A "stale" slot past game_history_count must not be read.
    path_keys = np.zeros(4, dtype=np.uint64)
    game_history_keys = np.array([np.uint64(0), np.uint64(777)], dtype=np.uint64)
    assert s._is_repetition(np.uint64(777), 0, path_keys, game_history_keys, 1) is False


# -- search-level effect via try_move -----------------------------------

_KQK_FEN = "7k/8/8/3Q4/8/8/8/4K3 w - - 0 1"


def test_winning_move_scores_positive_when_not_a_repeat():
    arrays = s.SearchArrays()
    legal, score = _try_move(_KQK_FEN, "d5", "d4", arrays)
    assert legal is True
    assert score > 100  # White is up a whole queen with a bare enemy king


def test_same_move_scores_below_zero_for_the_winning_side_when_it_is_a_known_repeat():
    """Pre-contempt (2026-09) this scored exactly 0 -- see the module
    docstring. Contempt (see ENABLE_CONTEMPT in cb_nb_search.py) now
    scales the draw score by the mover's own static eval at the
    repetition node: here the mover-after-White's-move is Black, down a
    whole queen, so Black's contempt-adjusted "draw" is capped at
    +CONTEMPT_EVAL_CAP*CONTEMPT_SCALE = +45 from Black's own
    perspective -- which negates back to -45 from White's perspective,
    exactly the desired effect (a won queen endgame thrown away by
    repetition should score clearly worse than 0 for the winning side,
    not as a neutral wash)."""
    arrays = s.SearchArrays()
    repeated_key = _key_after(_KQK_FEN, "d5", "d4")
    arrays.record_game_position(repeated_key)
    legal, score = _try_move(_KQK_FEN, "d5", "d4", arrays)
    assert legal is True
    assert score == -45


def test_a_different_move_is_unaffected_by_an_unrelated_repeat_entry():
    arrays = s.SearchArrays()
    unrelated_key = _key_after(_KQK_FEN, "d5", "d4")
    arrays.record_game_position(unrelated_key)
    # Qd5-b5 leads to a different position -- the repeat entry above must
    # not suppress it.
    legal, score = _try_move(_KQK_FEN, "d5", "b5", arrays)
    assert legal is True
    assert score > 100


def test_record_game_position_accumulates_across_calls():
    arrays = s.SearchArrays()
    assert arrays.game_history_count == 0
    arrays.record_game_position(np.uint64(1))
    arrays.record_game_position(np.uint64(2))
    assert arrays.game_history_count == 2
    assert arrays.game_history_keys[0] == 1
    assert arrays.game_history_keys[1] == 2


# -- _contempt_draw_score: direct, hand-picked FENs ----------------------
# Assumes ENABLE_CONTEMPT's default-on setting (CB_NB_ENABLE_CONTEMPT unset
# or != "0"); the njit function bakes ENABLE_CONTEMPT in as a compile-time
# global, so it can't be toggled per-test the way a plain constant could.

def _contempt_for_fen(fen):
    pieces, mailbox, meta = f.fen_to_state(fen)
    eval_state = np.zeros(3, dtype=np.int64)
    mg, eg, phase = f.compute_eval_state(pieces)
    eval_state[0], eval_state[1], eval_state[2] = mg, eg, phase
    return s._contempt_draw_score(
        pieces, mailbox, meta, eval_state,
        _T.rook_masks, _T.rook_magics, _T.rook_shifts, _T.rook_offsets, _T.rook_table,
        _T.bishop_masks, _T.bishop_magics, _T.bishop_shifts, _T.bishop_offsets, _T.bishop_table,
        _T.knight_attacks, _T.king_attacks,
    )


def test_contempt_is_positive_when_mover_is_clearly_losing():
    # Black to move, down a whole queen -- a draw should look attractive.
    score = _contempt_for_fen("7k/8/8/3Q4/8/8/8/4K3 b - - 0 1")
    assert score > 0


def test_contempt_is_negative_when_mover_is_clearly_winning():
    # White to move, up a whole queen -- a draw should look unattractive.
    score = _contempt_for_fen("7k/8/8/3Q4/8/8/8/4K3 w - - 0 1")
    assert score < 0


def test_contempt_is_small_near_the_starting_position():
    score = _contempt_for_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert abs(score) < 50


def test_contempt_is_capped_at_extreme_material_imbalance():
    # White up two queens and a rook: static eval is far past
    # CONTEMPT_EVAL_CAP in magnitude, so the offset must clamp rather
    # than keep growing with how far ahead/behind the mover is.
    score = _contempt_for_fen("7k/3Q4/8/3Q4/3R4/8/8/4K3 w - - 0 1")
    assert score == -int(s.CONTEMPT_SCALE * s.CONTEMPT_EVAL_CAP)
