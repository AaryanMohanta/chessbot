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
    zobrist = np.zeros(1, dtype=np.uint64)
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
    zobrist = np.zeros(1, dtype=np.uint64)
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


def test_same_move_scores_exactly_zero_when_its_result_is_a_known_repeat():
    arrays = s.SearchArrays()
    repeated_key = _key_after(_KQK_FEN, "d5", "d4")
    arrays.record_game_position(repeated_key)
    legal, score = _try_move(_KQK_FEN, "d5", "d4", arrays)
    assert legal is True
    assert score == 0


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
