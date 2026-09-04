"""Stage 1 gate for the numba bitboard experiment: FEN round-trip against
python-chess. A smaller sample here (fast, always run) than the one that
originally validated this (400 random games / 11,859 positions, 0 errors)
— this is a regression check, not the full validation run.
"""
import random

import chess

import cb_nb_board as bb


def _random_positions(n_games: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    positions = []
    for _ in range(n_games):
        board = chess.Board()
        for _ in range(rng.randint(0, 40)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            positions.append(board.fen())
    return positions


def test_fen_roundtrip_matches_python_chess():
    positions = _random_positions(n_games=40, seed=99)
    assert len(positions) > 100

    for fen in positions:
        pc_board = chess.Board(fen)
        my_board = bb.from_fen(fen)

        for square in range(64):
            pc_piece = pc_board.piece_at(square)
            my_pt = my_board.piece_type_at(square)
            if pc_piece is None:
                assert my_pt == -1, f"{fen}: square {square} should be empty"
            else:
                expected_pt = pc_piece.piece_type - 1
                expected_color = 0 if pc_piece.color else 1
                assert my_pt == expected_pt, f"{fen}: square {square} wrong piece type"
                assert my_board.color_at(square) == expected_color, f"{fen}: square {square} wrong color"

        assert my_board.turn == (0 if pc_board.turn else 1), fen

        expected_ep = pc_board.ep_square if pc_board.ep_square is not None else -1
        assert my_board.ep_square == expected_ep, fen

        assert bb.to_fen(my_board) == pc_board.fen(), fen
