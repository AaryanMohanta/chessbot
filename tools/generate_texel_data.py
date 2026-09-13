"""Generates a labeled dataset for Texel tuning: (FEN, game_result) pairs
sampled from self-play games, where result is 1.0/0.5/0.0 from WHITE's
perspective.

Runs entirely in one process -- no subprocess spawning, unlike the SPRT
harness. Each game normally pays a fresh ~20-40s numba JIT compile per
process (see ratings/sprt.py's module docstring); doing that once here
and then playing hundreds of games in a loop is what makes this fast
enough to generate a real dataset before the deadline, instead of paying
that cost per game.

Quiet-position sampling (standard Texel tuning practice): skip the
opening (first few plies), skip positions where the side to move is in
check, and sample sparsely (one position per few plies) so adjacent
samples from the same game aren't near-duplicates.

Usage: python tools/generate_texel_data.py [n_games] [output_path]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import chess  # noqa: E402

import cb_nb_fast as F  # noqa: E402
import cb_nb_search as S  # noqa: E402

SKIP_OPENING_PLIES = 10
SAMPLE_EVERY_N_PLIES = 4
MAX_PLIES = 240
SOFT_MS = 30.0
HARD_MS = 120.0


def play_one_game(arrays_a: "S.SearchArrays", arrays_b: "S.SearchArrays", rng: random.Random):
    """Self-play one game, alternating which SearchArrays (== TT/killers/
    history) plays which color each call so state doesn't leak between
    games in a way that biases one side -- both are the same compiled
    search, this is just bookkeeping hygiene, not an engine identity."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board_pieces, mailbox, meta = F.fen_to_state(fen)
    positions = []  # (fen_at_that_point, ply)

    ply = 0
    generation = 0
    while ply < MAX_PLIES:
        arrays = arrays_a if meta[0] == 0 else arrays_b
        generation += 1
        move, score, info = S.search(fen, SOFT_MS, HARD_MS, arrays, generation, max_depth=32)
        if move == -1:
            break  # no legal move: checkmate or stalemate

        if ply >= SKIP_OPENING_PLIES and ply % SAMPLE_EVERY_N_PLIES == 0:
            in_chk = F.in_check(
                board_pieces, meta[0],
                F._TABLES.rook_masks, F._TABLES.rook_magics, F._TABLES.rook_shifts, F._TABLES.rook_offsets, F._TABLES.rook_table,
                F._TABLES.bishop_masks, F._TABLES.bishop_magics, F._TABLES.bishop_shifts, F._TABLES.bishop_offsets, F._TABLES.bishop_table,
                F._TABLES.knight_attacks, F._TABLES.king_attacks, F._TABLES.pawn_attacks,
            )
            is_capture = mailbox[(move >> 6) & 0x3F] != -1
            if not in_chk and not is_capture:
                positions.append(fen)

        board = chess.Board(fen)
        uci = F.move_to_uci(move)
        board.push(chess.Move.from_uci(uci))
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                result = 0.5
            else:
                result = 1.0 if outcome.winner else 0.0
            return positions, result
        fen = board.fen()
        board_pieces, mailbox, meta = F.fen_to_state(fen)
        ply += 1

    return positions, 0.5  # hit MAX_PLIES: adjudicate as a draw for labeling purposes


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "ratings" / "texel_data.csv"

    rng = random.Random(42)
    arrays_a = S.SearchArrays()
    arrays_b = S.SearchArrays()
    # warm up the JIT once
    S.search("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1, 50, arrays_a)

    all_rows = []
    for g in range(n_games):
        positions, result = play_one_game(arrays_a, arrays_b, rng)
        for fen in positions:
            all_rows.append((fen, result))
        if (g + 1) % 10 == 0:
            print(f"  game {g+1}/{n_games}: {len(all_rows)} positions so far", flush=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("fen,result\n")
        for fen, result in all_rows:
            f.write(f'"{fen}",{result}\n')
    print(f"Wrote {len(all_rows)} positions from {n_games} games to {out_path}")


if __name__ == "__main__":
    main()
