"""Bitboard position representation + FEN parse/serialize (numba movegen
experiment, stage 1). Plain Python/numpy for now, deliberately not
njit-decorated yet — logic gets validated in an easily-debuggable form
first, then converted to jitted functions as a separate, re-verified step
(see the module docstring in cb_nb_movegen.py for why).

Layout: pieces[color][piece_type] as uint64 bitboards (color 0=white,
1=black; piece_type 0..5 = pawn,knight,bishop,rook,queen,king) — the same
square numbering as cb_nb_tables (a1=0..h8=63, matching python-chess).
"""
from __future__ import annotations

import numpy as np

from cb_nb_tables import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK, WHITE, BLACK

PIECE_CHARS = "pnbrqk"
CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8


class BBBoard:
    __slots__ = ("pieces", "turn", "castling_rights", "ep_square", "halfmove_clock", "fullmove_number")

    def __init__(self):
        self.pieces = np.zeros((2, 6), dtype=np.uint64)
        self.turn = WHITE
        self.castling_rights = 0
        self.ep_square = -1  # -1 = none, else 0..63
        self.halfmove_clock = 0
        self.fullmove_number = 1

    def occupied_co(self, color: int) -> int:
        occ = 0
        for pt in range(6):
            occ |= int(self.pieces[color][pt])
        return occ

    def occupied(self) -> int:
        return self.occupied_co(WHITE) | self.occupied_co(BLACK)

    def piece_type_at(self, square: int) -> int:
        mask = 1 << square
        for color in range(2):
            for pt in range(6):
                if int(self.pieces[color][pt]) & mask:
                    return pt
        return -1

    def color_at(self, square: int) -> int:
        mask = 1 << square
        for color in range(2):
            if self.occupied_co(color) & mask:
                return color
        return -1

    def set_piece_at(self, square: int, color: int, piece_type: int) -> None:
        self.pieces[color][piece_type] |= np.uint64(1 << square)

    def remove_piece_at(self, square: int) -> None:
        mask = np.uint64(~(1 << square) & ((1 << 64) - 1))
        for color in range(2):
            for pt in range(6):
                self.pieces[color][pt] &= mask


def from_fen(fen: str) -> BBBoard:
    board = BBBoard()
    parts = fen.split()
    placement, active, castling, ep, halfmove, fullmove = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]

    rank = 7
    file = 0
    for ch in placement:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            color = WHITE if ch.isupper() else BLACK
            piece_type = PIECE_CHARS.index(ch.lower())
            square = rank * 8 + file
            board.set_piece_at(square, color, piece_type)
            file += 1

    board.turn = WHITE if active == "w" else BLACK

    rights = 0
    if "K" in castling:
        rights |= CASTLE_WK
    if "Q" in castling:
        rights |= CASTLE_WQ
    if "k" in castling:
        rights |= CASTLE_BK
    if "q" in castling:
        rights |= CASTLE_BQ
    board.castling_rights = rights

    if ep == "-":
        board.ep_square = -1
    else:
        file_idx = ord(ep[0]) - ord("a")
        rank_idx = int(ep[1]) - 1
        board.ep_square = rank_idx * 8 + file_idx

    board.halfmove_clock = int(halfmove)
    board.fullmove_number = int(fullmove)
    return board


def to_fen(board: BBBoard) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            square = rank * 8 + file
            pt = board.piece_type_at(square)
            if pt == -1:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                color = board.color_at(square)
                ch = PIECE_CHARS[pt]
                row += ch.upper() if color == WHITE else ch
        if empty:
            row += str(empty)
        rows.append(row)
    placement = "/".join(rows)

    active = "w" if board.turn == WHITE else "b"

    castling = ""
    if board.castling_rights & CASTLE_WK:
        castling += "K"
    if board.castling_rights & CASTLE_WQ:
        castling += "Q"
    if board.castling_rights & CASTLE_BK:
        castling += "k"
    if board.castling_rights & CASTLE_BQ:
        castling += "q"
    if not castling:
        castling = "-"

    if board.ep_square == -1:
        ep = "-"
    else:
        file_idx = board.ep_square % 8
        rank_idx = board.ep_square // 8
        ep = chr(ord("a") + file_idx) + str(rank_idx + 1)

    return f"{placement} {active} {castling} {ep} {board.halfmove_clock} {board.fullmove_number}"
