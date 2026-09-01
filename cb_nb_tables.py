"""Bitboard constants and attack tables for the numba movegen experiment.

Magic numbers (ROOK_MAGICS/BISHOP_MAGICS below) were found by a one-time
offline random search (tools/magic_gen/find_magics.py, seeded, ~100s) —
NOT regenerated at import. Regenerating a magic-number search at every
process start would burn a meaningful chunk of the 60s init budget for a
result that never changes; only the (cheap, deterministic) attack tables
built *from* these fixed numbers happen at import time, which is what the
design doc actually means by "attack tables built at import".

Squares: a1=0 .. h8=63 (rank*8+file), matching python-chess, so every
table here is directly index-comparable against a python-chess Board for
validation.
"""
from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5

RANK_MASKS = np.array([np.uint64(0xFF << (8 * r)) for r in range(8)], dtype=np.uint64)
FILE_MASKS = np.array([np.uint64(0x0101010101010101 << f) for f in range(8)], dtype=np.uint64)


def sq(rank: int, file: int) -> int:
    return rank * 8 + file


def in_bounds(rank: int, file: int) -> bool:
    return 0 <= rank < 8 and 0 <= file < 8


# ---- Non-sliding piece attacks (knight, king, pawn) ----

def _knight_attacks(square: int) -> int:
    rank, file = divmod(square, 8)
    attacks = 0
    for dr, df in ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2)):
        r, f = rank + dr, file + df
        if in_bounds(r, f):
            attacks |= 1 << sq(r, f)
    return attacks


def _king_attacks(square: int) -> int:
    rank, file = divmod(square, 8)
    attacks = 0
    for dr in (-1, 0, 1):
        for df in (-1, 0, 1):
            if dr == 0 and df == 0:
                continue
            r, f = rank + dr, file + df
            if in_bounds(r, f):
                attacks |= 1 << sq(r, f)
    return attacks


def _pawn_attacks(square: int, color: int) -> int:
    rank, file = divmod(square, 8)
    dr = 1 if color == WHITE else -1
    attacks = 0
    for df in (-1, 1):
        r, f = rank + dr, file + df
        if in_bounds(r, f):
            attacks |= 1 << sq(r, f)
    return attacks


KNIGHT_ATTACKS = np.array([_knight_attacks(s) for s in range(64)], dtype=np.uint64)
KING_ATTACKS = np.array([_king_attacks(s) for s in range(64)], dtype=np.uint64)
PAWN_ATTACKS = np.array([[_pawn_attacks(s, c) for s in range(64)] for c in range(2)], dtype=np.uint64)


# ---- Magic bitboard sliding attacks (rook, bishop) ----
# fmt: off
ROOK_MAGICS = [
    0x3580001040062084, 0x4840002000100540, 0x2100200040110008, 0x0A00085042000420,
    0x0200041039204A00, 0x4100040048020100, 0x0080010002001180, 0x8200058201022054,
    0x0000802080004008, 0x0100400020005001, 0x200A802000881000, 0x0000803002880080,
    0x0060800800804C00, 0x290A000A00502438, 0x0080800600430080, 0x01C10008805A0100,
    0x044000800220844C, 0x0090004020004000, 0x2490808060021000, 0x060C090010002101,
    0x0802020004205008, 0x2020808004004200, 0x8201040008121041, 0x800012000C028061,
    0x1000400280019024, 0x02002000401001C9, 0x8009118200220040, 0x8405190100201000,
    0x810A040080080080, 0x1802002200051018, 0x6011104400110802, 0x0602040A00008041,
    0x0840012044801081, 0x04C000A800A01000, 0x0000820022004090, 0x0012100029002100,
    0x0088801800800400, 0x0601000401004802, 0x800200140A000811, 0x1003800840800900,
    0x0081008202420020, 0x4020100041204002, 0x8815004620010010, 0x0082480010008080,
    0x0848009805010010, 0x3002000430420008, 0x1080283002040003, 0x0051405102920004,
    0x4680002000400140, 0xD104400482210500, 0x2006C02001005100, 0x0023003002210900,
    0x3030080080040080, 0x00020008500C0200, 0x0000182110224400, 0x6A0000C104078200,
    0x1218412410800105, 0x4022A0C082090112, 0x0103000830200241, 0x0009041001000921,
    0x800200181004E002, 0x4C15000A040008C1, 0x2080180110321084, 0x003D000022024491,
]
ROOK_SHIFTS = [
    52, 53, 53, 53, 53, 53, 53, 52, 53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53, 53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53, 53, 54, 54, 54, 54, 54, 54, 53,
    53, 54, 54, 54, 54, 54, 54, 53, 52, 53, 53, 53, 53, 53, 53, 52,
]
BISHOP_MAGICS = [
    0x11501000808C0642, 0x0010100101012000, 0x00102C00C0420010, 0x40CC0400800C0600,
    0x8102021002000402, 0x88082450080000A0, 0x0080482809080000, 0x0180708410021004,
    0x0021108410408200, 0x0080840108030D10, 0x0831100080850810, 0x0040042403800120,
    0x010D020210100102, 0x4002090120504010, 0x0041040202112422, 0x0448030100900480,
    0x1440000508080114, 0x8070000410024042, 0x20820010006600A0, 0x8000802802014020,
    0x1001000090400804, 0x2002A10200942000, 0x2000840048141000, 0x0808C3008201B002,
    0x4488200441030200, 0x0401184010100100, 0x0B00240008080C21, 0x00400C008080A080,
    0x0403010000104001, 0x0000820001011082, 0x01008211E1051000, 0x40008D0002014200,
    0x20A802402009A820, 0x0488041304440814, 0x0054020812410040, 0x4022220080880082,
    0x0000410040840040, 0x84448081000A0121, 0x2028010421210090, 0x6D08020040829840,
    0x008090104840100C, 0x1084108470050400, 0x100898440A001000, 0x4240004012061040,
    0x0004880101480400, 0x0048310802043C20, 0x0010014220882404, 0x048C210041020600,
    0x005406030420C000, 0x0026110086104402, 0x3804044404040421, 0x0000202442020800,
    0x020000C048360400, 0x8000404801030000, 0x0004082881040208, 0x02383000C0810000,
    0x0008A02802282001, 0x0200050052022002, 0x002800108400C800, 0x0104000400840440,
    0x02000420A0274406, 0x8A80104004080220, 0x1824214889080188, 0x0E405800D40040C0,
]
BISHOP_SHIFTS = [
    58, 59, 59, 59, 59, 59, 59, 58, 59, 59, 59, 59, 59, 59, 59, 59,
    59, 59, 57, 57, 57, 57, 59, 59, 59, 59, 57, 55, 55, 57, 59, 59,
    59, 59, 57, 55, 55, 57, 59, 59, 59, 59, 57, 57, 57, 57, 59, 59,
    59, 59, 59, 59, 59, 59, 59, 59, 58, 59, 59, 59, 59, 59, 59, 58,
]
# fmt: on


def _rook_mask(square: int) -> int:
    rank, file = divmod(square, 8)
    mask = 0
    for dr, df in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, f = rank + dr, file + df
        while in_bounds(r + dr, f + df):
            mask |= 1 << sq(r, f)
            r += dr
            f += df
    return mask


def _bishop_mask(square: int) -> int:
    rank, file = divmod(square, 8)
    mask = 0
    for dr, df in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        r, f = rank + dr, file + df
        while in_bounds(r + dr, f + df):
            mask |= 1 << sq(r, f)
            r += dr
            f += df
    return mask


def _rook_attacks_slow(square: int, occupied: int) -> int:
    rank, file = divmod(square, 8)
    attacks = 0
    for dr, df in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, f = rank + dr, file + df
        while in_bounds(r, f):
            attacks |= 1 << sq(r, f)
            if occupied & (1 << sq(r, f)):
                break
            r += dr
            f += df
    return attacks


def _bishop_attacks_slow(square: int, occupied: int) -> int:
    rank, file = divmod(square, 8)
    attacks = 0
    for dr, df in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        r, f = rank + dr, file + df
        while in_bounds(r, f):
            attacks |= 1 << sq(r, f)
            if occupied & (1 << sq(r, f)):
                break
            r += dr
            f += df
    return attacks


def _subsets_of(mask: int):
    subset = 0
    while True:
        yield subset
        subset = (subset - mask) & mask
        if subset == 0:
            break


def _build_magic_table(square, mask, magic, shift, attacks_fn):
    size = 1 << (64 - shift)
    table = np.zeros(size, dtype=np.uint64)
    for subset in _subsets_of(mask):
        index = ((subset * magic) & MASK64) >> shift
        table[index] = attacks_fn(square, subset)
    return table


ROOK_MASKS = np.array([_rook_mask(s) for s in range(64)], dtype=np.uint64)
BISHOP_MASKS = np.array([_bishop_mask(s) for s in range(64)], dtype=np.uint64)

# Offsets into flat concatenated tables, so the whole thing is one numpy
# array numba can index without a list-of-arrays (which njit can't handle
# well for ragged sizes).
_rook_tables = [_build_magic_table(s, int(ROOK_MASKS[s]), ROOK_MAGICS[s], ROOK_SHIFTS[s], _rook_attacks_slow) for s in range(64)]
_bishop_tables = [_build_magic_table(s, int(BISHOP_MASKS[s]), BISHOP_MAGICS[s], BISHOP_SHIFTS[s], _bishop_attacks_slow) for s in range(64)]

ROOK_OFFSETS = np.zeros(65, dtype=np.int64)
for _s in range(64):
    ROOK_OFFSETS[_s + 1] = ROOK_OFFSETS[_s] + len(_rook_tables[_s])
ROOK_ATTACK_TABLE = np.concatenate(_rook_tables)

BISHOP_OFFSETS = np.zeros(65, dtype=np.int64)
for _s in range(64):
    BISHOP_OFFSETS[_s + 1] = BISHOP_OFFSETS[_s] + len(_bishop_tables[_s])
BISHOP_ATTACK_TABLE = np.concatenate(_bishop_tables)

ROOK_MAGICS_ARR = np.array(ROOK_MAGICS, dtype=np.uint64)
ROOK_SHIFTS_ARR = np.array(ROOK_SHIFTS, dtype=np.int64)
BISHOP_MAGICS_ARR = np.array(BISHOP_MAGICS, dtype=np.uint64)
BISHOP_SHIFTS_ARR = np.array(BISHOP_SHIFTS, dtype=np.int64)


def rook_attacks(square: int, occupied: int) -> int:
    blockers = occupied & int(ROOK_MASKS[square])
    index = (blockers * ROOK_MAGICS[square]) & MASK64
    index >>= ROOK_SHIFTS[square]
    return int(ROOK_ATTACK_TABLE[ROOK_OFFSETS[square] + index])


def bishop_attacks(square: int, occupied: int) -> int:
    blockers = occupied & int(BISHOP_MASKS[square])
    index = (blockers * BISHOP_MAGICS[square]) & MASK64
    index >>= BISHOP_SHIFTS[square]
    return int(BISHOP_ATTACK_TABLE[BISHOP_OFFSETS[square] + index])


def queen_attacks(square: int, occupied: int) -> int:
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)


# ---- Zobrist keys, fixed seed at import (design doc §5) ----
# numpy's randint can't sample a full unsigned 64-bit range directly
# (its bounds check works in signed space), so build each key from two
# independent 32-bit halves instead.
_rng = np.random.RandomState(20260901)


def _random_u64(shape):
    hi = _rng.randint(0, 2**32, size=shape, dtype=np.int64).astype(np.uint64)
    lo = _rng.randint(0, 2**32, size=shape, dtype=np.int64).astype(np.uint64)
    return (hi << np.uint64(32)) | lo


ZOBRIST_PIECE = _random_u64((2, 6, 64))
ZOBRIST_CASTLING = _random_u64(16)
ZOBRIST_EP_FILE = _random_u64(8)
ZOBRIST_SIDE = _random_u64(())
