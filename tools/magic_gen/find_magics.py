"""One-time offline search for magic bitboard multipliers (rook + bishop,
all 64 squares). Not part of the runtime path — this is a build tool, run
once, whose output (magic numbers + shift counts) gets hardcoded as
constants in cb_nb_tables.py. Regenerating magics via random search at
every import would blow the init budget for no reason: the numbers
themselves don't change, only the (cheap, deterministic) attack tables
built from them need to happen at import time.

Squares numbered a1=0 .. h8=63 (rank*8+file), matching python-chess, so
cross-validation against python-chess is a direct square-index comparison.
"""
import random

RANK_MASKS = [0xFF << (8 * r) for r in range(8)]
FILE_MASKS = [0x0101010101010101 << f for f in range(8)]
EDGE_MASK = RANK_MASKS[0] | RANK_MASKS[7] | FILE_MASKS[0] | FILE_MASKS[7]

MASK64 = (1 << 64) - 1


def sq(rank, file):
    return rank * 8 + file


def in_bounds(rank, file):
    return 0 <= rank < 8 and 0 <= file < 8


def rook_mask(square):
    rank, file = divmod(square, 8)
    mask = 0
    for dr, df in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        r, f = rank + dr, file + df
        while in_bounds(r + dr, f + df):  # stop one short of the edge
            mask |= 1 << sq(r, f)
            r += dr
            f += df
    return mask


def bishop_mask(square):
    rank, file = divmod(square, 8)
    mask = 0
    for dr, df in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        r, f = rank + dr, file + df
        while in_bounds(r + dr, f + df):
            mask |= 1 << sq(r, f)
            r += dr
            f += df
    return mask


def rook_attacks_slow(square, occupied):
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


def bishop_attacks_slow(square, occupied):
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


def subsets_of(mask):
    """Yield every subset of `mask`'s set bits, including 0 and mask
    itself (the standard carry-rippler enumeration)."""
    subset = 0
    while True:
        yield subset
        subset = (subset - mask) & mask
        if subset == 0:
            break


def random_sparse_u64():
    return random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)


def find_magic(square, mask, attacks_fn, max_tries=1_000_000):
    bits = bin(mask).count("1")
    size = 1 << bits
    subsets = list(subsets_of(mask))
    attack_of = [attacks_fn(square, s) for s in subsets]
    shift = 64 - bits

    for _ in range(max_tries):
        magic = random_sparse_u64()
        table = [None] * size
        ok = True
        for subset, attacks in zip(subsets, attack_of):
            index = ((subset * magic) & MASK64) >> shift
            if table[index] is None:
                table[index] = attacks
            elif table[index] != attacks:
                ok = False
                break
        if ok:
            return magic, shift
    raise RuntimeError(f"no magic found for square {square} after {max_tries} tries")


def main():
    random.seed(20260901)  # deterministic output across reruns
    rook_magics, rook_shifts = [], []
    bishop_magics, bishop_shifts = [], []

    for square in range(64):
        m, s = find_magic(square, rook_mask(square), rook_attacks_slow)
        rook_magics.append(m)
        rook_shifts.append(s)

    for square in range(64):
        m, s = find_magic(square, bishop_mask(square), bishop_attacks_slow)
        bishop_magics.append(m)
        bishop_shifts.append(s)

    print("ROOK_MAGICS = [")
    for m in rook_magics:
        print(f"    0x{m:016X},")
    print("]")
    print("ROOK_SHIFTS = ", rook_shifts)
    print()
    print("BISHOP_MAGICS = [")
    for m in bishop_magics:
        print(f"    0x{m:016X},")
    print("]")
    print("BISHOP_SHIFTS = ", bishop_shifts)


if __name__ == "__main__":
    main()
