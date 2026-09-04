"""Plain-Python (chess.Board-based) re-implementation of cb_nb_fast.py's
tunable eval terms, expressed as STRUCTURAL FEATURES (counts) rather than
already-weighted values -- e.g. "how many isolated-pawn files does each
side have" instead of "isolated_count * ISOLATED_PENALTY_MG".

Why this exists as a separate implementation instead of just calling the
real njit functions: numba freezes module-level constants (scalars AND
arrays) into the compiled machine code at JIT-compile time -- changing a
Python global afterward does NOT affect an already-compiled njit
function without a full recompile (a real, previously-discovered quirk
of this codebase's njit usage). Texel tuning needs to re-evaluate the
whole dataset under thousands of different candidate weight vectors;
paying a recompile per trial would be far too slow. Expressing each term
as (feature_count, weight) instead lets the tuner recompute totals via
plain arithmetic against a NumPy feature matrix -- fast, and independent
of numba entirely.

Every term here is checked against the real cb_nb_fast.py output in
tests/test_texel_features.py before being trusted for tuning -- a
mismatch here would silently tune weights for the wrong formula.

Feature naming convention: every feature is defined so that
term_value = feature_count * corresponding_weight, and the sign
convention is white-minus-black EXCEPT the king-safety features, which
are (as in the real eval) a penalty subtracted for white and added for
black -- so king-safety features are defined as (black_raw - white_raw)
so that term_value = feature_count * weight matches the real eval's sign
too.
"""
from __future__ import annotations

import chess

# Feature names, in a fixed order shared by the tuner and the initial
# weight vector below.
FEATURE_NAMES = [
    "isolated_mg", "isolated_eg",
    "doubled_mg", "doubled_eg",
    "passed_r1_mg", "passed_r2_mg", "passed_r3_mg", "passed_r4_mg", "passed_r5_mg", "passed_r6_mg",
    "passed_r1_eg", "passed_r2_eg", "passed_r3_eg", "passed_r4_eg", "passed_r5_eg", "passed_r6_eg",
    "king_pawn_race_eg",
    "rook_open_mg", "rook_open_eg",
    "rook_semiopen_mg", "rook_semiopen_eg",
    "bishop_pair_mg", "bishop_pair_eg",
    "mob_knight_mg", "mob_knight_eg",
    "mob_bishop_mg", "mob_bishop_eg",
    "mob_rook_mg", "mob_rook_eg",
    "mob_queen_mg", "mob_queen_eg",
    "king_shield_mg",
    "king_shield_material_mg",
    "king_open_file_mg",
    "king_semiopen_file_mg",
    "king_attacker_mg",
    "king_castled_mg",
    "king_uncastled_exposed_mg",
]

# Currently-shipped values (from cb_nb_fast.py, CB_NB_TEXEL_TUNED=1
# default) -- the tuner's starting point/prior for the NEXT retune, not
# a permanent record of the original hand-set guesses (those are now
# only reachable via CB_NB_TEXEL_TUNED=0, see cb_nb_fast.py). Updated
# 2026-09 after the first real tune (ratings/texel_tune.py against
# ratings/texel_data.csv, 8,578 positions, reg_lambda=5e-7); three
# tuner outputs that came back slightly negative (mob_rook_mg,
# mob_queen_mg, mob_knight_eg) were clamped to 0 before shipping, so
# they're 0 here too, matching cb_nb_fast.py exactly. isolated/doubled/
# passed/rook/bishop/mobility have identical MG and EG weight structure
# to the real file; king safety is MG-only (no EG term).
INITIAL_WEIGHTS = {
    "isolated_mg": 11, "isolated_eg": 32,
    "doubled_mg": 15, "doubled_eg": 18,
    "passed_r1_mg": 3, "passed_r2_mg": 6, "passed_r3_mg": 20, "passed_r4_mg": 32, "passed_r5_mg": 56, "passed_r6_mg": 96,
    "passed_r1_eg": 8, "passed_r2_eg": 13, "passed_r3_eg": 34, "passed_r4_eg": 54, "passed_r5_eg": 91, "passed_r6_eg": 145,
    "king_pawn_race_eg": 5,
    "rook_open_mg": 14, "rook_open_eg": 2,
    "rook_semiopen_mg": 21, "rook_semiopen_eg": 13,
    "bishop_pair_mg": 16, "bishop_pair_eg": 29,
    "mob_knight_mg": 5, "mob_knight_eg": 0,
    "mob_bishop_mg": 5, "mob_bishop_eg": 7,
    "mob_rook_mg": 0, "mob_rook_eg": 7,
    "mob_queen_mg": 0, "mob_queen_eg": 18,
    "king_shield_mg": 26,
    "king_shield_material_mg": 1,
    "king_open_file_mg": 35,
    "king_semiopen_file_mg": 20,
    "king_attacker_mg": 12,
    "king_castled_mg": 33,
    "king_uncastled_exposed_mg": 37,
}

_KING_ATTACKER_WEIGHT = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
_ATTACKING_MATERIAL_QUEEN_UNITS = 3
_ATTACKING_MATERIAL_ROOK_UNITS = 1


def _file_of(square: int) -> int:
    return chess.square_file(square)


def _isolated_and_doubled(board: chess.Board, color: bool) -> tuple[int, int]:
    """Returns (isolated_file_count, doubled_excess_count) for one side,
    matching _pawn_structure_score's per-file logic exactly."""
    pawns = board.pieces(chess.PAWN, color)
    isolated = 0
    doubled = 0
    for file in range(8):
        count = sum(1 for sq in pawns if _file_of(sq) == file)
        if count == 0:
            continue
        if count > 1:
            doubled += count - 1
        has_adjacent = any(
            _file_of(sq) in (file - 1, file + 1)
            for sq in pawns
        )
        if not has_adjacent:
            isolated += 1
    return isolated, doubled


def _passed_pawn_buckets(board: chess.Board, color: bool) -> list[int]:
    """Returns a length-6 list: count of passed pawns at relative rank
    1..6 (index 0 == relative rank 1), matching _passed_pawn_score's
    "no enemy pawn in the passed-pawn mask ahead" rule but expressed
    directly against python-chess's board instead of a precomputed mask
    table -- a pawn is passed if no enemy pawn on its own or adjacent
    file is on a rank strictly ahead of it (from that pawn's direction)."""
    buckets = [0] * 6
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    for sq in own_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        blocked = False
        for enemy_sq in enemy_pawns:
            ef = chess.square_file(enemy_sq)
            er = chess.square_rank(enemy_sq)
            if ef not in (file - 1, file, file + 1):
                continue
            if (color and er > rank) or (not color and er < rank):
                blocked = True
                break
        if blocked:
            continue
        rel_rank = rank if color else 7 - rank
        if 1 <= rel_rank <= 6:
            buckets[rel_rank - 1] += 1
    return buckets


def _king_pawn_race_feature(board: chess.Board, color: bool) -> int:
    """Sum over this side's passed pawns of (enemy king's Chebyshev
    distance to the pawn's promotion square minus this side's own king's
    distance to it) -- matches _passed_pawn_score's king_bonus_eg
    computation exactly, just returning the raw distance-difference sum
    rather than a weighted cp value (the weight is applied uniformly in
    eval_extra_terms, like every other feature here)."""
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    own_king_sq = board.king(color)
    enemy_king_sq = board.king(not color)
    total = 0
    for sq in own_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        blocked = False
        for enemy_sq in enemy_pawns:
            ef = chess.square_file(enemy_sq)
            er = chess.square_rank(enemy_sq)
            if ef not in (file - 1, file, file + 1):
                continue
            if (color and er > rank) or (not color and er < rank):
                blocked = True
                break
        if blocked:
            continue
        promo_sq = chess.square(file, 7 if color else 0)
        own_dist = chess.square_distance(own_king_sq, promo_sq)
        enemy_dist = chess.square_distance(enemy_king_sq, promo_sq)
        total += enemy_dist - own_dist
    return total


def _rook_files(board: chess.Board, color: bool) -> tuple[int, int]:
    """Returns (open_file_rook_count, semi_open_file_rook_count) for one
    side, matching _rook_file_score."""
    own_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    own_files = {chess.square_file(sq) for sq in own_pawns}
    enemy_files = {chess.square_file(sq) for sq in enemy_pawns}
    open_count = 0
    semi_count = 0
    for sq in board.pieces(chess.ROOK, color):
        file = chess.square_file(sq)
        if file in own_files:
            continue
        if file in enemy_files:
            semi_count += 1
        else:
            open_count += 1
    return open_count, semi_count


def _mobility(board: chess.Board, color: bool) -> dict[str, int]:
    """Returns per-piece-type reachable-square counts (pseudo-legal,
    matching _mobility_score's own-occupancy-only exclusion -- python-
    chess's attacks() already respects blockers for sliding pieces the
    same way the magic-bitboard version does)."""
    own_occ = board.occupied_co[color]
    counts = {"knight": 0, "bishop": 0, "rook": 0, "queen": 0}
    for pt, key in ((chess.KNIGHT, "knight"), (chess.BISHOP, "bishop"), (chess.ROOK, "rook"), (chess.QUEEN, "queen")):
        for sq in board.pieces(pt, color):
            attacks = board.attacks(sq)
            n = chess.popcount(int(attacks) & ~own_occ)
            counts[key] += n
    return counts


def _king_safety_raw(board: chess.Board, color: bool) -> dict[str, float]:
    """Returns this side's raw (unweighted) king-safety sub-features,
    matching _king_safety_score's per-color computation exactly."""
    enemy = not color
    king_sq = board.king(color)
    king_rank = chess.square_rank(king_sq)
    king_file = chess.square_file(king_sq)
    own_pawns = board.pieces(chess.PAWN, color)

    enemy_queens = len(board.pieces(chess.QUEEN, enemy))
    enemy_rooks = len(board.pieces(chess.ROOK, enemy))
    attacking_units = enemy_queens * _ATTACKING_MATERIAL_QUEEN_UNITS + enemy_rooks * _ATTACKING_MATERIAL_ROOK_UNITS

    shield_rank = king_rank + 1 if color else king_rank - 1
    shield_missing = 0
    if 0 <= shield_rank < 8:
        for f in (king_file - 1, king_file, king_file + 1):
            if 0 <= f < 8:
                shield_sq = chess.square(f, shield_rank)
                if shield_sq not in own_pawns:
                    shield_missing += 1

    own_file_pawns = any(chess.square_file(sq) == king_file for sq in own_pawns)
    enemy_file_pawns = any(chess.square_file(sq) == king_file for sq in board.pieces(chess.PAWN, enemy))
    open_file = 1 if (not own_file_pawns and not enemy_file_pawns) else 0
    semi_open_file = 1 if (not own_file_pawns and enemy_file_pawns) else 0

    zone = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
    attacker_weight = 0
    for pt, weight in _KING_ATTACKER_WEIGHT.items():
        for sq in board.pieces(pt, enemy):
            if board.attacks(sq) & zone:
                attacker_weight += weight

    # Raw castling_rights bitmask, NOT has_kingside/queenside_castling_
    # rights() -- those validate against an actual rook being present on
    # the corresponding home square, but fen_to_state (the real engine's
    # FEN parser) reads the castling field literally, with no such
    # validation. Must match that literal parsing exactly.
    king_bb = chess.BB_H1 if color else chess.BB_H8
    queen_bb = chess.BB_A1 if color else chess.BB_A8
    has_rights = bool(board.castling_rights & (king_bb | queen_bb))
    castled_sq = chess.G1 if color else chess.G8
    castled_sq_q = chess.C1 if color else chess.C8
    is_on_castled_square = king_sq in (castled_sq, castled_sq_q)

    castled = 1 if (not has_rights and is_on_castled_square) else 0
    uncastled_exposed = 1 if (not has_rights and not is_on_castled_square and attacking_units > 0) else 0

    return {
        "shield_missing": shield_missing,
        "shield_missing_times_material": shield_missing * attacking_units,
        "open_file": open_file,
        "semiopen_file": semi_open_file,
        "attacker_weight": attacker_weight,
        "castled": castled,
        "uncastled_exposed": uncastled_exposed,
    }


def extract_features(board: chess.Board) -> dict[str, float]:
    """Returns the full feature dict for a position, in the sign
    convention documented at the top of this file."""
    w_iso, w_dbl = _isolated_and_doubled(board, chess.WHITE)
    b_iso, b_dbl = _isolated_and_doubled(board, chess.BLACK)

    w_passed = _passed_pawn_buckets(board, chess.WHITE)
    b_passed = _passed_pawn_buckets(board, chess.BLACK)

    w_rook_open, w_rook_semi = _rook_files(board, chess.WHITE)
    b_rook_open, b_rook_semi = _rook_files(board, chess.BLACK)

    w_bishop_pair = 1 if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2 else 0
    b_bishop_pair = 1 if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2 else 0

    w_mob = _mobility(board, chess.WHITE)
    b_mob = _mobility(board, chess.BLACK)

    w_king = _king_safety_raw(board, chess.WHITE)
    b_king = _king_safety_raw(board, chess.BLACK)

    w_king_race = _king_pawn_race_feature(board, chess.WHITE)
    b_king_race = _king_pawn_race_feature(board, chess.BLACK)

    features = {
        "king_pawn_race_eg": w_king_race - b_king_race,
        "isolated_mg": b_iso - w_iso, "isolated_eg": b_iso - w_iso,
        "doubled_mg": b_dbl - w_dbl, "doubled_eg": b_dbl - w_dbl,
        "rook_open_mg": w_rook_open - b_rook_open, "rook_open_eg": w_rook_open - b_rook_open,
        "rook_semiopen_mg": w_rook_semi - b_rook_semi, "rook_semiopen_eg": w_rook_semi - b_rook_semi,
        "bishop_pair_mg": w_bishop_pair - b_bishop_pair, "bishop_pair_eg": w_bishop_pair - b_bishop_pair,
        "mob_knight_mg": w_mob["knight"] - b_mob["knight"], "mob_knight_eg": w_mob["knight"] - b_mob["knight"],
        "mob_bishop_mg": w_mob["bishop"] - b_mob["bishop"], "mob_bishop_eg": w_mob["bishop"] - b_mob["bishop"],
        "mob_rook_mg": w_mob["rook"] - b_mob["rook"], "mob_rook_eg": w_mob["rook"] - b_mob["rook"],
        "mob_queen_mg": w_mob["queen"] - b_mob["queen"], "mob_queen_eg": w_mob["queen"] - b_mob["queen"],
        "king_shield_mg": b_king["shield_missing"] - w_king["shield_missing"],
        "king_shield_material_mg": b_king["shield_missing_times_material"] - w_king["shield_missing_times_material"],
        "king_open_file_mg": b_king["open_file"] - w_king["open_file"],
        "king_semiopen_file_mg": b_king["semiopen_file"] - w_king["semiopen_file"],
        "king_attacker_mg": b_king["attacker_weight"] - w_king["attacker_weight"],
        "king_castled_mg": w_king["castled"] - b_king["castled"],
        "king_uncastled_exposed_mg": b_king["uncastled_exposed"] - w_king["uncastled_exposed"],
    }
    for i in range(6):
        features[f"passed_r{i+1}_mg"] = w_passed[i] - b_passed[i]
        features[f"passed_r{i+1}_eg"] = w_passed[i] - b_passed[i]
    return features


def eval_extra_terms(features: dict[str, float], weights: dict[str, float]) -> tuple[float, float]:
    """Returns (extra_mg, extra_eg) = sum(feature * weight) split by
    suffix -- this is exactly what gets added to the material+PST base
    before phase-tapering, mirroring evaluate_from_state's
    ``mg += pp_mg + ps_mg + ...`` line."""
    mg = 0.0
    eg = 0.0
    for name, count in features.items():
        w = weights[name]
        if name.endswith("_mg"):
            mg += count * w
        else:
            eg += count * w
    return mg, eg
