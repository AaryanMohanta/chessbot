"""Stage 3 + 4 gates: incremental Zobrist hash and incremental
material+PST (mg/eg/phase) both match a from-scratch recompute at every
node, over full games -- not just isolated positions. Checked after every
make_move AND every unmake_move (unmake must restore the exact pre-move
values, not just something self-consistent).

Off-by-one errors in incremental eval maintenance are silent (the number
is just wrong, nothing crashes) and poison the search's evaluation
everywhere -- this is exactly the kind of bug a debug-mode assertion over
many real games is meant to catch that a handful of hand-picked positions
would miss.
"""
import random

import numpy as np

import cb_nb_fast as f

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_incremental_hash_and_eval_match_recompute_over_full_games():
    rng = random.Random(777)
    total_checks = 0

    for game_idx in range(30):
        pieces, mailbox, meta = f.fen_to_state(STARTPOS)
        zobrist = np.zeros(1, dtype=np.uint64)
        zobrist[0] = np.uint64(f.compute_hash(pieces, meta))
        eval_state = f.new_eval_state(pieces)

        history = []
        for _ply in range(60):
            legal = f.generate_legal_moves_simple(pieces, mailbox, meta)
            if not legal:
                break
            move = rng.choice(legal)

            undo = f.make_move_simple(pieces, mailbox, meta, move, zobrist, eval_state)
            total_checks += 1
            assert int(zobrist[0]) == f.compute_hash(pieces, meta), f"game {game_idx}, hash after make_move {move}"
            assert tuple(eval_state) == f.compute_eval_state(pieces), f"game {game_idx}, eval_state after make_move {move}"
            history.append((move, undo))

        for move, undo in reversed(history):
            f.unmake_move_simple(pieces, mailbox, meta, move, undo, zobrist, eval_state)
            total_checks += 1
            assert int(zobrist[0]) == f.compute_hash(pieces, meta), f"game {game_idx}, hash after unmake_move {move}"
            assert tuple(eval_state) == f.compute_eval_state(pieces), f"game {game_idx}, eval_state after unmake_move {move}"

    assert total_checks > 1000  # sanity: the loop above actually ran


def test_material_pst_component_matches_v1_eval():
    """Cross-check the material+PST component against v1's cb_eval.py on
    the same position -- true only when both sides are using the SAME
    PST data (cb_tables.py's hand-set tables). Since CB_NB_TEXEL_TUNED_PST
    (2026-09) lets cb_nb_fast swap in ratings/pst_tune.py's fit instead
    -- deliberately different from v1's, which was never in scope for
    that retune -- this check is only meaningful in hand-set mode.
    evaluate_from_state's own passed-pawn term is deliberately excluded
    here -- v1 doesn't have it (see
    test_passed_pawn_score_matches_hand_count below for that term's own
    check) -- by replicating the tapering formula directly from
    eval_state rather than calling evaluate_from_state itself."""
    import chess
    import pytest

    import cb_eval as v1_eval

    if f._TEXEL_TUNED_PST:
        pytest.skip("cb_nb_fast is using the tuned PST (CB_NB_TEXEL_TUNED_PST=1) -- "
                    "this cross-check only holds against v1 in hand-set mode")

    fen = "r1bq1rk1/ppp2ppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 8"
    pieces, mailbox, meta = f.fen_to_state(fen)
    mg, eg, phase = f.new_eval_state(pieces)
    phase = min(phase, f.MAX_PHASE)
    phase_256 = (phase * 256) // f.MAX_PHASE
    material_pst_only = (mg * phase_256 + eg * (256 - phase_256)) // 256
    got = material_pst_only if meta[0] == 0 else -material_pst_only

    expected = v1_eval.evaluate(chess.Board(fen))
    assert got == expected


_ONE_PAWN_EACH_FEN = "4k3/p7/8/8/4P3/8/8/4K3 w - - 0 1"  # white pawn e4, black pawn a7
_WIDE_WINDOW = 1_000_000  # alpha/beta wide enough the lazy short-circuit never fires


def _eval_full(pieces, meta):
    """evaluate_from_state with a window wide enough to always take the
    full-terms path (not the lazy material+PST-only shortcut)."""
    t = f._TABLES
    return f.evaluate_from_state(
        f.new_eval_state(pieces), meta[0], pieces, meta[1], -_WIDE_WINDOW, _WIDE_WINDOW,
        t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks, t.king_attacks,
    )


def test_passed_pawn_score_matches_hand_count():
    """A position with exactly one pawn per side, on different files far
    apart -- both are trivially passed (no enemy pawn anywhere near
    either), checked against a hand count, not derived from the
    implementation under test.

    White pawn e4: 0-indexed rank 3 (a1=rank 0), "ahead" ranks 4-7 on
    d/e/f files are all empty of black pawns -> passed, relative rank 3.
    Black pawn a7: 0-indexed rank 6, "ahead" (towards rank 0) on a/b
    files (no c-1 file) are all empty of white pawns -> passed, relative
    rank 7-6=1.

    Kings are e1/e8 (from _ONE_PAWN_EACH_FEN), so the EG-only king-to-
    passed-pawn-race term (2026-09) is also nonzero here and hand-
    computed too, via Chebyshev distance to each pawn's promotion
    square: white's pawn promotes on e8 (dist: white king e1->e8 is 7,
    black king e8->e8 is 0); black's pawn promotes on a1 (dist: black
    king e8->a1 is 7, white king e1->a1 is 4)."""
    pieces, mailbox, meta = f.fen_to_state(_ONE_PAWN_EACH_FEN)

    mg, eg = f._passed_pawn_score(pieces)
    expected_mg = f.PASSED_PAWN_BONUS_MG[3] - f.PASSED_PAWN_BONUS_MG[1]
    white_king_bonus_eg = f.KING_PAWN_PROXIMITY_WEIGHT_EG * (0 - 7)  # enemy(black,e8) - own(white,e1) dist to e8
    black_king_bonus_eg = f.KING_PAWN_PROXIMITY_WEIGHT_EG * (4 - 7)  # enemy(white,e1) - own(black,e8) dist to a1
    expected_eg = (
        (f.PASSED_PAWN_BONUS_EG[3] + white_king_bonus_eg)
        - (f.PASSED_PAWN_BONUS_EG[1] + black_king_bonus_eg)
    )
    assert (mg, eg) == (expected_mg, expected_eg)


def test_isolated_pawn_penalty_alone():
    """Single white pawn on e4, nothing on d or f files, nothing else on
    the e-file either -- isolated, not doubled. No black pawns."""
    fen = "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    mg, eg = f._pawn_structure_score(pieces)
    assert (mg, eg) == (-f.ISOLATED_PENALTY_MG, -f.ISOLATED_PENALTY_EG)


def test_doubled_pawn_penalty_alone():
    """White pawns on e2 and e4 (doubled), plus a white pawn on d3 so the
    e-file pawns are NOT also isolated (d-file has a friendly pawn) --
    isolates the doubled penalty from the isolated one. No black pawns."""
    fen = "4k3/8/8/8/4P3/3P4/4P3/4K3 w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    mg, eg = f._pawn_structure_score(pieces)
    # e-file has 2 pawns -> one "extra" pawn's worth of doubled penalty;
    # d-file has 1 pawn, not isolated (e-file is adjacent and occupied).
    assert (mg, eg) == (-f.DOUBLED_PENALTY_MG, -f.DOUBLED_PENALTY_EG)


def test_bishop_pair_bonus():
    """White has two bishops, black has none."""
    fen = "4k3/8/8/8/8/8/8/2B2B1K w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    mg, eg = f._bishop_pair_score(pieces)
    assert (mg, eg) == (f.BISHOP_PAIR_BONUS_MG, f.BISHOP_PAIR_BONUS_EG)


def test_single_bishop_gets_no_pair_bonus():
    fen = "4k3/8/8/8/8/8/8/4B2K w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    mg, eg = f._bishop_pair_score(pieces)
    assert (mg, eg) == (0, 0)


def test_knight_mobility_on_open_board():
    """Lone white knight on d4 with nothing else on the board except the
    two kings (placed off its attack squares): all 8 knight-move squares
    from d4 are reachable, giving a hand-countable mobility total."""
    fen = "4k3/8/8/8/3N4/8/8/4K3 w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    t = f._TABLES
    mg, eg = f._mobility_score(
        pieces, t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks,
    )
    n = 8  # b3,b5,c2,c6,e2,e6,f3,f5 -- all empty
    assert (mg, eg) == (n * f.MOBILITY_UNIT_MG[f.KNIGHT], n * f.MOBILITY_UNIT_EG[f.KNIGHT])


def test_rook_mobility_on_open_board():
    """Lone white rook on d4, kings on the a-file (off its rank/file) --
    all 14 squares along the d-file and 4th rank are reachable."""
    fen = "k7/8/8/8/3R4/8/8/K7 w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    t = f._TABLES
    mg, eg = f._mobility_score(
        pieces, t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks,
    )
    n = 14  # 4 up (d5-d8) + 3 down (d1-d3) + 3 left (a4-c4) + 4 right (e4-h4)
    assert (mg, eg) == (n * f.MOBILITY_UNIT_MG[f.ROOK], n * f.MOBILITY_UNIT_EG[f.ROOK])


def test_rook_on_fully_open_file():
    """White rook on e1, no pawns anywhere -- e-file is open for it."""
    fen = "4k3/8/8/8/8/8/8/4R1K1 w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    mg, eg = f._rook_file_score(pieces)
    assert (mg, eg) == (f.ROOK_OPEN_FILE_BONUS_MG, f.ROOK_OPEN_FILE_BONUS_EG)


def test_rook_on_semi_open_file():
    """White rook on e1, black pawn on e7 (enemy pawn present, no own
    pawn) -- semi-open, not fully open."""
    fen = "4k3/4p3/8/8/8/8/8/4R1K1 w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    mg, eg = f._rook_file_score(pieces)
    assert (mg, eg) == (f.ROOK_SEMI_OPEN_FILE_BONUS_MG, f.ROOK_SEMI_OPEN_FILE_BONUS_EG)


def test_rook_behind_own_pawn_gets_no_bonus():
    """White rook on e1, white pawn on e4 -- own pawn on the file, so
    neither open nor semi-open."""
    fen = "4k3/8/8/8/4P3/8/8/4R1K1 w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    mg, eg = f._rook_file_score(pieces)
    assert (mg, eg) == (0, 0)


_KS_BASELINE_FEN = "1k6/ppp5/8/8/8/8/5PPP/6K1 w KQkq - 0 1"
# White king g1 with full f2/g2/h2 shield; black king b8 with full
# a7/b7/c7 shield -- files g and b never overlap, so each side's terms
# can be tested by perturbing only that side without touching the
# other's shield/file/attacker status. Castling rights are set (despite
# the kings already being off e1/e8) purely to keep the new castling
# bonus/penalty term inert for these tests -- fen_to_state reads the
# castling field literally, independent of king/rook placement, so this
# is a legitimate way to isolate the shield/file/attacker terms this
# block is actually testing. See test_castling_bonus_and_uncastled_
# exposure_penalty below for that term's own dedicated tests.


def _king_safety(pieces, castling_rights):
    t = f._TABLES
    return f._king_safety_score(
        pieces, castling_rights, t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks, t.king_attacks,
    )


def test_king_safety_zero_when_both_kings_are_safe():
    pieces, mailbox, meta = f.fen_to_state(_KS_BASELINE_FEN)
    assert _king_safety(pieces, meta[1]) == 0


def test_king_shield_penalty_alone():
    """White's f2 pawn removed (g2 kept, so the king's own file still has
    an own pawn and no file penalty triggers) -- isolates one missing
    shield pawn from the file term. No enemy queen/rook on the board, so
    the material-scaled part of the shield penalty is zero here (see
    test_king_shield_penalty_scales_with_enemy_attacking_material)."""
    fen = "1k6/ppp5/8/8/8/8/6PP/6K1 w KQkq - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    assert _king_safety(pieces, meta[1]) == -f.KING_SHIELD_PENALTY_MG


def test_king_open_file_penalty_includes_the_shared_shield_square():
    """White's g2 pawn removed and black has no pawn on the g-file
    either (black's pawns are queenside) -- fully open file for white's
    king. Necessarily also drops one shield pawn (g2 is both the king's
    own-file shield square and the file itself), so the expected penalty
    is both terms together, not the file term alone -- that overlap is
    structural (removing the king's own-file pawn can't help but affect
    both checks), not a test looseness."""
    fen = "1k6/ppp5/8/8/8/8/5P1P/6K1 w KQkq - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    expected = -(f.KING_SHIELD_PENALTY_MG + f.KING_OPEN_FILE_PENALTY_MG)
    assert _king_safety(pieces, meta[1]) == expected


def test_king_semi_open_file_penalty():
    """Same as the open-file case but with a black pawn added on g7 --
    semi-open (enemy pawn present) instead of fully open. The stray g7
    pawn doesn't touch black's own shield/file checks (black's king is
    on b8)."""
    fen = "1k6/ppp3p1/8/8/8/8/5P1P/6K1 w KQkq - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    expected = -(f.KING_SHIELD_PENALTY_MG + f.KING_SEMI_OPEN_FILE_PENALTY_MG)
    assert _king_safety(pieces, meta[1]) == expected


def test_king_attacker_penalty_alone():
    """Black rook on g6 attacks down the g-file into g2, which is inside
    white king's zone (king_attacks[g1] = f1/h1/f2/g2/h2) -- everything
    else stays at the safe baseline, isolating the attacker term. The
    rook also counts as attacking material for the shield-scaling term,
    but there's no missing shield pawn here for that to multiply against."""
    fen = "1k6/ppp5/6r1/8/8/8/5PPP/6K1 w KQkq - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    expected = -(f.KING_ATTACKER_WEIGHT[f.ROOK] * f.KING_ATTACKER_PENALTY_PER_UNIT_MG)
    assert _king_safety(pieces, meta[1]) == expected


def test_king_pawn_race_favours_the_closer_king():
    """Classic king-and-pawn-race shape: a lone white a-pawn on a5
    (passed), white king far away on h1, black king much closer on c6.
    Black's king is closer to the promotion square (a8) than white's,
    so the EG term should favour BLACK here despite it being white's
    passed pawn -- the term cares about the race, not whose pawn it is.
    White king h1->a8: Chebyshev distance 7. Black king c6->a8: distance
    2. bonus = WEIGHT * (7 - 2), and since it's white's pawn being
    raced, that bonus applies with a NEGATIVE sign to white's total (the
    pawn's own side is doing badly in this particular race)."""
    fen = "8/8/2k5/P7/8/8/8/7K w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    pp_mg, pp_eg = f._passed_pawn_score(pieces)
    expected_eg = f.PASSED_PAWN_BONUS_EG[4] + f.KING_PAWN_PROXIMITY_WEIGHT_EG * (2 - 7)
    assert pp_eg == expected_eg
    assert pp_eg < f.PASSED_PAWN_BONUS_EG[4]  # confirms the king term actively hurts white here


def test_king_shield_penalty_scales_with_enemy_attacking_material():
    """Same missing f2 shield pawn as test_king_shield_penalty_alone, but
    with a black rook added on g6 -- attacking material present, so the
    shield penalty must include the material-scaled bonus on top of the
    base per-pawn penalty (and the rook's own attacker-zone penalty,
    since g6-rook also attacks into g1's zone here)."""
    fen = "1k6/ppp5/6r1/8/8/8/6PP/6K1 w KQkq - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    shield = f.KING_SHIELD_PENALTY_MG + f.KING_SHIELD_MATERIAL_BONUS_PER_UNIT_MG * f.ATTACKING_MATERIAL_ROOK_UNITS
    attacker = f.KING_ATTACKER_WEIGHT[f.ROOK] * f.KING_ATTACKER_PENALTY_PER_UNIT_MG
    assert _king_safety(pieces, meta[1]) == -(shield + attacker)


_CASTLED_SAFE_FEN = "4k3/3ppp2/8/8/8/8/5PPP/6K1 w k - 0 1"
# White king g1, full f2/g2/h2 shield, no rights left (a completed
# kingside castle); black king e8, full d7/e7/f7 shield, kingside rights
# still held (so black's castling term stays inert) and no pieces that
# could attack into white's zone -- isolates the castled bonus from
# every other king-safety term.


def test_castling_bonus_for_castled_king():
    pieces, mailbox, meta = f.fen_to_state(_CASTLED_SAFE_FEN)
    assert _king_safety(pieces, meta[1]) == f.CASTLED_BONUS_MG


def test_uncastled_exposed_penalty_requires_enemy_attacking_material():
    """White's king has wandered to e2 (not a castled square) and lost
    all rights, with a full pawn shield around it (isolating this term
    from the shield/file terms) and black's rook still on the board, far
    from e2's zone -- exactly the round-4 pattern this term exists for:
    a central, rights-less king while the opponent retains real
    attacking material. Black keeps full rights and a home-square,
    fully-shielded king, so only white's exposure penalty is nonzero."""
    fen = "4k2r/3ppp2/8/8/8/3PPP2/4K3/8 w kq - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    assert _king_safety(pieces, meta[1]) == -f.UNCASTLED_EXPOSED_PENALTY_MG


def test_uncastled_king_with_no_enemy_major_pieces_is_not_penalised():
    """Same wandered, rights-less, fully-shielded white king as above,
    but black has no queen or rook left -- the exposure penalty must not
    fire (an uncastled king in a queenless, rookless endgame is normal,
    not unsafe)."""
    fen = "4k3/3ppp2/8/8/8/3PPP2/4K3/8 w - - 0 1"
    pieces, mailbox, meta = f.fen_to_state(fen)
    assert _king_safety(pieces, meta[1]) == 0


def test_evaluate_from_state_matches_v1_plus_pawn_structure_terms():
    """evaluate_from_state = material+PST (matches v1) + the passed-pawn
    and isolated/doubled terms (each verified independently above) --
    checked together once so a sign or wiring mistake combining them
    can't hide behind any single term's test passing in isolation.
    _ONE_PAWN_EACH_FEN's two pawns are isolated as well as passed (no
    pawns anywhere on either's adjacent files), so both terms are
    nonzero here."""
    import chess

    import cb_eval as v1_eval

    pieces, mailbox, meta = f.fen_to_state(_ONE_PAWN_EACH_FEN)
    eval_state = f.new_eval_state(pieces)
    t = f._TABLES
    # Wide alpha/beta so the lazy short-circuit (tested separately below)
    # never fires here -- this test is specifically about the full path.
    got = f.evaluate_from_state(
        eval_state, meta[0], pieces, meta[1], -_WIDE_WINDOW, _WIDE_WINDOW,
        t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks, t.king_attacks,
    )

    pp_mg, pp_eg = f._passed_pawn_score(pieces)
    ps_mg, ps_eg = f._pawn_structure_score(pieces)
    rf_mg, rf_eg = f._rook_file_score(pieces)  # zero here (no rooks on the board), included for completeness
    bp_mg, bp_eg = f._bishop_pair_score(pieces)  # zero here too (no bishops), same reason
    mob_mg, mob_eg = f._mobility_score(
        pieces, t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks,
    )  # zero here too (no knights/bishops/rooks/queens), same reason
    ks_mg = f._king_safety_score(
        pieces, meta[1], t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks, t.king_attacks,
    )  # nonzero: both kings have a broken "shield" and sit on open files here
    mg = eval_state[0] + pp_mg + ps_mg + rf_mg + bp_mg + mob_mg + ks_mg
    eg = eval_state[1] + pp_eg + ps_eg + rf_eg + bp_eg + mob_eg
    phase = eval_state[2]
    phase = min(phase, f.MAX_PHASE)
    phase_256 = (phase * 256) // f.MAX_PHASE
    combined = (mg * phase_256 + eg * (256 - phase_256)) // 256
    expected_from_formula = combined if meta[0] == 0 else -combined
    assert got == expected_from_formula

    v1_material_pst = v1_eval.evaluate(chess.Board(_ONE_PAWN_EACH_FEN))
    # v1 has neither term, so its score should differ from the numba one.
    assert got != v1_material_pst


def _material_pst_only(eval_state, turn):
    mg, eg, phase = eval_state[0], eval_state[1], eval_state[2]
    phase = min(phase, f.MAX_PHASE)
    phase_256 = (phase * 256) // f.MAX_PHASE
    score = (mg * phase_256 + eg * (256 - phase_256)) // 256
    return score if turn == 0 else -score


def test_lazy_eval_short_circuits_outside_the_window():
    """When material+PST alone already clears beta by more than
    LAZY_EVAL_MARGIN, evaluate_from_state must return that raw score --
    not the fuller score that would include this position's (nonzero)
    passed-pawn/isolated-pawn contributions. A window pinned far below
    the position's actual score forces the short-circuit; matching the
    raw material+PST value (and NOT the full value) is proof the extra
    terms were genuinely skipped, not coincidentally equal."""
    pieces, mailbox, meta = f.fen_to_state(_ONE_PAWN_EACH_FEN)
    eval_state = f.new_eval_state(pieces)
    material_pst_only = _material_pst_only(eval_state, meta[0])

    alpha = beta = material_pst_only - 10_000
    t = f._TABLES
    got = f.evaluate_from_state(
        eval_state, meta[0], pieces, meta[1], alpha, beta,
        t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks, t.king_attacks,
    )
    assert got == material_pst_only

    full = _eval_full(pieces, meta)
    assert full != material_pst_only  # sanity: the full path really differs here


def test_lazy_eval_margin_boundary_does_not_short_circuit():
    """lazy_score exactly at beta + LAZY_EVAL_MARGIN must NOT trigger the
    short-circuit -- only strictly beyond it does (the condition is a
    strict '>', not '>=')."""
    pieces, mailbox, meta = f.fen_to_state(_ONE_PAWN_EACH_FEN)
    eval_state = f.new_eval_state(pieces)
    material_pst_only = _material_pst_only(eval_state, meta[0])

    beta = material_pst_only - f.LAZY_EVAL_MARGIN  # lazy_score == beta + margin exactly
    alpha = beta - 1000
    t = f._TABLES
    got = f.evaluate_from_state(
        eval_state, meta[0], pieces, meta[1], alpha, beta,
        t.rook_masks, t.rook_magics, t.rook_shifts, t.rook_offsets, t.rook_table,
        t.bishop_masks, t.bishop_magics, t.bishop_shifts, t.bishop_offsets, t.bishop_table,
        t.knight_attacks, t.king_attacks,
    )
    assert got == _eval_full(pieces, meta)
