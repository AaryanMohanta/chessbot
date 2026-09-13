"""PST tuning: fits all 768 piece-square-table cells (6 piece types x 64
squares x {mg, eg}) by the same sigmoid/game-result objective as
ratings/texel_tune.py, holding material values and the 37 already-tuned
scalar eval terms FIXED (frozen at their current shipped values from
ratings/texel_features.py's INITIAL_WEIGHTS).

Why a separate script rather than folding PST into texel_tune.py's
coordinate descent: coordinate descent perturbs one parameter at a time,
which is fine for 37 parameters but does ~20x too many full-dataset
passes to converge on 768 -- Adam (design doc sec.6 explicitly allows it
as an alternative to coordinate descent) updates every parameter every
iteration from one gradient, needing two orders of magnitude fewer
dataset passes. The per-position feature is also a fundamentally
different shape: "which piece sits on which square" is sparse (at most
32 of 384 columns nonzero per position, out of ~725k rows), so this uses
scipy.sparse throughout -- a dense (725000, 384) float64 matrix would be
~2.2GB and force a slow dense matmul on almost-all-zero rows for no
reason.

Scope note (design doc sec.6): "resist bucketing PSTs by king square, it
breaks incremental PST updates" -- not attempted here, this is a
straight per-piece-type-and-square table, same shape as the hand-set
one it replaces.

Usage: python ratings/pst_tune.py [data_csv] [output_json] [reg_lambda] [iterations]
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import chess  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import cb_nb_fast as F  # noqa: E402
import cb_tables as V1_TABLES  # noqa: E402
from ratings.texel_features import INITIAL_WEIGHTS, extract_features, eval_extra_terms  # noqa: E402

K = math.log(10) / 400  # same Elo-logistic scaling as texel_tune.py

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_PT_INDEX = {pt: i for i, pt in enumerate(_PIECE_TYPES)}  # 0-indexed, matches cb_nb_fast's PAWN..KING
N_PST_FEATURES = 6 * 64  # 384: one canonical (piece_type, square) table, white's perspective


def _current_pst_flat() -> tuple[np.ndarray, np.ndarray]:
    """The hand-set PST_MG/PST_EG from cb_tables.py, flattened to (384,)
    each in [piece_type*64 + square] order -- both the optimizer's prior
    and what's subtracted out of the combined njit eval to recover
    material-only base_mg/base_eg (see module docstring)."""
    mg = np.zeros(N_PST_FEATURES)
    eg = np.zeros(N_PST_FEATURES)
    for pt in _PIECE_TYPES:
        idx = _PT_INDEX[pt]
        mg[idx * 64:(idx + 1) * 64] = V1_TABLES.PST_MG[pt]
        eg[idx * 64:(idx + 1) * 64] = V1_TABLES.PST_EG[pt]
    return mg, eg


def _pst_feature_row(board: chess.Board) -> tuple[list[int], list[float]]:
    """(col_indices, values) for one position's sparse PST feature row --
    +1 per white piece at (piece_type, square), -1 per black piece at
    (piece_type, mirrored_square) -- exactly mirroring cb_nb_fast.py's
    _build_signed_pst's black-side sign+mirror convention, so a tuned
    weight vector equal to the current hand-set table reconstructs the
    exact same combined eval bit-for-bit."""
    cols = []
    vals = []
    for square, piece in board.piece_map().items():
        idx = _PT_INDEX[piece.piece_type]
        if piece.color == chess.WHITE:
            cols.append(idx * 64 + square)
            vals.append(1.0)
        else:
            mirrored = square ^ 56
            cols.append(idx * 64 + mirrored)
            vals.append(-1.0)
    return cols, vals


def load_dataset(csv_path: Path):
    import csv

    fens = []
    results = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fens.append(row["fen"])
            results.append(float(row["result"]))

    n = len(fens)
    prior_mg, prior_eg = _current_pst_flat()

    base_mg = np.zeros(n)
    base_eg = np.zeros(n)
    phase = np.zeros(n)
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []

    t0 = time.perf_counter()
    for i, fen in enumerate(fens):
        pieces, mailbox, meta = F.fen_to_state(fen)
        combined_mg, combined_eg, ph = F.compute_eval_state(pieces)
        phase[i] = ph

        board = chess.Board(fen)
        cols, vals = _pst_feature_row(board)
        pst_mg_now = sum(v * prior_mg[c] for c, v in zip(cols, vals))
        pst_eg_now = sum(v * prior_eg[c] for c, v in zip(cols, vals))

        features = extract_features(board)
        extra_mg, extra_eg = eval_extra_terms(features, INITIAL_WEIGHTS)

        base_mg[i] = combined_mg - pst_mg_now + extra_mg
        base_eg[i] = combined_eg - pst_eg_now + extra_eg

        indices.extend(cols)
        data.extend(vals)
        indptr.append(len(indices))

        if (i + 1) % 50_000 == 0:
            print(f"  ...{i + 1}/{n} positions ({time.perf_counter() - t0:.1f}s)", flush=True)

    pst_matrix = sp.csr_matrix((data, indices, indptr), shape=(n, N_PST_FEATURES))
    return base_mg, base_eg, phase, pst_matrix, np.array(results), prior_mg, prior_eg


def _predict(base_mg, base_eg, phase, pst_matrix, pst_mg, pst_eg):
    phase_256 = (np.minimum(phase, F.MAX_PHASE) * 256) // F.MAX_PHASE
    mg = base_mg + pst_matrix @ pst_mg
    eg = base_eg + pst_matrix @ pst_eg
    white_eval = (mg * phase_256 + eg * (256 - phase_256)) / 256
    return 1.0 / (1.0 + np.exp(-K * white_eval)), phase_256


def error(base_mg, base_eg, phase, pst_matrix, results, pst_mg, pst_eg, prior_mg, prior_eg, reg_lambda):
    predicted, _ = _predict(base_mg, base_eg, phase, pst_matrix, pst_mg, pst_eg)
    mse = float(np.mean((results - predicted) ** 2))
    penalty = reg_lambda * float(np.sum((pst_mg - prior_mg) ** 2) + np.sum((pst_eg - prior_eg) ** 2))
    return mse, penalty


def adam_fit(base_mg, base_eg, phase, pst_matrix, results, prior_mg, prior_eg,
             reg_lambda, iterations, lr=2.0, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam over the 768-dim (pst_mg, pst_eg) vector, gradient derived
    from the chain rule through the tapered eval and the sigmoid (see
    module docstring for why Adam over coordinate descent at this
    dimensionality). lr is in centipawn-ish units, not a typical ML
    learning rate, since the parameters themselves are centipawn PST
    cells -- tuned empirically against this dataset's gradient scale."""
    n = len(results)
    pst_mg = prior_mg.copy()
    pst_eg = prior_eg.copy()
    m_mg = np.zeros_like(pst_mg); v_mg = np.zeros_like(pst_mg)
    m_eg = np.zeros_like(pst_eg); v_eg = np.zeros_like(pst_eg)
    pst_matrix_T = pst_matrix.T.tocsr()

    mse0, pen0 = error(base_mg, base_eg, phase, pst_matrix, results, pst_mg, pst_eg, prior_mg, prior_eg, reg_lambda)
    print(f"initial: mse={mse0:.6f} penalty={pen0:.6f} total={mse0 + pen0:.6f}", flush=True)

    for it in range(1, iterations + 1):
        predicted, phase_256 = _predict(base_mg, base_eg, phase, pst_matrix, pst_mg, pst_eg)
        residual = -2.0 * (results - predicted) / n
        sigmoid_deriv = K * predicted * (1.0 - predicted)
        weighted = residual * sigmoid_deriv  # (n,)

        grad_mg = pst_matrix_T @ (weighted * (phase_256 / 256.0)) + 2 * reg_lambda * (pst_mg - prior_mg)
        grad_eg = pst_matrix_T @ (weighted * (1.0 - phase_256 / 256.0)) + 2 * reg_lambda * (pst_eg - prior_eg)

        for name, grad, w, m, v in (("mg", grad_mg, pst_mg, m_mg, v_mg), ("eg", grad_eg, pst_eg, m_eg, v_eg)):
            m *= beta1; m += (1 - beta1) * grad
            v *= beta2; v += (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** it)
            v_hat = v / (1 - beta2 ** it)
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if it % 50 == 0 or it == iterations:
            mse, pen = error(base_mg, base_eg, phase, pst_matrix, results, pst_mg, pst_eg, prior_mg, prior_eg, reg_lambda)
            print(f"iter {it}/{iterations}: mse={mse:.6f} penalty={pen:.6f} total={mse + pen:.6f}", flush=True)

    return pst_mg, pst_eg


def main():
    data_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "ratings" / "data" / "quiet_labeled.csv"
    out_json = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "ratings" / "pst_tuned_weights.json"
    reg_lambda = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-8
    iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 400

    print(f"loading dataset from {data_csv} ...", flush=True)
    base_mg, base_eg, phase, pst_matrix, results, prior_mg, prior_eg = load_dataset(data_csv)
    n = len(results)
    print(f"{n} positions loaded, pst_matrix nnz={pst_matrix.nnz}", flush=True)

    # 80/20 held-out split (design doc sec.10/13: "hold out 20% of
    # positions; validate by SPRT, not training error" -- 768 parameters
    # is exactly the regime that warning is for, unlike the 37-parameter
    # scalar tune). Fixed seed for reproducibility across reruns with
    # different reg_lambda/iterations.
    rng = np.random.default_rng(20260905)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    train_idx, holdout_idx = perm[:n_train], perm[n_train:]

    train_pst = pst_matrix[train_idx]
    holdout_pst = pst_matrix[holdout_idx]

    tuned_mg, tuned_eg = adam_fit(
        base_mg[train_idx], base_eg[train_idx], phase[train_idx], train_pst, results[train_idx],
        prior_mg, prior_eg, reg_lambda=reg_lambda, iterations=iterations,
    )

    holdout_mse_tuned, _ = error(base_mg[holdout_idx], base_eg[holdout_idx], phase[holdout_idx],
                                  holdout_pst, results[holdout_idx], tuned_mg, tuned_eg, prior_mg, prior_eg, 0.0)
    holdout_mse_prior, _ = error(base_mg[holdout_idx], base_eg[holdout_idx], phase[holdout_idx],
                                  holdout_pst, results[holdout_idx], prior_mg, prior_eg, prior_mg, prior_eg, 0.0)
    print(f"\nheld-out ({len(holdout_idx)} positions, never seen during fitting):")
    print(f"  hand-set PST (prior):  mse={holdout_mse_prior:.6f}")
    print(f"  tuned PST:             mse={holdout_mse_tuned:.6f}")
    if holdout_mse_tuned < holdout_mse_prior:
        print(f"  -> tuned PST generalizes better ({100 * (1 - holdout_mse_tuned / holdout_mse_prior):.2f}% lower held-out error)")
    else:
        print("  -> WARNING: tuned PST is WORSE than the hand-set prior on held-out data -- overfit, do not ship as-is")

    # Refit on the FULL dataset for the shipped weights (the split above
    # is purely a validation check, not how the final numbers are
    # produced -- same convention as texel_tune.py's single-fit-on-
    # everything, now with an honest generalization check alongside it).
    print("\nrefitting on the full dataset for the weights actually written out...", flush=True)
    tuned_mg, tuned_eg = adam_fit(base_mg, base_eg, phase, pst_matrix, results, prior_mg, prior_eg,
                                   reg_lambda=reg_lambda, iterations=iterations)

    out = {"pst_mg": {}, "pst_eg": {}}
    piece_names = {chess.PAWN: "PAWN", chess.KNIGHT: "KNIGHT", chess.BISHOP: "BISHOP",
                   chess.ROOK: "ROOK", chess.QUEEN: "QUEEN", chess.KING: "KING"}
    for pt in _PIECE_TYPES:
        idx = _PT_INDEX[pt]
        name = piece_names[pt]
        out["pst_mg"][name] = [round(float(x)) for x in tuned_mg[idx * 64:(idx + 1) * 64]]
        out["pst_eg"][name] = [round(float(x)) for x in tuned_eg[idx * 64:(idx + 1) * 64]]

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote tuned PST tables to {out_json}")


if __name__ == "__main__":
    main()
