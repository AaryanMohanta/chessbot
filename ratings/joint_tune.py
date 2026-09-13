"""Joint material + PST tuning: extends ratings/pst_tune.py's 768-cell PST
fit with 5 more piece-value parameters per phase (pawn/knight/bishop/
rook/queen; king excluded, its material value is always 0 and has no
meaningful gradient), for 778 total tunable parameters. The 37 scalar
eval terms stay frozen at their already-tuned values, same as pst_tune.py.

Why this needed its own script rather than just widening pst_tune.py in
place: pst_tune.py already shipped and passed its own held-out
validation (2.10% generalization gain) -- this is a strictly riskier,
higher-dimensional fit (material values interact with EVERY position,
unlike a PST cell that's only "live" when that piece sits on that
square), so it gets an independent held-out check and an independent
output file. If this doesn't generalize, pst_tune.py's already-validated
result stays the fallback with zero risk of this experiment touching it.

Mechanically: reuses pst_tune.py's whole linear-in-features framework
unchanged (the Adam step, the sigmoid/MSE/L2 objective, the tapered
mg/eg combine) -- only the feature vector widens from 384 (PST-only) to
389 (PST + 5 material counts), and load_dataset's per-position base_mg/
base_eg now subtracts out material's current contribution too, not just
PST's, so a fit that reproduces the current hand-set values exactly
reconstructs the same combined eval bit-for-bit (same invariant
pst_tune.py maintains for PST alone).

Usage: python ratings/joint_tune.py [data_csv] [output_json] [reg_lambda] [iterations]
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

K = math.log(10) / 400

_PST_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
_PST_PT_INDEX = {pt: i for i, pt in enumerate(_PST_PIECE_TYPES)}
N_PST_FEATURES = 6 * 64  # 384

_MATERIAL_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]  # king excluded
_MAT_PT_INDEX = {pt: i for i, pt in enumerate(_MATERIAL_PIECE_TYPES)}
N_MATERIAL_FEATURES = len(_MATERIAL_PIECE_TYPES)  # 5

N_FEATURES = N_PST_FEATURES + N_MATERIAL_FEATURES  # 389


def _current_prior() -> tuple[np.ndarray, np.ndarray]:
    """(389,) prior for mg and eg: PST cells (from cb_tables.py) followed
    by the 5 material values (also from cb_tables.py -- currently
    identical for mg/eg there, since PIECE_VALUES_EG = dict(PIECE_VALUES_MG),
    but tuned independently here same as everything else in this file)."""
    mg = np.zeros(N_FEATURES)
    eg = np.zeros(N_FEATURES)
    for pt in _PST_PIECE_TYPES:
        idx = _PST_PT_INDEX[pt]
        mg[idx * 64:(idx + 1) * 64] = V1_TABLES.PST_MG[pt]
        eg[idx * 64:(idx + 1) * 64] = V1_TABLES.PST_EG[pt]
    for pt in _MATERIAL_PIECE_TYPES:
        idx = _MAT_PT_INDEX[pt]
        mg[N_PST_FEATURES + idx] = V1_TABLES.PIECE_VALUES_MG[pt]
        eg[N_PST_FEATURES + idx] = V1_TABLES.PIECE_VALUES_EG[pt]
    return mg, eg


def _feature_row(board: chess.Board) -> tuple[list[int], list[float]]:
    """(col_indices, values) combining the PST occupancy row (see
    pst_tune.py's _pst_feature_row -- identical convention, same sign/
    mirror rule) with 5 material-count columns appended at N_PST_FEATURES
    + material index: (white piece count of that type) - (black count)."""
    cols = []
    vals = []
    mat_counts = [0] * N_MATERIAL_FEATURES
    for square, piece in board.piece_map().items():
        idx = _PST_PT_INDEX[piece.piece_type]
        if piece.color == chess.WHITE:
            cols.append(idx * 64 + square)
            vals.append(1.0)
        else:
            mirrored = square ^ 56
            cols.append(idx * 64 + mirrored)
            vals.append(-1.0)
        if piece.piece_type != chess.KING:
            mat_idx = _MAT_PT_INDEX[piece.piece_type]
            mat_counts[mat_idx] += 1 if piece.color == chess.WHITE else -1
    for mat_idx, count in enumerate(mat_counts):
        if count != 0:
            cols.append(N_PST_FEATURES + mat_idx)
            vals.append(float(count))
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
    prior_mg, prior_eg = _current_prior()

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
        cols, vals = _feature_row(board)
        cur_mg = sum(v * prior_mg[c] for c, v in zip(cols, vals))
        cur_eg = sum(v * prior_eg[c] for c, v in zip(cols, vals))

        features = extract_features(board)
        extra_mg, extra_eg = eval_extra_terms(features, INITIAL_WEIGHTS)

        base_mg[i] = combined_mg - cur_mg + extra_mg
        base_eg[i] = combined_eg - cur_eg + extra_eg

        indices.extend(cols)
        data.extend(vals)
        indptr.append(len(indices))

        if (i + 1) % 50_000 == 0:
            print(f"  ...{i + 1}/{n} positions ({time.perf_counter() - t0:.1f}s)", flush=True)

    feature_matrix = sp.csr_matrix((data, indices, indptr), shape=(n, N_FEATURES))
    return base_mg, base_eg, phase, feature_matrix, np.array(results), prior_mg, prior_eg


def _predict(base_mg, base_eg, phase, feature_matrix, w_mg, w_eg):
    phase_256 = (np.minimum(phase, F.MAX_PHASE) * 256) // F.MAX_PHASE
    mg = base_mg + feature_matrix @ w_mg
    eg = base_eg + feature_matrix @ w_eg
    white_eval = (mg * phase_256 + eg * (256 - phase_256)) / 256
    return 1.0 / (1.0 + np.exp(-K * white_eval)), phase_256


def error(base_mg, base_eg, phase, feature_matrix, results, w_mg, w_eg, prior_mg, prior_eg, reg_lambda):
    predicted, _ = _predict(base_mg, base_eg, phase, feature_matrix, w_mg, w_eg)
    mse = float(np.mean((results - predicted) ** 2))
    penalty = reg_lambda * float(np.sum((w_mg - prior_mg) ** 2) + np.sum((w_eg - prior_eg) ** 2))
    return mse, penalty


def adam_fit(base_mg, base_eg, phase, feature_matrix, results, prior_mg, prior_eg,
             reg_lambda, iterations, lr=2.0, beta1=0.9, beta2=0.999, eps=1e-8):
    n = len(results)
    w_mg = prior_mg.copy()
    w_eg = prior_eg.copy()
    m_mg = np.zeros_like(w_mg); v_mg = np.zeros_like(w_mg)
    m_eg = np.zeros_like(w_eg); v_eg = np.zeros_like(w_eg)
    feature_matrix_T = feature_matrix.T.tocsr()

    mse0, pen0 = error(base_mg, base_eg, phase, feature_matrix, results, w_mg, w_eg, prior_mg, prior_eg, reg_lambda)
    print(f"initial: mse={mse0:.6f} penalty={pen0:.6f} total={mse0 + pen0:.6f}", flush=True)

    for it in range(1, iterations + 1):
        predicted, phase_256 = _predict(base_mg, base_eg, phase, feature_matrix, w_mg, w_eg)
        residual = -2.0 * (results - predicted) / n
        sigmoid_deriv = K * predicted * (1.0 - predicted)
        weighted = residual * sigmoid_deriv

        grad_mg = feature_matrix_T @ (weighted * (phase_256 / 256.0)) + 2 * reg_lambda * (w_mg - prior_mg)
        grad_eg = feature_matrix_T @ (weighted * (1.0 - phase_256 / 256.0)) + 2 * reg_lambda * (w_eg - prior_eg)

        for grad, w, m, v in ((grad_mg, w_mg, m_mg, v_mg), (grad_eg, w_eg, m_eg, v_eg)):
            m *= beta1; m += (1 - beta1) * grad
            v *= beta2; v += (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** it)
            v_hat = v / (1 - beta2 ** it)
            w -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if it % 50 == 0 or it == iterations:
            mse, pen = error(base_mg, base_eg, phase, feature_matrix, results, w_mg, w_eg, prior_mg, prior_eg, reg_lambda)
            mat_mg = w_mg[N_PST_FEATURES:]
            print(f"iter {it}/{iterations}: mse={mse:.6f} penalty={pen:.6f} total={mse + pen:.6f}  "
                  f"material_mg={[round(x) for x in mat_mg]}", flush=True)

    return w_mg, w_eg


def main():
    data_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "ratings" / "data" / "quiet_labeled.csv"
    out_json = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "ratings" / "joint_tuned_weights.json"
    reg_lambda = float(sys.argv[3]) if len(sys.argv) > 3 else 5e-8
    iterations = int(sys.argv[4]) if len(sys.argv) > 4 else 400

    print(f"loading dataset from {data_csv} ...", flush=True)
    base_mg, base_eg, phase, feature_matrix, results, prior_mg, prior_eg = load_dataset(data_csv)
    n = len(results)
    print(f"{n} positions loaded, feature_matrix nnz={feature_matrix.nnz}", flush=True)

    rng = np.random.default_rng(20260905)
    perm = rng.permutation(n)
    n_train = int(n * 0.8)
    train_idx, holdout_idx = perm[:n_train], perm[n_train:]

    train_fm = feature_matrix[train_idx]
    holdout_fm = feature_matrix[holdout_idx]

    tuned_mg, tuned_eg = adam_fit(
        base_mg[train_idx], base_eg[train_idx], phase[train_idx], train_fm, results[train_idx],
        prior_mg, prior_eg, reg_lambda=reg_lambda, iterations=iterations,
    )

    holdout_mse_tuned, _ = error(base_mg[holdout_idx], base_eg[holdout_idx], phase[holdout_idx],
                                  holdout_fm, results[holdout_idx], tuned_mg, tuned_eg, prior_mg, prior_eg, 0.0)
    holdout_mse_prior, _ = error(base_mg[holdout_idx], base_eg[holdout_idx], phase[holdout_idx],
                                  holdout_fm, results[holdout_idx], prior_mg, prior_eg, prior_mg, prior_eg, 0.0)
    print(f"\nheld-out ({len(holdout_idx)} positions, never seen during fitting):")
    print(f"  hand-set material+PST (prior):  mse={holdout_mse_prior:.6f}")
    print(f"  jointly tuned material+PST:     mse={holdout_mse_tuned:.6f}")
    if holdout_mse_tuned < holdout_mse_prior:
        print(f"  -> generalizes better ({100 * (1 - holdout_mse_tuned / holdout_mse_prior):.2f}% lower held-out error)")
    else:
        print("  -> WARNING: WORSE than the hand-set prior on held-out data -- overfit, do not ship")

    print("\nrefitting on the full dataset for the weights actually written out...", flush=True)
    tuned_mg, tuned_eg = adam_fit(base_mg, base_eg, phase, feature_matrix, results, prior_mg, prior_eg,
                                   reg_lambda=reg_lambda, iterations=iterations)

    out = {"pst_mg": {}, "pst_eg": {}, "material_mg": {}, "material_eg": {}}
    piece_names = {chess.PAWN: "PAWN", chess.KNIGHT: "KNIGHT", chess.BISHOP: "BISHOP",
                   chess.ROOK: "ROOK", chess.QUEEN: "QUEEN", chess.KING: "KING"}
    for pt in _PST_PIECE_TYPES:
        idx = _PST_PT_INDEX[pt]
        name = piece_names[pt]
        out["pst_mg"][name] = [round(float(x)) for x in tuned_mg[idx * 64:(idx + 1) * 64]]
        out["pst_eg"][name] = [round(float(x)) for x in tuned_eg[idx * 64:(idx + 1) * 64]]
    for pt in _MATERIAL_PIECE_TYPES:
        idx = _MAT_PT_INDEX[pt]
        name = piece_names[pt]
        out["material_mg"][name] = round(float(tuned_mg[N_PST_FEATURES + idx]))
        out["material_eg"][name] = round(float(tuned_eg[N_PST_FEATURES + idx]))
    out["material_mg"]["KING"] = 0
    out["material_eg"]["KING"] = 0

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote tuned material+PST tables to {out_json}")
    print("Material (old -> new):")
    for pt in _MATERIAL_PIECE_TYPES:
        name = piece_names[pt]
        print(f"  {name:8s} mg: {V1_TABLES.PIECE_VALUES_MG[pt]:4d} -> {out['material_mg'][name]:4d}   "
              f"eg: {V1_TABLES.PIECE_VALUES_EG[pt]:4d} -> {out['material_eg'][name]:4d}")


if __name__ == "__main__":
    main()
