"""Texel tuning: coordinate descent over the scalar eval weights in
ratings/texel_features.py, minimizing squared error between a sigmoid of
the (material+PST+extra-terms) eval and real game outcomes.

Scope (deliberate): tunes the ~37 scalar weights (pawn structure, rook
files, bishop pair, mobility, king safety) that were hand-set throughout
this project. Does NOT tune material values or the 768-cell piece-square
tables -- those are baked into cb_nb_fast.py's incrementally-maintained
eval_state at import time (via _build_signed_pst), so tuning them would
need a different, more invasive approach and a much larger dataset to
converge well. This is the fast, high-value slice: exactly the terms
that were "just be nonzero, Texel finds the real values" guesses.

Usage: python ratings/texel_tune.py [data_csv] [output_json]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import chess  # noqa: E402
import numpy as np  # noqa: E402

import cb_nb_fast as F  # noqa: E402
from ratings.texel_features import FEATURE_NAMES, INITIAL_WEIGHTS, extract_features  # noqa: E402

# Standard Elo-logistic scaling, matching this project's own SPRT model
# (ratings/sprt.py's _elo_to_p) rather than fitting K separately --
# keeps the tuned weights on the same "centipawns" scale everything else
# in the codebase already uses.
K = math.log(10) / 400


def load_dataset(csv_path: Path):
    """Returns (base_mg, base_eg, phase, feature_matrix, results) as
    NumPy arrays -- base material+PST and phase are computed once via
    the real (untuned) njit function and never change during tuning;
    only the feature-weighted extra terms do."""
    import csv

    fens = []
    results = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fens.append(row["fen"])
            results.append(float(row["result"]))

    n = len(fens)
    base_mg = np.zeros(n)
    base_eg = np.zeros(n)
    phase = np.zeros(n)
    stm_white = np.zeros(n, dtype=bool)
    feature_matrix = np.zeros((n, len(FEATURE_NAMES)))

    for i, fen in enumerate(fens):
        pieces, mailbox, meta = F.fen_to_state(fen)
        mg, eg, ph = F.compute_eval_state(pieces)
        base_mg[i] = mg
        base_eg[i] = eg
        phase[i] = ph
        stm_white[i] = meta[0] == 0
        board = chess.Board(fen)
        feats = extract_features(board)
        feature_matrix[i] = [feats[name] for name in FEATURE_NAMES]

    return base_mg, base_eg, phase, stm_white, feature_matrix, np.array(results)


def _mg_eg_masks():
    mg_mask = np.array([1.0 if n.endswith("_mg") else 0.0 for n in FEATURE_NAMES])
    eg_mask = np.array([1.0 if n.endswith("_eg") else 0.0 for n in FEATURE_NAMES])
    return mg_mask, eg_mask


_MG_MASK, _EG_MASK = _mg_eg_masks()


def extra_mg(feature_matrix, weights_vec):
    return feature_matrix @ (weights_vec * _MG_MASK)


def extra_eg(feature_matrix, weights_vec):
    return feature_matrix @ (weights_vec * _EG_MASK)


def texel_error(base_mg, base_eg, phase, stm_white, feature_matrix, results, weights_vec,
                 initial_vec=None, reg_lambda=0.0):
    """Mean squared error between the sigmoid of the eval and the real
    game result, plus an L2 penalty pulling weights back toward
    ``initial_vec`` (the hand-set starting point, i.e. real chess
    domain knowledge) scaled by 1/n_positions -- with a limited
    self-play dataset (thousands, not the hundreds of thousands a full
    from-scratch tune would want), coordinate descent WILL happily
    overfit incidental correlations into nonsensical weights (a bishop
    pair bonus flipping negative, a castling bonus flipping negative)
    if left unregularized; this keeps the search anchored near a
    sensible prior unless the data gives real, sample-size-appropriate
    evidence to move away from it."""
    phase_capped = np.minimum(phase, F.MAX_PHASE)
    phase_256 = (phase_capped * 256) // F.MAX_PHASE
    mg = base_mg + extra_mg(feature_matrix, weights_vec)
    eg = base_eg + extra_eg(feature_matrix, weights_vec)
    white_eval = (mg * phase_256 + eg * (256 - phase_256)) / 256
    predicted = 1.0 / (1.0 + np.exp(-K * white_eval))
    mse = float(np.mean((results - predicted) ** 2))
    if initial_vec is None or reg_lambda == 0.0:
        return mse
    # Deliberately NOT normalized by n_positions: the MSE term is already
    # a per-position average that stays roughly constant as the dataset
    # grows, so a /n penalty would fade out relatively as more data comes
    # in even if the dataset is still too small for 37 free parameters --
    # this keeps regularization strength calibrated directly against the
    # dataset actually used (see the reg_lambda chosen in main()), not a
    # formula that assumes it self-scales correctly.
    penalty = reg_lambda * float(np.sum((weights_vec - initial_vec) ** 2))
    return mse + penalty


def coordinate_descent(base_mg, base_eg, phase, stm_white, feature_matrix, results, initial_vec,
                        steps=(20, 10, 5, 2, 1), reg_lambda=0.0):
    weights = initial_vec.copy()
    best_error = texel_error(base_mg, base_eg, phase, stm_white, feature_matrix, results, weights, initial_vec, reg_lambda)
    print(f"initial error (reg_lambda={reg_lambda}): {best_error:.6f}", flush=True)

    for step in steps:
        improved_this_pass = True
        while improved_this_pass:
            improved_this_pass = False
            for i in range(len(weights)):
                for delta in (step, -step):
                    trial = weights.copy()
                    trial[i] += delta
                    err = texel_error(base_mg, base_eg, phase, stm_white, feature_matrix, results, trial, initial_vec, reg_lambda)
                    if err < best_error:
                        weights = trial
                        best_error = err
                        improved_this_pass = True
        print(f"step={step}: error={best_error:.6f}", flush=True)

    return weights, best_error


def main():
    data_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "ratings" / "texel_data.csv"
    out_json = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "ratings" / "texel_tuned_weights.json"
    reg_lambda = float(sys.argv[3]) if len(sys.argv) > 3 else 8000.0

    print(f"loading dataset from {data_csv} ...", flush=True)
    base_mg, base_eg, phase, stm_white, feature_matrix, results = load_dataset(data_csv)
    print(f"{len(results)} positions loaded", flush=True)

    initial_vec = np.array([INITIAL_WEIGHTS[n] for n in FEATURE_NAMES], dtype=float)

    tuned_vec, final_error = coordinate_descent(base_mg, base_eg, phase, stm_white, feature_matrix, results, initial_vec, reg_lambda=reg_lambda)

    tuned = {name: float(v) for name, v in zip(FEATURE_NAMES, tuned_vec)}
    print("\nTuned weights (name: old -> new):")
    for name in FEATURE_NAMES:
        old = INITIAL_WEIGHTS[name]
        new = tuned[name]
        marker = "  <-- changed" if abs(new - old) >= 1 else ""
        print(f"  {name:28s} {old:7.1f} -> {new:7.1f}{marker}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(tuned, f, indent=2)
    print(f"\nWrote tuned weights to {out_json}")
    print(f"Final error: {final_error:.6f}")


if __name__ == "__main__":
    main()
