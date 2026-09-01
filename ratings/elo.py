"""Joint rating fit by logistic maximum likelihood over the whole game
database at once (BayesElo/Ordo-style) — not incremental Elo updates.

Model: standard Bradley-Terry/Elo logistic,
    E(white wins | r_w, r_b) = 1 / (1 + 10^-((r_w - r_b) / 400))
fit by minimizing cross-entropy between E and the actual score (1/0.5/0)
across every game simultaneously, with one version's rating pinned at 0
(the anchor) to fix the otherwise-unidentifiable overall scale.

Draws are folded into the same logistic likelihood as a score of 0.5 for
both sides (the common simplification most DIY joint-rating tools use);
this is not full BayesElo's separate draw-probability parameter, which
would need its own fitted term. Flagged as a known simplification — see
the README.

A small L2 prior (`r ~ N(0, PRIOR_STD_ELO^2)`, independently per
non-anchor version) is added to the objective. This is the actual "Bayes"
in BayesElo: without it, the plain MLE is undefined or diverges in two
cases the brief calls out explicitly:

  - A version with no losses: nothing pulls its rating down, so the
    unregularized MLE for it is +infinity. The prior bounds it at a large
    but finite value; the report flags such versions as "undefeated" so
    the number isn't read as a precise estimate.
  - A version disconnected from the rest of the pool (zero recorded
    games): the likelihood carries no information about it at all, so its
    Hessian block is exactly the prior's — the fit correctly reports it at
    the anchor (0) with a wide confidence interval, which *is* the right
    answer for "we know nothing." The report flags these as "no games."

Confidence intervals are the Laplace approximation to the posterior:
inverse of the Hessian of (negative log-likelihood + prior) at the MAP
estimate, i.e. this is a Bayesian credible interval under the stated
prior, not a distribution-free frequentist interval.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from ratings.db import GameRecord

ELO_SCALE = math.log(10) / 400  # k, s.t. E = sigmoid(k * (r_w - r_b))
PRIOR_STD_ELO = 700.0  # weak prior; large enough not to distort well-connected estimates
Z_95 = 1.959963985  # two-sided 95% normal quantile


@dataclasses.dataclass
class RatingEntry:
    version: str
    rating: float
    ci95: float  # report as rating +/- ci95
    games: int
    wins: float  # counts draws as 0.5
    losses: float
    draws: int
    flags: list[str]


@dataclasses.dataclass
class RatingFit:
    anchor: str
    entries: list[RatingEntry]  # sorted by rating, descending

    def by_version(self) -> dict[str, RatingEntry]:
        return {e.version: e for e in self.entries}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_ratings(games: list[GameRecord], anchor: str) -> RatingFit:
    versions = {g.white_version for g in games} | {g.black_version for g in games} | {anchor}
    others = sorted(v for v in versions if v != anchor)
    index = {v: i for i, v in enumerate(others)}
    n = len(others)

    lam = 1.0 / (2.0 * PRIOR_STD_ELO**2)  # L2 penalty weight: lam * sum(r_i^2)

    # Per-game (white_idx_or_None, black_idx_or_None, white_score) with
    # None meaning "this side is the anchor, rating fixed at 0".
    parsed = [(index.get(g.white_version), index.get(g.black_version), g.white_score) for g in games]

    def grad_and_hess(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        grad = 2.0 * lam * r
        hess = np.diag(np.full(n, 2.0 * lam))
        for wi, bi, s in parsed:
            r_w = r[wi] if wi is not None else 0.0
            r_b = r[bi] if bi is not None else 0.0
            z = ELO_SCALE * (r_w - r_b)
            e = _sigmoid(np.array(z))
            g = ELO_SCALE * (e - s)
            h = (ELO_SCALE**2) * e * (1 - e)
            if wi is not None:
                grad[wi] += g
                hess[wi, wi] += h
            if bi is not None:
                grad[bi] -= g
                hess[bi, bi] += h
            if wi is not None and bi is not None:
                hess[wi, bi] -= h
                hess[bi, wi] -= h
        return grad, hess

    r = np.zeros(n)
    for _ in range(200):
        grad, hess = grad_and_hess(r)
        if np.linalg.norm(grad) < 1e-9:
            break
        step = np.linalg.solve(hess, grad)
        step = np.clip(step, -800.0, 800.0)  # see fit_rating_vs_known_opponents's note on Newton divergence
        r = r - step

    _, hess_final = grad_and_hess(r)
    covariance = np.linalg.inv(hess_final)
    variances = np.clip(np.diag(covariance), 0.0, None)
    stderr = np.sqrt(variances)

    stats: dict[str, dict] = {v: {"games": 0, "wins": 0.0, "losses": 0.0, "draws": 0} for v in versions}
    for g in games:
        stats[g.white_version]["games"] += 1
        stats[g.black_version]["games"] += 1
        if g.result == "1-0":
            stats[g.white_version]["wins"] += 1
            stats[g.black_version]["losses"] += 1
        elif g.result == "0-1":
            stats[g.black_version]["wins"] += 1
            stats[g.white_version]["losses"] += 1
        else:
            stats[g.white_version]["draws"] += 1
            stats[g.black_version]["draws"] += 1

    entries = []
    for v in versions:
        rating = 0.0 if v == anchor else float(r[index[v]])
        ci95 = 0.0 if v == anchor else float(Z_95 * stderr[index[v]])
        s = stats[v]
        flags = []
        if s["games"] == 0:
            flags.append("no_games")
        elif s["losses"] == 0 and s["wins"] > 0 and v != anchor:
            # No losses *and* at least one win is what actually makes the
            # unregularized MLE diverge; all-draws alone is a perfectly
            # well-behaved rating of "equal to whoever it drew" and needs
            # no flag. Doesn't apply to the anchor itself: its rating is
            # pinned at 0 regardless of W/L, so "undefeated" would be
            # true-but-misleading there.
            flags.append("undefeated")
        if v == anchor:
            flags.append("anchor")
        entries.append(RatingEntry(v, rating, ci95, s["games"], s["wins"], s["losses"], s["draws"], flags))

    entries.sort(key=lambda e: e.rating, reverse=True)
    return RatingFit(anchor=anchor, entries=entries)


def fit_rating_vs_known_opponents(observations: list[tuple[float, float]]) -> tuple[float, float]:
    """Single-parameter MLE: our rating R against opponents of *known,
    fixed* Elo (e.g. Stockfish capped via UCI_Elo) — the calibration-ladder
    case, distinct from the joint multi-version fit above where every
    rating is simultaneously unknown.

    ``observations``: [(opponent_elo, our_score), ...], one per game, our
    score in {0, 0.5, 1}. Returns (rating, ci95) — report as rating +/-
    ci95. Same weak L2 prior as the joint fit, for the same reason: it
    costs nothing when the data is informative and keeps the estimate
    finite in a degenerate sweep against every tested level.
    """
    lam = 1.0 / (2.0 * PRIOR_STD_ELO**2)
    opp_elos = np.array([o for o, _ in observations])
    scores = np.array([s for _, s in observations])

    def grad_and_hess(r: float) -> tuple[float, float]:
        z = ELO_SCALE * (r - opp_elos)
        e = _sigmoid(z)
        grad = ELO_SCALE * np.sum(e - scores) + 2.0 * lam * r
        hess = (ELO_SCALE**2) * np.sum(e * (1 - e)) + 2.0 * lam
        return grad, hess

    # Newton's method needs a starting point where the sigmoid has real
    # curvature to work with. Starting at 0 while opponents sit at
    # 1300-1900 puts z deep in the tail (Hessian ~0), which produced a
    # divergent oscillation the first time this ran (r bouncing between
    # -1410 and +9872 forever, never converging). Starting at the mean
    # opponent Elo puts z near 0 immediately. A step clamp is kept as a
    # second line of defense in case of a similarly unlucky sample.
    r = float(np.mean(opp_elos)) if len(opp_elos) else 0.0
    for _ in range(200):
        grad, hess = grad_and_hess(r)
        if abs(grad) < 1e-9:
            break
        step = grad / hess
        step = max(-800.0, min(800.0, step))
        r -= step

    _, hess_final = grad_and_hess(r)
    variance = 1.0 / hess_final
    ci95 = Z_95 * math.sqrt(max(variance, 0.0))
    return r, ci95
