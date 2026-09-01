"""SPRT: sequential test for "is candidate stronger than baseline", the
tool for day-to-day A/B decisions (the full joint rating fit in elo.py is
for tracking the whole pool over time — different purpose, different tool).

Two modes:

``run_sprt`` — the original per-game Bernoulli model. Each game's score s
in {0, 0.5, 1} is treated as a Bernoulli-like observation with mean p,
p0 = P(win | elo0), p1 = P(win | elo1) from the standard logistic Elo
model. Simple, but wastes information: two games from the same opening
with reversed colours are correlated (an opening biased toward one side
inflates or deflates both), and treating them as independent overstates
the sample's real information content.

``run_sprt_pentanomial`` — pairs games from the same opening with reversed
colours and tests on the PAIR's combined score (0, 0.5, 1, 1.5, or 2 —
five buckets, hence "pentanomial"). This is the standard variance-reduction
trick used by engine testers generally (not tied to any particular
external tool): pairing cancels most of the opening-driven correlation,
typically cutting required games by 15-20% for the same statistical power.
Implemented as a Generalized SPRT (GSPRT) on the paired-score sample mean
against its own running empirical variance — the same test structure
fishtest uses for pentanomial LLR, expressed directly rather than adopted
from a specific tool (our agent speaks a custom wire protocol, not UCI, so
UCI-only tools like cutechess-cli/fastchess can't drive it regardless).

Standard bounds used elsewhere in this repo from here on: [0, 10] Elo for
a novel change, [-10, 0] (non-regression) for a change with a strong
literature prior. Narrower bounds need MORE games, not fewer — expected
sample size scales roughly as 1/(elo1-elo0)^2 — so don't reach for [0, 5]
just because it "sounds more rigorous"; it costs ~16x the games of [0, 20]
for the same power.
"""
from __future__ import annotations

import argparse
import math
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.match import DEFAULT_INCREMENT_MS, DEFAULT_TIME_MS, play_game  # noqa: E402
from ratings.book import OPENING_BOOK  # noqa: E402

FAST_TIME_MS = 10_000
FAST_INCREMENT_MS = 100


def _elo_to_p(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def llr_bounds(alpha: float, beta: float) -> tuple[float, float]:
    lower = math.log(beta / (1 - alpha))
    upper = math.log((1 - beta) / alpha)
    return lower, upper


def llr_increment(score: float, elo0: float, elo1: float) -> float:
    p0, p1 = _elo_to_p(elo0), _elo_to_p(elo1)
    return score * math.log(p1 / p0) + (1 - score) * math.log((1 - p1) / (1 - p0))


def run_sprt(
    candidate_path: str,
    baseline_path: str,
    elo0: float,
    elo1: float,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_games: int = 6000,
    time_ms: float = FAST_TIME_MS,
    increment_ms: float = FAST_INCREMENT_MS,
    candidate_env: dict | None = None,
    baseline_env: dict | None = None,
) -> dict:
    """Per-game Bernoulli SPRT. ``candidate_env``/``baseline_env`` let
    candidate and baseline be the *same* agent.py with different
    environment variables — e.g. isolating one search feature
    (CB_ENABLE_NULL_MOVE, CB_ENABLE_LMR) at a time via
    harness.match.play_game's white_env/black_env."""
    lower, upper = llr_bounds(alpha, beta)
    llr = 0.0
    wins = draws = losses = 0

    for i in range(max_games):
        opening_id, _moves, fen = OPENING_BOOK[i % len(OPENING_BOOK)]
        candidate_is_white = i % 2 == 0

        if candidate_is_white:
            result = play_game(candidate_path, baseline_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, white_env=candidate_env, black_env=baseline_env)
            candidate_score = {"white": 1.0, "black": 0.0, None: 0.5}[result.winner]
        else:
            result = play_game(baseline_path, candidate_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, white_env=baseline_env, black_env=candidate_env)
            candidate_score = {"black": 1.0, "white": 0.0, None: 0.5}[result.winner]

        if candidate_score == 1.0:
            wins += 1
        elif candidate_score == 0.0:
            losses += 1
        else:
            draws += 1

        llr += llr_increment(candidate_score, elo0, elo1)
        n = i + 1
        print(
            f"game {n}: score={candidate_score} ({opening_id}, {result.reason}) "
            f"W{wins}/D{draws}/L{losses}  LLR={llr:+.3f}  bounds=[{lower:.3f}, {upper:.3f}]",
            flush=True,
        )

        if llr >= upper:
            return {"decision": "H1", "reason": f"candidate is >= {elo1} Elo stronger", "games": n, "llr": llr, "wins": wins, "draws": draws, "losses": losses}
        if llr <= lower:
            return {"decision": "H0", "reason": f"candidate is not >= {elo0} Elo stronger", "games": n, "llr": llr, "wins": wins, "draws": draws, "losses": losses}

    return {"decision": "inconclusive", "reason": f"hit max_games={max_games} without crossing a bound", "games": max_games, "llr": llr, "wins": wins, "draws": draws, "losses": losses}


def _play_pair(candidate_path, baseline_path, fen, time_ms, increment_ms, candidate_env, baseline_env):
    """One pair: candidate plays both colours from the same opening.
    Returns the candidate's combined score across the two games (0..2)."""
    r1 = play_game(candidate_path, baseline_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, white_env=candidate_env, black_env=baseline_env)
    s1 = {"white": 1.0, "black": 0.0, None: 0.5}[r1.winner]

    r2 = play_game(baseline_path, candidate_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, white_env=baseline_env, black_env=candidate_env)
    s2 = {"black": 1.0, "white": 0.0, None: 0.5}[r2.winner]

    return s1 + s2, (r1.reason, r2.reason)


def run_sprt_pentanomial(
    candidate_path: str,
    baseline_path: str,
    elo0: float,
    elo1: float,
    alpha: float = 0.05,
    beta: float = 0.05,
    max_pairs: int = 3000,
    time_ms: float = FAST_TIME_MS,
    increment_ms: float = FAST_INCREMENT_MS,
    concurrency: int = 8,
    openings: list[tuple[str, str]] | None = None,
    candidate_env: dict | None = None,
    baseline_env: dict | None = None,
) -> dict:
    """GSPRT on paired (same-opening, reversed-colour) game scores.

    ``openings``: [(opening_id, fen), ...] to draw from, cycling; defaults
    to the built-in 51-line book. Pass ratings.pgn_book.load_openings(...)
    for a much larger, more diverse set.

    Runs ``concurrency`` pairs at a time (a batch), checking the stopping
    rule after each batch completes — a pair in flight when the test would
    conclude is allowed to finish rather than cancelled, so the final game
    count can overshoot the crossing point by up to concurrency-1 pairs.
    """
    if openings is None:
        openings = [(oid, fen) for oid, _moves, fen in OPENING_BOOK]

    p0, p1 = _elo_to_p(elo0), _elo_to_p(elo1)
    mu0, mu1 = 2 * p0, 2 * p1  # expected *pair* score under each hypothesis
    lower, upper = llr_bounds(alpha, beta)

    lock = threading.Lock()
    pair_scores: list[float] = []

    def llr_from_pairs() -> float:
        n = len(pair_scores)
        if n < 2:
            return 0.0
        mean = sum(pair_scores) / n
        variance = sum((x - mean) ** 2 for x in pair_scores) / (n - 1)
        variance = max(variance, 1e-6)  # floor: avoid a division blowup on a near-zero-variance run of luck
        # GSPRT: log-likelihood ratio for a normal sample's mean under two
        # hypothesized means with a shared (here: empirically estimated)
        # variance.
        return n * (mu1 - mu0) * (mean - (mu0 + mu1) / 2) / variance

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        pairs_done = 0
        while pairs_done < max_pairs:
            batch = min(concurrency, max_pairs - pairs_done)
            futures = []
            for i in range(batch):
                opening_id, fen = openings[(pairs_done + i) % len(openings)]
                futures.append((opening_id, pool.submit(_play_pair, candidate_path, baseline_path, fen, time_ms, increment_ms, candidate_env, baseline_env)))

            for opening_id, future in futures:
                pair_score, reasons = future.result()
                # pair_score in {0, 0.5, 1, 1.5, 2}. Not decomposed into a
                # W/D/L breakdown here: a pair scoring 1.0 could be either
                # two draws or one win and one loss, and the two aren't
                # equivalent at the individual-game level — the pentanomial
                # bucket itself (this pair_score) is the meaningful unit,
                # not a derived per-game count.
                with lock:
                    pair_scores.append(pair_score)
                pairs_done += 1

            current_llr = llr_from_pairs()
            n_pairs = len(pair_scores)
            mean_score = sum(pair_scores) / n_pairs
            print(
                f"pair {n_pairs}/{max_pairs} (batch of {batch}): mean_pair_score={mean_score:.3f}/2  "
                f"LLR={current_llr:+.3f}  bounds=[{lower:.3f}, {upper:.3f}]",
                flush=True,
            )

            if current_llr >= upper:
                return {"decision": "H1", "reason": f"candidate is >= {elo1} Elo stronger", "pairs": n_pairs, "games": n_pairs * 2, "llr": current_llr, "mean_pair_score": mean_score}
            if current_llr <= lower:
                return {"decision": "H0", "reason": f"candidate is not >= {elo0} Elo stronger", "pairs": n_pairs, "games": n_pairs * 2, "llr": current_llr, "mean_pair_score": mean_score}

    final_llr = llr_from_pairs()
    return {"decision": "inconclusive", "reason": f"hit max_pairs={max_pairs} without crossing a bound", "pairs": len(pair_scores), "games": len(pair_scores) * 2, "llr": final_llr, "mean_pair_score": sum(pair_scores) / len(pair_scores)}


def main() -> int:
    parser = argparse.ArgumentParser(description="SPRT: is candidate stronger than baseline?")
    parser.add_argument("candidate_path")
    parser.add_argument("baseline_path")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--time-ms", type=float, default=FAST_TIME_MS)
    parser.add_argument("--increment-ms", type=float, default=FAST_INCREMENT_MS)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--simple", action="store_true", help="use the older per-game Bernoulli SPRT instead of pentanomial pairing")
    args = parser.parse_args()

    if args.simple:
        outcome = run_sprt(args.candidate_path, args.baseline_path, args.elo0, args.elo1, alpha=args.alpha, beta=args.beta, max_games=args.max_pairs, time_ms=args.time_ms, increment_ms=args.increment_ms)
    else:
        outcome = run_sprt_pentanomial(args.candidate_path, args.baseline_path, args.elo0, args.elo1, alpha=args.alpha, beta=args.beta, max_pairs=args.max_pairs, time_ms=args.time_ms, increment_ms=args.increment_ms, concurrency=args.concurrency)

    print()
    print(f"DECISION: {outcome['decision']} - {outcome['reason']} after {outcome['games']} games (LLR={outcome['llr']:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
