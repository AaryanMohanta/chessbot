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

Standard bounds used elsewhere in this repo from here on: [0, 15] Elo for
a novel change, [-10, 0] (non-regression) for a change with a strong
literature prior. Narrower bounds need MORE games, not fewer — expected
sample size scales roughly as 1/(elo1-elo0)^2 — [0, 15] was widened from an
earlier [0, 10] for exactly that reason: precision we weren't acting on
was costing real wall-clock time we needed elsewhere.

Concurrency and CPU pinning (2026-09, revised): each pentanomial pair
runs its two games sequentially within one thread-pool slot (not
simultaneously), so one slot only ever has one game's two processes
(white+black) alive at a time. ``cpu_pin=True`` hard-pins each slot to
its own two cores instead of letting the OS scheduler place unpinned
processes freely -- this removes the main reason concurrency was kept
low: unpinned processes (especially numba's LLVM JIT, which can
transiently use extra threads) were observed to steal cycles from their
neighbors under high concurrency, a test-harness artifact, not real
engine weakness.

This module previously hard-pinned slots to naive consecutive core
indices (2*i, 2*i+1) and defaulted concurrency=7 with a comment claiming
"Linux/taskset only" pinning -- both wrong. harness.match.AgentProcess
pins on Windows too (via psutil), and naive pairing is actively harmful
on a hybrid P-core/E-core machine: see ratings/cpu_topology.py's module
docstring for the full story (this is the exact same bug that was found
independently in ratings/run_variant_round_robin.py). Both callers now
share ratings.cpu_topology.PREFERRED_CORE_PAIRS (same-class core pairs
only, capping safe concurrency at 5 on this dev machine, not the naive
7) and safe_concurrency(), which additionally caps concurrency against
*current* free RAM -- concurrent numba/LLVM compiles are memory-hungry
enough that a correctly-pinned, topology-safe concurrency can still fail
outright if background apps have eaten most of the machine's RAM, a
failure mode that's invisible to anything that only looks at cores.
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

from harness.match import DEFAULT_INCREMENT_MS, DEFAULT_TIME_MS, HARNESS_ARTIFACT_REASONS, play_game  # noqa: E402
from ratings.book import OPENING_BOOK  # noqa: E402
from ratings.cpu_topology import PREFERRED_CORE_PAIRS, safe_concurrency  # noqa: E402

MAX_RETRIES_PER_PAIR = 2  # a transient harness crash shouldn't just permanently discard that sample

FAST_TIME_MS = 5_000
FAST_INCREMENT_MS = 50


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


def _play_pair(candidate_path, baseline_path, fen, time_ms, increment_ms, candidate_env, baseline_env, cpu_a=None, cpu_b=None):
    """One pair: candidate plays both colours from the same opening.
    Returns the candidate's combined score across the two games (0..2).

    The pair's two games run sequentially (not concurrently) within
    whichever thread-pool slot called this, so the same two cores
    (``cpu_a``, ``cpu_b``) are reused for both -- one slot, two cores,
    regardless of which side is which colour in each game.

    Raises RuntimeError if either game's reason is a harness artifact
    (see harness.match.HARNESS_ARTIFACT_REASONS) -- a compile-budget
    timeout comes back as an ordinary GameResult, not an exception, so
    without this check it would silently score as a real loss for
    whichever side got starved. Found 2026-09 while auditing this
    function against the same class of bug already fixed in
    ratings/run_variant_round_robin.py: this path had NO check at all,
    so every prior SPRT run using pentanomial pairing under any
    concurrency was vulnerable to exactly this contamination."""
    r1 = play_game(candidate_path, baseline_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, white_env=candidate_env, black_env=baseline_env, white_cpu=cpu_a, black_cpu=cpu_b)
    if r1.reason in HARNESS_ARTIFACT_REASONS:
        raise RuntimeError(f"harness artifact (game 1): {r1.reason}")
    s1 = {"white": 1.0, "black": 0.0, None: 0.5}[r1.winner]

    r2 = play_game(baseline_path, candidate_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, white_env=baseline_env, black_env=candidate_env, white_cpu=cpu_a, black_cpu=cpu_b)
    if r2.reason in HARNESS_ARTIFACT_REASONS:
        raise RuntimeError(f"harness artifact (game 2): {r2.reason}")
    s2 = {"black": 1.0, "white": 0.0, None: 0.5}[r2.winner]

    return s1 + s2, (r1.reason, r2.reason)


def _play_pair_with_retries(candidate_path, baseline_path, fen, time_ms, increment_ms, candidate_env, baseline_env, cpu_a=None, cpu_b=None):
    """Wraps _play_pair with bounded retries -- a transient harness
    crash (memory pressure, an unlucky compile under contention)
    shouldn't just permanently discard that sample. Re-raises the last
    exception once retries are exhausted, matching _play_pair's own
    contract so callers don't need to know retries happened."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES_PER_PAIR + 2):
        try:
            return _play_pair(candidate_path, baseline_path, fen, time_ms, increment_ms, candidate_env, baseline_env, cpu_a, cpu_b)
        except Exception as exc:
            last_exc = exc
            if attempt <= MAX_RETRIES_PER_PAIR:
                print(f"  RETRY {attempt}/{MAX_RETRIES_PER_PAIR}: pair for opening at {fen[:20]}... after {exc!r}", flush=True)
    raise last_exc


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
    concurrency: int = 5,  # this dev box's calibrated ceiling; see ratings/cpu_topology.py
    openings: list[tuple[str, str]] | None = None,
    candidate_env: dict | None = None,
    baseline_env: dict | None = None,
    cpu_pin: bool = True,
) -> dict:
    """GSPRT on paired (same-opening, reversed-colour) game scores.

    ``openings``: [(opening_id, fen), ...] to draw from, cycling; defaults
    to the built-in 51-line book. Pass ratings.pgn_book.load_openings(...)
    for a much larger, more diverse set.

    Runs ``concurrency`` pairs at a time (a batch), checking the stopping
    rule after each batch completes — a pair in flight when the test would
    conclude is allowed to finish rather than cancelled, so the final game
    count can overshoot the crossing point by up to concurrency-1 pairs.

    ``cpu_pin``: hard-pin each concurrent slot to its own 2 same-class
    cores (ratings.cpu_topology.PREFERRED_CORE_PAIRS) instead of naive
    consecutive indices. ``concurrency`` is re-derived via
    ratings.cpu_topology.safe_concurrency() regardless of what's passed
    in, capping it against both this machine's known-safe core pairs and
    its *current* free RAM (checked fresh every call, since it changes
    independently of this process) -- pass a lower number to request
    less, never a way to force more than what's currently safe.
    """
    if openings is None:
        openings = [(oid, fen) for oid, _moves, fen in OPENING_BOOK]

    if cpu_pin:
        concurrency, cpu_pin = safe_concurrency(concurrency)
    else:
        concurrency = max(1, concurrency)

    p0, p1 = _elo_to_p(elo0), _elo_to_p(elo1)
    mu0, mu1 = 2 * p0, 2 * p1  # expected *pair* score under each hypothesis
    lower, upper = llr_bounds(alpha, beta)

    lock = threading.Lock()
    pair_scores: list[float] = []
    flag_counts = {"candidate": 0, "baseline": 0}
    games_played = 0
    errored_pairs = 0

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
                cpu_a, cpu_b = PREFERRED_CORE_PAIRS[i] if cpu_pin else (None, None)
                futures.append((opening_id, pool.submit(
                    _play_pair_with_retries, candidate_path, baseline_path, fen, time_ms, increment_ms,
                    candidate_env, baseline_env, cpu_a, cpu_b,
                )))

            for opening_id, future in futures:
                try:
                    pair_score, reasons = future.result()
                except Exception as exc:
                    # A transient failure (observed cause: numba/LLVM
                    # compilation occasionally erroring under heavy
                    # concurrent load -- up to 2x concurrency simultaneous
                    # compiles when both candidate and baseline are numba
                    # agents, vs. 1x when one side is a plain-Python
                    # baseline) must not take down an entire overnight
                    # SPRT run over one bad pair. Skipped from
                    # pair_scores/flag_counts (it carries no chess
                    # information either way) but still counted against
                    # pairs_done so a persistently-crashing setup can't
                    # spin forever without ever hitting max_pairs.
                    with lock:
                        errored_pairs += 1
                    print(f"  WARNING: pair for opening {opening_id!r} errored, skipping: {exc!r}", flush=True)
                    pairs_done += 1
                    continue
                # pair_score in {0, 0.5, 1, 1.5, 2}. Not decomposed into a
                # W/D/L breakdown here: a pair scoring 1.0 could be either
                # two draws or one win and one loss, and the two aren't
                # equivalent at the individual-game level — the pentanomial
                # bucket itself (this pair_score) is the meaningful unit,
                # not a derived per-game count.
                # reasons = (r1.reason, r2.reason): r1 has candidate as
                # white/baseline as black, r2 is the reversed colours --
                # a permanent tracked metric (see ratings/
                # report_containerized_calibration.py's flag_rate_report),
                # not a one-off diagnostic. Should always be zero; the
                # bug that motivated tracking this was invisible to every
                # existing unit test, only a real per-game flag rate
                # surfaced it.
                r1_reason, r2_reason = reasons
                side_flags = {"candidate": 0, "baseline": 0}
                if r1_reason == "white_flag":
                    side_flags["candidate"] += 1
                elif r1_reason == "black_flag":
                    side_flags["baseline"] += 1
                if r2_reason == "white_flag":
                    side_flags["baseline"] += 1
                elif r2_reason == "black_flag":
                    side_flags["candidate"] += 1
                with lock:
                    pair_scores.append(pair_score)
                    flag_counts["candidate"] += side_flags["candidate"]
                    flag_counts["baseline"] += side_flags["baseline"]
                pairs_done += 1

            current_llr = llr_from_pairs()
            n_pairs = len(pair_scores)
            games_played = n_pairs * 2
            mean_score = sum(pair_scores) / n_pairs if n_pairs else 0.0
            print(
                f"pair {n_pairs}/{max_pairs} (batch of {batch}, {errored_pairs} errored so far): "
                f"mean_pair_score={mean_score:.3f}/2  "
                f"LLR={current_llr:+.3f}  bounds=[{lower:.3f}, {upper:.3f}]  "
                f"flags: candidate={flag_counts['candidate']}/{games_played} "
                f"({100*flag_counts['candidate']/games_played if games_played else 0:.1f}%), "
                f"baseline={flag_counts['baseline']}/{games_played} "
                f"({100*flag_counts['baseline']/games_played if games_played else 0:.1f}%)",
                flush=True,
            )

            if current_llr >= upper:
                return {"decision": "H1", "reason": f"candidate is >= {elo1} Elo stronger", "pairs": n_pairs, "games": n_pairs * 2, "llr": current_llr, "mean_pair_score": mean_score, "flag_counts": dict(flag_counts), "errored_pairs": errored_pairs}
            if current_llr <= lower:
                return {"decision": "H0", "reason": f"candidate is not >= {elo0} Elo stronger", "pairs": n_pairs, "games": n_pairs * 2, "llr": current_llr, "mean_pair_score": mean_score, "flag_counts": dict(flag_counts), "errored_pairs": errored_pairs}

    final_llr = llr_from_pairs()
    final_n = len(pair_scores)
    return {"decision": "inconclusive", "reason": f"hit max_pairs={max_pairs} without crossing a bound", "pairs": final_n, "games": final_n * 2, "llr": final_llr, "mean_pair_score": (sum(pair_scores) / final_n if final_n else 0.0), "flag_counts": dict(flag_counts), "errored_pairs": errored_pairs}


def main() -> int:
    parser = argparse.ArgumentParser(description="SPRT: is candidate stronger than baseline?")
    parser.add_argument("candidate_path")
    parser.add_argument("baseline_path")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=15.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--time-ms", type=float, default=FAST_TIME_MS)
    parser.add_argument("--increment-ms", type=float, default=FAST_INCREMENT_MS)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--no-cpu-pin", action="store_true", help="disable core pinning (use on a small box, or where concurrency*2 exceeds available cores)")
    parser.add_argument("--simple", action="store_true", help="use the older per-game Bernoulli SPRT instead of pentanomial pairing")
    parser.add_argument("--candidate-env", action="append", default=[], metavar="KEY=VALUE",
                         help="env var to set for the candidate process only (repeatable) -- lets "
                              "candidate_path and baseline_path be the SAME agent.py isolating one "
                              "flag, e.g. --candidate-env CB_NB_ENABLE_CORRHIST=1")
    parser.add_argument("--baseline-env", action="append", default=[], metavar="KEY=VALUE",
                         help="env var to set for the baseline process only (repeatable)")
    args = parser.parse_args()

    def _parse_env(pairs: list[str]) -> dict:
        env = {}
        for pair in pairs:
            key, _, value = pair.partition("=")
            if not _:
                raise SystemExit(f"--candidate-env/--baseline-env expects KEY=VALUE, got: {pair!r}")
            env[key] = value
        return env

    candidate_env = _parse_env(args.candidate_env)
    baseline_env = _parse_env(args.baseline_env)

    if args.simple:
        outcome = run_sprt(args.candidate_path, args.baseline_path, args.elo0, args.elo1, alpha=args.alpha, beta=args.beta, max_games=args.max_pairs, time_ms=args.time_ms, increment_ms=args.increment_ms, candidate_env=candidate_env, baseline_env=baseline_env)
    else:
        outcome = run_sprt_pentanomial(args.candidate_path, args.baseline_path, args.elo0, args.elo1, alpha=args.alpha, beta=args.beta, max_pairs=args.max_pairs, time_ms=args.time_ms, increment_ms=args.increment_ms, concurrency=args.concurrency, cpu_pin=not args.no_cpu_pin, candidate_env=candidate_env, baseline_env=baseline_env)

    print()
    print(f"DECISION: {outcome['decision']} - {outcome['reason']} after {outcome['games']} games (LLR={outcome['llr']:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
