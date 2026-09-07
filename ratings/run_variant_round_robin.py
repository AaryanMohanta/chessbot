"""Round-robin local gauntlet among this session's flag variants -- each
"player" is the same agent.py, distinguished only by which CB_NB_ENABLE_*
environment variables are set for its subprocess (see cb_nb_search.py's
and cb_nb_fast.py's own flags, all default off). This is local self-play,
not the real ladder -- a fast first-look signal while real ladder rounds
(14/day) are spent more carefully, not a replacement for them.

Known limitation carried over from this session's SPRT work: numba-vs-
numba self-play under concurrent load sees an elevated
init_budget_exceeded rate (each side independently JIT-compiles, so
concurrency N here means up to 2N simultaneous compiles, worse than the
~15-18% observed at concurrency 7 for the asymmetric numba-vs-Stockfish
case). Those games are a harness artifact, not a real chess result --
excluded from the standings below, not counted as losses, and reported
separately so the exclusion rate itself stays visible.

Usage: python -m ratings.run_variant_round_robin [--games-per-pairing N] [--concurrency C]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.match import play_game  # noqa: E402
from ratings.book import OPENING_BOOK  # noqa: E402

AGENT_PATH = str(REPO_ROOT / "agent.py")

VARIANTS = {
    "baseline": {},
    "corrhist": {"CB_NB_ENABLE_CORRHIST": "1"},
    "conthist": {"CB_NB_ENABLE_CONTHIST": "1"},
    "eval_batch3": {"CB_NB_ENABLE_TEMPO": "1", "CB_NB_ENABLE_THREATS": "1", "CB_NB_ENABLE_IIR": "1"},
    "lmr_depth_scaling": {"CB_NB_ENABLE_LMR_DEPTH_SCALING": "1"},
}

DEFAULT_GAMES_PER_PAIRING = 10  # even, so it splits evenly across both colours
DEFAULT_TIME_MS = 10_000
DEFAULT_INCREMENT_MS = 100
DEFAULT_CONCURRENCY = 10

HARNESS_ARTIFACT_REASONS = ("white_init_budget_exceeded", "black_init_budget_exceeded")


def _plan_games(games_per_pairing: int) -> list[tuple[str, str, str, str]]:
    """[(white_name, black_name, opening_id, fen), ...] -- every unordered
    pair of variants, alternating which one is white across the pairing's
    own games, cycling through the opening book for variety."""
    plan = []
    names = sorted(VARIANTS)
    for a, b in itertools.combinations(names, 2):
        for i in range(games_per_pairing):
            opening_id, _moves, fen = OPENING_BOOK[i % len(OPENING_BOOK)]
            if i % 2 == 0:
                plan.append((a, b, opening_id, fen))
            else:
                plan.append((b, a, opening_id, fen))
    return plan


def _play_one(white_name, black_name, opening_id, fen, time_ms, increment_ms):
    """Returns (white_name, black_name, opening_id, result_or_none) --
    None means a harness-level crash (e.g. numba/LLVM erroring or one
    side never sending a ready signal under heavy concurrent compile
    load -- see ratings/sprt.py's own _play_pair for the same transient
    failure, there too caught and excluded rather than allowed to take
    down an entire run over one bad pairing)."""
    try:
        result = play_game(
            AGENT_PATH, AGENT_PATH,
            time_ms=time_ms, increment_ms=increment_ms, start_fen=fen,
            white_env=VARIANTS[white_name], black_env=VARIANTS[black_name],
        )
    except Exception as exc:
        print(f"  WARNING: {white_name} vs {black_name} ({opening_id}) crashed, excluding: {exc!r}", flush=True)
        return white_name, black_name, opening_id, None
    return white_name, black_name, opening_id, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games-per-pairing", type=int, default=DEFAULT_GAMES_PER_PAIRING)
    parser.add_argument("--time-ms", type=float, default=DEFAULT_TIME_MS)
    parser.add_argument("--increment-ms", type=float, default=DEFAULT_INCREMENT_MS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args()

    plan = _plan_games(args.games_per_pairing)
    print(f"Round robin: {len(VARIANTS)} variants, {len(plan)} games total, concurrency={args.concurrency}\n", flush=True)

    scores: dict[str, float] = {name: 0.0 for name in VARIANTS}
    games_played: dict[str, int] = {name: 0 for name in VARIANTS}
    head_to_head: dict[tuple[str, str], list[float]] = {}
    errors = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(_play_one, w, b, oid, fen, args.time_ms, args.increment_ms)
            for w, b, oid, fen in plan
        ]
        done = 0
        for future in as_completed(futures):
            white_name, black_name, opening_id, result = future.result()
            done += 1
            if result is None or result.reason in HARNESS_ARTIFACT_REASONS:
                errors += 1
                reason = "harness crash" if result is None else result.reason
                print(f"  [{done}/{len(plan)}] ERROR {white_name} vs {black_name}: {reason} (excluded)", flush=True)
                continue

            white_score = {"white": 1.0, "black": 0.0, None: 0.5}[result.winner]
            black_score = 1.0 - white_score
            scores[white_name] += white_score
            scores[black_name] += black_score
            games_played[white_name] += 1
            games_played[black_name] += 1

            key = tuple(sorted((white_name, black_name)))
            if key not in head_to_head:
                head_to_head[key] = [0.0, 0]
            h2h_score = white_score if key[0] == white_name else black_score
            head_to_head[key][0] += h2h_score
            head_to_head[key][1] += 1

            print(f"  [{done}/{len(plan)}] {white_name} vs {black_name}: "
                  f"{result.winner or 'draw'} ({result.reason}, {result.plies} plies)", flush=True)

    print("\n=== Standings (score rate across all games played) ===")
    ranked = sorted(VARIANTS, key=lambda n: -(scores[n] / max(games_played[n], 1)))
    for name in ranked:
        g = games_played[name]
        rate = scores[name] / g if g else 0.0
        print(f"  {name:20s} {scores[name]:.1f}/{g}  ({rate:.1%})")

    print("\n=== Head-to-head ===")
    for (a, b), (a_score, n) in sorted(head_to_head.items()):
        print(f"  {a} vs {b}: {a_score:.1f}-{n - a_score:.1f} ({n} games)")

    if errors:
        print(f"\n{errors}/{len(plan)} game(s) hit init_budget_exceeded under concurrency "
              f"and were excluded, not counted as losses (see this module's docstring).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
