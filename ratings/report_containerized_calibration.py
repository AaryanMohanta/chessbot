"""Fits and reports the joint MLE rating from a
ratings/run_containerized_calibration.py-style results file (one JSON
object per line: elo, opening_id, agent_is_white, score, reason, plies,
elapsed_s).

Games that never reached a real chess outcome (orchestrator timeout/
error, a container crash, malformed output -- anything with a "reason"
that isn't a genuine game-ending state) are excluded from the rating fit
entirely rather than counted as a 0.5: they carry no information about
playing strength, and including them would silently bias the estimate
toward "more draws" for no chess reason. They're still reported as a
separate count so a large number of them is visible rather than quietly
absorbed into the score.

Also reports flag rate per side -- a permanent tracked metric alongside
nps/median depth (see ratings/benchmark_nb.py), not just a one-off
diagnostic. It should always be 0%; a calibration or SPRT run showing
anything above that means something regressed (this is exactly the
metric that would have surfaced the real-clock time-management bug
before it needed a manual audit to find -- see cb_nb_search.py's
search() docstring and tests/test_nb_real_clock.py).

Usage: python -m ratings.report_containerized_calibration <results.jsonl> [--skip-benchmark]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ratings.elo import fit_rating_vs_known_opponents  # noqa: E402

_NON_GAME_REASON_PREFIXES = ("orchestrator_timeout", "orchestrator_error", "container_error", "bad_output")


def flag_rate_report(results: list[dict]) -> str:
    """Per-side flag rate over every game that reached a real outcome
    (excludes orchestrator/container issues -- those aren't a chess flag
    either way). Should always read 0.0% / 0.0%."""
    our_games = our_flags = opp_games = opp_flags = 0
    for r in results:
        if r["reason"].startswith(_NON_GAME_REASON_PREFIXES):
            continue
        our_side = "white" if r["agent_is_white"] else "black"
        opp_side = "black" if r["agent_is_white"] else "white"
        our_games += 1
        opp_games += 1
        if r["reason"] == f"{our_side}_flag":
            our_flags += 1
        elif r["reason"] == f"{opp_side}_flag":
            opp_flags += 1
    our_pct = 100 * our_flags / our_games if our_games else 0.0
    opp_pct = 100 * opp_flags / opp_games if opp_games else 0.0
    return (
        f"Flag rate -- our agent: {our_flags}/{our_games} ({our_pct:.1f}%), "
        f"opponent: {opp_flags}/{opp_games} ({opp_pct:.1f}%)"
    )


def load_results(path: Path) -> list[dict]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_path")
    parser.add_argument("--skip-benchmark", action="store_true", help="skip the median middlegame depth/nps report")
    args = parser.parse_args()

    results = load_results(Path(args.results_path))

    by_level: dict[int, list[dict]] = defaultdict(list)
    excluded = []
    for r in results:
        if r["reason"].startswith(_NON_GAME_REASON_PREFIXES):
            excluded.append(r)
        else:
            by_level[r["elo"]].append(r)

    if excluded:
        print(f"WARNING: excluding {len(excluded)} non-game results from the fit (orchestrator/container issues, not chess outcomes):")
        for r in excluded[:10]:
            print(f"  elo={r['elo']} opening={r['opening_id']} reason={r['reason'][:120]!r}")
        if len(excluded) > 10:
            print(f"  ... and {len(excluded) - 10} more")
        print()

    observations = []
    print(f"{'='*70}\nRESULTS\n{'='*70}")
    for elo in sorted(by_level):
        level = by_level[elo]
        wins = sum(1 for r in level if r["score"] == 1.0)
        draws = sum(1 for r in level if r["score"] == 0.5)
        losses = sum(1 for r in level if r["score"] == 0.0)
        score_pct = sum(r["score"] for r in level) / len(level) * 100
        avg_elapsed = sum(r["elapsed_s"] for r in level) / len(level)
        print(f"  UCI_Elo {elo:5d}: {wins:3d}-{draws:3d}-{losses:3d}  ({score_pct:5.1f}% score, {len(level)} games, avg {avg_elapsed:.0f}s/game)")
        observations.extend((elo, r["score"]) for r in level)

    print(f"\n{flag_rate_report(results)}")
    if not args.skip_benchmark:
        from ratings.benchmark_nb import median_middlegame_depth_and_nps_nb

        bench = median_middlegame_depth_and_nps_nb()
        print(f"Median depth (middlegame only): {bench['median_depth']}  (range {bench['min_depth']}-{bench['max_depth']})")
        print(f"Median nps:                      {bench['median_nps']:,.0f}")

    if len(observations) < 2:
        print("\nNot enough completed games yet for a fit.")
        return 0

    rating, ci95 = fit_rating_vs_known_opponents(observations)
    print(f"\nFitted rating (joint MLE across {len(observations)} games): {rating:.0f} +/- {ci95:.0f} (95% CI)")
    print("NOTE: this is Stockfish's own UCI_Elo scale, not FIDE/CCRL -- treat it as an internal yardstick.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
