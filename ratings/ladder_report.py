"""CLI: summarize ratings/ladder_games.csv (see ratings/ingest_ladder_logs.py)
grouped by build_tag -- the actual payoff of item 1 ("use the ladder as
our test harness"): after playing two different build/config variants on
the real ladder for a while (LMR depth-scaling being the first named
candidate -- 209 local SPRT pairs never resolved it), run this to see
whether the ladder's real, cross-opponent results actually favour one of
them, rather than relying only on local self-play SPRT.

Usage: python -m ratings.ladder_report

Per build_tag: game count, win/draw/loss, score rate, and -- whenever
enough games in that group carry a numeric opponent Elo (from the PGN's
WhiteElo/BlackElo header) -- a performance rating + 95% CI via the same
fixed-known-opponent MLE ratings/calibrate_stockfish.py already uses
(ratings.elo.fit_rating_vs_known_opponents), since ladder opponents vary
game to game and a plain score rate alone doesn't account for facing a
harder or easier mix under one tag than the other. Also reports each
group's median-depth average and v1-fallback rate as an engine-health
sanity check -- a tag that's "winning" only because it happened to draw
easier opponents, or that's quietly falling back to v1 more often, is
worth knowing about before trusting the score-rate number alone.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ratings.elo import fit_rating_vs_known_opponents  # noqa: E402
from ratings.ladder_db import DB_PATH, LadderGameRecord, load_games  # noqa: E402

MIN_GAMES_FOR_RATING = 10


def _group_by_tag(games: list[LadderGameRecord]) -> dict[str, list[LadderGameRecord]]:
    groups: dict[str, list[LadderGameRecord]] = {}
    for g in games:
        groups.setdefault(g.build_tag, []).append(g)
    return groups


def summarize_tag(tag: str, games: list[LadderGameRecord]) -> dict:
    wins = sum(1 for g in games if g.score_for_us == 1.0)
    draws = sum(1 for g in games if g.score_for_us == 0.5)
    losses = sum(1 for g in games if g.score_for_us == 0.0)
    score_rate = sum(g.score_for_us for g in games) / len(games) if games else 0.0

    depths = [g.median_depth for g in games if g.median_depth is not None]
    v1_rate = sum(1 for g in games if g.v1_ever_played) / len(games) if games else 0.0
    collapse_total = sum(g.depth_collapse_count for g in games)

    observations = [
        (float(g.opponent_elo), g.score_for_us)
        for g in games if g.opponent_elo not in ("", None)
    ]
    rating = ci95 = None
    if len(observations) >= MIN_GAMES_FOR_RATING:
        rating, ci95 = fit_rating_vs_known_opponents(observations)

    return {
        "tag": tag,
        "games": len(games),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score_rate": score_rate,
        "avg_median_depth": statistics.mean(depths) if depths else None,
        "v1_ever_played_rate": v1_rate,
        "depth_collapse_total": collapse_total,
        "rating": rating,
        "rating_ci95": ci95,
        "rated_games": len(observations),
    }


def print_report(db_path: Path = DB_PATH) -> None:
    games = load_games(db_path)
    if not games:
        print(f"No ladder games recorded yet in {db_path} -- see ratings/ingest_ladder_logs.py.")
        return

    groups = _group_by_tag(games)
    summaries = [summarize_tag(tag, gs) for tag, gs in groups.items()]
    summaries.sort(key=lambda s: s["score_rate"], reverse=True)

    print(f"Ladder games by build_tag ({len(games)} total, {len(groups)} tag(s)):\n")
    for s in summaries:
        print(f"  {s['tag']}")
        print(f"    games={s['games']}  W{s['wins']}/D{s['draws']}/L{s['losses']}  "
              f"score_rate={s['score_rate']:.3f}")
        if s["rating"] is not None:
            print(f"    performance_rating={s['rating']:.0f} +/- {s['rating_ci95']:.0f} "
                  f"(from {s['rated_games']} games with known opponent Elo)")
        else:
            print(f"    performance_rating=n/a (only {s['rated_games']}/{MIN_GAMES_FOR_RATING} "
                  f"games have a known opponent Elo)")
        depth_str = f"{s['avg_median_depth']:.1f}" if s["avg_median_depth"] is not None else "n/a"
        print(f"    avg_median_depth={depth_str}  v1_ever_played_rate={s['v1_ever_played_rate']:.2f}  "
              f"depth_collapses_total={s['depth_collapse_total']}")
        print()

    if len(summaries) >= 2 and all(s["rating"] is not None for s in summaries[:2]):
        a, b = summaries[0], summaries[1]
        diff = a["rating"] - b["rating"]
        combined_ci = (a["rating_ci95"] + b["rating_ci95"])  # conservative, not a proper pooled CI
        note = "likely real" if abs(diff) > combined_ci else "within noise at this sample size"
        print(f"Top two by score rate: {a['tag']} vs {b['tag']}, rating diff {diff:+.0f} -- {note}.")


if __name__ == "__main__":
    print_report()
