"""Saved command: calibrate an agent's real Elo against a real Stockfish
binary, capped via UCI_LimitStrength + UCI_Elo, at several fixed levels.

Re-run this after every significant engine change — same command, same
defaults — to get a calibrated number at each milestone rather than a
one-off measurement:

    python -m ratings.calibrate_stockfish agent.py --stockfish PATH

IMPORTANT: Stockfish's UCI_Elo is its own internal scale, calibrated by
the Stockfish team against their own test conditions — it does not map
exactly onto FIDE or CCRL ratings. Treat every number this script prints
as a consistent *internal yardstick* for comparing our own versions over
time, not as a claim about human rating or CCRL placement.

Defaults match the standard recipe: UCI_Elo in {1320, 1500, 1700, 1900},
200 games/level, 10+0.1 (fast — nobody SPRTs or calibrates at the
tournament time control; 120+0.5 is reserved for final validation and the
wire-protocol smoke test), concurrency 8, alternating colours, real
opening variety from the 8moves_v3.pgn book (falls back to the small
built-in book if the download is unavailable), Stockfish pinned to 1
thread / 16MB hash so it isn't getting a hardware edge over our
single-core, low-memory-footprint engine.

The four levels' games are pooled into a single joint MLE fit (our rating
is the one unknown; each Stockfish level's UCI_Elo is treated as a fixed,
known anchor) rather than eyeballing where two adjacent levels cross 50%
— see ratings.elo.fit_rating_vs_known_opponents.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.match import play_game  # noqa: E402
from ratings.book import OPENING_BOOK  # noqa: E402
from ratings.elo import fit_rating_vs_known_opponents  # noqa: E402

DEFAULT_ELO_LADDER = [1320, 1500, 1700, 1900]
DEFAULT_GAMES_PER_LEVEL = 200
DEFAULT_TIME_MS = 10_000
DEFAULT_INCREMENT_MS = 100
DEFAULT_CONCURRENCY = 8
DEFAULT_THREADS = 1
DEFAULT_HASH_MB = 16

STOCKFISH_AGENT = str(REPO_ROOT / "baselines" / "stockfish_agent.py")


def _load_openings(limit: int) -> list[tuple[str, str]]:
    try:
        from ratings.pgn_book import load_openings

        openings = load_openings(limit=limit)
        print(f"Using {len(openings)} openings from 8moves_v3.pgn", flush=True)
        return openings
    except Exception as exc:
        print(f"Could not load 8moves_v3.pgn ({exc}); falling back to the built-in 51-line book", flush=True)
        return [(oid, fen) for oid, _moves, fen in OPENING_BOOK]


def _play_one(agent_path, stockfish_path, fen, elo, agent_is_white, time_ms, increment_ms, threads, hash_mb):
    env = {
        "STOCKFISH_PATH": stockfish_path,
        "STOCKFISH_ELO": str(elo),
        "STOCKFISH_THREADS": str(threads),
        "STOCKFISH_HASH_MB": str(hash_mb),
    }
    if agent_is_white:
        result = play_game(agent_path, STOCKFISH_AGENT, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, black_env=env)
        our_score = {"white": 1.0, "black": 0.0, None: 0.5}[result.winner]
    else:
        result = play_game(STOCKFISH_AGENT, agent_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, white_env=env)
        our_score = {"black": 1.0, "white": 0.0, None: 0.5}[result.winner]
    return our_score, result.reason, result.plies


def run_level(agent_path: str, stockfish_path: str, elo: float, games: int, openings: list[tuple[str, str]], time_ms: float, increment_ms: float, concurrency: int, threads: int, hash_mb: int) -> dict:
    jobs = []
    for i in range(games):
        opening_id, fen = openings[i % len(openings)]
        jobs.append((opening_id, fen, i % 2 == 0))  # alternate colours

    scores = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(_play_one, agent_path, stockfish_path, fen, elo, is_white, time_ms, increment_ms, threads, hash_mb): (opening_id, is_white)
            for opening_id, fen, is_white in jobs
        }
        done = 0
        for future in as_completed(futures):
            score, reason, plies = future.result()
            scores.append(score)
            done += 1
            if done % 20 == 0 or done == games:
                print(f"  [UCI_Elo={elo}] {done}/{games} done, running score {sum(scores)}/{done} ({sum(scores)/done*100:.1f}%)", flush=True)

    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    losses = sum(1 for s in scores if s == 0.0)
    return {"elo": elo, "games": games, "wins": wins, "draws": draws, "losses": losses, "scores": scores, "score_pct": sum(scores) / games * 100}


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate agent Elo against a real Stockfish binary (saved command — re-run after every milestone).")
    parser.add_argument("agent_path")
    parser.add_argument("--stockfish", required=True, help="path to the Stockfish binary")
    parser.add_argument("--elo-ladder", type=int, nargs="+", default=DEFAULT_ELO_LADDER)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES_PER_LEVEL, help="games per level (default 200)")
    parser.add_argument("--time-ms", type=float, default=DEFAULT_TIME_MS)
    parser.add_argument("--increment-ms", type=float, default=DEFAULT_INCREMENT_MS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--hash-mb", type=int, default=DEFAULT_HASH_MB)
    parser.add_argument("--openings-limit", type=int, default=2000, help="how many distinct PGN-book openings to draw from")
    parser.add_argument("--skip-benchmark", action="store_true", help="skip the median middlegame depth/nps report")
    args = parser.parse_args()

    openings = _load_openings(args.openings_limit)

    print(f"\n{'='*70}\nSTOCKFISH CALIBRATION — {args.agent_path}\n"
          f"levels={args.elo_ladder}  games/level={args.games}  tc={args.time_ms}+{args.increment_ms}ms  "
          f"concurrency={args.concurrency}  stockfish: {args.threads} thread(s), {args.hash_mb}MB hash\n{'='*70}\n", flush=True)

    level_results = []
    observations = []  # (opponent_elo, our_score) pairs, pooled across all levels
    for elo in args.elo_ladder:
        print(f"=== UCI_Elo = {elo} ===", flush=True)
        level = run_level(args.agent_path, args.stockfish, elo, args.games, openings, args.time_ms, args.increment_ms, args.concurrency, args.threads, args.hash_mb)
        level_results.append(level)
        observations.extend((elo, s) for s in level["scores"])
        print(f"  -> {level['wins']}-{level['draws']}-{level['losses']} ({level['score_pct']:.1f}% score)\n", flush=True)

    rating, ci95 = fit_rating_vs_known_opponents(observations)

    print(f"{'='*70}\nRESULTS — {args.agent_path}\n{'='*70}")
    for level in level_results:
        print(f"  UCI_Elo {level['elo']:5d}: {level['wins']:3d}-{level['draws']:3d}-{level['losses']:3d}  ({level['score_pct']:5.1f}% score, {level['games']} games)")
    print(f"\nFitted rating (joint MLE across all {len(observations)} games): {rating:.0f} +/- {ci95:.0f} (95% CI)")
    print("NOTE: this is Stockfish's own UCI_Elo scale, not FIDE/CCRL — treat it as an internal yardstick.")

    if not args.skip_benchmark:
        print(f"\n{'='*70}\nSEARCH BENCHMARK (median over real middlegame positions, 1.5s soft / 4.5s hard)\n{'='*70}")
        from ratings.benchmark import median_middlegame_depth_and_nps

        bench = median_middlegame_depth_and_nps()
        print(f"median depth: {bench['median_depth']}  (range {bench['min_depth']}-{bench['max_depth']})")
        print(f"median nps:   {bench['median_nps']:,.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
