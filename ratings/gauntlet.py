"""Gauntlet runner: plays a version against a configured set of zoo
opponents, alternating colours, drawing start positions from the opening
book, appending every game to the database as it completes.

Resumable: before launching, it loads which (white, black, opening_id)
triples are already in the database and skips those — interrupting and
rerunning the same command picks up where it left off rather than
replaying finished games.

Reuses harness.match.play_game for the actual game play; this module only
adds the pairing/scheduling/bookkeeping layer on top.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.match import DEFAULT_INCREMENT_MS, DEFAULT_TIME_MS, play_game  # noqa: E402
from ratings import zoo  # noqa: E402
from ratings.book import OPENING_BOOK  # noqa: E402
from ratings.db import DB_PATH, GameRecord, append_game, now_iso, played_pairings, result_string  # noqa: E402


def _resolve_opponent(opponent: str) -> str:
    """Opponents are usually zoo tags, but a direct path to any
    wire-protocol-conforming agent.py (e.g. a baseline) also works — the
    zoo is for tracked, rateable candidate versions specifically; fixed
    reference opponents don't have to live there to be usable here."""
    try:
        return zoo.agent_path_for(opponent)
    except FileNotFoundError:
        if Path(opponent).exists():
            return opponent
        raise


def _plan_games(new_tag: str, opponent_tag: str, games_per_opponent: int) -> list[tuple[str, str, str, str]]:
    """[(white_tag, black_tag, opening_id, fen), ...], alternating colours,
    cycling through the opening book for variety."""
    plan = []
    for i in range(games_per_opponent):
        opening_id, _moves, fen = OPENING_BOOK[i % len(OPENING_BOOK)]
        if i % 2 == 0:
            plan.append((new_tag, opponent_tag, opening_id, fen))
        else:
            plan.append((opponent_tag, new_tag, opening_id, fen))
    return plan


def _play_and_record(white_tag, white_path, black_tag, black_path, opening_id, fen, time_ms, increment_ms, db_path):
    result = play_game(white_path, black_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen)
    record = GameRecord(
        timestamp=now_iso(),
        white_version=white_tag,
        black_version=black_tag,
        opening_id=opening_id,
        result=result_string(result.winner),
        termination=result.reason,
        plies=result.plies,
        moves_uci=" ".join(result.moves_uci),
    )
    append_game(record, db_path=db_path)
    return white_tag, black_tag, opening_id, result


def run_gauntlet(
    new_tag: str,
    new_path: str,
    opponent_tags: list[str],
    games_per_opponent: int,
    concurrency: int = 1,
    time_ms: float = DEFAULT_TIME_MS,
    increment_ms: float = DEFAULT_INCREMENT_MS,
    db_path: Path = DB_PATH,
) -> None:
    already = played_pairings(db_path)

    jobs = []
    for opponent_tag in opponent_tags:
        opponent_path = _resolve_opponent(opponent_tag)
        for white_tag, black_tag, opening_id, fen in _plan_games(new_tag, opponent_tag, games_per_opponent):
            if (white_tag, black_tag, opening_id) in already:
                continue
            white_path = new_path if white_tag == new_tag else opponent_path
            black_path = new_path if black_tag == new_tag else opponent_path
            jobs.append((white_tag, white_path, black_tag, black_path, opening_id, fen))

    total_planned = len(opponent_tags) * games_per_opponent
    print(f"gauntlet: {len(jobs)} games to play ({total_planned - len(jobs)} already recorded, resuming)")

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [
            pool.submit(_play_and_record, w, wp, b, bp, oid, fen, time_ms, increment_ms, db_path)
            for w, wp, b, bp, oid, fen in jobs
        ]
        for future in as_completed(futures):
            white_tag, black_tag, opening_id, result = future.result()
            done += 1
            print(
                f"[{done}/{len(jobs)}] {white_tag} vs {black_tag} ({opening_id}) "
                f"-> {result_string(result.winner)} ({result.reason}, {result.plies} plies)",
                flush=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a gauntlet: new version vs. zoo opponents.")
    parser.add_argument("new_tag", help="version tag for the agent under test (need not be in the zoo yet)")
    parser.add_argument("new_path", help="path to the agent.py under test")
    parser.add_argument("--opponents", nargs="+", required=True, help="zoo tags to play against")
    parser.add_argument("--games", type=int, default=20, help="games per opponent (default 20)")
    parser.add_argument("--concurrency", type=int, default=1, help="parallel games (default 1)")
    parser.add_argument("--time-ms", type=float, default=DEFAULT_TIME_MS)
    parser.add_argument("--increment-ms", type=float, default=DEFAULT_INCREMENT_MS)
    args = parser.parse_args()

    run_gauntlet(
        args.new_tag,
        args.new_path,
        args.opponents,
        args.games,
        concurrency=args.concurrency,
        time_ms=args.time_ms,
        increment_ms=args.increment_ms,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
