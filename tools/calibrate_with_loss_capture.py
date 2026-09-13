"""Same 3-level Stockfish calibration as ratings/calibrate_stockfish.py,
but additionally saves the full move list + starting FEN + termination
reason for every LOST game to a JSON file, so losses can actually be
inspected afterward -- the original script only ever reports aggregate
scores, nothing per-game.

Usage: python tools/calibrate_with_loss_capture.py <agent_path> <stockfish_path> [out_json]
       [--elo-ladder E1 E2 ...] [--games N] [--time-ms T] [--increment-ms I] [--concurrency C]

Defaults to information-dense levels (near the agent's own strength, not
500+ Elo above it -- see ratings/calibrate_stockfish.py's own "nobody
calibrates at real TC, it's too slow" convention for why fast TC is the
default here too).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.match import play_game  # noqa: E402
from ratings.pgn_book import load_openings  # noqa: E402

STOCKFISH_AGENT = str(REPO_ROOT / "baselines" / "stockfish_agent.py")


def _play_one(agent_path, stockfish_path, elo, opening_id, fen, agent_is_white, time_ms, increment_ms):
    env = {"STOCKFISH_PATH": stockfish_path, "STOCKFISH_ELO": str(elo)}
    if agent_is_white:
        result = play_game(agent_path, STOCKFISH_AGENT, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, black_env=env)
        agent_score = {"white": 1.0, "black": 0.0, None: 0.5}[result.winner]
    else:
        result = play_game(STOCKFISH_AGENT, agent_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, white_env=env)
        agent_score = {"black": 1.0, "white": 0.0, None: 0.5}[result.winner]
    return elo, opening_id, fen, agent_is_white, agent_score, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_path", nargs="?", default="agent.py")
    parser.add_argument("stockfish_path")
    parser.add_argument("out_json", nargs="?", default=None)
    parser.add_argument("--elo-ladder", type=int, nargs="+", default=[1900, 2100])
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--time-ms", type=float, default=10_000)
    parser.add_argument("--increment-ms", type=float, default=100)
    parser.add_argument("--concurrency", type=int, default=7)
    args = parser.parse_args()

    agent_path = args.agent_path
    stockfish_path = args.stockfish_path
    out_json = Path(args.out_json) if args.out_json else REPO_ROOT / "ratings" / "calibration_losses.json"

    openings = load_openings(limit=max(300, args.games))
    losses = []
    summary = {}

    for elo in args.elo_ladder:
        print(f"=== UCI_Elo = {elo} ===", flush=True)
        jobs = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for i in range(args.games):
                opening_id, fen = openings[i % len(openings)]
                agent_is_white = i % 2 == 0
                jobs.append(pool.submit(_play_one, agent_path, stockfish_path, elo, opening_id, fen, agent_is_white, args.time_ms, args.increment_ms))

            total = 0.0
            wins = draws = plosses = 0
            done = 0
            for future in as_completed(jobs):
                elo_, opening_id, fen, agent_is_white, score, result = future.result()
                total += score
                done += 1
                if score == 1.0:
                    wins += 1
                elif score == 0.5:
                    draws += 1
                else:
                    plosses += 1
                    losses.append({
                        "elo": elo_,
                        "opening_id": opening_id,
                        "start_fen": fen,
                        "agent_color": "white" if agent_is_white else "black",
                        "reason": result.reason,
                        "plies": result.plies,
                        "moves_uci": result.moves_uci,
                    })
                if done % 20 == 0 or done == args.games:
                    print(f"  [{elo}] {done}/{args.games} done, running score {total:.1f}/{done} ({100*total/done:.1f}%)", flush=True)

        summary[elo] = {"score": total, "games": args.games, "wins": wins, "draws": draws, "losses": plosses}
        print(f"  -> {wins}-{plosses}-{draws} ({100*total/args.games:.1f}%)\n", flush=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "losses": losses}, f, indent=2)
    print(f"Wrote {len(losses)} loss record(s) to {out_json}")
    print("\nFinal summary:")
    for elo, s in summary.items():
        print(f"  {elo}: {s['wins']}-{s['losses']}-{s['draws']} ({100*s['score']/s['games']:.1f}%)")


if __name__ == "__main__":
    main()
