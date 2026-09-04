"""Runs exactly one calibration game against Stockfish and prints the
result as a single JSON line to stdout.

Exists so a game can be run inside its own isolated process/container --
each invocation of this script is meant to be one `docker run
--cpus=1 --memory=2g ...`, matching the competition's per-process
resource constraint exactly (rather than several games sharing one
container's CPU budget, which is what running ratings/calibrate_stockfish.py
directly inside a single container would do). See
ratings/run_containerized_calibration.py-style orchestration for how many
of these get launched in parallel across separate containers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.match import play_game  # noqa: E402

STOCKFISH_AGENT = str(REPO_ROOT / "baselines" / "stockfish_agent.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent_path")
    parser.add_argument("--stockfish", required=True, help="path to the Stockfish binary")
    parser.add_argument("--elo", type=int, required=True, help="Stockfish UCI_Elo for this game")
    parser.add_argument("--fen", required=True, help="starting position")
    parser.add_argument("--agent-is-white", action="store_true")
    parser.add_argument("--time-ms", type=float, required=True)
    parser.add_argument("--increment-ms", type=float, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=16)
    parser.add_argument("--agent-cpu", type=int, default=None, help="taskset core for our agent")
    parser.add_argument("--stockfish-cpu", type=int, default=None, help="taskset core for Stockfish")
    args = parser.parse_args()

    env = {
        "STOCKFISH_PATH": args.stockfish,
        "STOCKFISH_ELO": str(args.elo),
        "STOCKFISH_THREADS": str(args.threads),
        "STOCKFISH_HASH_MB": str(args.hash_mb),
    }
    if args.agent_is_white:
        result = play_game(args.agent_path, STOCKFISH_AGENT, time_ms=args.time_ms, increment_ms=args.increment_ms, start_fen=args.fen, black_env=env, white_cpu=args.agent_cpu, black_cpu=args.stockfish_cpu)
        our_score = {"white": 1.0, "black": 0.0, None: 0.5}[result.winner]
    else:
        result = play_game(STOCKFISH_AGENT, args.agent_path, time_ms=args.time_ms, increment_ms=args.increment_ms, start_fen=args.fen, white_env=env, white_cpu=args.stockfish_cpu, black_cpu=args.agent_cpu)
        our_score = {"black": 1.0, "white": 0.0, None: 0.5}[result.winner]

    print(json.dumps({"score": our_score, "reason": result.reason, "plies": result.plies}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
