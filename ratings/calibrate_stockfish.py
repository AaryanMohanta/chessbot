"""Calibrate our agent's real-world Elo by playing it against a real
Stockfish binary at several capped strength settings (UCI_LimitStrength +
UCI_Elo as the primary ladder, Skill Level as a cross-check), bisecting
toward whichever setting our agent draws roughly even with.

Local-only: Stockfish never ships in the submission zip (not in
tools/build_zip.py's whitelist) — the third-party-engine ban is on the
zip's contents, not on how we test locally.

Usage:
    python -m ratings.calibrate_stockfish <agent_path> --stockfish PATH
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.match import DEFAULT_INCREMENT_MS, DEFAULT_TIME_MS, play_game  # noqa: E402
from ratings.book import OPENING_BOOK  # noqa: E402

DEFAULT_ELO_LADDER = [1320, 1500, 1700, 1900, 2100, 2400]
DEFAULT_SKILL_LADDER = [0, 3, 6, 10, 14, 18]


def play_leg(agent_path: str, stockfish_path: str, opt_name: str, opt_value, games: int, time_ms: float, increment_ms: float) -> dict:
    env = {"STOCKFISH_PATH": stockfish_path, opt_name: str(opt_value)}
    stockfish_agent = str(REPO_ROOT / "baselines" / "stockfish_agent.py")

    wins = draws = losses = 0
    for i in range(games):
        _opening_id, _moves, fen = OPENING_BOOK[i % len(OPENING_BOOK)]
        if i % 2 == 0:
            result = play_game(agent_path, stockfish_agent, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, black_env=env)
            our_score = {"white": 1.0, "black": 0.0, None: 0.5}[result.winner]
        else:
            result = play_game(stockfish_agent, agent_path, time_ms=time_ms, increment_ms=increment_ms, start_fen=fen, white_env=env)
            our_score = {"black": 1.0, "white": 0.0, None: 0.5}[result.winner]

        if our_score == 1.0:
            wins += 1
        elif our_score == 0.0:
            losses += 1
        else:
            draws += 1
        print(f"  [{opt_name}={opt_value}] game {i+1}/{games}: score={our_score} ({result.reason}, {result.plies} plies)", flush=True)

    score_pct = (wins + 0.5 * draws) / games * 100
    return {"setting": opt_value, "wins": wins, "draws": draws, "losses": losses, "score_pct": score_pct}


def run_ladder(agent_path: str, stockfish_path: str, opt_name: str, ladder: list, games_per_setting: int, time_ms: float, increment_ms: float) -> list[dict]:
    results = []
    for value in ladder:
        print(f"=== {opt_name} = {value} ===", flush=True)
        leg = play_leg(agent_path, stockfish_path, opt_name, value, games_per_setting, time_ms, increment_ms)
        results.append(leg)
        print(f"  -> {leg['wins']}-{leg['draws']}-{leg['losses']} ({leg['score_pct']:.0f}% score)\n", flush=True)
    return results


def estimate_crossover(results: list[dict]) -> str:
    """Where our score_pct crosses 50% is roughly where the opponent's
    setting matches our real strength (a 50% score against a known-Elo
    opponent means, by definition of Elo, an equal rating)."""
    above = [r for r in results if r["score_pct"] >= 50]
    below = [r for r in results if r["score_pct"] < 50]
    if not below:
        return f">= {results[-1]['setting']} (never dropped to 50% score across the tested ladder)"
    if not above:
        return f"<= {results[0]['setting']} (never reached 50% score across the tested ladder)"
    last_above = above[-1]
    first_below = below[0]
    return f"between {last_above['setting']} (scored {last_above['score_pct']:.0f}%) and {first_below['setting']} (scored {first_below['score_pct']:.0f}%)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate agent Elo against a real Stockfish binary.")
    parser.add_argument("agent_path")
    parser.add_argument("--stockfish", required=True, help="path to the Stockfish binary")
    parser.add_argument("--games", type=int, default=8, help="games per strength setting (default 8; 4 was tried "
                         "first and was too noisy to trust for bisection — a 12-game direct v1-vs-v2 gauntlet "
                         "showed 8-4-0 in v2's favor the same day a 4-games-per-level Stockfish ladder implied "
                         "the opposite)")
    parser.add_argument("--elo-ladder", type=int, nargs="+", default=DEFAULT_ELO_LADDER)
    parser.add_argument("--skill-ladder", type=int, nargs="+", default=DEFAULT_SKILL_LADDER)
    parser.add_argument("--time-ms", type=float, default=DEFAULT_TIME_MS)
    parser.add_argument("--increment-ms", type=float, default=DEFAULT_INCREMENT_MS)
    parser.add_argument("--skip-skill-ladder", action="store_true", help="only run the UCI_Elo ladder")
    args = parser.parse_args()

    print("##### UCI_Elo ladder (primary calibration) #####")
    elo_results = run_ladder(args.agent_path, args.stockfish, "STOCKFISH_ELO", args.elo_ladder, args.games, args.time_ms, args.increment_ms)
    print(f"\nEstimated real Elo (UCI_Elo ladder): {estimate_crossover(elo_results)}\n")

    if not args.skip_skill_ladder:
        print("##### Skill Level ladder (cross-check) #####")
        skill_results = run_ladder(args.agent_path, args.stockfish, "STOCKFISH_SKILL", args.skill_ladder, args.games, args.time_ms, args.increment_ms)
        print(f"\nSkill Level crossover (cross-check, not directly Elo): {estimate_crossover(skill_results)}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
