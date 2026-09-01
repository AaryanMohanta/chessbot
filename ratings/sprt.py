"""SPRT: sequential test for "is candidate stronger than baseline", the
tool for day-to-day A/B decisions (the full joint rating fit in elo.py is
for tracking the whole pool over time — different purpose, different tool,
per the design brief).

Model: each game's score s in {0, 0.5, 1} (from the candidate's
perspective) is treated as a Bernoulli-like observation with mean p, where
p0 = P(win | elo0) and p1 = P(win | elo1) come from the standard logistic
Elo model. The per-game log-likelihood-ratio contribution is

    llr += s * ln(p1/p0) + (1-s) * ln((1-p1)/(1-p0))

which is exact for genuine Bernoulli trials and, same as elo.py, treats a
draw as a score of 0.5 rather than modeling draws as their own outcome
with a separate probability (the simpler approximation most SPRT tools for
chess use in practice, not the full trinomial/pentanomial model).

Runs games via harness.match.play_game (alternating colours, cycling the
opening book) until the running LLR crosses one of the two Wald bounds:

    accept H1 (candidate is >= elo1 stronger) when llr >= ln((1-beta)/alpha)
    accept H0 (candidate is <= elo0 stronger) when llr <= ln(beta/(1-alpha))
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harness.match import DEFAULT_INCREMENT_MS, DEFAULT_TIME_MS, play_game  # noqa: E402
from ratings.book import OPENING_BOOK  # noqa: E402


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
    max_games: int = 4000,
    time_ms: float = DEFAULT_TIME_MS,
    increment_ms: float = DEFAULT_INCREMENT_MS,
    candidate_env: dict | None = None,
    baseline_env: dict | None = None,
) -> dict:
    """``candidate_env``/``baseline_env`` let candidate and baseline be the
    *same* agent.py with different environment variables — e.g. isolating
    one search feature (CB_ENABLE_NULL_MOVE, CB_ENABLE_LMR) at a time via
    harness.match.play_game's white_env/black_env, with no need for two
    separate zoo snapshots just to A/B a flag."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="SPRT: is candidate stronger than baseline?")
    parser.add_argument("candidate_path")
    parser.add_argument("baseline_path")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--max-games", type=int, default=4000)
    parser.add_argument("--time-ms", type=float, default=DEFAULT_TIME_MS)
    parser.add_argument("--increment-ms", type=float, default=DEFAULT_INCREMENT_MS)
    args = parser.parse_args()

    outcome = run_sprt(
        args.candidate_path,
        args.baseline_path,
        args.elo0,
        args.elo1,
        alpha=args.alpha,
        beta=args.beta,
        max_games=args.max_games,
        time_ms=args.time_ms,
        increment_ms=args.increment_ms,
    )
    print()
    print(f"DECISION: {outcome['decision']} - {outcome['reason']} after {outcome['games']} games "
          f"(W{outcome['wins']}/D{outcome['draws']}/L{outcome['losses']}, LLR={outcome['llr']:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
