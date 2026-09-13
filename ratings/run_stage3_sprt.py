"""Stage-3 SPRT gate: null-move pruning and LMR, each tested in isolation
against the previous version, per the design doc's requirement that every
pruning technique earn its place with its own SPRT rather than being
bundled in on faith.

Wide elo bounds (0/20) are used deliberately: both techniques are
well-established with an expected effect size the design doc itself prices
at +70/+80 Elo, so a wide bound needs far fewer games to resolve than the
tight 0/5 bound used for genuine close calls — appropriate here given the
wall-clock cost of each game.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ratings.sprt import run_sprt

AGENT = str(REPO_ROOT / "agent.py")
TIME_MS = 20_000
INCREMENT_MS = 200
ELO0, ELO1 = 0.0, 20.0
MAX_GAMES = 100

print("##### SPRT: null-move pruning (LMR off in both, to isolate it) #####", flush=True)
result_a = run_sprt(
    AGENT, AGENT,
    elo0=ELO0, elo1=ELO1, max_games=MAX_GAMES, time_ms=TIME_MS, increment_ms=INCREMENT_MS,
    candidate_env={"CB_ENABLE_NULL_MOVE": "1", "CB_ENABLE_LMR": "0"},
    baseline_env={"CB_ENABLE_NULL_MOVE": "0", "CB_ENABLE_LMR": "0"},
)
print(f"\nNULL-MOVE DECISION: {result_a['decision']} - {result_a['reason']} "
      f"after {result_a['games']} games (W{result_a['wins']}/D{result_a['draws']}/L{result_a['losses']}, LLR={result_a['llr']:+.3f})\n")

print("##### SPRT: LMR (null-move on in both, since it's already validated) #####", flush=True)
result_b = run_sprt(
    AGENT, AGENT,
    elo0=ELO0, elo1=ELO1, max_games=MAX_GAMES, time_ms=TIME_MS, increment_ms=INCREMENT_MS,
    candidate_env={"CB_ENABLE_NULL_MOVE": "1", "CB_ENABLE_LMR": "1"},
    baseline_env={"CB_ENABLE_NULL_MOVE": "1", "CB_ENABLE_LMR": "0"},
)
print(f"\nLMR DECISION: {result_b['decision']} - {result_b['reason']} "
      f"after {result_b['games']} games (W{result_b['wins']}/D{result_b['draws']}/L{result_b['losses']}, LLR={result_b['llr']:+.3f})\n")
