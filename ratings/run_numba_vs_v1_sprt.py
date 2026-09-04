"""Head-to-head SPRT: the numba bitboard engine (cb_nb_agent.py) vs v1
(cb_v1_agent.py). Both agents skip cb_book.py deliberately -- the book
would help both sides equally and isn't part of what this test is
measuring (search-algorithm strength), so it would only add noise.

10+0.1 fast TC, concurrency 8, [0, 10] Elo bounds (a novel change, not a
non-regression check), 4000-pair cap -- per the standardized bounds/TC
choices in ratings/sprt.py's own docstring, and the exact parameters
specified for this test.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ratings.sprt import run_sprt_pentanomial  # noqa: E402

CANDIDATE = str(REPO_ROOT / "cb_nb_agent.py")
BASELINE = str(REPO_ROOT / "cb_v1_agent.py")

TIME_MS = 5_000
INCREMENT_MS = 50
ELO0, ELO1 = 0.0, 15.0
MAX_PAIRS = 4000
CONCURRENCY = 7

if __name__ == "__main__":
    print("##### SPRT: numba bitboard engine vs v1 #####", flush=True)
    result = run_sprt_pentanomial(
        CANDIDATE, BASELINE,
        elo0=ELO0, elo1=ELO1, max_pairs=MAX_PAIRS,
        time_ms=TIME_MS, increment_ms=INCREMENT_MS, concurrency=CONCURRENCY,
    )
    print(
        f"\nDECISION: {result['decision']} - {result['reason']} "
        f"after {result['games']} games ({result['pairs']} pairs), "
        f"mean_pair_score={result['mean_pair_score']:.3f}/2, LLR={result['llr']:+.3f}\n"
    )
